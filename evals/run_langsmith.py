import os
import re
import tempfile
import uuid

from langsmith import Client, evaluate

from backend.agents import run_assistant
from backend.knowledge import delete_document_chunks, ingest_file

DATASET_NAME = "knowledge-assistant-quality-v1"
EVALUATION_USER_ID = "langsmith-evaluation"
EVALUATION_DOCUMENT_ID = "langsmith-synthetic-policy"
CASES = [
    {
        "inputs": {"question": "Calculate 19 * 23"},
        "outputs": {"answer": "437", "route": "tool"},
    },
    {
        "inputs": {"question": "What time is it now?"},
        "outputs": {"answer": "UTC", "route": "tool"},
    },
    {
        "inputs": {"question": "According to the annual leave policy, how many paid days are granted?"},
        "outputs": {
            "answer": "15 paid days per calendar year",
            "route": "rag",
            "expected_source": "evaluation-policy.txt",
        },
    },
]


def answer_relevance(run, example) -> dict:
    answer = str((run.outputs or {}).get("answer", "")).lower()
    expected = str((example.outputs or {}).get("answer", "")).lower()
    expected_tokens = set(re.findall(r"[a-z0-9]+", expected))
    answer_tokens = set(re.findall(r"[a-z0-9]+", answer))
    score = len(expected_tokens & answer_tokens) / max(len(expected_tokens), 1)
    return {"key": "expected_fact_recall", "score": score}


def faithfulness(run, example) -> dict:
    answer = str((run.outputs or {}).get("answer", "")).lower()
    expected = str((example.outputs or {}).get("answer", "")).lower()
    facts = set(re.findall(r"[a-z0-9]+", expected))
    answer_tokens = set(re.findall(r"[a-z0-9]+", answer))
    score = len(facts & answer_tokens) / max(len(facts), 1)
    return {"key": "expected_fact_faithfulness", "score": score}


def tool_selection_accuracy(run, example) -> dict:
    expected_route = (example.outputs or {}).get("route")
    if expected_route is None:
        return {"key": "tool_selection_accuracy", "value": "not_applicable"}
    return {
        "key": "tool_selection_accuracy",
        "score": float((run.outputs or {}).get("route") == expected_route),
    }


def context_relevance(run, example) -> dict:
    expected_route = (example.outputs or {}).get("route")
    if expected_route != "rag":
        return {"key": "context_relevance", "value": "not_applicable"}
    question = str((example.inputs or {}).get("question", "")).lower()
    citations = (run.outputs or {}).get("citations", [])
    excerpts = " ".join(str(item.get("excerpt", "")) for item in citations).lower()
    query_tokens = set(re.findall(r"[a-z0-9]{3,}", question))
    context_tokens = set(re.findall(r"[a-z0-9]{3,}", excerpts))
    score = len(query_tokens & context_tokens) / max(len(query_tokens), 1)
    return {"key": "context_relevance", "score": score}


def retrieval_quality(run, example) -> dict:
    expected_source = (example.outputs or {}).get("expected_source")
    if expected_source is None:
        return {"key": "retrieval_source_match", "value": "not_applicable"}
    actual_sources = {
        item.get("source") for item in (run.outputs or {}).get("citations", [])
    }
    return {"key": "retrieval_source_match", "score": float(expected_source in actual_sources)}


def target(inputs: dict) -> dict:
    result = run_assistant(
        inputs["question"],
        user_id=EVALUATION_USER_ID,
        conversation_id=str(uuid.uuid4()),
    )
    return {
        "answer": result.get("answer", ""),
        "route": result.get("route", ""),
        "citations": result.get("citations", []),
    }


def main() -> None:
    if not os.getenv("LANGSMITH_API_KEY") and not os.getenv("LANGCHAIN_API_KEY"):
        raise RuntimeError("Set LANGSMITH_API_KEY to create and run LangSmith evaluations.")
    client = Client()
    datasets = list(client.list_datasets(dataset_name=DATASET_NAME))
    if datasets:
        dataset = datasets[0]
    else:
        dataset = client.create_dataset(DATASET_NAME, description="Basic answer correctness smoke tests.")
        client.create_examples(
            dataset_id=dataset.id,
            examples=[{"inputs": case["inputs"], "outputs": case["outputs"]} for case in CASES],
        )
    temporary_path = None
    indexed = False
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".txt", delete=False) as file:
            file.write("Annual leave policy: full-time employees receive 15 paid days per calendar year.")
            temporary_path = file.name
        ingest_file(
            temporary_path,
            "evaluation-policy.txt",
            EVALUATION_USER_ID,
            EVALUATION_DOCUMENT_ID,
        )
        indexed = True
        evaluate(
            target,
            data=dataset.name,
            evaluators=[
                answer_relevance,
                faithfulness,
                tool_selection_accuracy,
                context_relevance,
                retrieval_quality,
            ],
            experiment_prefix="assistant-quality",
            metadata={"evaluation": "synthetic smoke set; add representative reviewed examples"},
        )
    finally:
        if indexed:
            delete_document_chunks(EVALUATION_DOCUMENT_ID, EVALUATION_USER_ID)
        if temporary_path is not None:
            os.unlink(temporary_path)


if __name__ == "__main__":
    main()
