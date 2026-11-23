from types import SimpleNamespace

from evals.run_langsmith import (
    answer_relevance,
    context_relevance,
    faithfulness,
    retrieval_quality,
    tool_selection_accuracy,
)


def test_evaluators_score_answer_route_context_and_source() -> None:
    run = SimpleNamespace(
        outputs={
            "answer": "The policy provides 15 paid days per calendar year.",
            "route": "rag",
            "citations": [
                {
                    "source": "evaluation-policy.txt",
                    "excerpt": "Annual leave policy gives 15 paid days each calendar year.",
                }
            ],
        }
    )
    example = SimpleNamespace(
        inputs={"question": "How many annual leave days are paid?"},
        outputs={
            "answer": "15 paid days per calendar year",
            "route": "rag",
            "expected_source": "evaluation-policy.txt",
        },
    )

    assert answer_relevance(run, example)["score"] > 0.5
    assert faithfulness(run, example)["score"] > 0.5
    assert tool_selection_accuracy(run, example)["score"] == 1.0
    assert context_relevance(run, example)["score"] > 0
    assert retrieval_quality(run, example)["score"] == 1.0
