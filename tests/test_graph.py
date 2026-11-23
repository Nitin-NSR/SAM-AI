from backend import db
from backend.config import Settings
from backend.graph import nodes, workflow
from backend import llm
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage


def prepare_isolated_graph(tmp_path, monkeypatch) -> None:
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'assistant.db'}",
        checkpoint_db_url=f"sqlite:///{tmp_path / 'checkpoints.db'}",
        environment="test",
    )
    monkeypatch.setattr(db, "get_settings", lambda: settings)
    monkeypatch.setattr(nodes, "get_settings", lambda: settings)
    monkeypatch.setattr(workflow, "get_settings", lambda: settings)
    db.get_engine.cache_clear()
    db.get_session_factory.cache_clear()
    workflow.get_checkpointer.cache_clear()
    workflow.get_graph.cache_clear()
    db.create_database()


def test_graph_preserves_thread_history_without_paid_model_calls(tmp_path, monkeypatch) -> None:
    prepare_isolated_graph(tmp_path, monkeypatch)
    monkeypatch.setattr(nodes, "invoke_prompt", lambda _system, _user: "Mock answer.")

    workflow.run_assistant("What should I call you?", "user-1", "thread-1")
    result = workflow.run_assistant("And what did I ask?", "user-1", "thread-1")

    assert result["answer"] == "Mock answer."
    assert len(result["messages"]) == 4
    assert result["messages"][-2].content == "And what did I ask?"
    workflow.delete_checkpoint("user-1", "thread-1")
    assert not workflow.get_graph().get_state(
        {"configurable": {"thread_id": "user-1:thread-1"}}
    ).values


def test_agent_tool_loop_is_bounded_and_runs_planned_tools(tmp_path, monkeypatch) -> None:
    prepare_isolated_graph(tmp_path, monkeypatch)
    monkeypatch.setattr(nodes, "invoke_prompt", lambda _system, user: user)

    result = workflow.run_assistant(
        "Calculate 2 + 3 and tell me the current time",
        "user-1",
        "thread-tools",
    )

    assert result["iteration"] == 2
    assert result["tool_results"][0] == "calculator: 5"
    assert result["tool_results"][1].startswith("clock: Current UTC time:")


def test_database_tool_uses_only_fixed_user_scoped_counts(tmp_path, monkeypatch) -> None:
    prepare_isolated_graph(tmp_path, monkeypatch)

    metrics = nodes.execute_database_tool(
        {"question": "Count my documents", "user_id": "user-1"}
    )
    restricted = nodes.execute_database_tool(
        {"question": "What is the employee headcount?", "user_id": "user-1"}
    )

    assert metrics["tool_result"] == "You have 0 indexed documents in your account."
    assert "does not have access to company HR" in restricted["tool_result"]


def test_graph_streams_fake_model_chunks(tmp_path, monkeypatch) -> None:
    prepare_isolated_graph(tmp_path, monkeypatch)
    monkeypatch.setattr(llm, "get_chat_model", lambda: FakeListChatModel(responses=["Mock streamed answer."]))
    config = {"configurable": {"thread_id": "user-1:stream-test"}}
    chunks = list(
        workflow.get_graph().stream(
            {
                "question": "Explain a general concept",
                "user_id": "user-1",
                "messages": [HumanMessage(content="Explain a general concept")],
            },
            config=config,
            stream_mode="messages",
        )
    )

    text = "".join(
        message.content
        for message, _metadata in chunks
        if isinstance(message.content, str)
    )
    assert "Mock streamed answer." in text
