import sqlite3
from functools import lru_cache
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy.engine import make_url

from ..config import get_settings
from .edges import (
    route_after_memory,
    route_after_planning,
    route_after_tool_evaluation,
)
from .nodes import (
    classify_intent,
    execute_database_tool,
    execute_tool,
    execute_web_search,
    evaluate_tool_result,
    final_output,
    generate_response,
    plan_task,
    process_input,
    retrieve_memory,
    retrieve_rag,
    rewrite_query,
    validate_response,
)
from .state import AssistantState


@lru_cache
def get_checkpointer() -> SqliteSaver:
    url = make_url(get_settings().checkpoint_db_url)
    if not url.drivername.startswith("sqlite"):
        raise ValueError("The LangGraph checkpoint store must be a SQLite URL.")
    database = url.database or ":memory:"
    if database != ":memory:":
        Path(database).parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database, check_same_thread=False)
    saver = SqliteSaver(connection)
    saver.setup()
    return saver


@lru_cache
def get_graph() -> CompiledStateGraph:
    graph = StateGraph(AssistantState)
    graph.add_node("input_processing", process_input)
    graph.add_node("intent_classification", classify_intent)
    graph.add_node("memory_retrieval", retrieve_memory)
    graph.add_node("query_rewriting", rewrite_query)
    graph.add_node("rag_retrieval", retrieve_rag)
    graph.add_node("agent_planning", plan_task)
    graph.add_node("tool_execution", execute_tool)
    graph.add_node("web_search", execute_web_search)
    graph.add_node("database_tool", execute_database_tool)
    graph.add_node("tool_evaluation", evaluate_tool_result)
    graph.add_node("response_generation", generate_response)
    graph.add_node("response_validation", validate_response)
    graph.add_node("final_output", final_output)

    graph.set_entry_point("input_processing")
    graph.add_edge("input_processing", "intent_classification")
    graph.add_edge("intent_classification", "memory_retrieval")
    graph.add_conditional_edges(
        "memory_retrieval",
        route_after_memory,
        {
            "direct": "response_generation",
            "rag": "query_rewriting",
            "tool": "agent_planning",
            "web": "agent_planning",
            "sql": "agent_planning",
        },
    )
    graph.add_edge("query_rewriting", "rag_retrieval")
    graph.add_edge("rag_retrieval", "agent_planning")
    graph.add_conditional_edges(
        "agent_planning",
        route_after_planning,
        {
            "tool_execution": "tool_execution",
            "web_search": "web_search",
            "database_tool": "database_tool",
            "response_generation": "response_generation",
        },
    )
    for node in ("tool_execution", "web_search", "database_tool"):
        graph.add_edge(node, "tool_evaluation")
    graph.add_conditional_edges(
        "tool_evaluation",
        route_after_tool_evaluation,
        {"tool_execution": "tool_execution", "response_generation": "response_generation"},
    )
    graph.add_edge("response_generation", "response_validation")
    graph.add_edge("response_validation", "final_output")
    graph.add_edge("final_output", END)
    return graph.compile(checkpointer=get_checkpointer())


def run_assistant(question: str, user_id: str, conversation_id: str) -> AssistantState:
    config = {"configurable": {"thread_id": f"{user_id}:{conversation_id}"}}
    return get_graph().invoke(
        {
            "question": question,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "messages": [HumanMessage(content=question)],
        },
        config=config,
    )


def delete_checkpoint(user_id: str, conversation_id: str) -> None:
    get_checkpointer().delete_thread(f"{user_id}:{conversation_id}")
