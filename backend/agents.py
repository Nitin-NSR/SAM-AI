"""Compatibility exports for the original starter agent module."""

from .graph.nodes import classify_intent
from .graph.state import AssistantState
from .graph.workflow import get_graph, run_assistant as run_persistent_assistant
from langgraph.graph.state import CompiledStateGraph


def route_question(state: AssistantState) -> AssistantState:
    return classify_intent(state)


def build_graph() -> CompiledStateGraph:
    return get_graph()


def run_assistant(
    question: str,
    user_id: str = "anonymous",
    conversation_id: str = "default",
) -> AssistantState:
    return run_persistent_assistant(question, user_id, conversation_id)
