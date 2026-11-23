from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


Route = Literal["direct", "rag", "tool", "web", "sql"]


class AssistantState(TypedDict, total=False):
    question: str
    user_id: str
    conversation_id: str
    route: Route
    rewritten_query: str
    documents: list
    memories: list[str]
    plan: str
    planned_tools: list[str]
    tool_results: list[str]
    tool_result: str
    iteration: int
    answer: str
    citations: list[dict[str, Any]]
    messages: Annotated[list[AnyMessage], add_messages]
