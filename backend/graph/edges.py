from .state import AssistantState


def route_after_memory(state: AssistantState) -> str:
    return state.get("route", "direct")


def route_after_planning(state: AssistantState) -> str:
    route = state.get("route", "direct")
    return {
        "direct": "response_generation",
        "rag": "response_generation",
        "tool": "tool_execution",
        "web": "web_search",
        "sql": "database_tool",
    }[route]


def route_after_tool_evaluation(state: AssistantState) -> str:
    iteration = state.get("iteration", 0)
    if state.get("route") == "tool" and iteration < min(len(state.get("planned_tools", [])), 3):
        return "tool_execution"
    return "response_generation"
