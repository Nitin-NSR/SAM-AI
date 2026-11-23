from backend.agents import route_question


def test_routes_current_questions_to_web() -> None:
    assert route_question({"question": "What is the latest company news?"})["route"] == "web"


def test_routes_metrics_to_sql() -> None:
    assert route_question({"question": "What is our headcount?"})["route"] == "sql"


def test_routes_policy_questions_to_rag() -> None:
    assert route_question({"question": "What is the parental leave policy?"})["route"] == "rag"


def test_rag_route_is_preserved_for_follow_up_questions() -> None:
    assert route_question({"question": "What about contractors?", "route": "rag"})["route"] == "rag"
