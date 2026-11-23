from collections.abc import Generator
from threading import Event

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessageChunk
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from backend.db import Base, get_db
from backend.main import app
from backend.api.routes import _active_streams, _stream_lock
from backend import main
from backend.config import Settings
from backend.models import Conversation, Memory, User
from backend.security import create_access_token, hash_password, get_current_user


@pytest.fixture
def api_client(
    tmp_path,
) -> Generator[tuple[TestClient, tuple[User, User, str, list[User], sessionmaker]], None, None]:
    engine = create_engine(f"sqlite:///{tmp_path / 'api-test.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as session:
        users = [
            User(email="one@example.test", password_hash=hash_password("test-password-one")),
            User(email="two@example.test", password_hash=hash_password("test-password-two")),
        ]
        session.add_all(users)
        session.commit()
        for user in users:
            session.refresh(user)
        session.add(Memory(user_id=users[0].id, content="Prefers concise answers", category="preference"))
        session.add(Conversation(user_id=users[0].id, title="Private chat", messages=[]))
        session.commit()
        conversation_id = session.scalar(
            select(Conversation.id).where(Conversation.user_id == users[0].id)
        )

    active_user = [users[0]]

    def override_db() -> Generator[Session, None, None]:
        with factory() as session:
            yield session

    def override_user() -> User:
        return active_user[0]

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_current_user] = override_user
    client = TestClient(app)
    try:
        yield client, (users[0], users[1], conversation_id, active_user, factory)
    finally:
        app.dependency_overrides.clear()
        engine.dispose()


def test_memory_listing_is_user_isolated(api_client) -> None:
    client, (_, other_user, _, active_user, _) = api_client

    assert len(client.get("/api/memories").json()) == 1
    active_user[0] = other_user
    assert client.get("/api/memories").json() == []


def test_conversation_lookup_does_not_cross_user_boundary(api_client) -> None:
    client, (_, other_user, conversation_id, active_user, _) = api_client
    active_user[0] = other_user

    assert client.get(f"/api/conversations/{conversation_id}").status_code == 404


def test_chat_requires_authentication(api_client) -> None:
    client, _ = api_client
    app.dependency_overrides.pop(get_current_user)
    response = client.post("/api/chat", json={"question": "Hello there"})

    assert response.status_code == 401


def test_registration_and_login_return_access_tokens(api_client) -> None:
    client, _ = api_client
    credentials = {"email": "new@example.test", "password": "a-strong-password"}

    registered = client.post("/api/auth/register", json=credentials)
    logged_in = client.post("/api/auth/token", json=credentials)

    assert registered.status_code == 201
    assert logged_in.status_code == 200
    assert registered.json()["access_token"]
    assert logged_in.json()["access_token"]


def test_signed_token_authenticates_the_token_owner(api_client) -> None:
    client, (owner, _, _, _, _) = api_client
    app.dependency_overrides.pop(get_current_user)
    token = create_access_token(owner.id)

    authorized = client.get("/api/memories", headers={"Authorization": f"Bearer {token}"})
    rejected = client.get("/api/memories", headers={"Authorization": "Bearer invalid"})

    assert authorized.status_code == 200
    assert len(authorized.json()) == 1
    assert rejected.status_code == 401


def test_stream_endpoint_emits_tokens_and_citations(api_client, monkeypatch) -> None:
    client, (_, _, _, _, factory) = api_client

    class FakeGraph:
        def stream(self, _input, config, stream_mode):
            assert stream_mode == "messages"
            yield AIMessageChunk(content="Grounded "), {"node": "response_generation"}
            yield AIMessageChunk(content="answer."), {"node": "response_generation"}

        def get_state(self, _config):
            class State:
                values = {"answer": "Grounded answer.", "citations": [{"source": "policy.txt", "excerpt": "Evidence"}]}

            return State()

    monkeypatch.setattr("backend.api.routes.get_graph", lambda: FakeGraph())
    monkeypatch.setattr("backend.db.get_session_factory", lambda: factory)
    response = client.post(
        "/api/chat/stream",
        json={"question": "What does this document say?"},
    )

    assert response.status_code == 200
    assert "Grounded " in response.text
    assert "answer." in response.text
    assert "policy.txt" in response.text
    assert "event: done" in response.text


def test_stop_endpoint_signals_only_an_owned_conversation(api_client) -> None:
    client, (owner, other_user, conversation_id, active_user, _) = api_client
    thread_id = f"{owner.id}:{conversation_id}"
    event = Event()
    with _stream_lock:
        _active_streams[thread_id] = event

    active_user[0] = other_user
    denied = client.post(f"/api/conversations/{conversation_id}/stop")
    active_user[0] = owner
    stopped = client.post(f"/api/conversations/{conversation_id}/stop")

    with _stream_lock:
        _active_streams.pop(thread_id, None)
    assert denied.status_code == 404
    assert stopped.status_code == 202
    assert event.is_set()


def test_application_lifespan_initializes_database(tmp_path, monkeypatch) -> None:
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'startup.db'}",
        checkpoint_db_url=f"sqlite:///{tmp_path / 'checkpoints.db'}",
        environment="test",
    )
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr("backend.db.get_settings", lambda: settings)
    main.get_redis.cache_clear()
    from backend.db import get_engine, get_session_factory

    get_engine.cache_clear()
    get_session_factory.cache_clear()
    with TestClient(app) as client:
        assert client.get("/api/health").json() == {"status": "ok"}
    get_engine.cache_clear()
    get_session_factory.cache_clear()
