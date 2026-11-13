import json

import pytest
from fastapi.testclient import TestClient

from mini_dify.llm.fake import FakeProvider
from mini_dify.llm.registry import get_provider
from mini_dify.main import app


@pytest.fixture
def client():
    # Swap the real provider for the fake one, for this test only.
    app.dependency_overrides[get_provider] = FakeProvider
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_chat_returns_full_reply(client):
    response = client.post("/v1/chat", json={"messages": [{"role": "user", "content": "hi"}]})

    assert response.status_code == 200
    body = response.json()
    assert body["content"] == "echo: hi"
    assert body["finish_reason"] == "stop"
    assert body["usage"]["completion_tokens"] == 2


def test_chat_streams_server_sent_events(client):
    payload = {"messages": [{"role": "user", "content": "hello world"}], "stream": True}

    with client.stream("POST", "/v1/chat", json=payload) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = [line.removeprefix("data: ") for line in response.iter_lines() if line]

    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    assert "".join(c.get("delta", "") for c in chunks) == "echo: hello world"
    assert chunks[-1]["done"] is True


def test_chat_rejects_empty_messages(client):
    response = client.post("/v1/chat", json={"messages": []})
    assert response.status_code == 422
