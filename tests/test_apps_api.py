import json

import pytest
from fastapi.testclient import TestClient

from mini_dify.llm.fake import FakeProvider
from mini_dify.llm.registry import get_provider
from mini_dify.main import app


class RecordingProvider(FakeProvider):
    """The fake provider, but it also remembers every prompt it was sent."""

    def __init__(self):
        self.prompts = []

    async def stream_chat(self, model, messages, temperature=0.7, max_tokens=None):
        self.prompts.append([(m.role, m.content) for m in messages])
        async for chunk in super().stream_chat(model, messages, temperature, max_tokens):
            yield chunk


@pytest.fixture
def provider():
    return RecordingProvider()


@pytest.fixture
def client(clean_db, provider):
    app.dependency_overrides[get_provider] = lambda: provider
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def create_app(client, **fields):
    body = {"name": "test app", **fields}
    response = client.post("/v1/apps", json=body)
    assert response.status_code == 201, response.text
    return response.json()


def test_create_app_extracts_variables(client):
    created = create_app(client, system_prompt="You help {{name}} learn {{topic}}.")

    assert created["variables"] == ["name", "topic"]
    assert client.get(f"/v1/apps/{created['id']}").json()["name"] == "test app"
    assert [a["id"] for a in client.get("/v1/apps").json()] == [created["id"]]


def test_unknown_app_is_404(client):
    response = client.get("/v1/apps/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404


def test_missing_inputs_are_rejected(client):
    created = create_app(client, system_prompt="You help {{name}}.")
    response = client.post(f"/v1/apps/{created['id']}/chat-messages", json={"query": "hi"})
    assert response.status_code == 422
    assert "name" in response.json()["detail"]


def test_chat_saves_messages_and_remembers_history(client, provider):
    created = create_app(client, system_prompt="You help {{name}}.")
    url = f"/v1/apps/{created['id']}/chat-messages"

    first = client.post(url, json={"query": "hello", "inputs": {"name": "Abhi"}}).json()
    assert first["answer"] == "echo: hello"

    second = client.post(url, json={"query": "again", "conversation_id": first["conversation_id"]})
    assert second.status_code == 200
    assert second.json()["conversation_id"] == first["conversation_id"]

    # The second prompt contains the system prompt, the first exchange, then the new question.
    assert provider.prompts[1] == [
        ("system", "You help Abhi."),
        ("user", "hello"),
        ("assistant", "echo: hello"),
        ("user", "again"),
    ]

    history = client.get(f"/v1/conversations/{first['conversation_id']}/messages").json()
    assert [(m["role"], m["content"]) for m in history] == [
        ("user", "hello"),
        ("assistant", "echo: hello"),
        ("user", "again"),
        ("assistant", "echo: again"),
    ]


def test_memory_window_limits_history(client, provider):
    created = create_app(client, memory_window=2)
    url = f"/v1/apps/{created['id']}/chat-messages"

    conversation_id = client.post(url, json={"query": "one"}).json()["conversation_id"]
    client.post(url, json={"query": "two", "conversation_id": conversation_id})
    client.post(url, json={"query": "three", "conversation_id": conversation_id})

    # Only the last 2 stored messages (the "two" exchange) are sent, plus the new question.
    assert provider.prompts[2] == [
        ("user", "two"),
        ("assistant", "echo: two"),
        ("user", "three"),
    ]


def test_streamed_chat_sends_ids_and_saves_messages(client):
    created = create_app(client)
    url = f"/v1/apps/{created['id']}/chat-messages"

    with client.stream("POST", url, json={"query": "hi there", "stream": True}) as response:
        assert response.status_code == 200
        events = [line.removeprefix("data: ") for line in response.iter_lines() if line]

    assert events[-1] == "[DONE]"
    payloads = [json.loads(e) for e in events[:-1]]
    conversation_id = payloads[0]["conversation_id"]
    assert "message_id" in payloads[-1]
    assert "".join(p.get("delta", "") for p in payloads) == "echo: hi there"

    history = client.get(f"/v1/conversations/{conversation_id}/messages").json()
    assert [m["content"] for m in history] == ["hi there", "echo: hi there"]
