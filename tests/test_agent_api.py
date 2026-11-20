import json

import pytest
from fastapi.testclient import TestClient

from mini_dify.llm.embeddings import FakeEmbedding
from mini_dify.llm.fake import ScriptedProvider
from mini_dify.llm.registry import get_embedder, get_provider
from mini_dify.llm.types import ToolCall
from mini_dify.main import app

MULTIPLY = [ToolCall(name="calculator", arguments={"expression": "6 * 7"})]


@pytest.fixture
def provider():
    # Each test sets the script it needs with provider.replies = [...]
    return ScriptedProvider([])


@pytest.fixture
def client(clean_db, provider):
    embedder = FakeEmbedding()
    app.dependency_overrides[get_provider] = lambda: provider
    app.dependency_overrides[get_embedder] = lambda: embedder
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def create_agent_app(client, **fields):
    body = {"name": "Agent", "system_prompt": "You are helpful.", "tools": ["calculator"]}
    response = client.post("/v1/apps", json={**body, **fields})
    assert response.status_code == 201, response.text
    return response.json()


def test_list_tools(client):
    tools = client.get("/v1/tools").json()
    assert [tool["name"] for tool in tools] == ["calculator", "current_time", "wikipedia"]
    assert tools[0]["parameters"]["required"] == ["expression"]


def test_unknown_tool_is_rejected(client):
    response = client.post("/v1/apps", json={"name": "x", "tools": ["rm_rf"]})
    assert response.status_code == 422
    assert "rm_rf" in response.json()["detail"]


def test_agent_app_answers_with_steps_and_saves_them(client, provider):
    created = create_agent_app(client)
    assert created["agent_mode"] == "function_calling"
    assert created["max_iterations"] == 5

    provider.replies = [MULTIPLY, "6 * 7 is 42."]
    response = client.post(
        f"/v1/apps/{created['id']}/chat-messages", json={"query": "What is 6 * 7?"}
    )
    body = response.json()

    assert response.status_code == 200
    assert body["answer"] == "6 * 7 is 42."
    assert body["agent_steps"][0]["tool"] == "calculator"
    assert body["agent_steps"][0]["observation"] == "6 * 7 = 42"
    assert body["usage"] == {"prompt_tokens": 20, "completion_tokens": 10}

    messages = client.get(f"/v1/conversations/{body['conversation_id']}/messages").json()
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[0]["agent_steps"] is None
    assert messages[1]["agent_steps"][0]["arguments"] == {"expression": "6 * 7"}


def test_react_mode(client, provider):
    created = create_agent_app(client, agent_mode="react")
    provider.replies = [
        'Thought: multiply\nAction: calculator\nAction Input: {"expression": "6 * 7"}',
        "Final Answer: 42",
    ]
    body = client.post(
        f"/v1/apps/{created['id']}/chat-messages", json={"query": "What is 6 * 7?"}
    ).json()
    assert body["answer"] == "42"
    assert body["agent_steps"][0]["thought"] == "multiply"


def test_streamed_agent_sends_steps_then_answer(client, provider):
    created = create_agent_app(client)
    provider.replies = [MULTIPLY, "42."]
    with client.stream(
        "POST",
        f"/v1/apps/{created['id']}/chat-messages",
        json={"query": "What is 6 * 7?", "stream": True},
    ) as response:
        lines = [line.removeprefix("data: ") for line in response.iter_lines() if line]

    assert lines[-1] == "[DONE]"
    events = [json.loads(line) for line in lines[:-1]]
    assert "conversation_id" in events[0]
    assert events[1]["agent_step"]["observation"] == "6 * 7 = 42"
    assert events[2]["delta"] == "42." and events[2]["done"] is True
    assert "message_id" in events[3]


def test_agent_with_dataset_gets_knowledge_search(client, provider):
    dataset = client.post("/v1/datasets", json={"name": "Docs", "chunk_size": 200}).json()
    client.post(
        f"/v1/datasets/{dataset['id']}/documents",
        files={"file": ("faq.txt", b"Support is open Monday to Friday, 9am to 5pm.")},
    )
    created = create_agent_app(client, dataset_ids=[dataset["id"]])

    provider.replies = [
        [ToolCall(name="knowledge_search", arguments={"query": "support hours"})],
        "Monday to Friday, 9 to 5 [1].",
    ]
    body = client.post(
        f"/v1/apps/{created['id']}/chat-messages", json={"query": "When is support open?"}
    ).json()

    offered = [tool["function"]["name"] for tool in provider.calls[0]["tools"]]
    assert offered == ["calculator", "knowledge_search"]
    assert body["agent_steps"][0]["observation"].startswith("[1] (from faq.txt)")
    assert body["citations"][0]["number"] == 1
    assert body["citations"][0]["filename"] == "faq.txt"
