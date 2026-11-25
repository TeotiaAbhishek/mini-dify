import json

import pytest
from fastapi.testclient import TestClient

from mini_dify.config import settings
from mini_dify.llm.embeddings import FakeEmbedding
from mini_dify.llm.fake import FakeProvider, ScriptedProvider
from mini_dify.llm.registry import get_embedder, get_provider
from mini_dify.llm.types import ProviderError, ToolCall
from mini_dify.main import app


class BrokenProvider(FakeProvider):
    """A provider that is always down."""

    async def stream_chat(self, model, messages, temperature=0.7, max_tokens=None, tools=None):
        raise ProviderError("model is down")
        yield  # makes this an async generator


@pytest.fixture
def provider():
    return FakeProvider()


@pytest.fixture
def client(clean_db, provider):
    embedder = FakeEmbedding()
    app.dependency_overrides[get_provider] = lambda: provider
    app.dependency_overrides[get_embedder] = lambda: embedder
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def create_app(client, **fields):
    response = client.post("/v1/apps", json={"name": "Helper", **fields})
    assert response.status_code == 201, response.text
    return response.json()


def ask(client, app_id, query="How many vacation days?"):
    response = client.post(f"/v1/apps/{app_id}/chat-messages", json={"query": query})
    assert response.status_code == 200, response.text
    return response.json()


def kinds(node):
    """The tree as nested (kind, name) pairs, easy to compare."""
    return [(child["kind"], child["name"]) for child in node["children"]]


def test_chat_with_knowledge_is_traced(client):
    dataset = client.post("/v1/datasets", json={"name": "Handbook"}).json()
    client.post(
        f"/v1/datasets/{dataset['id']}/documents",
        files={"file": ("leave.txt", b"Every employee gets 25 vacation days per year.")},
    )
    created = create_app(client, dataset_ids=[dataset["id"]])

    body = ask(client, created["id"])
    assert body["trace_id"]

    trace = client.get(f"/v1/traces/{body['trace_id']}").json()
    assert trace["name"] == "chat"
    assert trace["status"] == "ok"
    assert trace["inputs"]["query"] == "How many vacation days?"
    assert trace["span_count"] == 3
    assert trace["llm_calls"] == 1
    assert trace["completion_tokens"] > 0

    root = trace["root"]
    assert root["outputs"]["answer"] == body["answer"]
    assert [kind for kind, _ in kinds(root)] == ["retriever", "llm"]
    retriever, llm = root["children"]
    assert retriever["outputs"]["chunks"][0]["filename"] == "leave.txt"
    assert llm["attributes"]["provider"] == "fake"
    assert llm["outputs"]["content"] == body["answer"]
    assert llm["inputs"]["messages"][-1]["content"].startswith("How many")

    # The saved message points at the same trace.
    messages = client.get(f"/v1/conversations/{body['conversation_id']}/messages").json()
    assert messages[1]["trace_id"] == body["trace_id"]


@pytest.mark.parametrize(
    "provider",
    [ScriptedProvider([[ToolCall(name="calculator", arguments={"expression": "6 * 7"})], "42"])],
)
def test_agent_trace_shows_the_tool_call(client):
    created = create_app(client, tools=["calculator"])
    body = ask(client, created["id"], "What is 6 * 7?")

    root = client.get(f"/v1/traces/{body['trace_id']}").json()["root"]
    assert kinds(root) == [("agent", "agent")]
    agent = root["children"][0]
    assert [kind for kind, _ in kinds(agent)] == ["llm", "tool", "llm"]
    tool = agent["children"][1]
    assert tool["name"] == "calculator"
    assert tool["inputs"] == {"arguments": {"expression": "6 * 7"}}
    assert tool["outputs"] == {"result": "6 * 7 = 42"}
    assert agent["outputs"]["iterations"] == 2


def test_streamed_chat_sends_the_trace_id(client):
    created = create_app(client)
    payload = {"query": "hello", "stream": True}
    with client.stream("POST", f"/v1/apps/{created['id']}/chat-messages", json=payload) as r:
        events = [line.removeprefix("data: ") for line in r.iter_lines() if line]

    assert events[-1] == "[DONE]"
    end = json.loads(events[-2])
    trace = client.get(f"/v1/traces/{end['trace_id']}").json()
    assert trace["status"] == "ok"
    assert trace["root"]["outputs"]["answer"] == "echo: hello"


def test_workflow_run_is_traced(client):
    graph = {
        "nodes": [
            {"id": "start", "type": "start", "data": {"variables": [{"name": "q"}]}},
            {"id": "llm", "type": "llm", "title": "Answer", "data": {"prompt": "{{#start.q#}}"}},
            {
                "id": "end",
                "type": "end",
                "data": {"outputs": [{"name": "a", "value": "{{#llm.text#}}"}]},
            },
        ],
        "edges": [
            {"source": "start", "target": "llm", "source_handle": "source"},
            {"source": "llm", "target": "end", "source_handle": "source"},
        ],
    }
    workflow = client.post("/v1/workflows", json={"name": "Echo", "graph": graph}).json()
    run = client.post(f"/v1/workflows/{workflow['id']}/run", json={"inputs": {"q": "hi"}}).json()
    assert run["trace_id"]

    trace = client.get(f"/v1/traces/{run['trace_id']}").json()
    assert trace["name"] == "workflow Echo"
    root = trace["root"]
    assert root["attributes"]["workflow_run_id"] == run["id"]
    assert kinds(root) == [("chain", "start"), ("chain", "Answer"), ("chain", "end")]
    # The LLM call is a child of the node that made it.
    assert kinds(root["children"][1]) == [("llm", f"llm {settings.default_model}")]


def test_list_traces_and_filter_by_status(client, provider):
    created = create_app(client)
    first = ask(client, created["id"], "one")
    second = ask(client, created["id"], "two")

    traces = client.get("/v1/traces").json()
    assert [t["trace_id"] for t in traces] == [second["trace_id"], first["trace_id"]]
    assert traces[0]["span_count"] == 2
    assert client.get("/v1/traces?limit=1").json()[0]["trace_id"] == second["trace_id"]
    assert client.get("/v1/traces?status=error").json() == []


def test_text_view_draws_the_tree(client):
    created = create_app(client)
    body = ask(client, created["id"], "hi")

    response = client.get(f"/v1/traces/{body['trace_id']}/text")
    assert response.status_code == 200
    lines = response.text.splitlines()
    assert lines[0].startswith("[chain] chat")
    assert lines[1].startswith("└─ [llm] llm ")
    assert "tokens" in lines[1]


@pytest.mark.parametrize("provider", [BrokenProvider()])
def test_failed_llm_call_is_saved_as_an_error_trace(client):
    created = create_app(client)
    response = client.post(f"/v1/apps/{created['id']}/chat-messages", json={"query": "hi"})
    assert response.status_code == 502

    traces = client.get("/v1/traces?status=error").json()
    assert len(traces) == 1
    assert "model is down" in traces[0]["error"]
    root = client.get(f"/v1/traces/{traces[0]['trace_id']}").json()["root"]
    assert root["children"][0]["status"] == "error"


def test_unknown_trace_is_404(client):
    missing = "00000000-0000-0000-0000-000000000000"
    assert client.get(f"/v1/traces/{missing}").status_code == 404
    assert client.get(f"/v1/traces/{missing}/text").status_code == 404


def test_tracing_can_be_turned_off(client, monkeypatch):
    monkeypatch.setattr(settings, "tracing_enabled", False)
    created = create_app(client)
    body = ask(client, created["id"], "hi")

    assert body["trace_id"] is None
    assert client.get("/v1/traces").json() == []
