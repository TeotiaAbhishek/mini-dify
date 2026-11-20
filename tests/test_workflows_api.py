import json

import httpx
import pytest
from fastapi.testclient import TestClient

from mini_dify.api.workflows import get_http_transport
from mini_dify.llm.embeddings import FakeEmbedding
from mini_dify.llm.fake import FakeProvider
from mini_dify.llm.registry import get_embedder, get_provider
from mini_dify.main import app


@pytest.fixture
def client(clean_db):
    provider, embedder = FakeProvider(), FakeEmbedding()
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text="pong"))
    app.dependency_overrides[get_provider] = lambda: provider
    app.dependency_overrides[get_embedder] = lambda: embedder
    app.dependency_overrides[get_http_transport] = lambda: transport
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def edge(source, target, handle="source"):
    return {"source": source, "target": target, "source_handle": handle}


def summary_graph(dataset_id=None):
    """start -> (knowledge) -> llm -> end: answer a question, from a dataset if given."""
    nodes = [
        {"id": "start", "type": "start", "data": {"variables": [{"name": "question"}]}},
        {
            "id": "llm",
            "type": "llm",
            "title": "Answer",
            "data": {"prompt": "Answer: {{#start.question#}}"},
        },
        {
            "id": "end",
            "type": "end",
            "data": {"outputs": [{"name": "answer", "value": "{{#llm.text#}}"}]},
        },
    ]
    edges = [edge("start", "llm"), edge("llm", "end")]
    if dataset_id:
        nodes.insert(
            1,
            {
                "id": "kb",
                "type": "knowledge_retrieval",
                "data": {"dataset_ids": [dataset_id], "query": "{{#start.question#}}"},
            },
        )
        nodes[2]["data"]["prompt"] = "Context: {{#kb.text#}}\nAnswer: {{#start.question#}}"
        edges = [edge("start", "kb"), edge("kb", "llm"), edge("llm", "end")]
    return {"nodes": nodes, "edges": edges}


def create_workflow(client, graph=None):
    response = client.post("/v1/workflows", json={"name": "Q&A", "graph": graph or summary_graph()})
    assert response.status_code == 201, response.text
    return response.json()


def test_create_get_list_and_update(client):
    created = create_workflow(client)
    assert client.get(f"/v1/workflows/{created['id']}").json()["name"] == "Q&A"
    assert [w["id"] for w in client.get("/v1/workflows").json()] == [created["id"]]

    response = client.put(
        f"/v1/workflows/{created['id']}", json={"name": "Renamed", "graph": summary_graph()}
    )
    assert response.status_code == 200
    assert response.json()["name"] == "Renamed"


def test_broken_graph_is_rejected_with_the_reason(client):
    graph = summary_graph()
    graph["edges"].append(edge("end", "llm"))
    response = client.post("/v1/workflows", json={"name": "bad", "graph": graph})
    assert response.status_code == 422
    assert response.json()["detail"] == "End node 'end' can't have outgoing edges"


def test_run_returns_outputs_and_saves_a_trace(client):
    workflow = create_workflow(client)
    response = client.post(
        f"/v1/workflows/{workflow['id']}/run", json={"inputs": {"question": "What is RAG?"}}
    )
    body = response.json()

    assert response.status_code == 200, response.text
    assert body["status"] == "succeeded"
    assert body["outputs"] == {"answer": "echo: Answer: What is RAG?"}
    assert [n["node_id"] for n in body["node_runs"]] == ["start", "llm", "end"]
    assert body["node_runs"][1]["title"] == "Answer"
    assert body["node_runs"][1]["outputs"]["completion_tokens"] == 5

    # The same record can be read back later.
    saved = client.get(f"/v1/workflow-runs/{body['id']}").json()
    assert saved == body


def test_missing_input_is_rejected(client):
    workflow = create_workflow(client)
    response = client.post(f"/v1/workflows/{workflow['id']}/run", json={"inputs": {}})
    assert response.status_code == 422
    assert response.json()["detail"] == "Missing inputs: question"


def test_streamed_run_sends_events(client):
    workflow = create_workflow(client)
    with client.stream(
        "POST",
        f"/v1/workflows/{workflow['id']}/run",
        json={"inputs": {"question": "hi"}, "stream": True},
    ) as response:
        lines = [line for line in response.iter_lines() if line.startswith("data: ")]

    assert lines[-1] == "data: [DONE]"
    events = [json.loads(line.removeprefix("data: ")) for line in lines[:-1]]
    assert events[0]["event"] == "workflow_started"
    assert events[-1]["event"] == "workflow_finished"
    assert events[-1]["outputs"] == {"answer": "echo: Answer: hi"}
    assert events[0]["workflow_run_id"] == events[-1]["workflow_run_id"]

    saved = client.get(f"/v1/workflow-runs/{events[-1]['workflow_run_id']}").json()
    assert saved["status"] == "succeeded"
    assert len(saved["node_runs"]) == 3


def test_knowledge_node_reads_a_dataset(client):
    dataset = client.post("/v1/datasets", json={"name": "Handbook"}).json()
    client.post(
        f"/v1/datasets/{dataset['id']}/documents",
        files={"file": ("leave.txt", b"Every employee gets 25 vacation days per year.")},
    )
    workflow = create_workflow(client, summary_graph(dataset["id"]))

    body = client.post(
        f"/v1/workflows/{workflow['id']}/run",
        json={"inputs": {"question": "How many vacation days?"}},
    ).json()

    assert body["status"] == "succeeded", body
    kb = body["node_runs"][1]
    assert kb["node_id"] == "kb"
    assert kb["outputs"]["result"][0]["filename"] == "leave.txt"
    assert "[1] (leave.txt) Every employee gets 25 vacation days" in body["outputs"]["answer"]


def test_failed_run_is_saved_as_failed(client):
    graph = {
        "nodes": [
            {"id": "start", "type": "start", "data": {}},
            {"id": "call", "type": "http_request", "data": {"url": "http://127.0.0.1:6379/"}},
            {"id": "end", "type": "end", "data": {}},
        ],
        "edges": [edge("start", "call"), edge("call", "end")],
    }
    workflow = create_workflow(client, graph)
    body = client.post(f"/v1/workflows/{workflow['id']}/run", json={}).json()

    assert body["status"] == "failed"
    assert "private address" in body["error"]
    assert [(n["node_id"], n["status"]) for n in body["node_runs"]] == [
        ("start", "succeeded"),
        ("call", "failed"),
    ]


def test_unknown_workflow_and_run(client):
    missing = "00000000-0000-0000-0000-000000000000"
    assert client.get(f"/v1/workflows/{missing}").status_code == 404
    assert client.post(f"/v1/workflows/{missing}/run", json={}).status_code == 404
    assert client.get(f"/v1/workflow-runs/{missing}").status_code == 404
