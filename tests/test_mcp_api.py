import httpx
import pytest
from fastapi.testclient import TestClient

from mini_dify.api.mcp_servers import get_mcp_transport
from mini_dify.config import settings
from mini_dify.llm.embeddings import FakeEmbedding
from mini_dify.llm.fake import ScriptedProvider
from mini_dify.llm.registry import get_embedder, get_provider
from mini_dify.llm.types import ToolCall
from mini_dify.main import app

# mini-dify's own /mcp endpoint, reached in-process: the MCP client talks to the MCP
# server through httpx.ASGITransport, without a network. "testserver" isn't a real
# host, so private addresses are allowed for these tests.
SELF_URL = "http://testserver/mcp"

GREETER = {
    "name": "Greeter",
    "description": "Greets someone loudly.",
    "graph": {
        "nodes": [
            {"id": "start", "type": "start", "data": {"variables": [{"name": "who"}]}},
            {"id": "t", "type": "template", "data": {"template": "HELLO {{#start.who#}}!"}},
            {
                "id": "end",
                "type": "end",
                "data": {"outputs": [{"name": "greeting", "value": "{{#t.output#}}"}]},
            },
        ],
        "edges": [{"source": "start", "target": "t"}, {"source": "t", "target": "end"}],
    },
}


@pytest.fixture
def provider():
    return ScriptedProvider([])


@pytest.fixture
def client(clean_db, provider, monkeypatch):
    monkeypatch.setattr(settings, "mcp_allow_private_urls", True)
    app.dependency_overrides[get_provider] = lambda: provider
    app.dependency_overrides[get_embedder] = FakeEmbedding
    app.dependency_overrides[get_mcp_transport] = lambda: httpx.ASGITransport(app=app)
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def register(client, **fields):
    body = {"name": "self", "url": SELF_URL, "headers": {"Authorization": "Bearer secret"}}
    return client.post("/v1/mcp-servers", json={**body, **fields})


def test_register_lists_and_caches_the_tools(client):
    client.post("/v1/workflows", json=GREETER)
    response = register(client)
    body = response.json()

    assert response.status_code == 201, response.text
    assert body["protocol_version"] == "2025-11-25"
    assert body["server_info"]["name"] == "mini-dify"
    assert [tool["name"] for tool in body["tools"]] == ["greeter"]
    assert body["header_names"] == ["Authorization"]
    assert "secret" not in response.text  # header values are never sent back

    assert [s["name"] for s in client.get("/v1/mcp-servers").json()] == ["self"]
    assert register(client).status_code == 409  # names are unique


def test_refresh_picks_up_new_tools(client):
    server = register(client).json()
    assert server["tools"] == []
    client.post("/v1/workflows", json=GREETER)

    refreshed = client.post(f"/v1/mcp-servers/{server['id']}/refresh").json()
    assert [tool["name"] for tool in refreshed["tools"]] == ["greeter"]


def test_unreachable_server_is_a_502(client):
    def down(request):
        raise httpx.ConnectError("connection refused")

    app.dependency_overrides[get_mcp_transport] = lambda: httpx.MockTransport(down)
    response = register(client)
    assert response.status_code == 502
    assert "Can't reach the MCP server" in response.json()["detail"]


def test_private_url_is_refused_when_not_allowed(client, monkeypatch):
    monkeypatch.setattr(settings, "mcp_allow_private_urls", False)
    response = register(client, url="http://127.0.0.1:8000/mcp")
    assert response.status_code == 502
    assert "private address" in response.json()["detail"]


def test_agent_app_calls_an_mcp_tool(client, provider):
    client.post("/v1/workflows", json=GREETER)
    server = register(client).json()
    created = client.post(
        "/v1/apps",
        json={"name": "Agent", "system_prompt": "Be helpful.", "mcp_server_ids": [server["id"]]},
    ).json()
    assert created["mcp_server_ids"] == [server["id"]]

    provider.replies = [
        [ToolCall(name="self__greeter", arguments={"who": "abhi"})],
        "The greeting is HELLO abhi!",
    ]
    body = client.post(
        f"/v1/apps/{created['id']}/chat-messages", json={"query": "Greet abhi"}
    ).json()

    # The model saw the MCP tool, with the server's schema...
    spec = provider.calls[0]["tools"][0]["function"]
    assert spec["name"] == "self__greeter"
    assert spec["parameters"]["required"] == ["who"]
    # ...called it, and the call ran the workflow through MCP.
    assert body["agent_steps"][0]["tool"] == "self__greeter"
    assert body["agent_steps"][0]["observation"] == '{"greeting": "HELLO abhi!"}'
    assert body["answer"] == "The greeting is HELLO abhi!"


def test_mcp_tool_errors_reach_the_model_as_text(client, provider):
    client.post("/v1/workflows", json=GREETER)
    server = register(client).json()
    created = client.post(
        "/v1/apps", json={"name": "Agent", "mcp_server_ids": [server["id"]]}
    ).json()
    provider.replies = [[ToolCall(name="self__greeter", arguments={})], "Sorry."]
    body = client.post(f"/v1/apps/{created['id']}/chat-messages", json={"query": "Hi"}).json()
    assert body["agent_steps"][0]["observation"] == "Error: Missing inputs: who"


def test_delete_and_unknown_server(client):
    server = register(client).json()
    missing = "00000000-0000-0000-0000-000000000000"
    response = client.post("/v1/apps", json={"name": "x", "mcp_server_ids": [missing]})
    assert response.status_code == 422
    assert client.delete(f"/v1/mcp-servers/{server['id']}").status_code == 204
    assert client.get(f"/v1/mcp-servers/{server['id']}").status_code == 404
