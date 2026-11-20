import pytest
from fastapi.testclient import TestClient

from mini_dify.llm.embeddings import FakeEmbedding
from mini_dify.llm.fake import FakeProvider
from mini_dify.llm.registry import get_embedder, get_provider
from mini_dify.main import app

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
def client(clean_db):
    app.dependency_overrides[get_provider] = FakeProvider
    app.dependency_overrides[get_embedder] = FakeEmbedding
    with TestClient(app) as test_client:
        test_client.post("/v1/workflows", json=GREETER)
        yield test_client
    app.dependency_overrides.clear()


def rpc(client, method, params=None, request_id=1, headers=None):
    message = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        message["params"] = params
    return client.post("/mcp", json=message, headers=headers or {})


def test_initialize_agrees_on_a_version(client):
    reply = rpc(client, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {}})
    body = reply.json()
    assert reply.status_code == 200
    assert body["id"] == 1
    assert body["result"]["protocolVersion"] == "2025-06-18"
    assert body["result"]["capabilities"] == {"tools": {"listChanged": False}}
    assert body["result"]["serverInfo"]["name"] == "mini-dify"

    # A version we don't know: we offer our newest instead.
    newer = rpc(client, "initialize", {"protocolVersion": "2099-01-01"}).json()
    assert newer["result"]["protocolVersion"] == "2025-11-25"


def test_notifications_get_202_and_no_body(client):
    reply = client.post("/mcp", json={"jsonrpc": "2.0", "method": "notifications/initialized"})
    assert reply.status_code == 202
    assert reply.content == b""


def test_workflows_are_listed_as_tools(client):
    tools = rpc(client, "tools/list").json()["result"]["tools"]
    assert tools == [
        {
            "name": "greeter",
            "title": "Greeter",
            "description": "Greets someone loudly.",
            "inputSchema": {
                "type": "object",
                "properties": {"who": {"description": "The workflow input 'who'"}},
                "required": ["who"],
            },
        }
    ]


def test_same_names_get_a_suffix(client):
    client.post("/v1/workflows", json=GREETER)
    tools = rpc(client, "tools/list").json()["result"]["tools"]
    assert [tool["name"] for tool in tools] == ["greeter", "greeter_2"]


def test_call_runs_the_workflow(client):
    result = rpc(client, "tools/call", {"name": "greeter", "arguments": {"who": "abhi"}}).json()
    assert result["result"] == {
        "content": [{"type": "text", "text": '{"greeting": "HELLO abhi!"}'}],
        "isError": False,
        "structuredContent": {"greeting": "HELLO abhi!"},
    }


def test_tool_problems_are_results_with_is_error(client):
    result = rpc(client, "tools/call", {"name": "greeter", "arguments": {}}).json()["result"]
    assert result["isError"] is True
    assert result["content"][0]["text"] == "Missing inputs: who"


def test_protocol_errors(client):
    unknown_tool = rpc(client, "tools/call", {"name": "nope", "arguments": {}}).json()
    assert unknown_tool["error"] == {"code": -32602, "message": "Unknown tool: nope"}

    unknown_method = rpc(client, "resources/list", request_id="abc").json()
    assert unknown_method == {
        "jsonrpc": "2.0",
        "id": "abc",
        "error": {"code": -32601, "message": "Method not found: resources/list"},
    }
    assert rpc(client, "ping").json()["result"] == {}


@pytest.mark.parametrize(
    ("content", "status", "code"),
    [
        (b"{not json", 400, -32700),
        (b'[{"jsonrpc": "2.0", "id": 1, "method": "ping"}]', 400, -32600),
        (b'{"id": 1, "method": "ping"}', 400, -32600),
    ],
)
def test_bad_messages(client, content, status, code):
    reply = client.post("/mcp", content=content, headers={"content-type": "application/json"})
    assert reply.status_code == status
    assert reply.json()["error"]["code"] == code


def test_unknown_protocol_version_header_is_refused(client):
    reply = rpc(client, "tools/list", headers={"MCP-Protocol-Version": "1999-01-01"})
    assert reply.status_code == 400
    assert (
        rpc(client, "tools/list", headers={"MCP-Protocol-Version": "2025-06-18"}).status_code == 200
    )


def test_browser_pages_from_other_sites_are_refused(client):
    assert rpc(client, "ping", headers={"Origin": "https://evil.example"}).status_code == 403
    assert rpc(client, "ping", headers={"Origin": "http://localhost:3000"}).status_code == 200


def test_no_server_stream(client):
    assert client.get("/mcp").status_code == 405
