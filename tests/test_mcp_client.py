import json

import httpx
import pytest

from mini_dify.mcp.client import MCPClient, MCPError, parse_sse

pytestmark = pytest.mark.anyio

URL = "http://93.184.215.14/mcp"  # a public IP address: the SSRF check needs no DNS


class FakeServer:
    """A stateful MCP server like the official SDK's: it hands out a session id,
    answers with SSE, and pages its tool list."""

    def __init__(self):
        self.received: list[dict] = []
        self.headers: list[httpx.Headers] = []
        self.deleted = False

    def sse(self, *messages: dict) -> httpx.Response:
        # A data-less priming event first, then each message: the client must skip the first.
        body = "id: 0\ndata:\n\n" + "".join(f"data: {json.dumps(m)}\n\n" for m in messages)
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    def handle(self, request: httpx.Request) -> httpx.Response:
        if request.method == "DELETE":
            self.deleted = True
            return httpx.Response(200)
        message = json.loads(request.content)
        self.received.append(message)
        self.headers.append(request.headers)
        method, request_id = message.get("method"), message.get("id")
        if request_id is None:
            return httpx.Response(202)
        if method == "initialize":
            reply = {
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fake", "version": "1"},
            }
            return httpx.Response(
                200,
                json={"jsonrpc": "2.0", "id": request_id, "result": reply},
                headers={"mcp-session-id": "session-123"},
            )
        if method == "tools/list":
            cursor = (message.get("params") or {}).get("cursor")
            if cursor is None:
                page = {"tools": [{"name": "add", "description": "Add."}], "nextCursor": "p2"}
            else:
                page = {"tools": [{"name": "echo", "inputSchema": {"type": "object"}}]}
            return self.sse({"jsonrpc": "2.0", "id": request_id, "result": page})
        if method == "tools/call":
            progress = {"jsonrpc": "2.0", "method": "notifications/progress", "params": {}}
            reply = {
                "content": [{"type": "text", "text": "42"}, {"type": "image", "data": "..."}],
                "structuredContent": {"result": 42},
            }
            return self.sse(progress, {"jsonrpc": "2.0", "id": request_id, "result": reply})
        error = {"code": -32601, "message": f"Method not found: {method}"}
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": request_id, "error": error})


async def test_handshake_tools_and_session():
    server = FakeServer()
    transport = httpx.MockTransport(server.handle)
    async with MCPClient(URL, {"Authorization": "Bearer x"}, transport=transport) as client:
        assert client.protocol_version == "2025-06-18"
        assert client.server_info == {"name": "fake", "version": "1"}
        tools = await client.list_tools()
        result = await client.call_tool("add", {"a": 40, "b": 2})

    assert [tool.name for tool in tools] == ["add", "echo"]  # both pages
    assert tools[0].input_schema == {"type": "object"}  # a default when none is given
    assert result.text == "42\n[image content]"
    assert result.structured == {"result": 42}
    assert result.is_error is False

    methods = [m["method"] for m in server.received]
    assert methods == [
        "initialize",
        "notifications/initialized",
        "tools/list",
        "tools/list",
        "tools/call",
    ]
    # After the handshake, every message carries the session and the agreed version.
    assert "mcp-session-id" not in server.headers[0]
    assert server.headers[-1]["mcp-session-id"] == "session-123"
    assert server.headers[-1]["mcp-protocol-version"] == "2025-06-18"
    assert server.headers[-1]["authorization"] == "Bearer x"
    assert server.deleted  # the session was ended on the way out


async def test_error_reply_becomes_mcp_error():
    server = FakeServer()
    async with MCPClient(URL, transport=httpx.MockTransport(server.handle)) as client:
        with pytest.raises(
            MCPError, match=r"prompts/list failed: Method not found: prompts/list \(-32601\)"
        ):
            await client._request("prompts/list", {})


async def test_unknown_protocol_version_is_refused():
    def handler(request):
        message = json.loads(request.content)
        reply = {"protocolVersion": "2030-01-01", "capabilities": {}, "serverInfo": {}}
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": message["id"], "result": reply})

    with pytest.raises(MCPError, match="2030-01-01"):
        async with MCPClient(URL, transport=httpx.MockTransport(handler)):
            pass


async def test_http_errors_become_mcp_error():
    transport = httpx.MockTransport(lambda request: httpx.Response(401, text="no token"))
    with pytest.raises(MCPError, match="replied 401: no token"):
        async with MCPClient(URL, transport=transport):
            pass


async def test_private_addresses_are_blocked():
    def handler(request):
        raise AssertionError("must not be called")

    with pytest.raises(MCPError, match="private address"):
        async with MCPClient("http://127.0.0.1:8000/mcp", transport=httpx.MockTransport(handler)):
            pass


def test_parse_sse():
    body = (
        'event: message\r\ndata: {"a":\r\ndata: 1}\r\n\r\n: comment\n\ndata:\n\ndata: {"b": 2}\n\n'
    )
    assert parse_sse(body) == [{"a": 1}, {"b": 2}]
