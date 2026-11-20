import json
from dataclasses import dataclass, field
from itertools import count
from typing import Any

import httpx

from mini_dify.mcp import protocol
from mini_dify.security import UnsafeURLError, check_url_is_public


class MCPError(Exception):
    """Talking to an MCP server failed: unreachable, a protocol error, or an error reply."""


@dataclass
class RemoteTool:
    """A tool as an MCP server describes it in `tools/list`."""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=lambda: {"type": "object"})


@dataclass
class ToolCallResult:
    text: str
    is_error: bool = False
    structured: dict[str, Any] | None = None


def parse_sse(body: str) -> list[dict[str, Any]]:
    """The JSON messages in a Server-Sent Events body.

    Events are separated by a blank line. An event's data can span several `data:` lines.
    Events with no data (some servers send these to keep the connection alive) are skipped.
    """
    messages = []
    for event in body.replace("\r\n", "\n").split("\n\n"):
        data = "\n".join(
            line[5:].removeprefix(" ") for line in event.split("\n") if line.startswith("data:")
        )
        if data.strip():
            messages.append(json.loads(data))
    return messages


class MCPClient:
    """Talks to one MCP server over Streamable HTTP.

        async with MCPClient("https://example.com/mcp") as client:
            tools = await client.list_tools()
            result = await client.call_tool("search", {"query": "dify"})

    Entering the `async with` runs the `initialize` handshake. Leaving it ends the session.
    """

    def __init__(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        allow_private: bool = False,
        timeout: float = 30.0,
    ):
        self.url = url
        self.headers = dict(headers or {})
        self.transport = transport
        self.allow_private = allow_private
        self.timeout = timeout
        self.session_id: str | None = None
        self.protocol_version: str | None = None
        self.server_info: dict[str, Any] = {}
        self._ids = count(1)
        self._http: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "MCPClient":
        if not self.allow_private:
            try:
                await check_url_is_public(httpx.URL(self.url))
            except UnsafeURLError as exc:
                raise MCPError(str(exc)) from exc
        # No redirects: a redirect could lead to an address we never checked.
        self._http = httpx.AsyncClient(
            transport=self.transport, timeout=self.timeout, follow_redirects=False
        )
        try:
            await self.initialize()
        except BaseException:
            await self._http.aclose()
            raise
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        assert self._http is not None
        if self.session_id:
            # Tell a stateful server it can forget this session. Best effort: some
            # servers don't allow DELETE, and the session is over for us either way.
            try:
                await self._http.delete(self.url, headers=self._headers())
            except httpx.HTTPError:
                pass
        await self._http.aclose()

    async def initialize(self) -> None:
        """The handshake: agree on a protocol version, then say we're ready."""
        reply = await self._request(
            "initialize",
            {
                "protocolVersion": protocol.LATEST_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "mini-dify", "version": "0.1.0"},
            },
        )
        version = reply.get("protocolVersion")
        if version not in protocol.PROTOCOL_VERSIONS:
            raise MCPError(f"The server wants protocol version {version!r}, which we don't speak")
        self.protocol_version = version
        self.server_info = reply.get("serverInfo", {})
        await self._notify("notifications/initialized")

    async def list_tools(self) -> list[RemoteTool]:
        """Every tool the server offers, following `nextCursor` across pages."""
        tools: list[RemoteTool] = []
        cursor = None
        while True:
            reply = await self._request("tools/list", {"cursor": cursor} if cursor else {})
            for tool in reply.get("tools", []):
                tools.append(
                    RemoteTool(
                        name=tool["name"],
                        description=tool.get("description") or "",
                        input_schema=tool.get("inputSchema") or {"type": "object"},
                    )
                )
            cursor = reply.get("nextCursor")
            if not cursor:
                return tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolCallResult:
        reply = await self._request("tools/call", {"name": name, "arguments": arguments})
        # A result is a list of content blocks. We keep the text ones and describe the rest.
        parts = []
        for block in reply.get("content", []):
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
            else:
                parts.append(f"[{block.get('type', 'unknown')} content]")
        return ToolCallResult(
            text="\n".join(parts),
            is_error=bool(reply.get("isError")),
            structured=reply.get("structuredContent"),
        )

    # ---------- the transport ----------

    def _headers(self) -> dict[str, str]:
        headers = {
            **self.headers,
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        }
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        if self.protocol_version:
            headers["MCP-Protocol-Version"] = self.protocol_version
        return headers

    async def _post(self, message: dict[str, Any]) -> httpx.Response:
        assert self._http is not None, "use `async with MCPClient(...)`"
        try:
            response = await self._http.post(self.url, json=message, headers=self._headers())
        except httpx.HTTPError as exc:
            raise MCPError(f"Can't reach the MCP server: {exc}") from exc
        if response.status_code == 404 and self.session_id:
            raise MCPError("The MCP server ended our session (404)")
        if response.status_code >= 400:
            raise MCPError(f"The MCP server replied {response.status_code}: {response.text[:300]}")
        return response

    async def _notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        await self._post(protocol.notification(method, params))

    async def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        request_id = next(self._ids)
        response = await self._post(protocol.request(request_id, method, params))
        if method == "initialize":
            # A stateful server gives us a session id to send with every later message.
            self.session_id = response.headers.get("mcp-session-id")

        # The server may answer with plain JSON, or with an SSE stream that carries
        # the reply (possibly after progress notifications). We handle both.
        content_type = response.headers.get("content-type", "")
        try:
            if content_type.startswith("text/event-stream"):
                messages = parse_sse(response.text)
            else:
                messages = [response.json()]
        except ValueError as exc:
            raise MCPError(f"The MCP server sent something that isn't JSON: {exc}") from exc

        for message in messages:
            if isinstance(message, dict) and message.get("id") == request_id:
                if "error" in message:
                    error = message["error"]
                    raise MCPError(f"{method} failed: {error.get('message')} ({error.get('code')})")
                return message.get("result") or {}
        raise MCPError(f"The MCP server sent no reply to {method}")
