from typing import Any

from mini_dify.config import settings
from mini_dify.mcp.client import MCPClient, MCPError
from mini_dify.models import MCPServer
from mini_dify.tools.base import Tool, ToolContext


class MCPTool(Tool):
    """A tool that lives on an MCP server. Calling it calls the server.

    Unlike the built-in tools, its argument schema comes from the server, so there's
    no pydantic `Args` model: the arguments go to the server as they are, and the
    server checks them.
    """

    def __init__(self, server: MCPServer, tool: dict[str, Any]):
        self.server_url = server.url
        self.server_headers = dict(server.headers)
        self.remote_name = tool["name"]
        # Prefix with the server's name, so two servers can both have a "search" tool.
        self.name = f"{server.name}__{tool['name']}"[:64]
        self.description = tool.get("description") or f"The {tool['name']} tool of {server.name}"
        self.schema = tool.get("input_schema") or {"type": "object"}

    def parameters(self) -> dict[str, Any]:
        return self.schema

    async def run(self, args: Any, ctx: ToolContext) -> str:
        raise NotImplementedError("MCP tools are run through invoke()")

    async def invoke(self, arguments: dict[str, Any], ctx: ToolContext) -> str:
        """Open a session, call the tool, close the session. Errors become text, as with
        every tool, so the model can see what went wrong."""
        try:
            async with MCPClient(
                self.server_url,
                self.server_headers,
                transport=ctx.mcp_transport,
                allow_private=settings.mcp_allow_private_urls,
            ) as client:
                result = await client.call_tool(self.remote_name, arguments)
        except MCPError as exc:
            return f"Error: {exc}"
        return f"Error: {result.text}" if result.is_error else result.text


def mcp_tools(servers: list[MCPServer]) -> list[Tool]:
    """Every cached tool of these servers, as tools an agent can use."""
    return [MCPTool(server, tool) for server in servers for tool in server.tools]
