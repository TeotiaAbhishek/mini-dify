import uuid
from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from mini_dify.agent.base import AgentRunner
from mini_dify.agent.function_calling import FunctionCallingAgent
from mini_dify.agent.react import ReActAgent
from mini_dify.llm.base import LLMProvider
from mini_dify.models import App, MCPServer
from mini_dify.tools.base import Tool, ToolContext
from mini_dify.tools.builtin import KnowledgeSearchTool
from mini_dify.tools.mcp_tool import mcp_tools
from mini_dify.tools.registry import get_tools

AGENT_MODES: dict[str, type[AgentRunner]] = {
    "function_calling": FunctionCallingAgent,
    "react": ReActAgent,
}


def is_agent_app(app: App) -> bool:
    return bool(app.tools or app.mcp_server_ids)


async def load_mcp_servers(session: AsyncSession, app: App) -> list[MCPServer]:
    """The MCP servers this app uses, with their cached tool lists."""
    if not app.mcp_server_ids:
        return []
    ids = [uuid.UUID(server_id) for server_id in app.mcp_server_ids]
    result = await session.scalars(select(MCPServer).where(MCPServer.id.in_(ids)))
    return list(result)


def app_tools(app: App, mcp_servers: Sequence[MCPServer] = ()) -> list[Tool]:
    """The app's chosen tools, knowledge_search if it has datasets, and its MCP servers' tools."""
    tools = get_tools(app.tools)
    if app.dataset_ids:
        tools.append(KnowledgeSearchTool())
    tools.extend(mcp_tools(list(mcp_servers)))
    return tools


def make_agent(
    app: App,
    provider: LLMProvider,
    model: str,
    ctx: ToolContext,
    mcp_servers: Sequence[MCPServer] = (),
) -> AgentRunner:
    ctx.dataset_ids = [uuid.UUID(dataset_id) for dataset_id in app.dataset_ids]
    ctx.retrieval_top_k = app.retrieval_top_k
    runner_class = AGENT_MODES[app.agent_mode]
    return runner_class(
        provider,
        model,
        app_tools(app, mcp_servers),
        ctx,
        temperature=app.temperature,
        max_iterations=app.max_iterations,
    )
