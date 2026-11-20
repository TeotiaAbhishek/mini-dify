import uuid

from mini_dify.agent.base import AgentRunner
from mini_dify.agent.function_calling import FunctionCallingAgent
from mini_dify.agent.react import ReActAgent
from mini_dify.llm.base import LLMProvider
from mini_dify.models import App
from mini_dify.tools.base import Tool, ToolContext
from mini_dify.tools.builtin import KnowledgeSearchTool
from mini_dify.tools.registry import get_tools

AGENT_MODES: dict[str, type[AgentRunner]] = {
    "function_calling": FunctionCallingAgent,
    "react": ReActAgent,
}


def is_agent_app(app: App) -> bool:
    return bool(app.tools)


def app_tools(app: App) -> list[Tool]:
    """The app's chosen tools, plus knowledge_search if it has datasets."""
    tools = get_tools(app.tools)
    if app.dataset_ids:
        tools.append(KnowledgeSearchTool())
    return tools


def make_agent(app: App, provider: LLMProvider, model: str, ctx: ToolContext) -> AgentRunner:
    ctx.dataset_ids = [uuid.UUID(dataset_id) for dataset_id in app.dataset_ids]
    ctx.retrieval_top_k = app.retrieval_top_k
    runner_class = AGENT_MODES[app.agent_mode]
    return runner_class(
        provider,
        model,
        app_tools(app),
        ctx,
        temperature=app.temperature,
        max_iterations=app.max_iterations,
    )
