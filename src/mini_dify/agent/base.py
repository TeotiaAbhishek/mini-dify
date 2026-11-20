from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel

from mini_dify.llm.base import LLMProvider
from mini_dify.llm.types import Message, Usage
from mini_dify.tools.base import Tool, ToolContext


class AgentStep(BaseModel):
    """One tool call the agent made, and what came back."""

    iteration: int
    thought: str = ""
    tool: str
    arguments: dict[str, Any]
    observation: str


class AgentAnswer(BaseModel):
    """The agent's final reply."""

    content: str
    iterations: int
    usage: Usage
    # True when the agent ran out of iterations and had to answer with what it had.
    stopped_early: bool = False


AgentEvent = AgentStep | AgentAnswer

OUT_OF_STEPS = (
    "You have used all your tool calls. Answer the user now, using only what you found so far."
)


class AgentRunner(ABC):
    """The loop shared by every agent: ask the model, run tools, repeat, then answer."""

    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        tools: list[Tool],
        ctx: ToolContext,
        temperature: float = 0.7,
        max_iterations: int = 5,
    ):
        self.provider = provider
        self.model = model
        self.tools = {tool.name: tool for tool in tools}
        self.ctx = ctx
        self.temperature = temperature
        self.max_iterations = max_iterations
        self.usage = Usage()

    @abstractmethod
    def run(self, messages: list[Message]) -> AsyncIterator[AgentEvent]:
        """Yield an AgentStep for every tool call, then exactly one AgentAnswer."""

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        tool = self.tools.get(name)
        if tool is None:
            return f"Error: unknown tool {name!r}. Available tools: {', '.join(self.tools)}"
        return await tool.invoke(arguments, self.ctx)

    def add_usage(self, usage: Usage) -> None:
        self.usage = Usage(
            prompt_tokens=self.usage.prompt_tokens + usage.prompt_tokens,
            completion_tokens=self.usage.completion_tokens + usage.completion_tokens,
        )
