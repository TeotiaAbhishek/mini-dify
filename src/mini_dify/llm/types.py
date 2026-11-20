from typing import Any, Literal

from pydantic import BaseModel


class ToolCall(BaseModel):
    """The model asking us to run one tool with these arguments."""

    name: str
    arguments: dict[str, Any] = {}


class Message(BaseModel):
    """One turn in a conversation.

    - system / user / assistant: normal text turns.
    - assistant with `tool_calls`: the model asked to run tools instead of answering.
    - tool: the result of one tool call, sent back to the model. `tool_name` says which tool.
    """

    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[ToolCall] | None = None
    tool_name: str | None = None


class Usage(BaseModel):
    """Token counts reported by the model for one call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ChatChunk(BaseModel):
    """One piece of a streamed reply.

    Every chunk carries a bit of text in `delta`. A chunk can also carry tool calls.
    The last chunk has `done=True` and carries the token usage and why the model stopped.
    """

    delta: str = ""
    tool_calls: list[ToolCall] | None = None
    done: bool = False
    finish_reason: str | None = None
    usage: Usage | None = None


class ChatResult(BaseModel):
    """A complete (non-streamed) reply."""

    model: str
    content: str
    tool_calls: list[ToolCall] = []
    finish_reason: str | None
    usage: Usage
    latency_ms: float
    time_to_first_token_ms: float | None


class ProviderError(Exception):
    """Raised when a model provider fails (down, unknown model, bad response)."""
