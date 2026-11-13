from typing import Literal

from pydantic import BaseModel


class Message(BaseModel):
    """One turn in a conversation."""

    role: Literal["system", "user", "assistant"]
    content: str


class Usage(BaseModel):
    """Token counts reported by the model for one call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ChatChunk(BaseModel):
    """One piece of a streamed reply.

    Every chunk carries a bit of text in `delta`. The last chunk has `done=True`
    and carries the token usage and why the model stopped.
    """

    delta: str = ""
    done: bool = False
    finish_reason: str | None = None
    usage: Usage | None = None


class ChatResult(BaseModel):
    """A complete (non-streamed) reply."""

    model: str
    content: str
    finish_reason: str | None
    usage: Usage
    latency_ms: float
    time_to_first_token_ms: float | None


class ProviderError(Exception):
    """Raised when a model provider fails (down, unknown model, bad response)."""
