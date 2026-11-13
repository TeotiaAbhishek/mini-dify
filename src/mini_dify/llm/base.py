import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from mini_dify.llm.types import ChatChunk, ChatResult, Message, Usage


class LLMProvider(ABC):
    """The one interface every model provider must implement.

    The rest of the app only ever talks to this class, never to Ollama,
    OpenAI or Anthropic directly. Adding a provider means adding a subclass.
    """

    name: str

    @abstractmethod
    def stream_chat(
        self,
        model: str,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> AsyncIterator[ChatChunk]:
        """Yield the reply piece by piece as the model generates it."""

    async def aclose(self) -> None:
        """Release resources (like HTTP connections). Override if needed."""
        return None

    async def chat(
        self,
        model: str,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> ChatResult:
        """Return the whole reply at once, built on top of stream_chat."""
        start = time.perf_counter()
        first_token_at: float | None = None
        parts: list[str] = []
        usage = Usage()
        finish_reason = None

        async for chunk in self.stream_chat(model, messages, temperature, max_tokens):
            if chunk.delta:
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                parts.append(chunk.delta)
            if chunk.done:
                usage = chunk.usage or usage
                finish_reason = chunk.finish_reason

        end = time.perf_counter()
        return ChatResult(
            model=model,
            content="".join(parts),
            finish_reason=finish_reason,
            usage=usage,
            latency_ms=round((end - start) * 1000, 1),
            time_to_first_token_ms=(
                round((first_token_at - start) * 1000, 1) if first_token_at else None
            ),
        )
