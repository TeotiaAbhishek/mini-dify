from collections.abc import AsyncIterator

from mini_dify.llm.base import LLMProvider
from mini_dify.llm.types import ChatChunk, Message, Usage


class FakeProvider(LLMProvider):
    """A provider for tests: echoes the last user message back, word by word.

    It needs no GPU, no network and no model, so CI can test everything
    around the LLM without a real one.
    """

    name = "fake"

    async def stream_chat(
        self,
        model: str,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> AsyncIterator[ChatChunk]:
        words = f"echo: {messages[-1].content}".split(" ")
        for i, word in enumerate(words):
            yield ChatChunk(delta=word if i == 0 else f" {word}")
        yield ChatChunk(
            done=True,
            finish_reason="stop",
            usage=Usage(
                prompt_tokens=len(messages[-1].content.split()), completion_tokens=len(words)
            ),
        )
