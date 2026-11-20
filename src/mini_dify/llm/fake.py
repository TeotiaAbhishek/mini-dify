from collections.abc import AsyncIterator
from typing import Any

from mini_dify.llm.base import LLMProvider
from mini_dify.llm.types import ChatChunk, Message, ToolCall, Usage


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
        tools: list[dict[str, Any]] | None = None,
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


class ScriptedProvider(LLMProvider):
    """A provider for agent tests: plays back a script of replies, one per call.

    Each reply is either text, or a list of ToolCalls (the model asking to use tools).
    Every call is recorded in `calls`, so tests can check what the model was sent.
    """

    name = "scripted"

    def __init__(self, replies: list[str | list[ToolCall]]):
        self.replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    async def stream_chat(
        self,
        model: str,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[ChatChunk]:
        self.calls.append({"messages": list(messages), "tools": tools})
        if not self.replies:
            raise AssertionError("ScriptedProvider ran out of replies")
        reply = self.replies.pop(0)
        if isinstance(reply, str):
            yield ChatChunk(delta=reply)
        else:
            yield ChatChunk(tool_calls=reply)
        yield ChatChunk(
            done=True, finish_reason="stop", usage=Usage(prompt_tokens=10, completion_tokens=5)
        )
