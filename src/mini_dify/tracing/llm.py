import time
from collections.abc import AsyncIterator
from typing import Any

from mini_dify.llm.base import LLMProvider
from mini_dify.llm.types import ChatChunk, Message, Usage
from mini_dify.tracing.tracer import span


class TracedProvider(LLMProvider):
    """Wraps any provider and records each call as an "llm" span: the messages sent,
    the reply, the tokens, and how long until the first token."""

    def __init__(self, inner: LLMProvider):
        self.inner = inner
        self.name = inner.name

    async def stream_chat(
        self,
        model: str,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[ChatChunk]:
        inputs = {
            "messages": [message.model_dump(exclude_none=True) for message in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "tools": [tool["function"]["name"] for tool in tools or []],
        }
        async with span(f"llm {model}", "llm", inputs) as record:
            start = time.perf_counter()
            first_token_ms: float | None = None
            parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            usage = Usage()
            finish_reason = None

            async for chunk in self.inner.stream_chat(
                model, messages, temperature, max_tokens, tools
            ):
                if first_token_ms is None and (chunk.delta or chunk.tool_calls):
                    first_token_ms = round((time.perf_counter() - start) * 1000, 1)
                parts.append(chunk.delta)
                tool_calls.extend(call.model_dump() for call in chunk.tool_calls or [])
                if chunk.done:
                    usage = chunk.usage or usage
                    finish_reason = chunk.finish_reason
                yield chunk

            record.outputs = {"content": "".join(parts), "tool_calls": tool_calls}
            record.attributes = {
                "model": model,
                "provider": self.inner.name,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "time_to_first_token_ms": first_token_ms,
                "finish_reason": finish_reason,
            }

    async def aclose(self) -> None:
        await self.inner.aclose()
