import json
from collections.abc import AsyncIterator

import httpx

from mini_dify.llm.base import LLMProvider
from mini_dify.llm.types import ChatChunk, Message, ProviderError, Usage


class OllamaProvider(LLMProvider):
    """Talks to a local Ollama server over its HTTP API (/api/chat)."""

    name = "ollama"

    def __init__(self, base_url: str, client: httpx.AsyncClient | None = None):
        self.base_url = base_url.rstrip("/")
        # Generation can be slow, so allow a long read timeout but fail fast on connect.
        self.client = client or httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=5.0))

    async def aclose(self) -> None:
        await self.client.aclose()

    async def stream_chat(
        self,
        model: str,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> AsyncIterator[ChatChunk]:
        options: dict = {"temperature": temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        payload = {
            "model": model,
            "messages": [m.model_dump() for m in messages],
            "stream": True,
            "options": options,
        }

        try:
            async with self.client.stream(
                "POST", f"{self.base_url}/api/chat", json=payload
            ) as response:
                if response.status_code != 200:
                    body = (await response.aread()).decode()
                    raise ProviderError(f"Ollama returned {response.status_code}: {body}")

                # Ollama streams NDJSON: one JSON object per line.
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if "error" in data:
                        raise ProviderError(f"Ollama error: {data['error']}")

                    if data.get("done"):
                        yield ChatChunk(
                            delta=data.get("message", {}).get("content", ""),
                            done=True,
                            finish_reason=data.get("done_reason"),
                            usage=Usage(
                                prompt_tokens=data.get("prompt_eval_count", 0),
                                completion_tokens=data.get("eval_count", 0),
                            ),
                        )
                    else:
                        yield ChatChunk(delta=data["message"]["content"])
        except httpx.HTTPError as exc:
            raise ProviderError(f"Could not reach Ollama at {self.base_url}: {exc}") from exc
