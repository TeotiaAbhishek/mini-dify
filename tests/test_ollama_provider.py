import json

import httpx
import pytest

from mini_dify.llm.ollama import OllamaProvider
from mini_dify.llm.types import Message, ProviderError

# What Ollama's /api/chat really sends back when stream=true: one JSON object per line.
OLLAMA_LINES = [
    {"message": {"role": "assistant", "content": "Hello"}, "done": False},
    {"message": {"role": "assistant", "content": " there"}, "done": False},
    {
        "message": {"role": "assistant", "content": ""},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 12,
        "eval_count": 2,
    },
]


def make_provider(handler) -> OllamaProvider:
    """An OllamaProvider whose HTTP calls go to `handler` instead of the network."""
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OllamaProvider("http://ollama.test", client=client)


@pytest.mark.anyio
async def test_stream_chat_parses_ndjson():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        body = "\n".join(json.dumps(line) for line in OLLAMA_LINES)
        return httpx.Response(200, text=body)

    provider = make_provider(handler)
    chunks = [
        chunk
        async for chunk in provider.stream_chat(
            "llama3.2:3b", [Message(role="user", content="hi")], max_tokens=50
        )
    ]

    assert seen["url"] == "http://ollama.test/api/chat"
    assert seen["body"]["stream"] is True
    assert seen["body"]["options"]["num_predict"] == 50
    assert "".join(c.delta for c in chunks) == "Hello there"
    assert chunks[-1].done is True
    assert chunks[-1].usage.prompt_tokens == 12
    assert chunks[-1].usage.completion_tokens == 2


@pytest.mark.anyio
async def test_chat_collects_stream_into_result():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="\n".join(json.dumps(line) for line in OLLAMA_LINES))

    result = await make_provider(handler).chat("llama3.2:3b", [Message(role="user", content="hi")])

    assert result.content == "Hello there"
    assert result.finish_reason == "stop"
    assert result.usage.total_tokens == 14
    assert result.time_to_first_token_ms is not None


@pytest.mark.anyio
async def test_unknown_model_raises_provider_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model 'nope' not found"})

    with pytest.raises(ProviderError, match="not found"):
        await make_provider(handler).chat("nope", [Message(role="user", content="hi")])


@pytest.mark.anyio
async def test_ollama_down_raises_provider_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(ProviderError, match="Could not reach Ollama"):
        await make_provider(handler).chat("llama3.2:3b", [Message(role="user", content="hi")])
