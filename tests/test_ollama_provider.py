import json

import httpx
import pytest

from mini_dify.llm.ollama import OllamaProvider
from mini_dify.llm.types import Message, ProviderError, ToolCall

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


@pytest.mark.anyio
async def test_tool_calls_are_sent_and_parsed():
    seen = {}
    lines = [
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "calculator", "arguments": {"expression": "6*7"}}}
                ],
            },
            "done": False,
        },
        {"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop"},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, text="\n".join(json.dumps(line) for line in lines))

    tools = [{"type": "function", "function": {"name": "calculator", "parameters": {}}}]
    history = [
        Message(role="user", content="6*7?"),
        Message(role="assistant", tool_calls=[ToolCall(name="calculator", arguments={"x": 1})]),
        Message(role="tool", content="42", tool_name="calculator"),
    ]
    result = await make_provider(handler).chat("qwen2.5:7b", history, tools=tools)

    # What we sent: the tools, and each message in Ollama's own format.
    assert seen["body"]["tools"] == tools
    assert seen["body"]["messages"][0] == {"role": "user", "content": "6*7?"}
    assert seen["body"]["messages"][1]["tool_calls"] == [
        {"function": {"name": "calculator", "arguments": {"x": 1}}}
    ]
    assert seen["body"]["messages"][2] == {
        "role": "tool",
        "content": "42",
        "tool_name": "calculator",
    }
    # What we got back.
    assert result.content == ""
    assert result.tool_calls == [ToolCall(name="calculator", arguments={"expression": "6*7"})]
