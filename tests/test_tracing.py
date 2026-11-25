import asyncio

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from mini_dify.config import settings
from mini_dify.llm.fake import FakeProvider, ScriptedProvider
from mini_dify.llm.types import Message, ToolCall
from mini_dify.models import Span
from mini_dify.tracing.llm import TracedProvider
from mini_dify.tracing.tracer import MAX_TEXT, current_trace_id, span, to_json, trace

pytestmark = pytest.mark.anyio


@pytest.fixture
async def sessionmaker(clean_db):
    engine = create_async_engine(settings.database_url)
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


async def saved_spans(sessionmaker, trace_id):
    async with sessionmaker() as session:
        result = await session.scalars(
            select(Span).where(Span.trace_id == trace_id).order_by(Span.position)
        )
        return list(result)


async def test_spans_nest_and_are_saved_together(sessionmaker):
    async with trace("chat", "chain", {"query": "hi"}, sessionmaker) as root:
        trace_id = current_trace_id()
        async with span("retrieval", "retriever", {"query": "hi"}) as search:
            search.outputs = {"chunks": []}
        async with span("agent", "agent"):
            async with span("calculator", "tool", {"expression": "1+1"}):
                pass
        root.outputs = {"answer": "hello"}

    assert root.id == trace_id  # the root span's id is the trace's id
    spans = await saved_spans(sessionmaker, trace_id)
    by_name = {s.name: s for s in spans}
    assert [s.name for s in spans] == ["chat", "retrieval", "agent", "calculator"]
    assert by_name["chat"].parent_id is None
    assert by_name["retrieval"].parent_id == root.id
    assert by_name["calculator"].parent_id == by_name["agent"].id
    assert by_name["chat"].outputs == {"answer": "hello"}
    assert all(s.status == "ok" and s.duration_ms >= 0 for s in spans)


async def test_spans_outside_a_trace_record_nothing():
    async with span("lonely", "tool") as record:
        assert record.trace_id is None
    assert current_trace_id() is None


async def test_an_error_is_recorded_and_the_trace_still_saved(sessionmaker):
    with pytest.raises(ValueError):
        async with trace("chat", "chain", {}, sessionmaker):
            trace_id = current_trace_id()
            async with span("llm", "llm"):
                raise ValueError("model went away")

    spans = await saved_spans(sessionmaker, trace_id)
    assert [(s.name, s.status, s.error) for s in spans] == [
        ("chat", "error", "model went away"),
        ("llm", "error", "model went away"),
    ]


async def test_parallel_tasks_keep_their_own_parents(sessionmaker):
    async def node(name):
        async with span(name, "chain"):
            await asyncio.sleep(0.01)
            async with span(f"{name} llm", "llm"):
                await asyncio.sleep(0.01)

    async with trace("workflow", "chain", {}, sessionmaker):
        trace_id = current_trace_id()
        await asyncio.gather(node("a"), node("b"))

    by_name = {s.name: s for s in await saved_spans(sessionmaker, trace_id)}
    assert by_name["a llm"].parent_id == by_name["a"].id
    assert by_name["b llm"].parent_id == by_name["b"].id
    assert by_name["a"].parent_id == by_name["b"].parent_id == by_name["workflow"].id


async def test_a_trace_inside_a_trace_is_a_child_span(sessionmaker):
    async with trace("outer", "chain", {}, sessionmaker):
        outer_id = current_trace_id()
        async with trace("inner", "chain", {}, sessionmaker) as inner:
            assert current_trace_id() == outer_id
    assert inner.parent_id == outer_id


async def test_tracing_can_be_switched_off(sessionmaker, monkeypatch):
    monkeypatch.setattr(settings, "tracing_enabled", False)
    async with trace("chat", "chain", {}, sessionmaker):
        assert current_trace_id() is None


async def test_traced_provider_records_the_call(sessionmaker):
    provider = TracedProvider(FakeProvider())
    async with trace("chat", "chain", {}, sessionmaker):
        trace_id = current_trace_id()
        result = await provider.chat("m1", [Message(role="user", content="hello there")])

    assert result.content == "echo: hello there"
    llm = (await saved_spans(sessionmaker, trace_id))[1]
    assert llm.kind == "llm"
    assert llm.name == "llm m1"
    assert llm.inputs["messages"] == [{"role": "user", "content": "hello there"}]
    assert llm.outputs == {"content": "echo: hello there", "tool_calls": []}
    assert llm.attributes["prompt_tokens"] == 2
    assert llm.attributes["completion_tokens"] == 3
    assert llm.attributes["time_to_first_token_ms"] is not None


async def test_traced_provider_records_tool_calls(sessionmaker):
    calls = [ToolCall(name="calculator", arguments={"expression": "6*7"})]
    provider = TracedProvider(ScriptedProvider([calls]))
    tools = [{"type": "function", "function": {"name": "calculator"}}]
    async with trace("chat", "chain", {}, sessionmaker):
        trace_id = current_trace_id()
        await provider.chat("m1", [Message(role="user", content="6*7?")], tools=tools)

    llm = (await saved_spans(sessionmaker, trace_id))[1]
    assert llm.inputs["tools"] == ["calculator"]
    assert llm.outputs["tool_calls"] == [{"name": "calculator", "arguments": {"expression": "6*7"}}]


def test_to_json_cuts_long_text_and_stringifies_the_rest():
    value = to_json({"text": "x" * (MAX_TEXT + 5), "when": object, "items": [1, "a"]})
    assert value["text"].endswith("... [5 more characters]")
    assert len(value["text"]) == MAX_TEXT + len("... [5 more characters]")
    assert value["when"] == str(object)
    assert value["items"] == [1, "a"]
