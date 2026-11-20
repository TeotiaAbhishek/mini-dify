import asyncio
import json
import time

import httpx
import pytest

from mini_dify.llm.fake import FakeProvider
from mini_dify.workflow.engine import WorkflowEngine
from mini_dify.workflow.graph import Graph, GraphSpec
from mini_dify.workflow.nodes import NodeContext, build_nodes, check_condition

pytestmark = pytest.mark.anyio


class SlowProvider(FakeProvider):
    """Echoes like FakeProvider, but each call takes 0.2 seconds."""

    async def stream_chat(self, *args, **kwargs):
        await asyncio.sleep(0.2)
        async for chunk in super().stream_chat(*args, **kwargs):
            yield chunk


def make_ctx(inputs, provider=None, transport=None, allow_private=False):
    return NodeContext(
        provider=provider or FakeProvider(),
        embedder=None,
        sessionmaker=None,
        default_model="test-model",
        inputs=inputs,
        http_transport=transport,
        allow_private_http=allow_private,
    )


async def run_graph(nodes, edges, ctx):
    graph = Graph(GraphSpec.model_validate({"nodes": nodes, "edges": edges}))
    engine = WorkflowEngine(graph, build_nodes(graph), ctx)
    return [event async for event in engine.run()]


def edge(source, target, handle="source"):
    return {"source": source, "target": target, "source_handle": handle}


def finished(events):
    return {e.node_id: e for e in events if e.event == "node_finished"}


START = {"id": "start", "type": "start", "data": {"variables": [{"name": "topic"}]}}


async def test_linear_workflow_passes_values_along():
    nodes = [
        START,
        {"id": "llm", "type": "llm", "data": {"prompt": "Tell me about {{#start.topic#}}"}},
        {
            "id": "end",
            "type": "end",
            "data": {"outputs": [{"name": "answer", "value": "{{#llm.text#}}"}]},
        },
    ]
    events = await run_graph(
        nodes, [edge("start", "llm"), edge("llm", "end")], make_ctx({"topic": "GPUs"})
    )

    assert [e.event for e in events] == [
        "workflow_started",
        "node_started",
        "node_finished",
        "node_started",
        "node_finished",
        "node_started",
        "node_finished",
        "workflow_finished",
    ]
    llm = finished(events)["llm"]
    assert llm.inputs["prompt"] == "Tell me about GPUs"
    assert llm.inputs["model"] == "test-model"
    assert events[-1].status == "succeeded"
    assert events[-1].outputs == {"answer": "echo: Tell me about GPUs"}


async def test_independent_nodes_run_in_parallel():
    nodes = [
        START,
        {"id": "short", "type": "llm", "data": {"prompt": "Short note on {{#start.topic#}}"}},
        {"id": "long", "type": "llm", "data": {"prompt": "Essay on {{#start.topic#}}"}},
        {
            "id": "join",
            "type": "template",
            "data": {"template": "{{#short.text#}} | {{#long.text#}}"},
        },
        {
            "id": "end",
            "type": "end",
            "data": {"outputs": [{"name": "both", "value": "{{#join.output#}}"}]},
        },
    ]
    edges = [
        edge("start", "short"),
        edge("start", "long"),
        edge("short", "join"),
        edge("long", "join"),
        edge("join", "end"),
    ]
    start = time.perf_counter()
    events = await run_graph(nodes, edges, make_ctx({"topic": "rust"}, SlowProvider()))
    elapsed = time.perf_counter() - start

    assert elapsed < 0.35  # two 0.2 s calls at the same time, not one after the other
    assert events[-1].outputs["both"] == "echo: Short note on rust | echo: Essay on rust"
    # Both LLM nodes started before either finished.
    kinds = [(e.event, getattr(e, "node_id", None)) for e in events]
    assert kinds.index(("node_started", "long")) < kinds.index(("node_finished", "short"))


def branching_graph():
    nodes = [
        START,
        {
            "id": "check",
            "type": "if_else",
            "data": {
                "conditions": [
                    {"variable": "{{#start.topic#}}", "operator": "contains", "value": "urgent"}
                ]
            },
        },
        {"id": "fast", "type": "template", "data": {"template": "FAST: {{#start.topic#}}"}},
        {"id": "slow", "type": "template", "data": {"template": "slow: {{#start.topic#}}"}},
        {"id": "after_slow", "type": "template", "data": {"template": "{{#slow.output#}}!"}},
        {
            "id": "end_fast",
            "type": "end",
            "data": {"outputs": [{"name": "result", "value": "{{#fast.output#}}"}]},
        },
        {
            "id": "end_slow",
            "type": "end",
            "data": {"outputs": [{"name": "result", "value": "{{#after_slow.output#}}"}]},
        },
    ]
    edges = [
        edge("start", "check"),
        edge("check", "fast", "true"),
        edge("check", "slow", "false"),
        edge("slow", "after_slow"),
        edge("fast", "end_fast"),
        edge("after_slow", "end_slow"),
    ]
    return nodes, edges


async def test_if_else_takes_one_branch_and_skips_the_other():
    nodes, edges = branching_graph()
    events = await run_graph(nodes, edges, make_ctx({"topic": "urgent: disk full"}))
    status = {node_id: e.status for node_id, e in finished(events).items()}

    assert status == {
        "start": "succeeded",
        "check": "succeeded",
        "fast": "succeeded",
        "slow": "skipped",
        "after_slow": "skipped",  # the skip spreads down the branch
        "end_fast": "succeeded",
        "end_slow": "skipped",
    }
    assert finished(events)["check"].outputs == {"result": True}
    assert events[-1].outputs == {"result": "FAST: urgent: disk full"}


async def test_if_else_false_branch():
    nodes, edges = branching_graph()
    events = await run_graph(nodes, edges, make_ctx({"topic": "a new idea"}))
    assert finished(events)["fast"].status == "skipped"
    assert events[-1].outputs == {"result": "slow: a new idea!"}


async def test_branches_join_again():
    """A node after both branches runs if either branch reached it."""
    nodes = [
        START,
        {
            "id": "check",
            "type": "if_else",
            "data": {"conditions": [{"variable": "{{#start.topic#}}", "operator": "is_empty"}]},
        },
        {"id": "empty", "type": "template", "data": {"template": "nothing"}},
        {"id": "full", "type": "template", "data": {"template": "something"}},
        {"id": "end", "type": "end", "data": {"outputs": [{"name": "done", "value": "yes"}]}},
    ]
    edges = [
        edge("start", "check"),
        edge("check", "empty", "true"),
        edge("check", "full", "false"),
        edge("empty", "end"),
        edge("full", "end"),
    ]
    events = await run_graph(nodes, edges, make_ctx({"topic": "x"}))
    assert finished(events)["end"].status == "succeeded"
    assert events[-1].outputs == {"done": "yes"}


async def test_failed_node_stops_the_run():
    nodes = [
        START,
        {
            "id": "bad",
            "type": "if_else",
            "data": {
                "conditions": [
                    {"variable": "{{#start.topic#}}", "operator": "greater_than", "value": "3"}
                ]
            },
        },
        {"id": "yes", "type": "template", "data": {"template": "big"}},
        {"id": "no", "type": "template", "data": {"template": "small"}},
        {"id": "end", "type": "end", "data": {}},
    ]
    edges = [
        edge("start", "bad"),
        edge("bad", "yes", "true"),
        edge("bad", "no", "false"),
        edge("yes", "end"),
        edge("no", "end"),
    ]
    events = await run_graph(nodes, edges, make_ctx({"topic": "lots"}))

    assert finished(events)["bad"].status == "failed"
    assert finished(events)["bad"].error == "The value is not a number: 'lots'"
    assert "yes" not in finished(events)  # nothing after the failure runs
    assert events[-1].status == "failed"
    assert events[-1].error == "Node 'bad' failed: The value is not a number: 'lots'"


@pytest.mark.parametrize(
    ("actual", "operator", "value", "expected"),
    [
        ("hello world", "contains", "world", True),
        ("hello", "not_contains", "x", True),
        ("yes", "equals", "yes", True),
        ("yes", "not_equals", "yes", False),
        ("report.pdf", "ends_with", ".pdf", True),
        ("Error: x", "starts_with", "Error", True),
        ("", "is_empty", "", True),
        ([], "is_empty", "", True),
        ("x", "is_not_empty", "", True),
        (10, "greater_than", "3", True),
        ("2.5", "less_than", "3", True),
    ],
)
def test_conditions(actual, operator, value, expected):
    assert check_condition(actual, operator, value) is expected


# ---------- HTTP node ----------


def http_graph(url):
    return [
        START,
        {
            "id": "call",
            "type": "http_request",
            "data": {
                "method": "POST",
                "url": url,
                "headers": {"x-topic": "{{#start.topic#}}"},
                "body": '{"q": "{{#start.topic#}}"}',
            },
        },
        {
            "id": "end",
            "type": "end",
            "data": {
                "outputs": [
                    {"name": "status", "value": "{{#call.status_code#}}"},
                    {"name": "body", "value": "{{#call.body#}}"},
                ]
            },
        },
    ], [edge("start", "call"), edge("call", "end")]


async def test_http_node_sends_rendered_request():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["topic"] = request.headers["x-topic"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"ok": True})

    # An IP address, so the SSRF check needs no DNS lookup. 93.184.215.14 is public.
    nodes, edges = http_graph("http://93.184.215.14/search")
    ctx = make_ctx({"topic": "cats"}, transport=httpx.MockTransport(handler))
    events = await run_graph(nodes, edges, ctx)

    assert seen == {"url": "http://93.184.215.14/search", "topic": "cats", "body": {"q": "cats"}}
    assert events[-1].outputs == {"status": 200, "body": '{"ok":true}'}


@pytest.mark.parametrize(
    ("url", "message"),
    [
        ("http://127.0.0.1:5432/", "private address (127.0.0.1)"),
        ("http://localhost/admin", "private address"),
        ("http://169.254.169.254/latest/meta-data", "private address (169.254.169.254)"),
        ("http://10.0.0.5/", "private address (10.0.0.5)"),
        ("file:///etc/passwd", "Only http and https"),
    ],
)
async def test_http_node_blocks_private_addresses(url, message):
    def handler(request):
        raise AssertionError("the request must never be sent")

    nodes, edges = http_graph(url)
    ctx = make_ctx({"topic": "x"}, transport=httpx.MockTransport(handler))
    events = await run_graph(nodes, edges, ctx)

    assert events[-1].status == "failed"
    assert message in finished(events)["call"].error


async def test_private_addresses_allowed_when_switched_on():
    nodes, edges = http_graph("http://127.0.0.1:9000/")
    transport = httpx.MockTransport(lambda request: httpx.Response(204))
    events = await run_graph(
        nodes, edges, make_ctx({"topic": "x"}, transport=transport, allow_private=True)
    )
    assert events[-1].outputs["status"] == 204
