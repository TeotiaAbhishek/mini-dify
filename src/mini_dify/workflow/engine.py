import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any, Literal

from pydantic import BaseModel

from mini_dify.tracing.tracer import span
from mini_dify.workflow.graph import Graph
from mini_dify.workflow.nodes import Node, NodeContext
from mini_dify.workflow.variables import VariablePool

NodeStatus = Literal["succeeded", "failed", "skipped"]


# ---------- events: what the engine reports while it runs ----------


class WorkflowStarted(BaseModel):
    event: Literal["workflow_started"] = "workflow_started"
    workflow_run_id: uuid.UUID | None = None  # filled in by the service that saves the run
    trace_id: uuid.UUID | None = None
    inputs: dict[str, Any]


class NodeStarted(BaseModel):
    event: Literal["node_started"] = "node_started"
    node_id: str
    node_type: str
    title: str


class NodeFinished(BaseModel):
    event: Literal["node_finished"] = "node_finished"
    node_id: str
    node_type: str
    title: str
    status: NodeStatus
    inputs: dict[str, Any] = {}
    outputs: dict[str, Any] = {}
    error: str | None = None
    elapsed_ms: float = 0.0


class WorkflowFinished(BaseModel):
    event: Literal["workflow_finished"] = "workflow_finished"
    workflow_run_id: uuid.UUID | None = None
    trace_id: uuid.UUID | None = None
    status: Literal["succeeded", "failed"]
    outputs: dict[str, Any] = {}
    error: str | None = None
    elapsed_ms: float


WorkflowEvent = WorkflowStarted | NodeStarted | NodeFinished | WorkflowFinished


# ---------- the engine ----------


class WorkflowEngine:
    """Runs a checked graph. Nodes whose inputs are ready run at the same time.

    Each edge ends up "taken" or "skipped". A node runs once all its incoming edges
    are decided and at least one was taken. If none was taken (the branch it's on
    wasn't chosen), it is skipped, and so are its outgoing edges.
    """

    def __init__(self, graph: Graph, nodes: dict[str, Node], ctx: NodeContext):
        self.graph = graph
        self.nodes = nodes
        self.ctx = ctx

    async def run(self) -> AsyncIterator[WorkflowEvent]:
        start = time.perf_counter()
        pool = VariablePool()
        edge_state: dict[int, Literal["taken", "skipped"]] = {}
        node_status: dict[str, NodeStatus] = {}
        running: dict[asyncio.Task[tuple[NodeFinished, str]], str] = {}
        ready: list[str] = [self.graph.start_id]

        yield WorkflowStarted(inputs=self.ctx.inputs)

        def decide_edges(node_id: str, handle: str | None) -> list[str]:
            """Mark the node's outgoing edges, then return targets whose inputs are all decided.
            handle=None means the node was skipped, so every outgoing edge is skipped."""
            targets = []
            for index, edge in enumerate(self.graph.edges):
                if edge.source == node_id:
                    edge_state[index] = "taken" if edge.source_handle == handle else "skipped"
                    targets.append(edge.target)
            decided = []
            for target in dict.fromkeys(targets):  # unique, keeps order
                incoming = [i for i, e in enumerate(self.graph.edges) if e.target == target]
                if all(i in edge_state for i in incoming):
                    decided.append(target)
            return decided

        try:
            while ready or running:
                # 1. Start every ready node. Skipped ones finish at once and may skip others.
                while ready:
                    node_id = ready.pop(0)
                    node = self.nodes[node_id]
                    incoming = [i for i, e in enumerate(self.graph.edges) if e.target == node_id]
                    if incoming and not any(edge_state[i] == "taken" for i in incoming):
                        node_status[node_id] = "skipped"
                        yield NodeFinished(
                            node_id=node_id, node_type=node.type, title=node.title, status="skipped"
                        )
                        ready.extend(decide_edges(node_id, None))
                        continue
                    yield NodeStarted(node_id=node_id, node_type=node.type, title=node.title)
                    running[asyncio.create_task(self._execute(node, pool))] = node_id

                if not running:
                    break

                # 2. Wait until at least one running node finishes.
                done, _ = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
                for task in sorted(done, key=lambda t: self.graph.order.index(running[t])):
                    node_id = running.pop(task)
                    finished, handle = task.result()
                    node_status[node_id] = finished.status
                    yield finished
                    if finished.status == "failed":
                        yield WorkflowFinished(
                            status="failed",
                            error=f"Node {node_id!r} failed: {finished.error}",
                            elapsed_ms=elapsed_since(start),
                        )
                        return
                    pool.set_outputs(node_id, finished.outputs)
                    ready.extend(decide_edges(node_id, handle))
        finally:
            # A failure (or the client going away) stops the run: cancel what's still going.
            for task in running:
                task.cancel()
            if running:
                await asyncio.gather(*running, return_exceptions=True)

        outputs: dict[str, Any] = {}
        for node_id in self.graph.order:
            if self.nodes[node_id].type == "end" and node_status.get(node_id) == "succeeded":
                outputs.update(pool.get_outputs(node_id))
        yield WorkflowFinished(status="succeeded", outputs=outputs, elapsed_ms=elapsed_since(start))

    async def _execute(self, node: Node, pool: VariablePool) -> tuple[NodeFinished, str]:
        """Run one node inside its own span, so the trace shows every node with its
        inputs and outputs, and the LLM calls or searches it made as its children."""
        async with span(node.title, "chain") as record:
            record.attributes = {"node_id": node.id, "node_type": node.type}
            finished, handle = await self._run_node(node, pool)
            record.inputs, record.outputs = finished.inputs, finished.outputs
            if finished.status == "failed":
                record.fail(finished.error or "failed")
        return finished, handle

    async def _run_node(self, node: Node, pool: VariablePool) -> tuple[NodeFinished, str]:
        """Run one node and turn any error into a failed result (never raises).
        Also returns the handle whose edges to follow next."""
        start = time.perf_counter()
        base = {"node_id": node.id, "node_type": node.type, "title": node.title}
        try:
            result = await node.run(pool, self.ctx)
        except Exception as exc:  # any failure becomes part of the run's record
            message = str(exc) or type(exc).__name__
            failed = NodeFinished(
                **base, status="failed", error=message, elapsed_ms=elapsed_since(start)
            )
            return failed, ""
        finished = NodeFinished(
            **base,
            status="succeeded",
            inputs=result.inputs,
            outputs=result.outputs,
            elapsed_ms=elapsed_since(start),
        )
        return finished, result.handle


def elapsed_since(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)
