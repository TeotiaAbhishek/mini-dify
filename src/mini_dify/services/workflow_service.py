import uuid
from collections.abc import AsyncIterator
from typing import Any

from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from mini_dify.models import NodeRun, WorkflowRun
from mini_dify.workflow.engine import (
    NodeFinished,
    WorkflowEngine,
    WorkflowEvent,
    WorkflowFinished,
    WorkflowStarted,
)
from mini_dify.workflow.graph import Graph, GraphError, GraphSpec
from mini_dify.workflow.nodes import Node, NodeContext, StartNode, build_nodes


def compile_workflow(graph_json: dict[str, Any]) -> tuple[Graph, dict[str, Node]]:
    """Check a stored graph and build its nodes. Raises GraphError if it can't run."""
    try:
        spec = GraphSpec.model_validate(graph_json)
    except ValidationError as exc:
        raise GraphError(f"Invalid graph: {exc.errors()[0]['msg']}") from None
    graph = Graph(spec)
    return graph, build_nodes(graph)


def missing_inputs(graph: Graph, nodes: dict[str, Node], inputs: dict[str, Any]) -> list[str]:
    start = nodes[graph.start_id]
    assert isinstance(start, StartNode)
    return start.missing_inputs(inputs)


async def run_workflow(
    sessionmaker: async_sessionmaker[AsyncSession],
    workflow_id: uuid.UUID,
    graph_json: dict[str, Any],
    ctx: NodeContext,
) -> AsyncIterator[WorkflowEvent]:
    """Run a workflow, saving the run and every node's result as it goes."""
    graph, nodes = compile_workflow(graph_json)
    async with sessionmaker() as session:
        run = WorkflowRun(workflow_id=workflow_id, graph=graph_json, inputs=ctx.inputs)
        session.add(run)
        await session.commit()  # saved now, so the run shows up as "running" straight away

        position = 0
        async for event in WorkflowEngine(graph, nodes, ctx).run():
            if isinstance(event, (WorkflowStarted, WorkflowFinished)):
                event.workflow_run_id = run.id
            if isinstance(event, NodeFinished):
                session.add(
                    NodeRun(
                        run_id=run.id,
                        position=position,
                        **event.model_dump(mode="json", exclude={"event"}),
                    )
                )
                position += 1
            if isinstance(event, WorkflowFinished):
                run.status = event.status
                run.outputs = event.model_dump(mode="json")["outputs"]
                run.error = event.error
                run.elapsed_ms = event.elapsed_ms
                await session.commit()
            yield event
