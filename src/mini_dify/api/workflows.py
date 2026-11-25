import json
import uuid
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Annotated, Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from mini_dify.config import settings
from mini_dify.db import get_session
from mini_dify.llm.base import LLMProvider
from mini_dify.llm.embeddings import EmbeddingProvider
from mini_dify.llm.registry import get_embedder, get_provider
from mini_dify.models import NodeRun, Workflow, WorkflowRun
from mini_dify.services.workflow_service import compile_workflow, missing_inputs, run_workflow
from mini_dify.tracing.llm import TracedProvider
from mini_dify.workflow.graph import GraphError, GraphSpec
from mini_dify.workflow.nodes import NodeContext

router = APIRouter(prefix="/v1", tags=["workflows"])


def get_http_transport() -> httpx.AsyncBaseTransport | None:
    """The transport HTTP nodes use. None means the real network; tests override it."""
    return None


SessionDep = Annotated[AsyncSession, Depends(get_session)]
ProviderDep = Annotated[LLMProvider, Depends(get_provider)]
EmbedderDep = Annotated[EmbeddingProvider, Depends(get_embedder)]
TransportDep = Annotated[httpx.AsyncBaseTransport | None, Depends(get_http_transport)]


# ---------- Request and response shapes ----------


class WorkflowIn(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    description: str = ""
    graph: GraphSpec


class WorkflowOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    description: str
    graph: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class RunRequest(BaseModel):
    inputs: dict[str, Any] = {}
    stream: bool = False


class NodeRunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    node_id: str
    node_type: str
    title: str
    status: str
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    error: str | None
    elapsed_ms: float


class WorkflowRunOut(BaseModel):
    id: uuid.UUID
    workflow_id: uuid.UUID
    status: str
    inputs: dict[str, Any]
    outputs: dict[str, Any] | None
    error: str | None
    elapsed_ms: float | None
    trace_id: uuid.UUID | None
    created_at: datetime
    node_runs: list[NodeRunOut]


def check_graph(body: WorkflowIn) -> dict[str, Any]:
    """Refuse a graph that can't run, with a message saying why."""
    graph_json = body.graph.model_dump(mode="json")
    try:
        compile_workflow(graph_json)
    except GraphError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return graph_json


async def load_run(session: AsyncSession, run_id: uuid.UUID) -> WorkflowRunOut:
    run = await session.get(WorkflowRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Workflow run not found")
    node_runs = await session.scalars(
        select(NodeRun).where(NodeRun.run_id == run_id).order_by(NodeRun.position)
    )
    return WorkflowRunOut(
        id=run.id,
        workflow_id=run.workflow_id,
        status=run.status,
        inputs=run.inputs,
        outputs=run.outputs,
        error=run.error,
        elapsed_ms=run.elapsed_ms,
        trace_id=run.trace_id,
        created_at=run.created_at,
        node_runs=[NodeRunOut.model_validate(node_run) for node_run in node_runs],
    )


# ---------- Workflows ----------


@router.post("/workflows", response_model=WorkflowOut, status_code=201)
async def create_workflow(body: WorkflowIn, session: SessionDep):
    workflow = Workflow(name=body.name, description=body.description, graph=check_graph(body))
    session.add(workflow)
    await session.commit()
    return workflow


@router.get("/workflows", response_model=list[WorkflowOut])
async def list_workflows(session: SessionDep):
    result = await session.scalars(select(Workflow).order_by(Workflow.created_at.desc()))
    return result.all()


async def get_workflow_or_404(session: AsyncSession, workflow_id: uuid.UUID) -> Workflow:
    workflow = await session.get(Workflow, workflow_id)
    if workflow is None:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return workflow


@router.get("/workflows/{workflow_id}", response_model=WorkflowOut)
async def get_workflow(workflow_id: uuid.UUID, session: SessionDep):
    return await get_workflow_or_404(session, workflow_id)


@router.put("/workflows/{workflow_id}", response_model=WorkflowOut)
async def update_workflow(workflow_id: uuid.UUID, body: WorkflowIn, session: SessionDep):
    workflow = await get_workflow_or_404(session, workflow_id)
    workflow.graph = check_graph(body)
    workflow.name = body.name
    workflow.description = body.description
    await session.commit()
    return workflow


# ---------- Running a workflow ----------


@router.post("/workflows/{workflow_id}/run", response_model=WorkflowRunOut)
async def run(
    workflow_id: uuid.UUID,
    body: RunRequest,
    request: Request,
    session: SessionDep,
    provider: ProviderDep,
    embedder: EmbedderDep,
    transport: TransportDep,
):
    workflow = await get_workflow_or_404(session, workflow_id)
    graph, nodes = compile_workflow(workflow.graph)
    missing = missing_inputs(graph, nodes, body.inputs)
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing inputs: {', '.join(missing)}")

    sessionmaker = request.app.state.sessionmaker
    ctx = NodeContext(
        provider=TracedProvider(provider),
        embedder=embedder,
        sessionmaker=sessionmaker,
        default_model=settings.default_model,
        inputs=body.inputs,
        http_transport=transport,
        allow_private_http=settings.workflow_http_allow_private,
    )
    events = run_workflow(sessionmaker, workflow.id, workflow.graph, ctx, workflow.name)

    # Whole run at once: wait for the end, then return the saved record.
    if not body.stream:
        run_id = None
        async for event in events:
            if event.event == "workflow_finished":
                run_id = event.workflow_run_id
        async with sessionmaker() as read_session:
            return await load_run(read_session, run_id)

    # Streamed: one Server-Sent Event per engine event, as it happens.
    async def event_stream() -> AsyncIterator[str]:
        async for event in events:
            yield f"data: {json.dumps(event.model_dump(mode='json'))}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/workflow-runs/{run_id}", response_model=WorkflowRunOut)
async def get_run(run_id: uuid.UUID, session: SessionDep):
    """A past run, with every node's inputs, outputs and timing: the run's trace."""
    return await load_run(session, run_id)
