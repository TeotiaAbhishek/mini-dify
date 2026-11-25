import uuid
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from mini_dify.db import get_session
from mini_dify.models import Span

router = APIRouter(prefix="/v1", tags=["traces"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]


class TraceSummary(BaseModel):
    trace_id: uuid.UUID
    name: str
    status: str
    error: str | None
    inputs: dict[str, Any]
    started_at: datetime
    duration_ms: float
    span_count: int
    llm_calls: int
    prompt_tokens: int
    completion_tokens: int


class SpanNode(BaseModel):
    id: uuid.UUID
    name: str
    kind: str
    status: str
    error: str | None
    # When the span started, counted from the start of the trace.
    offset_ms: float
    duration_ms: float
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    attributes: dict[str, Any]
    children: list["SpanNode"] = []


class TraceDetail(TraceSummary):
    root: SpanNode


def summarize(spans: list[Span]) -> TraceSummary:
    """Totals for one trace: how many spans and LLM calls, and how many tokens."""
    root = next(s for s in spans if s.parent_id is None)
    llm_spans = [s for s in spans if s.kind == "llm"]
    return TraceSummary(
        trace_id=root.trace_id,
        name=root.name,
        status=root.status,
        error=root.error,
        inputs=root.inputs,
        started_at=root.started_at,
        duration_ms=root.duration_ms,
        span_count=len(spans),
        llm_calls=len(llm_spans),
        prompt_tokens=sum(s.attributes.get("prompt_tokens") or 0 for s in llm_spans),
        completion_tokens=sum(s.attributes.get("completion_tokens") or 0 for s in llm_spans),
    )


def build_tree(spans: list[Span]) -> SpanNode:
    """Turn the flat list of spans into a tree, using parent_id."""
    root = next(s for s in spans if s.parent_id is None)
    nodes = {
        s.id: SpanNode(
            id=s.id,
            name=s.name,
            kind=s.kind,
            status=s.status,
            error=s.error,
            offset_ms=round((s.started_at - root.started_at).total_seconds() * 1000, 1),
            duration_ms=s.duration_ms,
            inputs=s.inputs,
            outputs=s.outputs,
            attributes=s.attributes,
        )
        for s in spans
    }
    for s in spans:  # in start order, so children stay in the order they started
        if s.parent_id is not None and s.parent_id in nodes:
            nodes[s.parent_id].children.append(nodes[s.id])
    return nodes[root.id]


async def load_spans(session: AsyncSession, trace_id: uuid.UUID) -> list[Span]:
    result = await session.scalars(
        select(Span).where(Span.trace_id == trace_id).order_by(Span.position)
    )
    spans = list(result)
    if not spans:
        raise HTTPException(status_code=404, detail="Trace not found")
    return spans


@router.get("/traces", response_model=list[TraceSummary])
async def list_traces(
    session: SessionDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    status: str | None = None,
):
    """The newest traces first, with their totals."""
    roots = select(Span).where(Span.parent_id.is_(None))
    if status:
        roots = roots.where(Span.status == status)
    roots = roots.order_by(Span.started_at.desc()).limit(limit)
    trace_ids = [root.trace_id for root in await session.scalars(roots)]
    if not trace_ids:
        return []

    # One query for every span of these traces, grouped in Python.
    by_trace: dict[uuid.UUID, list[Span]] = {trace_id: [] for trace_id in trace_ids}
    for s in await session.scalars(
        select(Span).where(Span.trace_id.in_(trace_ids)).order_by(Span.position)
    ):
        by_trace[s.trace_id].append(s)
    return [summarize(by_trace[trace_id]) for trace_id in trace_ids]


@router.get("/traces/{trace_id}", response_model=TraceDetail)
async def get_trace(trace_id: uuid.UUID, session: SessionDep):
    """One trace as a tree of spans, with every input and output."""
    spans = await load_spans(session, trace_id)
    return TraceDetail(**summarize(spans).model_dump(), root=build_tree(spans))


def describe(node: SpanNode) -> str:
    """The short note after a span's name in the text view."""
    notes = []
    if node.kind == "llm":
        a = node.attributes
        notes.append(f"{a.get('prompt_tokens')}+{a.get('completion_tokens')} tokens")
        if a.get("time_to_first_token_ms") is not None:
            notes.append(f"first token {a['time_to_first_token_ms']:.0f}ms")
    if node.kind == "retriever":
        count = len(node.outputs.get("chunks", []))
        notes.append(f"{count} chunk{'' if count == 1 else 's'}")
    if node.error:
        notes.append(f"error: {node.error[:80]}")
    return f"  ({', '.join(notes)})" if notes else ""


def render_tree(node: SpanNode, prefix: str = "", last: bool = True, top: bool = True) -> list[str]:
    branch = "" if top else ("└─ " if last else "├─ ")
    line = (
        f"{prefix}{branch}[{node.kind}] {node.name}  "
        f"+{node.offset_ms:.0f}ms  {node.duration_ms:.0f}ms  {node.status}{describe(node)}"
    )
    lines = [line]
    child_prefix = prefix if top else prefix + ("   " if last else "│  ")
    for index, child in enumerate(node.children):
        lines += render_tree(child, child_prefix, index == len(node.children) - 1, top=False)
    return lines


@router.get("/traces/{trace_id}/text", response_class=PlainTextResponse)
async def get_trace_text(trace_id: uuid.UUID, session: SessionDep):
    """The same trace as a text tree, easy to read in a terminal."""
    spans = await load_spans(session, trace_id)
    return "\n".join(render_tree(build_tree(spans))) + "\n"
