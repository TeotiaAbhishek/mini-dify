import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic_core import to_jsonable_python
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from mini_dify.config import settings
from mini_dify.models import Span

logger = logging.getLogger(__name__)

# The same kinds as Dify's trace model: a chain of steps, an LLM call, a search,
# a tool call, an agent loop.
SpanKind = Literal["chain", "llm", "retriever", "tool", "agent"]

MAX_TEXT = 10_000  # longer strings are cut, so one huge prompt can't bloat the table


@dataclass
class SpanRecord:
    """One timed step, kept in memory until the whole trace is saved."""

    name: str
    kind: SpanKind
    trace_id: uuid.UUID | None
    parent_id: uuid.UUID | None
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    attributes: dict[str, Any] = field(default_factory=dict)
    status: Literal["ok", "error"] = "ok"
    error: str | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    ended_at: datetime | None = None
    duration_ms: float = 0.0
    _start: float = field(default_factory=time.perf_counter)

    def fail(self, error: BaseException | str) -> None:
        self.status = "error"
        if isinstance(error, asyncio.CancelledError):
            self.error = "cancelled"
        elif isinstance(error, BaseException):
            self.error = str(error) or type(error).__name__
        else:
            self.error = error

    def finish(self) -> None:
        self.ended_at = datetime.now(UTC)
        self.duration_ms = round((time.perf_counter() - self._start) * 1000, 1)


@dataclass
class TraceState:
    """Every span of one trace, in the order they started."""

    trace_id: uuid.UUID
    spans: list[SpanRecord] = field(default_factory=list)


# Which trace and which span are "current" for the code running right now. asyncio
# copies these into every task it creates, so spans started by parallel workflow
# nodes still find their parent.
_current_trace: ContextVar[TraceState | None] = ContextVar("current_trace", default=None)
_current_span: ContextVar[SpanRecord | None] = ContextVar("current_span", default=None)


def _reset(var: ContextVar, token: Token) -> None:
    try:
        var.reset(token)
    except ValueError:
        # The token was made in another context (a generator closed from elsewhere).
        var.set(None)


@asynccontextmanager
async def span(
    name: str, kind: SpanKind, inputs: dict[str, Any] | None = None
) -> AsyncIterator[SpanRecord]:
    """Time a step as a child of the current span. Outside a trace, nothing is recorded,
    so instrumented code works the same whether or not anyone is tracing it."""
    state = _current_trace.get()
    parent = _current_span.get()
    record = SpanRecord(
        name=name,
        kind=kind,
        trace_id=state.trace_id if state else None,
        parent_id=parent.id if parent else None,
        inputs=dict(inputs or {}),
    )
    if state is not None:
        if parent is None:
            record.id = state.trace_id  # the root span's id is the trace's id
        state.spans.append(record)
    token = _current_span.set(record)
    try:
        yield record
    except GeneratorExit:
        raise  # the consumer stopped reading: not an error of this step
    except BaseException as exc:
        record.fail(exc)
        raise
    finally:
        record.finish()
        _reset(_current_span, token)


@asynccontextmanager
async def trace(
    name: str,
    kind: SpanKind,
    inputs: dict[str, Any],
    sessionmaker: async_sessionmaker[AsyncSession],
) -> AsyncIterator[SpanRecord]:
    """Start a trace with a root span, and save every span when it ends (even on errors).
    Inside another trace, it's just a child span."""
    if not settings.tracing_enabled or _current_trace.get() is not None:
        async with span(name, kind, inputs) as record:
            yield record
        return

    state = TraceState(trace_id=uuid.uuid4())
    token = _current_trace.set(state)
    try:
        async with span(name, kind, inputs) as root:
            yield root
    finally:
        _reset(_current_trace, token)
        await save_trace(sessionmaker, state)


def current_trace_id() -> uuid.UUID | None:
    state = _current_trace.get()
    return state.trace_id if state else None


def to_json(value: Any) -> Any:
    """Make a value storable as JSON: unknown types become strings, long strings are cut."""
    value = to_jsonable_python(value, fallback=str)
    if isinstance(value, str) and len(value) > MAX_TEXT:
        return value[:MAX_TEXT] + f"... [{len(value) - MAX_TEXT} more characters]"
    if isinstance(value, dict):
        return {str(key): to_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_json(item) for item in value]
    return value


async def save_trace(sessionmaker: async_sessionmaker[AsyncSession], state: TraceState) -> None:
    """Write all spans in one transaction. A failure is logged, never raised: losing a
    trace must not break the chat or workflow it describes."""
    try:
        async with sessionmaker() as session:
            for position, record in enumerate(state.spans):
                session.add(
                    Span(
                        id=record.id,
                        trace_id=state.trace_id,
                        parent_id=record.parent_id,
                        name=record.name[:100],
                        kind=record.kind,
                        status=record.status,
                        error=record.error,
                        inputs=to_json(record.inputs),
                        outputs=to_json(record.outputs),
                        attributes=to_json(record.attributes),
                        started_at=record.started_at,
                        ended_at=record.ended_at or record.started_at,
                        duration_ms=record.duration_ms,
                        position=position,
                    )
                )
            await session.commit()
    except Exception:
        logger.exception("Could not save trace %s", state.trace_id)
