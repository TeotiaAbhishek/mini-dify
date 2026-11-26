import uuid
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from mini_dify.api.apps import AppOut
from mini_dify.config import settings
from mini_dify.db import get_session
from mini_dify.evals.runner import run_eval
from mini_dify.llm.base import LLMProvider
from mini_dify.llm.registry import get_provider
from mini_dify.models import App, EvalCase, EvalResult, EvalRun, EvalSet

router = APIRouter(prefix="/v1", tags=["evals"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]
ProviderDep = Annotated[LLMProvider, Depends(get_provider)]


# ---------- Request and response shapes ----------


class CaseIn(BaseModel):
    query: str = Field(min_length=1)
    inputs: dict[str, str] = {}
    expected_answer: str | None = None
    must_include: list[str] = []
    expected_sources: list[str] = []


class CaseOut(CaseIn):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    position: int


class EvalSetIn(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    description: str = ""
    cases: list[CaseIn] = Field(min_length=1, max_length=500)


class EvalSetOut(BaseModel):
    id: uuid.UUID
    name: str
    description: str
    created_at: datetime
    cases: list[CaseOut]


class RunIn(BaseModel):
    app_id: uuid.UUID
    # Grade answers with an LLM. Only cases with an expected_answer are graded.
    judge: bool = False
    judge_model: str | None = None  # defaults to DEFAULT_MODEL
    # How many cases run at the same time. 1 gives the most honest latencies.
    concurrency: int = Field(default=1, ge=1, le=4)


class ResultOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    case_id: uuid.UUID
    position: int
    answer: str | None
    error: str | None
    scores: dict[str, Any]
    judge_reason: str | None
    passed: bool
    latency_ms: float | None
    prompt_tokens: int
    completion_tokens: int
    trace_id: uuid.UUID | None


class RunSummaryOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    eval_set_id: uuid.UUID
    app_id: uuid.UUID
    judge_model: str | None
    status: str
    error: str | None
    summary: dict[str, Any] | None
    created_at: datetime
    finished_at: datetime | None


class RunOut(RunSummaryOut):
    app_config: dict[str, Any]
    results: list[ResultOut]


class CaseChange(BaseModel):
    position: int
    query: str
    baseline_passed: bool
    passed: bool
    baseline_scores: dict[str, Any]
    scores: dict[str, Any]
    answer: str | None
    trace_id: uuid.UUID | None


class CompareOut(BaseModel):
    run_id: uuid.UUID
    baseline_id: uuid.UUID
    # For each number in the summary: {"baseline": ..., "run": ..., "change": ...}
    summary: dict[str, dict[str, Any]]
    regressions: list[CaseChange]  # passed in the baseline, fails now
    fixes: list[CaseChange]  # failed in the baseline, passes now
    unchanged: int


# ---------- Helpers ----------


async def load_cases(session: AsyncSession, eval_set_id: uuid.UUID) -> list[EvalCase]:
    result = await session.scalars(
        select(EvalCase).where(EvalCase.eval_set_id == eval_set_id).order_by(EvalCase.position)
    )
    return list(result)


async def get_set_or_404(session: AsyncSession, eval_set_id: uuid.UUID) -> EvalSet:
    eval_set = await session.get(EvalSet, eval_set_id)
    if eval_set is None:
        raise HTTPException(status_code=404, detail="Eval set not found")
    return eval_set


async def set_out(session: AsyncSession, eval_set: EvalSet) -> EvalSetOut:
    cases = await load_cases(session, eval_set.id)
    return EvalSetOut(
        id=eval_set.id,
        name=eval_set.name,
        description=eval_set.description,
        created_at=eval_set.created_at,
        cases=[CaseOut.model_validate(case) for case in cases],
    )


async def get_run_or_404(session: AsyncSession, run_id: uuid.UUID) -> EvalRun:
    run = await session.get(EvalRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Eval run not found")
    return run


async def load_results(session: AsyncSession, run_id: uuid.UUID) -> list[EvalResult]:
    result = await session.scalars(
        select(EvalResult).where(EvalResult.run_id == run_id).order_by(EvalResult.position)
    )
    return list(result)


# ---------- Eval sets ----------


@router.post("/eval-sets", response_model=EvalSetOut, status_code=201)
async def create_eval_set(body: EvalSetIn, session: SessionDep):
    eval_set = EvalSet(name=body.name, description=body.description)
    session.add(eval_set)
    await session.flush()  # gives eval_set its id, for the cases below
    for position, case in enumerate(body.cases):
        session.add(EvalCase(eval_set_id=eval_set.id, position=position, **case.model_dump()))
    await session.commit()
    return await set_out(session, eval_set)


@router.get("/eval-sets", response_model=list[EvalSetOut])
async def list_eval_sets(session: SessionDep):
    sets = await session.scalars(select(EvalSet).order_by(EvalSet.created_at.desc()))
    return [await set_out(session, eval_set) for eval_set in sets.all()]


@router.get("/eval-sets/{eval_set_id}", response_model=EvalSetOut)
async def get_eval_set(eval_set_id: uuid.UUID, session: SessionDep):
    return await set_out(session, await get_set_or_404(session, eval_set_id))


# ---------- Runs ----------


@router.post("/eval-sets/{eval_set_id}/runs", response_model=RunOut, status_code=202)
async def start_run(
    eval_set_id: uuid.UUID,
    body: RunIn,
    request: Request,
    background_tasks: BackgroundTasks,
    session: SessionDep,
    provider: ProviderDep,
):
    """Start a run. It works in the background: poll GET /v1/eval-runs/{id} until it's done."""
    await get_set_or_404(session, eval_set_id)
    app = await session.get(App, body.app_id)
    if app is None:
        raise HTTPException(status_code=404, detail="App not found")

    run = EvalRun(
        eval_set_id=eval_set_id,
        app_id=app.id,
        app_config=AppOut.model_validate(app).model_dump(mode="json"),
        judge_model=(body.judge_model or settings.default_model) if body.judge else None,
    )
    session.add(run)
    await session.commit()

    background_tasks.add_task(
        run_eval,
        request.app,
        request.app.state.sessionmaker,
        provider,
        run.id,
        body.concurrency,
    )
    return RunOut(
        **RunSummaryOut.model_validate(run).model_dump(), app_config=run.app_config, results=[]
    )


@router.get("/eval-runs", response_model=list[RunSummaryOut])
async def list_runs(
    session: SessionDep,
    eval_set_id: uuid.UUID | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
):
    """The newest runs first, with their summaries."""
    query = select(EvalRun).order_by(EvalRun.created_at.desc()).limit(limit)
    if eval_set_id:
        query = query.where(EvalRun.eval_set_id == eval_set_id)
    return (await session.scalars(query)).all()


@router.get("/eval-runs/{run_id}", response_model=RunOut)
async def get_run(run_id: uuid.UUID, session: SessionDep):
    """One run with every case's answer, scores and trace id."""
    run = await get_run_or_404(session, run_id)
    results = await load_results(session, run_id)
    return RunOut(
        **RunSummaryOut.model_validate(run).model_dump(),
        app_config=run.app_config,
        results=[ResultOut.model_validate(r) for r in results],
    )


COMPARED_NUMBERS = [
    "pass_rate",
    "contains",
    "source_recall",
    "judge",
    "latency_p50_ms",
    "latency_p95_ms",
    "prompt_tokens",
    "completion_tokens",
]


@router.get("/eval-runs/{run_id}/compare", response_model=CompareOut)
async def compare_runs(run_id: uuid.UUID, baseline: uuid.UUID, session: SessionDep):
    """What changed between a baseline run and this one: the numbers, and which cases
    started failing (regressions) or started passing (fixes)."""
    run = await get_run_or_404(session, run_id)
    base = await get_run_or_404(session, baseline)
    if run.eval_set_id != base.eval_set_id:
        raise HTTPException(status_code=422, detail="Both runs must use the same eval set")
    if run.status != "succeeded" or base.status != "succeeded":
        raise HTTPException(status_code=409, detail="Both runs must have finished successfully")

    summary = {}
    for name in COMPARED_NUMBERS:
        before, after = base.summary.get(name), run.summary.get(name)
        change = round(after - before, 3) if before is not None and after is not None else None
        summary[name] = {"baseline": before, "run": after, "change": change}

    queries = {case.id: case.query for case in await load_cases(session, run.eval_set_id)}
    before_by_case = {r.case_id: r for r in await load_results(session, base.id)}
    regressions, fixes, unchanged = [], [], 0
    for result in await load_results(session, run.id):
        before = before_by_case.get(result.case_id)
        if before is None or before.passed == result.passed:
            unchanged += 1
            continue
        change = CaseChange(
            position=result.position,
            query=queries.get(result.case_id, ""),
            baseline_passed=before.passed,
            passed=result.passed,
            baseline_scores=before.scores,
            scores=result.scores,
            answer=result.answer,
            trace_id=result.trace_id,
        )
        (fixes if result.passed else regressions).append(change)

    return CompareOut(
        run_id=run.id,
        baseline_id=base.id,
        summary=summary,
        regressions=regressions,
        fixes=fixes,
        unchanged=unchanged,
    )
