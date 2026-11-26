import asyncio
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from mini_dify.api.console_auth import console_headers
from mini_dify.evals.judge import judge_answer
from mini_dify.evals.metrics import case_passed, contains_score, source_recall, summarize
from mini_dify.llm.base import LLMProvider
from mini_dify.models import EvalCase, EvalResult, EvalRun
from mini_dify.tracing.llm import TracedProvider
from mini_dify.tracing.tracer import trace

logger = logging.getLogger(__name__)

# One answer can take a while on a local model, so wait long enough.
APP_CALL_TIMEOUT = 300.0


async def ask_app(client: httpx.AsyncClient, app_id: uuid.UUID, case: EvalCase) -> dict[str, Any]:
    """Send one case to the app through its public API, exactly like a user would."""
    response = await client.post(
        f"/v1/apps/{app_id}/chat-messages", json={"query": case.query, "inputs": case.inputs}
    )
    if response.status_code != 200:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        return {"error": f"HTTP {response.status_code}: {detail}"}
    return response.json()


async def evaluate_case(
    case: EvalCase,
    run: EvalRun,
    client: httpx.AsyncClient,
    sessionmaker: async_sessionmaker[AsyncSession],
    judge: LLMProvider,
) -> dict[str, Any]:
    """Answer one case, score it, save the result, and return it for the summary."""
    reply = await ask_app(client, run.app_id, case)
    error = reply.get("error")
    answer = reply.get("answer")
    usage = reply.get("usage") or {}

    scores: dict[str, Any] = {}
    judge_reason = None
    if answer is not None:
        if case.must_include:
            scores["contains"] = contains_score(answer, case.must_include)
        if case.expected_sources:
            cited = [citation["filename"] for citation in reply.get("citations", [])]
            scores["source_recall"] = source_recall(cited, case.expected_sources)
        if run.judge_model and case.expected_answer:
            # Each grading is its own trace, so you can read what the judge saw.
            try:
                async with trace("judge", "chain", {"query": case.query}, sessionmaker) as root:
                    verdict = await judge_answer(
                        judge, run.judge_model, case.query, case.expected_answer, answer
                    )
                    root.outputs = {"score": verdict.score, "reason": verdict.reason}
                scores["judge"], judge_reason = verdict.score, verdict.reason
            except Exception as exc:  # a broken judge must not stop the whole run
                scores["judge"], judge_reason = None, f"Judge failed: {exc}"

    result = {
        "case_id": case.id,
        "position": case.position,
        "answer": answer,
        "error": error,
        "scores": scores,
        "judge_reason": judge_reason,
        "passed": case_passed(scores, error),
        "latency_ms": reply.get("latency_ms"),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "trace_id": uuid.UUID(reply["trace_id"]) if reply.get("trace_id") else None,
    }
    async with sessionmaker() as session:
        session.add(EvalResult(run_id=run.id, **result))
        await session.commit()
    return result


async def run_eval(
    asgi_app: Any,
    sessionmaker: async_sessionmaker[AsyncSession],
    provider: LLMProvider,
    run_id: uuid.UUID,
    concurrency: int = 1,
) -> None:
    """Run every case of an eval run, then save the summary. Runs in the background."""
    async with sessionmaker() as session:
        run = await session.get(EvalRun, run_id)
        cases = list(
            await session.scalars(
                select(EvalCase)
                .where(EvalCase.eval_set_id == run.eval_set_id)
                .order_by(EvalCase.position)
            )
        )

        # Talk to our own API in-process: no network, but the same code path as a real
        # client, so the eval measures what users actually get.
        transport = httpx.ASGITransport(app=asgi_app)
        limit = asyncio.Semaphore(concurrency)

        async def one(case: EvalCase) -> dict[str, Any]:
            async with limit:
                return await evaluate_case(
                    case, run, client, sessionmaker, TracedProvider(provider)
                )

        try:
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://mini-dify",
                headers=console_headers(),  # the chat endpoint is part of the console API
                timeout=APP_CALL_TIMEOUT,
            ) as client:
                results = await asyncio.gather(*(one(case) for case in cases))
            run.summary = summarize(list(results))
            run.status = "succeeded"
        except Exception as exc:
            logger.exception("Eval run %s failed", run_id)
            run.status, run.error = "failed", str(exc) or type(exc).__name__
        run.finished_at = datetime.now(UTC)
        await session.commit()
