import math
from typing import Any

# A case passes when every metric it uses reaches its threshold.
JUDGE_PASS_SCORE = 4  # on the judge's 1 to 5 scale


def contains_score(answer: str, must_include: list[str]) -> float | None:
    """The share of required phrases found in the answer, ignoring case.
    None when the case doesn't list any."""
    if not must_include:
        return None
    text = answer.lower()
    found = sum(1 for phrase in must_include if phrase.lower() in text)
    return round(found / len(must_include), 3)


def source_recall(cited: list[str], expected_sources: list[str]) -> float | None:
    """The share of expected files that the answer cited. None when none are expected."""
    if not expected_sources:
        return None
    cited_set = set(cited)
    hits = sum(1 for filename in expected_sources if filename in cited_set)
    return round(hits / len(expected_sources), 3)


def case_passed(scores: dict[str, Any], error: str | None) -> bool:
    """Pass if there was no error, and every metric that was measured is good enough.
    A judge that couldn't give a score (None) counts as a fail."""
    if error:
        return False
    if "contains" in scores and scores["contains"] < 1:
        return False
    if "source_recall" in scores and scores["source_recall"] < 1:
        return False
    if "judge" in scores and (scores["judge"] is None or scores["judge"] < JUDGE_PASS_SCORE):
        return False
    return True


def percentile(values: list[float], pct: float) -> float | None:
    """The value below which `pct` percent of the values fall (nearest rank)."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]


def mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Totals for a run. Each metric is averaged over the cases that measured it."""

    def measured(name: str) -> list[float]:
        return [r["scores"][name] for r in results if r["scores"].get(name) is not None]

    latencies = [r["latency_ms"] for r in results if r["latency_ms"] is not None]
    passed = sum(1 for r in results if r["passed"])
    return {
        "cases": len(results),
        "passed": passed,
        "pass_rate": round(passed / len(results), 3) if results else None,
        "errors": sum(1 for r in results if r["error"]),
        "contains": mean(measured("contains")),
        "source_recall": mean(measured("source_recall")),
        "judge": mean(measured("judge")),
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "prompt_tokens": sum(r["prompt_tokens"] for r in results),
        "completion_tokens": sum(r["completion_tokens"] for r in results),
    }
