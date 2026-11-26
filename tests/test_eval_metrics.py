from mini_dify.evals.judge import build_judge_messages, parse_verdict
from mini_dify.evals.metrics import (
    case_passed,
    contains_score,
    percentile,
    source_recall,
    summarize,
)


def test_contains_score_ignores_case_and_counts_each_phrase():
    assert contains_score("You get 25 Vacation days.", ["25", "vacation days"]) == 1.0
    assert contains_score("You get 25 days.", ["25", "vacation", "March"]) == 0.333
    assert contains_score("anything", []) is None


def test_source_recall():
    assert source_recall(["leave.txt", "pay.txt"], ["leave.txt"]) == 1.0
    assert source_recall(["pay.txt"], ["leave.txt", "pay.txt"]) == 0.5
    assert source_recall([], ["leave.txt"]) == 0.0
    assert source_recall(["leave.txt"], []) is None


def test_case_passed_needs_every_measured_metric():
    assert case_passed({}, None)
    assert case_passed({"contains": 1.0, "source_recall": 1.0, "judge": 4}, None)
    assert not case_passed({"contains": 0.5}, None)
    assert not case_passed({"source_recall": 0.0}, None)
    assert not case_passed({"judge": 3}, None)
    assert not case_passed({"judge": None}, None)  # the judge couldn't score it
    assert not case_passed({}, "HTTP 502: model is down")


def test_percentile_uses_the_nearest_rank():
    values = [100.0, 300.0, 200.0, 400.0]
    assert percentile(values, 50) == 200.0
    assert percentile(values, 95) == 400.0
    assert percentile([], 50) is None


def test_summarize_averages_only_what_was_measured():
    results = [
        {
            "scores": {"contains": 1.0, "judge": 5},
            "passed": True,
            "error": None,
            "latency_ms": 100.0,
            "prompt_tokens": 10,
            "completion_tokens": 5,
        },
        {
            "scores": {"contains": 0.5},
            "passed": False,
            "error": None,
            "latency_ms": 300.0,
            "prompt_tokens": 20,
            "completion_tokens": 5,
        },
        {
            "scores": {},
            "passed": False,
            "error": "HTTP 502: down",
            "latency_ms": None,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        },
    ]
    summary = summarize(results)
    assert summary["cases"] == 3
    assert summary["passed"] == 1
    assert summary["pass_rate"] == 0.333
    assert summary["errors"] == 1
    assert summary["contains"] == 0.75
    assert summary["judge"] == 5.0
    assert summary["source_recall"] is None
    assert summary["latency_p50_ms"] == 100.0
    assert summary["prompt_tokens"] == 30


def test_parse_verdict_reads_json_even_inside_other_text():
    assert parse_verdict('{"reason": "Correct.", "score": 5}').score == 5
    wrapped = 'Sure! ```json\n{"reason": "Misses the date.", "score": 3}\n```'
    verdict = parse_verdict(wrapped)
    assert (verdict.score, verdict.reason) == (3, "Misses the date.")


def test_parse_verdict_falls_back_and_gives_up_politely():
    assert parse_verdict("Score: 4. Mostly right.").score == 4
    assert parse_verdict('{"score": 9}').score is None  # out of range
    unreadable = parse_verdict("I think it's fine")
    assert unreadable.score is None
    assert unreadable.reason.startswith("Could not read")


def test_judge_prompt_has_all_three_parts():
    system, user = build_judge_messages("How many days?", "25 days", "You get 25.")
    assert system.role == "system" and "1 to 5" in system.content
    assert "How many days?" in user.content
    assert "Reference answer:\n25 days" in user.content
    assert "Assistant's answer:\nYou get 25." in user.content
