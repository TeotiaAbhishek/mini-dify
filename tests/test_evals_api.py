import json

import pytest
from fastapi.testclient import TestClient

from mini_dify.llm.base import LLMProvider
from mini_dify.llm.embeddings import FakeEmbedding
from mini_dify.llm.registry import get_embedder, get_provider
from mini_dify.llm.types import ChatChunk, Usage
from mini_dify.main import app

ZERO_UUID = "00000000-0000-0000-0000-000000000000"


class ExamProvider(LLMProvider):
    """Answers questions from a table, and grades like a judge when asked to.

    `answers` maps a word in the question to the reply. `verdict` is what the judge says.
    """

    name = "exam"

    def __init__(self):
        self.answers = {"vacation": "You get 25 vacation days [1].", "capital": "Paris."}
        self.verdict = '{"reason": "Matches the reference.", "score": 5}'
        self.judge_calls = 0

    async def stream_chat(self, model, messages, temperature=0.7, max_tokens=None, tools=None):
        if messages[0].role == "system" and messages[0].content.startswith("You grade"):
            self.judge_calls += 1
            reply = self.verdict
        else:
            question = messages[-1].content.lower()
            reply = next((a for word, a in self.answers.items() if word in question), "No idea.")
        yield ChatChunk(delta=reply)
        yield ChatChunk(
            done=True, finish_reason="stop", usage=Usage(prompt_tokens=7, completion_tokens=3)
        )


@pytest.fixture
def provider():
    return ExamProvider()


@pytest.fixture
def client(clean_db, provider):
    embedder = FakeEmbedding()
    app.dependency_overrides[get_provider] = lambda: provider
    app.dependency_overrides[get_embedder] = lambda: embedder
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def create_app(client, **fields):
    response = client.post("/v1/apps", json={"name": "HR helper", **fields})
    assert response.status_code == 201, response.text
    return response.json()


def create_set(client, cases):
    response = client.post("/v1/eval-sets", json={"name": "HR questions", "cases": cases})
    assert response.status_code == 201, response.text
    return response.json()


def run(client, eval_set, app_id, **options):
    """Start a run. TestClient waits for background tasks, so it's done on return."""
    response = client.post(
        f"/v1/eval-sets/{eval_set['id']}/runs", json={"app_id": app_id, **options}
    )
    assert response.status_code == 202, response.text
    assert response.json()["status"] == "running"
    return client.get(f"/v1/eval-runs/{response.json()['id']}").json()


CASES = [
    {"query": "How many vacation days do I get?", "must_include": ["25", "vacation"]},
    {"query": "What is the capital of France?", "must_include": ["Paris"]},
    {"query": "Who is the CEO?", "must_include": ["Dana"]},
]


def test_create_get_and_list_eval_sets(client):
    created = create_set(client, CASES)
    assert [case["position"] for case in created["cases"]] == [0, 1, 2]
    assert created["cases"][0]["must_include"] == ["25", "vacation"]
    assert created["cases"][0]["expected_answer"] is None

    assert client.get(f"/v1/eval-sets/{created['id']}").json() == created
    assert [s["id"] for s in client.get("/v1/eval-sets").json()] == [created["id"]]
    assert client.get(f"/v1/eval-sets/{ZERO_UUID}").status_code == 404


def test_an_eval_set_needs_cases(client):
    response = client.post("/v1/eval-sets", json={"name": "empty", "cases": []})
    assert response.status_code == 422


def test_run_scores_every_case_and_sums_up(client):
    helper = create_app(client, system_prompt="Be brief.")
    result = run(client, create_set(client, CASES), helper["id"])

    assert result["status"] == "succeeded"
    assert result["app_config"]["system_prompt"] == "Be brief."
    assert result["judge_model"] is None
    rows = [(r["position"], r["scores"], r["passed"]) for r in result["results"]]
    assert rows == [
        (0, {"contains": 1.0}, True),
        (1, {"contains": 1.0}, True),
        (2, {"contains": 0.0}, False),
    ]
    assert result["results"][2]["answer"] == "No idea."

    summary = result["summary"]
    assert summary["cases"] == 3
    assert summary["passed"] == 2
    assert summary["pass_rate"] == 0.667
    assert summary["contains"] == 0.667
    assert summary["prompt_tokens"] == 21
    assert summary["latency_p50_ms"] is not None

    # Every answer keeps its trace, so a failed case can be looked into.
    trace_id = result["results"][2]["trace_id"]
    trace = client.get(f"/v1/traces/{trace_id}").json()
    assert trace["inputs"]["query"] == "Who is the CEO?"


def test_source_recall_uses_the_citations(client):
    dataset = client.post("/v1/datasets", json={"name": "Handbook"}).json()
    client.post(
        f"/v1/datasets/{dataset['id']}/documents",
        files={"file": ("leave.txt", b"Every employee gets 25 vacation days per year.")},
    )
    helper = create_app(client, dataset_ids=[dataset["id"]])
    cases = [
        {"query": "How many vacation days?", "expected_sources": ["leave.txt"]},
        {"query": "How many vacation days?", "expected_sources": ["pay.txt"]},
    ]
    result = run(client, create_set(client, cases), helper["id"])

    assert [r["scores"] for r in result["results"]] == [
        {"source_recall": 1.0},
        {"source_recall": 0.0},
    ]
    assert result["summary"]["source_recall"] == 0.5


def test_the_judge_grades_cases_with_a_reference_answer(client, provider):
    helper = create_app(client)
    cases = [
        {"query": "How many vacation days?", "expected_answer": "25 days a year."},
        {"query": "What is the capital of France?"},  # no reference: not graded
    ]
    result = run(client, create_set(client, cases), helper["id"], judge=True, judge_model="judge-1")

    assert result["judge_model"] == "judge-1"
    first, second = result["results"]
    assert first["scores"] == {"judge": 5}
    assert first["judge_reason"] == "Matches the reference."
    assert first["passed"] is True
    assert second["scores"] == {}
    assert provider.judge_calls == 1
    assert result["summary"]["judge"] == 5.0

    # The grading is traced too, separately from the answer.
    judge_traces = [t for t in client.get("/v1/traces").json() if t["name"] == "judge"]
    assert len(judge_traces) == 1
    assert judge_traces[0]["llm_calls"] == 1


def test_a_low_or_unreadable_verdict_fails_the_case(client, provider):
    helper = create_app(client)
    eval_set = create_set(
        client, [{"query": "How many vacation days?", "expected_answer": "25 days."}]
    )

    provider.verdict = '{"reason": "Wrong number.", "score": 2}'
    low = run(client, eval_set, helper["id"], judge=True)["results"][0]
    assert (low["scores"], low["passed"]) == ({"judge": 2}, False)

    provider.verdict = "Looks good to me!"
    unreadable = run(client, eval_set, helper["id"], judge=True)["results"][0]
    assert unreadable["scores"] == {"judge": None}
    assert unreadable["passed"] is False
    assert unreadable["judge_reason"].startswith("Could not read")


def test_an_app_error_is_recorded_on_the_case(client):
    # The app needs a {{name}} input that the case doesn't give.
    helper = create_app(client, system_prompt="You help {{name}}.")
    result = run(client, create_set(client, [{"query": "hi"}]), helper["id"])

    case = result["results"][0]
    assert result["status"] == "succeeded"  # the run worked, the case didn't
    assert case["error"].startswith("HTTP 422")
    assert case["passed"] is False
    assert result["summary"]["errors"] == 1


def test_compare_shows_regressions_and_fixes(client, provider):
    helper = create_app(client)
    eval_set = create_set(client, CASES)
    baseline = run(client, eval_set, helper["id"])

    # The "new version" of the app knows the CEO, but forgot the capital.
    provider.answers = {
        "vacation": "You get 25 vacation days.",
        "capital": "Lyon.",
        "ceo": "Dana Smith.",
    }
    latest = run(client, eval_set, helper["id"])

    response = client.get(f"/v1/eval-runs/{latest['id']}/compare?baseline={baseline['id']}")
    assert response.status_code == 200
    diff = response.json()
    assert [c["query"] for c in diff["regressions"]] == ["What is the capital of France?"]
    assert diff["regressions"][0]["answer"] == "Lyon."
    assert [c["query"] for c in diff["fixes"]] == ["Who is the CEO?"]
    assert diff["unchanged"] == 1
    assert diff["summary"]["pass_rate"] == {"baseline": 0.667, "run": 0.667, "change": 0.0}

    listed = client.get(f"/v1/eval-runs?eval_set_id={eval_set['id']}").json()
    assert [r["id"] for r in listed] == [latest["id"], baseline["id"]]


def test_compare_needs_the_same_eval_set(client):
    helper = create_app(client)
    first = run(client, create_set(client, CASES[:1]), helper["id"])
    other = run(client, create_set(client, CASES[1:]), helper["id"])
    response = client.get(f"/v1/eval-runs/{other['id']}/compare?baseline={first['id']}")
    assert response.status_code == 422


def test_unknown_app_set_and_run(client):
    eval_set = create_set(client, CASES)
    response = client.post(f"/v1/eval-sets/{eval_set['id']}/runs", json={"app_id": ZERO_UUID})
    assert response.status_code == 404
    assert (
        client.post(f"/v1/eval-sets/{ZERO_UUID}/runs", json={"app_id": ZERO_UUID}).status_code
        == 404
    )
    assert client.get(f"/v1/eval-runs/{ZERO_UUID}").status_code == 404


def test_eval_set_can_be_loaded_from_a_json_file(client, tmp_path):
    # The README's way: keep cases in a file under version control, and post it.
    path = tmp_path / "hr.json"
    path.write_text(json.dumps({"name": "HR", "cases": CASES}))
    response = client.post(
        "/v1/eval-sets", content=path.read_bytes(), headers={"content-type": "application/json"}
    )
    assert response.status_code == 201
    assert len(response.json()["cases"]) == 3


def test_cases_can_run_at_the_same_time(client):
    helper = create_app(client)
    result = run(client, create_set(client, CASES), helper["id"], concurrency=3)

    assert [r["passed"] for r in result["results"]] == [True, True, False]
    # Each answer got its own trace, even though they ran together.
    trace_ids = {r["trace_id"] for r in result["results"]}
    assert len(trace_ids) == 3
    for r in result["results"]:
        assert client.get(f"/v1/traces/{r['trace_id']}").json()["span_count"] == 2
