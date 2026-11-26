"""Take the README screenshots from a running mini-dify, with real models.

It fills mini-dify with examples (a dataset, two apps, a workflow, an eval set with two
runs and an API key, all from examples/hr/), uses the console and the chat page in a
headless browser, and saves PNGs to docs/images/.

    uv run --with playwright playwright install chromium   # once
    uv run --with playwright python scripts/screenshots.py            # qwen2.5:7b
    uv run --with playwright python scripts/screenshots.py llama3.2:3b

If you set CONSOLE_TOKEN (in your shell or in .env), the script sends it.
"""

import json
import os
import sys
import time
from pathlib import Path

import httpx
from playwright.sync_api import Page, sync_playwright

BASE = "http://localhost:8000"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:7b"
ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples" / "hr"
OUT = ROOT / "docs" / "images"


def console_token() -> str:
    """CONSOLE_TOKEN from the shell, or else from .env, like mini-dify itself reads it."""
    if os.environ.get("CONSOLE_TOKEN"):
        return os.environ["CONSOLE_TOKEN"]
    env = ROOT / ".env"
    for line in env.read_text().splitlines() if env.exists() else []:
        name, _, value = line.partition("=")
        if name.strip() == "CONSOLE_TOKEN":
            return value.strip().strip("'\"")
    return ""


TOKEN = console_token()
SLOW_MS = 300_000  # the first answer also loads the model into memory

HR_PROMPT = (
    "You answer {{employee}}'s questions about the company handbook, from the context "
    "only. Cite sources like [1]. Be brief."
)


# ---------- example data, made through the API ----------


def stop_on_error(response: httpx.Response) -> None:
    """Stop at the first failed API call, and say why (a 401 means the token is missing)."""
    if response.is_error:
        response.read()
        request = response.request
        sys.exit(f"{request.method} {request.url.path}: {response.status_code} {response.text}")


def wait_for(check, what: str, seconds: int = 600):
    for _ in range(seconds):
        result = check()
        if result:
            return result
        time.sleep(1)
    sys.exit(f"Timed out waiting for {what}")


def setup(api: httpx.Client) -> dict[str, str]:
    dataset = api.post(
        "/v1/datasets", json={"name": "HR handbook", "description": "Leave and expenses policy"}
    ).json()
    for name in ("leave.txt", "expenses.txt"):
        data = (EXAMPLES / name).read_bytes()
        api.post(f"/v1/datasets/{dataset['id']}/documents", files={"file": (name, data)})

    def indexed():
        documents = api.get(f"/v1/datasets/{dataset['id']}/documents").json()
        return all(d["status"] == "completed" for d in documents)

    wait_for(indexed, "indexing")

    hr_body = {
        "name": "HR helper",
        "system_prompt": HR_PROMPT,
        "model": MODEL,
        "temperature": 0.2,
        "dataset_ids": [dataset["id"]],
        "retrieval_top_k": 1,
    }
    hr = api.post("/v1/apps", json=hr_body).json()
    api.post(
        "/v1/apps",
        json={
            "name": "Calculator agent",
            "system_prompt": "Use the calculator for any arithmetic. Be brief.",
            "model": MODEL,
            "temperature": 0.2,
            "tools": ["calculator", "current_time"],
        },
    )
    key = api.post(f"/v1/apps/{hr['id']}/api-keys", json={"name": "website"}).json()["key"]

    workflow = api.post(
        "/v1/workflows",
        json={
            "name": "Handbook Q&A",
            "description": "Look the question up, then answer from what was found",
            "graph": {
                "nodes": [
                    {"id": "start", "type": "start", "data": {"variables": [{"name": "question"}]}},
                    {
                        "id": "kb",
                        "type": "knowledge_retrieval",
                        "title": "Search the handbook",
                        "data": {"dataset_ids": [dataset["id"]], "query": "{{#start.question#}}"},
                    },
                    {
                        "id": "llm",
                        "type": "llm",
                        "title": "Answer",
                        "data": {
                            "model": MODEL,
                            "prompt": "Answer briefly from this context:\n{{#kb.text#}}\n\n"
                            "Question: {{#start.question#}}",
                        },
                    },
                    {
                        "id": "end",
                        "type": "end",
                        "data": {"outputs": [{"name": "answer", "value": "{{#llm.text#}}"}]},
                    },
                ],
                "edges": [
                    {"source": "start", "target": "kb"},
                    {"source": "kb", "target": "llm"},
                    {"source": "llm", "target": "end"},
                ],
            },
        },
    ).json()

    # Two eval runs: the app finds 1 chunk, then 3. The second run is compared with the first.
    eval_body = json.loads((EXAMPLES / "eval-set.json").read_text())
    for case in eval_body["cases"]:
        case["inputs"] = {"employee": "Alice"}  # the app's prompt has an {{employee}} variable
    eval_set = api.post("/v1/eval-sets", json=eval_body).json()

    def run_eval() -> str:
        run = api.post(f"/v1/eval-sets/{eval_set['id']}/runs", json={"app_id": hr["id"]}).json()

        def done():
            status = api.get(f"/v1/eval-runs/{run['id']}").json()["status"]
            return status != "running"

        wait_for(done, "the eval run", seconds=1200)
        return run["id"]

    run_eval()
    api.put(f"/v1/apps/{hr['id']}", json={**hr_body, "retrieval_top_k": 3})
    second_run = run_eval()

    return {
        "app": hr["id"],
        "dataset": dataset["id"],
        "workflow": workflow["id"],
        "run": second_run,
        "key": key,
    }


# ---------- screenshots ----------


def shot(page: Page, name: str, full_page: bool = False) -> None:
    page.wait_for_timeout(300)  # let the last paint finish
    page.screenshot(path=OUT / name, full_page=full_page)


def console_chat(page: Page, app_id: str, questions: list[str]) -> None:
    page.goto(f"{BASE}/console/#/apps/{app_id}")
    page.get_by_placeholder("employee").fill("Alice")
    for number, question in enumerate(questions, start=1):
        page.get_by_placeholder("Ask something…").fill(question)
        page.get_by_role("button", name="Send").click()
        # The last event of an answer adds a "trace" link under it.
        page.wait_for_function(
            f"document.querySelectorAll('.msg.assistant .meta a').length >= {number}",
            timeout=SLOW_MS,
        )


def public_chat(page: Page, key: str) -> None:
    page.goto(f"{BASE}/ui/")
    page.fill("#key", key)
    page.click("#connect")
    page.wait_for_selector("#query:not([disabled])")
    page.fill("#inputs input[name=employee]", "Bob")
    page.fill("#query", "Can I expense a taxi to the airport?")
    page.click("#send")
    page.wait_for_selector(".assistant details", timeout=SLOW_MS)
    page.locator(".assistant details summary").last.click()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}
    with httpx.Client(
        base_url=BASE,
        timeout=SLOW_MS / 1000,
        headers=headers,
        event_hooks={"response": [stop_on_error]},
    ) as api:
        ids = setup(api)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(
            viewport={"width": 1280, "height": 820}, device_scale_factor=2, color_scheme="light"
        )
        if TOKEN:  # the console reads its token from localStorage
            context.add_init_script(
                f"localStorage.setItem('mini-dify-console-token', {json.dumps(TOKEN)})"
            )
        page = context.new_page()

        console_chat(
            page,
            ids["app"],
            ["How many vacation days do I get?", "And when do unused ones expire?"],
        )
        shot(page, "console-chat.png")

        # The trace of the last answer, with its LLM call selected.
        page.locator(".msg.assistant .meta a").last.click()
        page.wait_for_selector(".span-row")
        page.locator(".span-row", has_text="llm").last.click()
        shot(page, "console-trace.png", full_page=True)

        page.goto(f"{BASE}/console/#/knowledge/{ids['dataset']}")
        page.get_by_placeholder("How many vacation days do I get?").fill("hotel per night")
        page.get_by_role("button", name="Search").click()
        page.wait_for_selector("form pre")
        shot(page, "console-knowledge.png", full_page=True)

        page.goto(f"{BASE}/console/#/workflows/{ids['workflow']}")
        page.locator("form label input").first.fill("How many sick days need no doctor's note?")
        page.get_by_role("button", name="Run").click()
        page.wait_for_selector("text=Outputs", timeout=SLOW_MS)
        shot(page, "console-workflow.png")

        page.goto(f"{BASE}/console/#/eval-runs/{ids['run']}")
        page.get_by_role("button", name="Compare").click()
        page.wait_for_selector("text=Regressions")
        shot(page, "console-evals.png", full_page=True)

        page.goto(f"{BASE}/console/#/apps")
        page.wait_for_selector("table")
        page.set_viewport_size({"width": 1280, "height": 480})
        shot(page, "console-apps.png")

        page.set_viewport_size({"width": 900, "height": 700})
        public_chat(page, ids["key"])
        shot(page, "chat-page.png", full_page=True)
        browser.close()

    for path in sorted(OUT.glob("*.png")):
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
