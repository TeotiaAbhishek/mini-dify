import pytest
from fastapi.testclient import TestClient

from mini_dify.config import settings
from mini_dify.llm.embeddings import FakeEmbedding
from mini_dify.llm.fake import FakeProvider
from mini_dify.llm.registry import get_embedder, get_provider
from mini_dify.main import app

ZERO_UUID = "00000000-0000-0000-0000-000000000000"


@pytest.fixture
def client(clean_db):
    provider, embedder = FakeProvider(), FakeEmbedding()
    app.dependency_overrides[get_provider] = lambda: provider
    app.dependency_overrides[get_embedder] = lambda: embedder
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def token(monkeypatch):
    """Turn the console token on for one test."""
    monkeypatch.setattr(settings, "console_token", "s3cret")
    return {"Authorization": "Bearer s3cret"}


def create_app(client, headers=None, **fields):
    response = client.post("/v1/apps", json={"name": "Helper", **fields}, headers=headers)
    assert response.status_code == 201, response.text
    return response.json()


# ---------- editing and deleting apps ----------


def test_update_an_app(client):
    helper = create_app(client, system_prompt="Hi.")
    changed = {"name": "Support", "system_prompt": "You help {{customer}}.", "temperature": 0.2}
    response = client.put(f"/v1/apps/{helper['id']}", json=changed)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["name"] == "Support"
    assert body["variables"] == ["customer"]  # worked out again from the new prompt
    assert body["temperature"] == 0.2
    assert body["memory_window"] == 10  # a field left out goes back to its default
    assert client.get(f"/v1/apps/{helper['id']}").json()["name"] == "Support"


def test_update_checks_the_body_like_create(client):
    helper = create_app(client)
    response = client.put(f"/v1/apps/{helper['id']}", json={"name": "x", "tools": ["nope"]})
    assert response.status_code == 422
    missing = client.put(f"/v1/apps/{ZERO_UUID}", json={"name": "x"})
    assert missing.status_code == 404


def test_delete_an_app_with_its_conversations_and_keys(client):
    helper = create_app(client)
    client.post(f"/v1/apps/{helper['id']}/chat-messages", json={"query": "hi"})
    key = client.post(f"/v1/apps/{helper['id']}/api-keys", json={"name": "web"}).json()

    assert client.delete(f"/v1/apps/{helper['id']}").status_code == 204
    assert client.get(f"/v1/apps/{helper['id']}").status_code == 404
    assert client.delete(f"/v1/apps/{helper['id']}").status_code == 404
    # The key went with the app.
    public = client.get("/public/v1/info", headers={"Authorization": f"Bearer {key['key']}"})
    assert public.status_code == 401


def test_conversation_logs(client):
    helper = create_app(client, system_prompt="You help {{customer}}.")
    url = f"/v1/apps/{helper['id']}/chat-messages"
    first = client.post(url, json={"query": "first", "inputs": {"customer": "Ann"}}).json()
    client.post(url, json={"query": "again", "conversation_id": first["conversation_id"]})
    client.post(url, json={"query": "second", "inputs": {"customer": "Bo"}, "user": "bob"})

    logs = client.get(f"/v1/apps/{helper['id']}/conversations").json()
    assert [c["first_query"] for c in logs] == ["second", "first"]  # newest first
    assert [c["message_count"] for c in logs] == [2, 4]
    assert logs[0]["end_user"] == "bob"
    assert logs[1]["end_user"] is None
    assert logs[1]["inputs"] == {"customer": "Ann"}

    newest = client.get(f"/v1/apps/{helper['id']}/conversations", params={"limit": 1}).json()
    assert [c["id"] for c in newest] == [logs[0]["id"]]
    assert client.get(f"/v1/apps/{ZERO_UUID}/conversations").status_code == 404


# ---------- the console token ----------


def test_no_token_needed_by_default(client):
    assert client.get("/v1/apps").status_code == 200


def test_the_console_token_guards_the_console_api(client, token):
    refused = client.get("/v1/apps")
    assert refused.status_code == 401
    assert refused.headers["WWW-Authenticate"] == "Bearer"
    wrong = client.get("/v1/apps", headers={"Authorization": "Bearer nope"})
    assert wrong.status_code == 401
    assert client.get("/v1/apps", headers=token).status_code == 200
    for path in ("/v1/traces", "/v1/datasets", "/v1/workflows", "/v1/eval-sets", "/v1/tools"):
        assert client.get(path).status_code == 401, path
    ping = {"jsonrpc": "2.0", "id": 1, "method": "ping"}
    assert client.post("/mcp", json=ping).status_code == 401


def test_the_token_leaves_public_routes_alone(client, token):
    helper = create_app(client, headers=token)
    key = client.post(f"/v1/apps/{helper['id']}/api-keys", json={"name": "w"}, headers=token).json()

    assert client.get("/health").status_code == 200
    assert client.get("/console/").status_code == 200  # the page itself, not its data
    assert client.get("/ui/").status_code == 200
    # The public API uses API keys, not the console token.
    info = client.get("/public/v1/info", headers={"Authorization": f"Bearer {key['key']}"})
    assert info.status_code == 200
    assert client.get("/public/v1/info", headers=token).status_code == 401


def test_evals_still_run_with_the_token_set(client, token):
    helper = create_app(client, headers=token)
    eval_set = client.post(
        "/v1/eval-sets",
        json={"name": "smoke", "cases": [{"query": "ping", "must_include": ["ping"]}]},
        headers=token,
    ).json()
    started = client.post(
        f"/v1/eval-sets/{eval_set['id']}/runs", json={"app_id": helper["id"]}, headers=token
    )
    assert started.status_code == 202
    finished = client.get(f"/v1/eval-runs/{started.json()['id']}", headers=token).json()
    # The runner calls the app through the console API, so it has to send the token too.
    assert finished["status"] == "succeeded", finished
    assert finished["results"][0]["passed"] is True


# ---------- the console page ----------


def test_the_console_is_served(client):
    root = client.get("/", follow_redirects=False)
    assert root.status_code == 307
    assert root.headers["location"] == "/console/"

    page = client.get("/console/")
    assert page.status_code == 200
    assert 'src="js/main.js"' in page.text

    script = client.get("/console/js/pages/apps.js")
    assert script.status_code == 200
    # Browsers only run ES modules served with a JavaScript type.
    assert "javascript" in script.headers["content-type"]
    assert client.get("/console/console.css").headers["content-type"].startswith("text/css")
