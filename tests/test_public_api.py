import json

import pytest
from fastapi.testclient import TestClient

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


def create_app(client, **fields):
    response = client.post("/v1/apps", json={"name": "Helper", **fields})
    assert response.status_code == 201, response.text
    return response.json()


def create_key(client, app_id, **fields):
    response = client.post(f"/v1/apps/{app_id}/api-keys", json={"name": "website", **fields})
    assert response.status_code == 201, response.text
    return response.json()


def auth(key):
    return {"Authorization": f"Bearer {key['key']}"}


def ask(client, key, query="hi", **fields):
    body = {"query": query, "user": "alice", **fields}
    return client.post("/public/v1/chat-messages", json=body, headers=auth(key))


# ---------- managing keys ----------


def test_create_list_and_revoke_keys(client):
    helper = create_app(client)
    key = create_key(client, helper["id"], rate_limit_per_minute=10)
    assert key["key"].startswith("mdk-")
    assert len(key["key"]) == 36
    assert key["key"].startswith(key["prefix"])
    assert key["rate_limit_per_minute"] == 10
    assert key["last_used_at"] is None

    listed = client.get(f"/v1/apps/{helper['id']}/api-keys").json()
    assert [k["id"] for k in listed] == [key["id"]]
    assert "key" not in listed[0]  # the full key is only shown once
    assert "key_hash" not in listed[0]

    assert client.delete(f"/v1/api-keys/{key['id']}").status_code == 204
    assert client.delete(f"/v1/api-keys/{key['id']}").status_code == 204  # still revoked
    assert client.get(f"/v1/apps/{helper['id']}/api-keys").json()[0]["revoked_at"] is not None


def test_key_endpoints_404(client):
    assert client.post(f"/v1/apps/{ZERO_UUID}/api-keys", json={"name": "x"}).status_code == 404
    assert client.get(f"/v1/apps/{ZERO_UUID}/api-keys").status_code == 404
    assert client.delete(f"/v1/api-keys/{ZERO_UUID}").status_code == 404


# ---------- authentication ----------


def test_requests_without_a_valid_key_are_refused(client):
    key = create_key(client, create_app(client)["id"])

    missing = client.get("/public/v1/info")
    assert missing.status_code == 401
    assert missing.headers["WWW-Authenticate"] == "Bearer"

    wrong = client.get("/public/v1/info", headers={"Authorization": "Bearer mdk-not-a-key"})
    assert wrong.status_code == 401
    assert wrong.json()["detail"] == "Invalid or revoked API key"

    assert client.get("/public/v1/info", headers=auth(key)).status_code == 200
    client.delete(f"/v1/api-keys/{key['id']}")
    assert client.get("/public/v1/info", headers=auth(key)).status_code == 401


def test_info_describes_the_keys_app(client):
    helper = create_app(client, name="Support", system_prompt="You help {{customer}}.")
    key = create_key(client, helper["id"])
    response = client.get("/public/v1/info", headers=auth(key))
    assert response.json() == {"name": "Support", "mode": "chat", "variables": ["customer"]}
    assert "system_prompt" not in response.json()


def test_last_used_is_recorded(client):
    helper = create_app(client)
    key = create_key(client, helper["id"])
    client.get("/public/v1/info", headers=auth(key))
    listed = client.get(f"/v1/apps/{helper['id']}/api-keys").json()
    assert listed[0]["last_used_at"] is not None


# ---------- chatting ----------


def test_chat_through_the_public_api(client):
    helper = create_app(client)
    key = create_key(client, helper["id"])

    response = ask(client, key, "hello there")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["answer"] == "echo: hello there"
    assert body["trace_id"]
    assert response.headers["X-RateLimit-Limit"] == "60"
    assert response.headers["X-RateLimit-Remaining"] == "59"

    # Continue the same conversation.
    again = ask(client, key, "more", conversation_id=body["conversation_id"])
    assert again.json()["conversation_id"] == body["conversation_id"]

    messages = client.get(
        f"/public/v1/conversations/{body['conversation_id']}/messages",
        params={"user": "alice"},
        headers=auth(key),
    ).json()
    assert [m["content"] for m in messages] == [
        "hello there",
        "echo: hello there",
        "more",
        "echo: more",
    ]


def test_the_user_is_required(client):
    key = create_key(client, create_app(client)["id"])
    response = client.post("/public/v1/chat-messages", json={"query": "hi"}, headers=auth(key))
    assert response.status_code == 422


def test_conversations_belong_to_one_user(client):
    key = create_key(client, create_app(client)["id"])
    conversation_id = ask(client, key, "secret").json()["conversation_id"]

    # Bob can't read or continue Alice's conversation.
    bob_reads = client.get(
        f"/public/v1/conversations/{conversation_id}/messages",
        params={"user": "bob"},
        headers=auth(key),
    )
    assert bob_reads.status_code == 404
    bob_writes = ask(client, key, "me too", user="bob", conversation_id=conversation_id)
    assert bob_writes.status_code == 404

    alice = client.get("/public/v1/conversations", params={"user": "alice"}, headers=auth(key))
    assert [c["id"] for c in alice.json()] == [conversation_id]
    bob = client.get("/public/v1/conversations", params={"user": "bob"}, headers=auth(key))
    assert bob.json() == []


def test_a_key_only_reaches_its_own_app(client):
    first, second = create_app(client, name="First"), create_app(client, name="Second")
    first_key, second_key = create_key(client, first["id"]), create_key(client, second["id"])
    conversation_id = ask(client, first_key).json()["conversation_id"]

    response = ask(client, second_key, conversation_id=conversation_id)
    assert response.status_code == 404
    assert client.get("/public/v1/info", headers=auth(second_key)).json()["name"] == "Second"


def test_streamed_answers_keep_the_rate_limit_headers(client):
    key = create_key(client, create_app(client)["id"])
    body = {"query": "stream me", "user": "alice", "stream": True}
    with client.stream("POST", "/public/v1/chat-messages", json=body, headers=auth(key)) as r:
        assert r.status_code == 200
        assert r.headers["X-RateLimit-Remaining"] == "59"
        events = [line.removeprefix("data: ") for line in r.iter_lines() if line]

    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    assert "".join(c.get("delta", "") for c in chunks) == "echo: stream me"
    assert chunks[-1]["trace_id"]


# ---------- rate limits ----------


def test_rate_limit_per_key(client):
    helper = create_app(client)
    limited = create_key(client, helper["id"], rate_limit_per_minute=2)
    other = create_key(client, helper["id"])

    statuses = [client.get("/public/v1/info", headers=auth(limited)).status_code for _ in range(3)]
    assert statuses == [200, 200, 429]

    refused = client.get("/public/v1/info", headers=auth(limited))
    assert refused.status_code == 429
    assert refused.headers["X-RateLimit-Remaining"] == "0"
    assert 1 <= int(refused.headers["Retry-After"]) <= 60
    assert "2 requests per minute" in refused.json()["detail"]

    # Each key has its own counter.
    assert client.get("/public/v1/info", headers=auth(other)).status_code == 200


# ---------- the chat page ----------


def test_the_chat_page_is_served(client):
    root = client.get("/", follow_redirects=False)
    assert root.status_code == 307
    assert root.headers["location"] == "/ui/"

    page = client.get("/ui/")
    assert page.status_code == 200
    assert page.headers["content-type"].startswith("text/html")
    assert "/public/v1/chat-messages" in page.text
