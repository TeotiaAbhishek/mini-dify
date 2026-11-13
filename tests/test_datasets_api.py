import json

import pytest
from fastapi.testclient import TestClient

from mini_dify.llm.embeddings import FakeEmbedding
from mini_dify.llm.fake import FakeProvider
from mini_dify.llm.registry import get_embedder, get_provider
from mini_dify.main import app

HANDBOOK = """Refund policy

Customers can ask for a refund within 30 days of purchase. Refunds go back to the
original payment method and take 5 to 7 business days to arrive.

Shipping

Orders ship from our Sydney warehouse. Standard shipping takes 3 business days
within Australia. Express shipping arrives the next business day.

Opening hours

The support team is available Monday to Friday, 9am to 5pm Sydney time."""


class RecordingProvider(FakeProvider):
    def __init__(self):
        self.prompts = []

    async def stream_chat(self, model, messages, temperature=0.7, max_tokens=None):
        self.prompts.append([(m.role, m.content) for m in messages])
        async for chunk in super().stream_chat(model, messages, temperature, max_tokens):
            yield chunk


@pytest.fixture
def provider():
    return RecordingProvider()


@pytest.fixture
def client(clean_db, provider):
    embedder = FakeEmbedding()
    app.dependency_overrides[get_provider] = lambda: provider
    app.dependency_overrides[get_embedder] = lambda: embedder
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def create_dataset(client, **fields):
    response = client.post("/v1/datasets", json={"name": "Handbook", **fields})
    assert response.status_code == 201, response.text
    return response.json()


def upload(client, dataset_id, filename, content: bytes):
    return client.post(f"/v1/datasets/{dataset_id}/documents", files={"file": (filename, content)})


def test_upload_indexes_document_into_chunks(client):
    dataset = create_dataset(client, chunk_size=200, chunk_overlap=30)

    response = upload(client, dataset["id"], "handbook.txt", HANDBOOK.encode())
    assert response.status_code == 202
    assert response.json()["status"] == "pending"

    # TestClient runs background tasks before returning, so indexing is already done.
    documents = client.get(f"/v1/datasets/{dataset['id']}/documents").json()
    assert documents[0]["status"] == "completed"
    assert documents[0]["chunk_count"] > 1

    chunks = client.get(f"/v1/documents/{documents[0]['id']}/chunks").json()
    assert [c["position"] for c in chunks] == list(range(len(chunks)))
    assert all(len(c["content"]) <= 200 for c in chunks)


def test_unsupported_file_is_rejected(client):
    dataset = create_dataset(client)
    response = upload(client, dataset["id"], "photo.png", b"not text")
    assert response.status_code == 415


def test_empty_file_ends_in_error_status(client):
    dataset = create_dataset(client)
    upload(client, dataset["id"], "empty.txt", b"   ")
    document = client.get(f"/v1/datasets/{dataset['id']}/documents").json()[0]
    assert document["status"] == "error"
    assert "No text" in document["error"]


def test_overlap_must_be_smaller_than_chunk_size(client):
    response = client.post(
        "/v1/datasets", json={"name": "bad", "chunk_size": 200, "chunk_overlap": 200}
    )
    assert response.status_code == 422


@pytest.mark.parametrize("mode", ["vector", "keyword", "hybrid"])
def test_retrieve_finds_the_relevant_chunk(client, mode):
    dataset = create_dataset(client, chunk_size=200, chunk_overlap=0)
    upload(client, dataset["id"], "handbook.txt", HANDBOOK.encode())

    response = client.post(
        f"/v1/datasets/{dataset['id']}/retrieve",
        json={"query": "how many days do refunds take", "top_k": 2, "mode": mode},
    )
    assert response.status_code == 200
    results = response.json()
    assert "refund" in results[0]["content"].lower()
    assert results[0]["filename"] == "handbook.txt"


def test_app_answers_with_retrieved_context_and_citations(client, provider):
    dataset = create_dataset(client, chunk_size=200, chunk_overlap=0)
    upload(client, dataset["id"], "handbook.txt", HANDBOOK.encode())
    created = client.post(
        "/v1/apps",
        json={
            "name": "Support bot",
            "system_prompt": "You are a support assistant.",
            "dataset_ids": [dataset["id"]],
            "retrieval_top_k": 2,
        },
    ).json()

    response = client.post(
        f"/v1/apps/{created['id']}/chat-messages", json={"query": "When is support open?"}
    )
    body = response.json()

    assert response.status_code == 200
    assert [c["number"] for c in body["citations"]] == [1, 2]
    assert "Monday to Friday" in body["citations"][0]["content"]

    system_prompt = provider.prompts[0][0][1]
    assert system_prompt.startswith("You are a support assistant.")
    assert "<context>" in system_prompt
    assert "[1] (from handbook.txt)" in system_prompt


def test_streamed_answer_ends_with_citations(client):
    dataset = create_dataset(client, chunk_size=200, chunk_overlap=0)
    upload(client, dataset["id"], "handbook.txt", HANDBOOK.encode())
    created = client.post(
        "/v1/apps", json={"name": "Support bot", "dataset_ids": [dataset["id"]]}
    ).json()

    with client.stream(
        "POST",
        f"/v1/apps/{created['id']}/chat-messages",
        json={"query": "express shipping", "stream": True},
    ) as response:
        events = [line.removeprefix("data: ") for line in response.iter_lines() if line]

    end_event = json.loads(events[-2])
    assert "message_id" in end_event
    assert "Express" in end_event["citations"][0]["content"]


def test_app_with_unknown_dataset_is_rejected(client):
    response = client.post(
        "/v1/apps",
        json={"name": "x", "dataset_ids": ["00000000-0000-0000-0000-000000000000"]},
    )
    assert response.status_code == 422
