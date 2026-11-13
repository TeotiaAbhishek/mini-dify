import json

import httpx
import pytest

from mini_dify.llm.embeddings import EMBEDDING_DIM, OllamaEmbedding
from mini_dify.llm.types import ProviderError


def make_embedder(handler) -> OllamaEmbedding:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OllamaEmbedding(
        "http://ollama.test",
        "nomic-embed-text",
        document_prefix="search_document: ",
        query_prefix="search_query: ",
        client=client,
    )


@pytest.mark.anyio
async def test_documents_and_queries_get_their_prefixes():
    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        sent.append(body)
        return httpx.Response(
            200, json={"embeddings": [[0.1] * EMBEDDING_DIM] * len(body["input"])}
        )

    embedder = make_embedder(handler)
    vectors = await embedder.embed_documents(["chunk one", "chunk two"])
    query_vector = await embedder.embed_query("a question")

    assert len(vectors) == 2 and len(query_vector) == EMBEDDING_DIM
    assert sent[0] == {
        "model": "nomic-embed-text",
        "input": ["search_document: chunk one", "search_document: chunk two"],
    }
    assert sent[1]["input"] == ["search_query: a question"]


@pytest.mark.anyio
async def test_wrong_dimension_is_rejected():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": [[0.1] * 384]})

    with pytest.raises(ProviderError, match="384 dimensions"):
        await make_embedder(handler).embed(["text"])


@pytest.mark.anyio
async def test_missing_model_raises_provider_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model 'nomic-embed-text' not found"})

    with pytest.raises(ProviderError, match="not found"):
        await make_embedder(handler).embed(["text"])
