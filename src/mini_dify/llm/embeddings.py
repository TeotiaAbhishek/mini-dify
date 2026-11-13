import hashlib
import math
import re
from abc import ABC, abstractmethod

import httpx

from mini_dify.llm.types import ProviderError

# Must match the vector column size in models.py. nomic-embed-text produces 768 numbers.
EMBEDDING_DIM = 768


class EmbeddingProvider(ABC):
    """Turns text into vectors (lists of numbers) whose distance reflects meaning."""

    name: str
    document_prefix: str = ""
    query_prefix: str = ""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns one vector per text, in the same order."""

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self.embed([self.document_prefix + text for text in texts])

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed([self.query_prefix + text]))[0]

    async def aclose(self) -> None:
        return None


class OllamaEmbedding(EmbeddingProvider):
    """Calls Ollama's /api/embed endpoint."""

    name = "ollama"

    def __init__(
        self,
        base_url: str,
        model: str,
        document_prefix: str = "",
        query_prefix: str = "",
        client: httpx.AsyncClient | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.document_prefix = document_prefix
        self.query_prefix = query_prefix
        self.client = client or httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=5.0))

    async def aclose(self) -> None:
        await self.client.aclose()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            response = await self.client.post(
                f"{self.base_url}/api/embed", json={"model": self.model, "input": texts}
            )
        except httpx.HTTPError as exc:
            raise ProviderError(f"Could not reach Ollama at {self.base_url}: {exc}") from exc
        if response.status_code != 200:
            raise ProviderError(f"Ollama returned {response.status_code}: {response.text}")

        vectors = response.json()["embeddings"]
        if vectors and len(vectors[0]) != EMBEDDING_DIM:
            raise ProviderError(
                f"{self.model} returns {len(vectors[0])} dimensions, "
                f"but the database expects {EMBEDDING_DIM}"
            )
        return vectors


class FakeEmbedding(EmbeddingProvider):
    """For tests: a bag-of-words vector. Texts sharing words end up close together.

    Deterministic, instant, and needs no model, so tests can check retrieval logic.
    """

    name = "fake"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    @staticmethod
    def _vector(text: str) -> list[float]:
        vector = [0.0] * EMBEDDING_DIM
        for word in re.findall(r"\w+", text.lower()):
            bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % EMBEDDING_DIM
            vector[bucket] += 1.0
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / norm for x in vector]
