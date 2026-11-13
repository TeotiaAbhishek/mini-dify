import re
import uuid
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Literal

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from mini_dify.llm.embeddings import EmbeddingProvider
from mini_dify.models import Chunk, Document

SearchMode = Literal["vector", "keyword", "hybrid"]


@dataclass
class RetrievedChunk:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    position: int
    content: str
    score: float


async def vector_search(
    session: AsyncSession, dataset_ids: list[uuid.UUID], query_vector: list[float], top_k: int
) -> list[RetrievedChunk]:
    """Chunks whose meaning is closest to the query (smallest cosine distance)."""
    distance = Chunk.embedding.cosine_distance(query_vector)
    rows = await session.execute(
        select(Chunk, Document.filename, distance.label("distance"))
        .join(Document, Chunk.document_id == Document.id)
        .where(Chunk.dataset_id.in_(dataset_ids))
        .order_by(distance)
        .limit(top_k)
    )
    return [
        RetrievedChunk(
            chunk.id,
            chunk.document_id,
            filename,
            chunk.position,
            chunk.content,
            score=round(1 - dist, 4),
        )
        for chunk, filename, dist in rows
    ]


async def keyword_search(
    session: AsyncSession, dataset_ids: list[uuid.UUID], query: str, top_k: int
) -> list[RetrievedChunk]:
    """Chunks that contain the query's words, ranked by Postgres full-text search."""
    words = re.findall(r"\w+", query.lower())
    if not words:
        return []
    # OR the words together: a chunk matching any word is a candidate, more matches rank higher.
    tsquery = func.to_tsquery("english", " | ".join(words))
    rank = func.ts_rank_cd(Chunk.content_tsv, tsquery)
    rows = await session.execute(
        select(Chunk, Document.filename, rank.label("rank"))
        .join(Document, Chunk.document_id == Document.id)
        .where(Chunk.dataset_id.in_(dataset_ids), Chunk.content_tsv.op("@@")(tsquery))
        .order_by(rank.desc())
        .limit(top_k)
    )
    return [
        RetrievedChunk(
            chunk.id,
            chunk.document_id,
            filename,
            chunk.position,
            chunk.content,
            score=round(float(rank_value), 4),
        )
        for chunk, filename, rank_value in rows
    ]


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievedChunk]], top_k: int, k: int = 60
) -> list[RetrievedChunk]:
    """Merge ranked lists: each chunk scores 1/(k + rank) in every list it appears in."""
    scores: dict[uuid.UUID, float] = defaultdict(float)
    chunks: dict[uuid.UUID, RetrievedChunk] = {}
    for results in result_lists:
        for rank, chunk in enumerate(results, start=1):
            scores[chunk.chunk_id] += 1 / (k + rank)
            chunks.setdefault(chunk.chunk_id, chunk)
    best = sorted(scores, key=scores.__getitem__, reverse=True)[:top_k]
    return [replace(chunks[chunk_id], score=round(scores[chunk_id], 4)) for chunk_id in best]


async def retrieve(
    session: AsyncSession,
    embedder: EmbeddingProvider,
    dataset_ids: list[uuid.UUID],
    query: str,
    top_k: int = 4,
    mode: SearchMode = "hybrid",
) -> list[RetrievedChunk]:
    """Find the chunks most relevant to `query` in the given datasets."""
    if not dataset_ids:
        return []
    if mode == "keyword":
        return await keyword_search(session, dataset_ids, query, top_k)

    query_vector = await embedder.embed_query(query)
    if mode == "vector":
        return await vector_search(session, dataset_ids, query_vector, top_k)

    # Hybrid: fetch extra candidates from both, then fuse the two rankings.
    candidates = top_k * 2
    by_meaning = await vector_search(session, dataset_ids, query_vector, candidates)
    by_keyword = await keyword_search(session, dataset_ids, query, candidates)
    return reciprocal_rank_fusion([by_meaning, by_keyword], top_k)
