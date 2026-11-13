import uuid

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from mini_dify.llm.embeddings import EmbeddingProvider
from mini_dify.models import Chunk, Dataset, Document
from mini_dify.rag.extract import clean_text, extract_text
from mini_dify.rag.splitter import split_text

# How many chunks to send to the embedding model in one request.
EMBED_BATCH_SIZE = 32


async def index_document(
    sessionmaker: async_sessionmaker[AsyncSession],
    embedder: EmbeddingProvider,
    document_id: uuid.UUID,
    data: bytes,
) -> None:
    """Extract -> clean -> split -> embed -> store. Runs in the background after upload."""
    async with sessionmaker() as session:
        document = await session.get(Document, document_id)
        dataset = await session.get(Dataset, document.dataset_id)
        document.status = "indexing"
        await session.commit()

        try:
            text = clean_text(extract_text(document.filename, data))
            pieces = split_text(text, dataset.chunk_size, dataset.chunk_overlap)
            if not pieces:
                raise ValueError("No text could be extracted from this file")

            vectors: list[list[float]] = []
            for start in range(0, len(pieces), EMBED_BATCH_SIZE):
                batch = pieces[start : start + EMBED_BATCH_SIZE]
                vectors.extend(await embedder.embed_documents(batch))

            session.add_all(
                Chunk(
                    dataset_id=dataset.id,
                    document_id=document.id,
                    position=position,
                    content=piece,
                    embedding=vector,
                )
                for position, (piece, vector) in enumerate(zip(pieces, vectors, strict=True))
            )
            document.char_count = len(text)
            document.chunk_count = len(pieces)
            document.status = "completed"
            await session.commit()
        except Exception as exc:
            # Undo any half-written chunks, then record what went wrong.
            await session.rollback()
            document = await session.get(Document, document_id)
            document.status = "error"
            document.error = str(exc)[:1000]
            await session.commit()
