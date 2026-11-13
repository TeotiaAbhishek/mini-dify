import uuid
from datetime import datetime
from typing import Annotated

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Request,
    UploadFile,
)
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from mini_dify.db import get_session
from mini_dify.llm.embeddings import EmbeddingProvider
from mini_dify.llm.registry import get_embedder
from mini_dify.llm.types import ProviderError
from mini_dify.models import Chunk, Dataset, Document
from mini_dify.rag.extract import SUPPORTED_EXTENSIONS
from mini_dify.rag.indexing import index_document
from mini_dify.rag.retrieval import SearchMode, retrieve

router = APIRouter(prefix="/v1", tags=["datasets"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]
EmbedderDep = Annotated[EmbeddingProvider, Depends(get_embedder)]

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


# ---------- Request and response shapes ----------


class DatasetCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    description: str = ""
    chunk_size: int = Field(default=1000, ge=100, le=4000)
    chunk_overlap: int = Field(default=150, ge=0, le=1000)

    @model_validator(mode="after")
    def overlap_smaller_than_size(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


class DatasetOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    description: str
    chunk_size: int
    chunk_overlap: int
    created_at: datetime


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    dataset_id: uuid.UUID
    filename: str
    status: str
    error: str | None
    char_count: int
    chunk_count: int
    created_at: datetime


class ChunkOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    position: int
    content: str


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=4, ge=1, le=20)
    mode: SearchMode = "hybrid"


class RetrievedChunkOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    position: int
    content: str
    score: float


async def get_dataset_or_404(session: AsyncSession, dataset_id: uuid.UUID) -> Dataset:
    dataset = await session.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset


# ---------- Datasets ----------


@router.post("/datasets", response_model=DatasetOut, status_code=201)
async def create_dataset(body: DatasetCreate, session: SessionDep):
    dataset = Dataset(**body.model_dump())
    session.add(dataset)
    await session.commit()
    return dataset


@router.get("/datasets", response_model=list[DatasetOut])
async def list_datasets(session: SessionDep):
    result = await session.scalars(select(Dataset).order_by(Dataset.created_at.desc()))
    return result.all()


# ---------- Documents ----------


@router.post("/datasets/{dataset_id}/documents", response_model=DocumentOut, status_code=202)
async def upload_document(
    dataset_id: uuid.UUID,
    file: UploadFile,
    request: Request,
    background_tasks: BackgroundTasks,
    session: SessionDep,
    embedder: EmbedderDep,
):
    dataset = await get_dataset_or_404(session, dataset_id)

    filename = file.filename or "upload"
    if not any(filename.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type. Allowed: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File is larger than 10 MB")

    document = Document(dataset_id=dataset.id, filename=filename)
    session.add(document)
    await session.commit()

    # Respond now; extract, split and embed after the response has been sent.
    background_tasks.add_task(
        index_document, request.app.state.sessionmaker, embedder, document.id, data
    )
    return document


@router.get("/datasets/{dataset_id}/documents", response_model=list[DocumentOut])
async def list_documents(dataset_id: uuid.UUID, session: SessionDep):
    await get_dataset_or_404(session, dataset_id)
    result = await session.scalars(
        select(Document)
        .where(Document.dataset_id == dataset_id)
        .order_by(Document.created_at.desc())
    )
    return result.all()


@router.get("/documents/{document_id}/chunks", response_model=list[ChunkOut])
async def list_chunks(document_id: uuid.UUID, session: SessionDep):
    if await session.get(Document, document_id) is None:
        raise HTTPException(status_code=404, detail="Document not found")
    result = await session.scalars(
        select(Chunk).where(Chunk.document_id == document_id).order_by(Chunk.position)
    )
    return result.all()


# ---------- Retrieval testing (Dify calls this "hit testing") ----------


@router.post("/datasets/{dataset_id}/retrieve", response_model=list[RetrievedChunkOut])
async def retrieve_chunks(
    dataset_id: uuid.UUID, body: RetrieveRequest, session: SessionDep, embedder: EmbedderDep
):
    await get_dataset_or_404(session, dataset_id)
    try:
        return await retrieve(session, embedder, [dataset_id], body.query, body.top_k, body.mode)
    except ProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
