import json
import time
import uuid
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from mini_dify.config import settings
from mini_dify.db import get_session
from mini_dify.llm.base import LLMProvider
from mini_dify.llm.embeddings import EmbeddingProvider
from mini_dify.llm.registry import get_embedder, get_provider
from mini_dify.llm.types import ProviderError, Usage
from mini_dify.models import App, Conversation, Dataset, Message
from mini_dify.prompts import MissingVariablesError, extract_variables, render_template
from mini_dify.rag.retrieval import RetrievedChunk, retrieve
from mini_dify.services.chat_service import build_prompt, load_history

router = APIRouter(prefix="/v1", tags=["apps"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]
ProviderDep = Annotated[LLMProvider, Depends(get_provider)]
EmbedderDep = Annotated[EmbeddingProvider, Depends(get_embedder)]


# ---------- Request and response shapes ----------


class AppCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    system_prompt: str = ""
    model: str | None = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    memory_window: int = Field(default=10, ge=0, le=100)
    dataset_ids: list[uuid.UUID] = []
    retrieval_top_k: int = Field(default=4, ge=1, le=20)


class AppOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    system_prompt: str
    variables: list[str]
    model: str | None
    temperature: float
    memory_window: int
    dataset_ids: list[uuid.UUID]
    retrieval_top_k: int
    created_at: datetime


class ChatMessageRequest(BaseModel):
    query: str = Field(min_length=1)
    inputs: dict[str, str] = {}
    conversation_id: uuid.UUID | None = None
    stream: bool = False


class Citation(BaseModel):
    """One retrieved chunk the answer was based on. `number` matches [1], [2]... in the answer."""

    number: int
    document_id: uuid.UUID
    filename: str
    chunk_id: uuid.UUID
    score: float
    content: str


class ChatMessageResponse(BaseModel):
    conversation_id: uuid.UUID
    message_id: uuid.UUID
    answer: str
    citations: list[Citation]
    usage: Usage
    latency_ms: float
    time_to_first_token_ms: float | None


def to_citations(chunks: list[RetrievedChunk]) -> list[Citation]:
    return [
        Citation(
            number=number,
            document_id=chunk.document_id,
            filename=chunk.filename,
            chunk_id=chunk.chunk_id,
            score=chunk.score,
            content=chunk.content,
        )
        for number, chunk in enumerate(chunks, start=1)
    ]


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    role: str
    content: str
    prompt_tokens: int | None
    completion_tokens: int | None
    created_at: datetime


# ---------- Apps ----------


@router.post("/apps", response_model=AppOut, status_code=201)
async def create_app(body: AppCreate, session: SessionDep):
    for dataset_id in body.dataset_ids:
        if await session.get(Dataset, dataset_id) is None:
            raise HTTPException(status_code=422, detail=f"Dataset {dataset_id} not found")
    # mode="json" turns the UUIDs into strings, which the JSON column can store.
    app = App(**body.model_dump(mode="json"), variables=extract_variables(body.system_prompt))
    session.add(app)
    await session.commit()
    return app


@router.get("/apps", response_model=list[AppOut])
async def list_apps(session: SessionDep):
    result = await session.scalars(select(App).order_by(App.created_at.desc()))
    return result.all()


@router.get("/apps/{app_id}", response_model=AppOut)
async def get_app(app_id: uuid.UUID, session: SessionDep):
    app = await session.get(App, app_id)
    if app is None:
        raise HTTPException(status_code=404, detail="App not found")
    return app


# ---------- Chatting with an app ----------


@router.post("/apps/{app_id}/chat-messages", response_model=ChatMessageResponse)
async def send_chat_message(
    app_id: uuid.UUID,
    body: ChatMessageRequest,
    request: Request,
    session: SessionDep,
    provider: ProviderDep,
    embedder: EmbedderDep,
):
    app = await session.get(App, app_id)
    if app is None:
        raise HTTPException(status_code=404, detail="App not found")

    # 1. Find the conversation, or start a new one with this request's inputs.
    if body.conversation_id is not None:
        conversation = await session.get(Conversation, body.conversation_id)
        if conversation is None or conversation.app_id != app.id:
            raise HTTPException(status_code=404, detail="Conversation not found")
    else:
        try:
            render_template(app.system_prompt, body.inputs)  # fail early if inputs are missing
        except MissingVariablesError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        conversation = Conversation(app_id=app.id, inputs=body.inputs)
        session.add(conversation)
        await session.commit()

    # 2. Knowledge: find the chunks most relevant to the question (if the app has datasets).
    try:
        context = await retrieve(
            session,
            embedder,
            [uuid.UUID(dataset_id) for dataset_id in app.dataset_ids],
            body.query,
            app.retrieval_top_k,
        )
    except ProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    citations = to_citations(context)

    # 3. Memory: the last N messages, then build what the model will see.
    history = await load_history(session, conversation.id, app.memory_window)
    prompt = build_prompt(app, conversation, history, body.query, context)
    model = app.model or settings.default_model

    # 4a. Whole answer at once.
    if not body.stream:
        try:
            result = await provider.chat(model, prompt, app.temperature)
        except ProviderError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        session.add(Message(conversation_id=conversation.id, role="user", content=body.query))
        answer = Message(
            conversation_id=conversation.id,
            role="assistant",
            content=result.content,
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            latency_ms=result.latency_ms,
        )
        session.add(answer)
        await session.commit()
        return ChatMessageResponse(
            conversation_id=conversation.id,
            message_id=answer.id,
            answer=result.content,
            citations=citations,
            usage=result.usage,
            latency_ms=result.latency_ms,
            time_to_first_token_ms=result.time_to_first_token_ms,
        )

    # 4b. Streamed answer. Messages are saved once the stream has finished.
    conversation_id = conversation.id
    sessionmaker = request.app.state.sessionmaker

    async def event_stream() -> AsyncIterator[str]:
        yield f"data: {json.dumps({'conversation_id': str(conversation_id)})}\n\n"
        start = time.perf_counter()
        parts: list[str] = []
        usage = Usage()
        try:
            async for chunk in provider.stream_chat(model, prompt, app.temperature):
                parts.append(chunk.delta)
                if chunk.usage:
                    usage = chunk.usage
                yield f"data: {json.dumps(chunk.model_dump(exclude_none=True))}\n\n"
        except ProviderError as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
            yield "data: [DONE]\n\n"
            return

        # The request's own session may already be closed, so open a fresh one.
        async with sessionmaker() as save_session:
            save_session.add(
                Message(conversation_id=conversation_id, role="user", content=body.query)
            )
            answer = Message(
                conversation_id=conversation_id,
                role="assistant",
                content="".join(parts),
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                latency_ms=round((time.perf_counter() - start) * 1000, 1),
            )
            save_session.add(answer)
            await save_session.commit()
        end_event = {
            "message_id": str(answer.id),
            "citations": [citation.model_dump(mode="json") for citation in citations],
        }
        yield f"data: {json.dumps(end_event)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/conversations/{conversation_id}/messages", response_model=list[MessageOut])
async def list_messages(conversation_id: uuid.UUID, session: SessionDep):
    conversation = await session.get(Conversation, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    result = await session.scalars(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at)
    )
    return result.all()
