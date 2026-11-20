import time
import uuid
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from mini_dify.agent.base import AgentAnswer, AgentStep
from mini_dify.api.chat import sse
from mini_dify.config import settings
from mini_dify.db import get_session
from mini_dify.llm.base import LLMProvider
from mini_dify.llm.embeddings import EmbeddingProvider
from mini_dify.llm.registry import get_embedder, get_provider
from mini_dify.llm.types import ProviderError, Usage
from mini_dify.models import App, Conversation, Dataset, Message
from mini_dify.prompts import MissingVariablesError, extract_variables, render_template
from mini_dify.rag.retrieval import RetrievedChunk, retrieve
from mini_dify.services.agent_service import is_agent_app, make_agent
from mini_dify.services.chat_service import build_prompt, load_history
from mini_dify.tools.base import ToolContext
from mini_dify.tools.registry import BUILTIN_TOOLS

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
    tools: list[str] = []
    agent_mode: Literal["function_calling", "react"] = "function_calling"
    max_iterations: int = Field(default=5, ge=1, le=10)


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
    tools: list[str]
    agent_mode: str
    max_iterations: int
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
    agent_steps: list[AgentStep] = []
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
    agent_steps: list[AgentStep] | None = None
    created_at: datetime


async def save_turn(
    session: AsyncSession,
    conversation_id: uuid.UUID,
    query: str,
    answer: str,
    usage: Usage,
    latency_ms: float,
    agent_steps: list[AgentStep] | None = None,
) -> Message:
    """Store the user's question and the assistant's answer."""
    session.add(Message(conversation_id=conversation_id, role="user", content=query))
    message = Message(
        conversation_id=conversation_id,
        role="assistant",
        content=answer,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        latency_ms=latency_ms,
        agent_steps=(
            [step.model_dump(mode="json") for step in agent_steps] if agent_steps else None
        ),
    )
    session.add(message)
    await session.commit()
    return message


# ---------- Apps ----------


@router.post("/apps", response_model=AppOut, status_code=201)
async def create_app(body: AppCreate, session: SessionDep):
    for dataset_id in body.dataset_ids:
        if await session.get(Dataset, dataset_id) is None:
            raise HTTPException(status_code=422, detail=f"Dataset {dataset_id} not found")
    unknown = [name for name in body.tools if name not in BUILTIN_TOOLS]
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown tools: {', '.join(unknown)}. Available: {', '.join(BUILTIN_TOOLS)}",
        )
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

    # Agent apps take a different path: the model decides which tools to call.
    if is_agent_app(app):
        return await agent_chat(app, conversation, body, request, session, provider, embedder)

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

        answer = await save_turn(
            session, conversation.id, body.query, result.content, result.usage, result.latency_ms
        )
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
        yield sse({"conversation_id": str(conversation_id)})
        start = time.perf_counter()
        parts: list[str] = []
        usage = Usage()
        try:
            async for chunk in provider.stream_chat(model, prompt, app.temperature):
                parts.append(chunk.delta)
                if chunk.usage:
                    usage = chunk.usage
                yield sse(chunk.model_dump(exclude_none=True))
        except ProviderError as exc:
            yield sse({"error": str(exc)})
            yield "data: [DONE]\n\n"
            return

        # The request's own session may already be closed, so open a fresh one.
        async with sessionmaker() as save_session:
            latency_ms = round((time.perf_counter() - start) * 1000, 1)
            answer = await save_turn(
                save_session, conversation_id, body.query, "".join(parts), usage, latency_ms
            )
        end_event = {
            "message_id": str(answer.id),
            "citations": [citation.model_dump(mode="json") for citation in citations],
        }
        yield sse(end_event)
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ---------- Chatting with an agent app ----------


async def agent_chat(
    app: App,
    conversation: Conversation,
    body: ChatMessageRequest,
    request: Request,
    session: AsyncSession,
    provider: LLMProvider,
    embedder: EmbeddingProvider,
):
    """Run the agent loop: the model calls tools until it can answer."""
    history = await load_history(session, conversation.id, app.memory_window)
    prompt = build_prompt(app, conversation, history, body.query)
    model = app.model or settings.default_model

    # Whole answer at once.
    if not body.stream:
        ctx = ToolContext(session=session, embedder=embedder)
        agent = make_agent(app, provider, model, ctx)
        start = time.perf_counter()
        steps: list[AgentStep] = []
        try:
            async for event in agent.run(prompt):
                if isinstance(event, AgentStep):
                    steps.append(event)
                else:
                    final: AgentAnswer = event
        except ProviderError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        latency_ms = round((time.perf_counter() - start) * 1000, 1)

        answer = await save_turn(
            session, conversation.id, body.query, final.content, final.usage, latency_ms, steps
        )
        return ChatMessageResponse(
            conversation_id=conversation.id,
            message_id=answer.id,
            answer=final.content,
            citations=to_citations(ctx.retrieved),
            agent_steps=steps,
            usage=final.usage,
            latency_ms=latency_ms,
            time_to_first_token_ms=None,
        )

    # Streamed: one event per tool call as it happens, then the answer.
    sessionmaker = request.app.state.sessionmaker

    async def event_stream() -> AsyncIterator[str]:
        yield sse({"conversation_id": str(conversation.id)})
        start = time.perf_counter()
        steps: list[AgentStep] = []
        # Tools need a database session that lives as long as this stream.
        async with sessionmaker() as agent_session:
            ctx = ToolContext(session=agent_session, embedder=embedder)
            agent = make_agent(app, provider, model, ctx)
            try:
                async for event in agent.run(prompt):
                    if isinstance(event, AgentStep):
                        steps.append(event)
                        yield sse({"agent_step": event.model_dump(mode="json")})
                    else:
                        final = event
            except ProviderError as exc:
                yield sse({"error": str(exc)})
                yield "data: [DONE]\n\n"
                return

            yield sse({"delta": final.content, "done": True, "usage": final.usage.model_dump()})
            latency_ms = round((time.perf_counter() - start) * 1000, 1)
            answer = await save_turn(
                agent_session,
                conversation.id,
                body.query,
                final.content,
                final.usage,
                latency_ms,
                steps,
            )
        citations = to_citations(ctx.retrieved)
        yield sse(
            {
                "message_id": str(answer.id),
                "citations": [citation.model_dump(mode="json") for citation in citations],
            }
        )
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
