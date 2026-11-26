import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from mini_dify.api.apps import (
    ChatMessageRequest,
    ChatMessageResponse,
    EmbedderDep,
    MessageOut,
    ProviderDep,
    send_chat_message,
)
from mini_dify.api.mcp_servers import MCPTransportDep
from mini_dify.config import settings
from mini_dify.db import get_session
from mini_dify.models import ApiKey, App, Conversation, Message
from mini_dify.public.keys import hash_key
from mini_dify.public.rate_limit import hit
from mini_dify.services.agent_service import is_agent_app

router = APIRouter(prefix="/public/v1", tags=["public api"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]
# auto_error=False: we send our own 401. It also adds the "Authorize" button to /docs.
bearer = HTTPBearer(auto_error=False)

# last_used_at is written at most once a minute per key, not on every request.
LAST_USED_PRECISION = timedelta(minutes=1)


@dataclass
class Caller:
    """Who is calling: the API key, and the one app it gives access to."""

    key: ApiKey
    app: App


def unauthorized(detail: str) -> HTTPException:
    return HTTPException(status_code=401, detail=detail, headers={"WWW-Authenticate": "Bearer"})


async def authenticate(
    request: Request,
    response: Response,
    session: SessionDep,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
) -> Caller:
    """Check the API key, then count the request against the key's rate limit."""
    if credentials is None:
        raise unauthorized("Send your API key as 'Authorization: Bearer <key>'")
    api_key = await session.scalar(
        select(ApiKey).where(
            ApiKey.key_hash == hash_key(credentials.credentials), ApiKey.revoked_at.is_(None)
        )
    )
    if api_key is None:
        raise unauthorized("Invalid or revoked API key")
    app = await session.get(App, api_key.app_id)

    limit = api_key.rate_limit_per_minute or settings.rate_limit_per_minute
    result = await hit(request.app.state.redis, f"key:{api_key.id}", limit)
    headers = {
        "X-RateLimit-Limit": str(result.limit),
        "X-RateLimit-Remaining": str(result.remaining),
    }
    if not result.allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit of {limit} requests per minute reached",
            headers={**headers, "Retry-After": str(result.retry_after)},
        )
    response.headers.update(headers)

    now = datetime.now(UTC)
    if api_key.last_used_at is None or now - api_key.last_used_at > LAST_USED_PRECISION:
        api_key.last_used_at = now
        await session.commit()
    return Caller(key=api_key, app=app)


CallerDep = Annotated[Caller, Depends(authenticate)]


async def owned_conversation(
    session: AsyncSession, caller: Caller, conversation_id: uuid.UUID, user: str
) -> Conversation:
    """The conversation, if it belongs to this app and this user. Otherwise 404, the same
    answer as for a conversation that doesn't exist, so ids can't be probed."""
    conversation = await session.get(Conversation, conversation_id)
    if (
        conversation is None
        or conversation.app_id != caller.app.id
        or conversation.end_user != user
    ):
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


# ---------- Request and response shapes ----------


class AppInfo(BaseModel):
    name: str
    mode: str  # "chat" or "agent"
    variables: list[str]  # the inputs a new conversation needs


class PublicChatRequest(BaseModel):
    query: str = Field(min_length=1)
    inputs: dict[str, str] = {}
    conversation_id: uuid.UUID | None = None
    stream: bool = False
    # Your own id for the person asking (any string). Conversations belong to it.
    user: str = Field(min_length=1, max_length=100)


class ConversationOut(BaseModel):
    id: uuid.UUID
    inputs: dict[str, str]
    created_at: datetime


# ---------- Endpoints ----------


@router.get("/info", response_model=AppInfo)
async def info(caller: CallerDep):
    """What the key's app needs: its name, and the inputs for a new conversation."""
    app = caller.app
    return AppInfo(
        name=app.name, mode="agent" if is_agent_app(app) else "chat", variables=app.variables
    )


@router.post("/chat-messages", response_model=ChatMessageResponse)
async def chat_messages(
    body: PublicChatRequest,
    caller: CallerDep,
    request: Request,
    response: Response,
    session: SessionDep,
    provider: ProviderDep,
    embedder: EmbedderDep,
    mcp_transport: MCPTransportDep,
):
    """The same chat as the console's, for the key's app only."""
    if body.conversation_id is not None:
        await owned_conversation(session, caller, body.conversation_id, body.user)
    result = await send_chat_message(
        caller.app.id,
        ChatMessageRequest(**body.model_dump()),
        request,
        session,
        provider,
        embedder,
        mcp_transport,
    )
    if isinstance(result, Response):
        # A streamed answer is its own Response, so the rate limit headers set on
        # `response` wouldn't reach the client. Copy them over.
        result.headers.update(
            {k: v for k, v in response.headers.items() if k.startswith("x-ratelimit")}
        )
    return result


@router.get("/conversations", response_model=list[ConversationOut])
async def list_conversations(user: str, caller: CallerDep, session: SessionDep):
    """This user's conversations with the key's app, newest first."""
    result = await session.scalars(
        select(Conversation)
        .where(Conversation.app_id == caller.app.id, Conversation.end_user == user)
        .order_by(Conversation.created_at.desc())
    )
    return [ConversationOut(id=c.id, inputs=c.inputs, created_at=c.created_at) for c in result]


@router.get("/conversations/{conversation_id}/messages", response_model=list[MessageOut])
async def list_messages(
    conversation_id: uuid.UUID, user: str, caller: CallerDep, session: SessionDep
):
    await owned_conversation(session, caller, conversation_id, user)
    result = await session.scalars(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at)
    )
    return result.all()
