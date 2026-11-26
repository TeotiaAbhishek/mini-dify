import uuid
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from mini_dify.db import get_session
from mini_dify.models import ApiKey, App
from mini_dify.public.keys import display_prefix, generate_key, hash_key

router = APIRouter(prefix="/v1", tags=["api keys"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]


class ApiKeyIn(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    rate_limit_per_minute: int | None = Field(default=None, ge=1, le=10_000)


class ApiKeyOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    app_id: uuid.UUID
    name: str
    prefix: str
    rate_limit_per_minute: int | None
    created_at: datetime
    last_used_at: datetime | None
    revoked_at: datetime | None


class ApiKeyCreated(ApiKeyOut):
    # The whole key. It's only in this one response: save it now.
    key: str


@router.post("/apps/{app_id}/api-keys", response_model=ApiKeyCreated, status_code=201)
async def create_api_key(app_id: uuid.UUID, body: ApiKeyIn, session: SessionDep):
    if await session.get(App, app_id) is None:
        raise HTTPException(status_code=404, detail="App not found")
    key = generate_key()
    api_key = ApiKey(
        app_id=app_id,
        name=body.name,
        prefix=display_prefix(key),
        key_hash=hash_key(key),
        rate_limit_per_minute=body.rate_limit_per_minute,
    )
    session.add(api_key)
    await session.commit()
    return ApiKeyCreated(**ApiKeyOut.model_validate(api_key).model_dump(), key=key)


@router.get("/apps/{app_id}/api-keys", response_model=list[ApiKeyOut])
async def list_api_keys(app_id: uuid.UUID, session: SessionDep):
    if await session.get(App, app_id) is None:
        raise HTTPException(status_code=404, detail="App not found")
    result = await session.scalars(
        select(ApiKey).where(ApiKey.app_id == app_id).order_by(ApiKey.created_at)
    )
    return result.all()


@router.delete("/api-keys/{key_id}", status_code=204)
async def revoke_api_key(key_id: uuid.UUID, session: SessionDep):
    """Revoke a key: it stops working at once. The row stays, so you can see it existed."""
    api_key = await session.get(ApiKey, key_id)
    if api_key is None:
        raise HTTPException(status_code=404, detail="API key not found")
    if api_key.revoked_at is None:
        api_key.revoked_at = datetime.now(UTC)
        await session.commit()
    return Response(status_code=204)
