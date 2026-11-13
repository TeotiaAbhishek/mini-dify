import json
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from mini_dify.config import settings
from mini_dify.llm.base import LLMProvider
from mini_dify.llm.registry import get_provider
from mini_dify.llm.types import ChatResult, Message, ProviderError

router = APIRouter(prefix="/v1", tags=["chat"])


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1)
    model: str | None = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)
    stream: bool = False


def sse(data: dict) -> str:
    """Format one Server-Sent Event."""
    return f"data: {json.dumps(data)}\n\n"


@router.post("/chat", response_model=ChatResult)
async def chat(request: ChatRequest, provider: Annotated[LLMProvider, Depends(get_provider)]):
    model = request.model or settings.default_model

    if not request.stream:
        try:
            return await provider.chat(
                model, request.messages, request.temperature, request.max_tokens
            )
        except ProviderError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    async def event_stream() -> AsyncIterator[str]:
        try:
            async for chunk in provider.stream_chat(
                model, request.messages, request.temperature, request.max_tokens
            ):
                yield sse(chunk.model_dump(exclude_none=True))
        except ProviderError as exc:
            # Headers (200 OK) were already sent, so report the error inside the stream.
            yield sse({"error": str(exc)})
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
