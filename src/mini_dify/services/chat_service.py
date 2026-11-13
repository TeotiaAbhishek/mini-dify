import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from mini_dify.llm.types import Message as LLMMessage
from mini_dify.models import App, Conversation, Message
from mini_dify.prompts import render_template


async def load_history(
    session: AsyncSession, conversation_id: uuid.UUID, limit: int
) -> list[Message]:
    """The last `limit` messages of a conversation, oldest first."""
    if limit <= 0:
        return []
    result = await session.scalars(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.desc())
        .limit(limit)
    )
    return list(reversed(result.all()))


def build_prompt(
    app: App, conversation: Conversation, history: list[Message], query: str
) -> list[LLMMessage]:
    """Turn app config + memory + the new question into the message list the model sees."""
    messages: list[LLMMessage] = []

    system_prompt = render_template(app.system_prompt, conversation.inputs)
    if system_prompt:
        messages.append(LLMMessage(role="system", content=system_prompt))

    for past in history:
        messages.append(LLMMessage(role=past.role, content=past.content))

    messages.append(LLMMessage(role="user", content=query))
    return messages
