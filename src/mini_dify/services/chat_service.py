import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from mini_dify.llm.types import Message as LLMMessage
from mini_dify.models import App, Conversation, Message
from mini_dify.prompts import render_template
from mini_dify.rag.retrieval import RetrievedChunk

CONTEXT_INSTRUCTIONS = (
    "Answer using the context below. Cite the sources you use with their number in "
    "square brackets, like [1]. If the context does not contain the answer, say that "
    "you don't know instead of guessing."
)


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


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Number each retrieved chunk so the model can cite it as [1], [2], ..."""
    sources = "\n\n".join(
        f"[{number}] (from {chunk.filename})\n{chunk.content}"
        for number, chunk in enumerate(chunks, start=1)
    )
    return f"{CONTEXT_INSTRUCTIONS}\n\n<context>\n{sources}\n</context>"


def build_prompt(
    app: App,
    conversation: Conversation,
    history: list[Message],
    query: str,
    context: list[RetrievedChunk] | None = None,
) -> list[LLMMessage]:
    """Turn app config + retrieved context + memory + the new question into model messages."""
    messages: list[LLMMessage] = []

    system_parts = [render_template(app.system_prompt, conversation.inputs)]
    if context:
        system_parts.append(format_context(context))
    system_prompt = "\n\n".join(part for part in system_parts if part)
    if system_prompt:
        messages.append(LLMMessage(role="system", content=system_prompt))

    for past in history:
        messages.append(LLMMessage(role=past.role, content=past.content))

    messages.append(LLMMessage(role="user", content=query))
    return messages
