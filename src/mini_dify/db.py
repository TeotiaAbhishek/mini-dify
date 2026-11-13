from collections.abc import AsyncIterator

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Parent class for every database table. Alembic reads Base.metadata."""


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    """FastAPI dependency: one database session per request, always closed afterwards."""
    async with request.app.state.sessionmaker() as session:
        yield session
