import asyncio
import os

# Tests use their own database so they never wipe your dev data.
# If DATABASE_URL is already set (as it is in CI), that wins.
os.environ.setdefault(
    "DATABASE_URL", "postgresql+asyncpg://minidify:minidify@localhost:5432/minidify_test"
)

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from mini_dify.config import settings


@pytest.fixture
def anyio_backend():
    # Run async tests on asyncio only (anyio would also try trio by default).
    return "asyncio"


@pytest.fixture(scope="session")
def migrated_db():
    """Run the Alembic migrations once per test run, so tests see the real schema."""
    command.upgrade(Config("alembic.ini"), "head")


@pytest.fixture
def clean_db(migrated_db):
    """Empty every table before the test, so tests can't affect each other."""

    async def truncate():
        engine = create_async_engine(settings.database_url)
        async with engine.begin() as conn:
            await conn.execute(text("TRUNCATE apps, conversations, messages CASCADE"))
        await engine.dispose()

    asyncio.run(truncate())
