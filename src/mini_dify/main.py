from contextlib import asynccontextmanager

import redis.asyncio as redis
from fastapi import FastAPI, Request
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from mini_dify.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once at startup: create shared connection pools.
    app.state.engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    app.state.redis = redis.from_url(settings.redis_url)
    yield
    # Runs once at shutdown: close them cleanly.
    await app.state.redis.aclose()
    await app.state.engine.dispose()


app = FastAPI(title="mini-dify", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    """Liveness: the process is up. Touches nothing else."""
    return {"status": "ok"}


@app.get("/ready")
async def ready(request: Request) -> dict:
    """Readiness: Postgres (with pgvector) and Redis are reachable."""
    async with request.app.state.engine.connect() as conn:
        pgvector = await conn.scalar(
            text("select extversion from pg_extension where extname = 'vector'")
        )
    await request.app.state.redis.ping()
    return {"status": "ready", "pgvector": pgvector}