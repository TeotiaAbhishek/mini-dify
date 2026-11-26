from contextlib import asynccontextmanager
from pathlib import Path

import redis.asyncio as redis
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from mini_dify.api.api_keys import router as api_keys_router
from mini_dify.api.apps import router as apps_router
from mini_dify.api.chat import router as chat_router
from mini_dify.api.datasets import router as datasets_router
from mini_dify.api.evals import router as evals_router
from mini_dify.api.mcp_servers import router as mcp_servers_router
from mini_dify.api.public import router as public_router
from mini_dify.api.tools import router as tools_router
from mini_dify.api.traces import router as traces_router
from mini_dify.api.workflows import router as workflows_router
from mini_dify.config import settings
from mini_dify.llm.registry import get_embedder, get_provider
from mini_dify.mcp.server import router as mcp_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once at startup: create shared connection pools.
    app.state.engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    app.state.sessionmaker = async_sessionmaker(app.state.engine, expire_on_commit=False)
    app.state.redis = redis.from_url(settings.redis_url)
    yield
    # Runs once at shutdown: close them cleanly.
    await app.state.redis.aclose()
    await app.state.engine.dispose()
    await get_provider().aclose()
    await get_embedder().aclose()


app = FastAPI(title="mini-dify", lifespan=lifespan)
app.include_router(chat_router)
app.include_router(apps_router)
app.include_router(datasets_router)
app.include_router(tools_router)
app.include_router(workflows_router)
app.include_router(mcp_servers_router)
app.include_router(mcp_router)
app.include_router(traces_router)
app.include_router(evals_router)
app.include_router(api_keys_router)
app.include_router(public_router)

# The chat page, a single HTML file that talks to the public API.
app.mount("/ui", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="ui")


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse("/ui/")


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
