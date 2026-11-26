from contextlib import asynccontextmanager
from pathlib import Path

import redis.asyncio as redis
from fastapi import Depends, FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from mini_dify.api.api_keys import router as api_keys_router
from mini_dify.api.apps import router as apps_router
from mini_dify.api.chat import router as chat_router
from mini_dify.api.console_auth import require_console_token
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

# The console API: everything for building apps. Guarded by CONSOLE_TOKEN when it's set.
console = [Depends(require_console_token)]
for router in (
    chat_router,
    apps_router,
    datasets_router,
    tools_router,
    workflows_router,
    mcp_servers_router,
    mcp_router,
    traces_router,
    evals_router,
    api_keys_router,
):
    app.include_router(router, dependencies=console)

# The public API: authenticated with per-app API keys instead.
app.include_router(public_router)

STATIC = Path(__file__).parent / "static"
# The console page (for you) and the chat page (for the users of one app). The pages
# themselves are public files; the data they show comes from the APIs above.
app.mount("/console", StaticFiles(directory=STATIC / "console", html=True), name="console")
app.mount("/ui", StaticFiles(directory=STATIC, html=True), name="ui")


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse("/console/")


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
