# mini-dify

A small, from-scratch rebuild of an LLM app platform in the style of [Dify](https://github.com/langgenius/dify), built phase by phase to learn how these platforms work: model providers, chat apps, RAG, agents and tool calling, a workflow engine, MCP, and observability.

This is a learning project, not a production fork. Each phase is a separate PR, and each one is tied to the part of Dify's codebase it mirrors.

## Stack

- **API:** Python 3.12, FastAPI, uvicorn
- **Data:** Postgres 16 + pgvector (relational data and embeddings in one database), SQLAlchemy (async)
- **Queue / cache:** Redis
- **Tooling:** uv, ruff, pytest, Docker Compose, GitHub Actions

## Running locally

Requirements: Docker, [uv](https://docs.astral.sh/uv/).

```bash
cp .env.example .env
docker compose up -d                                   # Postgres + pgvector, Redis
uv sync                                                # install dependencies
uv run uvicorn mini_dify.main:app --reload --app-dir src
```

Then:

- http://localhost:8000/health: liveness check
- http://localhost:8000/ready: readiness check (Postgres, pgvector, Redis)
- http://localhost:8000/docs: interactive API docs

## Tests and lint

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest -v
```

CI runs the same checks on every pull request, against real Postgres and Redis service containers.

## Roadmap

| Phase | What | Status |
|---|---|---|
| 0 | Project skeleton, local stack, CI | ✅ |
| 1 | Model provider layer, streaming, token and cost tracking | ⏳ |
| 2 | Chat apps, prompt templates, conversation memory | |
| 3 | RAG: ingestion, chunking, embeddings, hybrid retrieval, reranking, citations | |
| 4 | Tools and agents (function calling, ReAct) | |
| 5 | Workflow engine (graph of LLM, retrieval, branching, code and HTTP nodes) | |
| 6 | MCP client and server | |
| 7 | Tracing and evals | |
| 8 | Public API with keys and rate limits, minimal UI | |
