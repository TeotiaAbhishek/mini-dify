# mini-dify

A small, from-scratch rebuild of an LLM app platform in the style of [Dify](https://github.com/langgenius/dify), built phase by phase to learn how these platforms work: model providers, chat apps, RAG, agents and tool calling, a workflow engine, MCP, and observability.

This is a learning project, not a production fork. Each phase is a separate PR, and each one is tied to the part of Dify's codebase it mirrors.

## Stack

- **API:** Python 3.12, FastAPI, uvicorn
- **Data:** Postgres 16 + pgvector (relational data and embeddings in one database), SQLAlchemy (async)
- **Queue / cache:** Redis
- **LLM:** Ollama (local models on the GPU) behind a provider interface
- **Tooling:** uv, ruff, pytest, Docker Compose, GitHub Actions

## Running locally

Requirements: Docker, [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com) with a chat model and an embedding model pulled (`ollama pull llama3.2:3b` and `ollama pull nomic-embed-text`).

```bash
cp .env.example .env
docker compose up -d                                   # Postgres + pgvector, Redis
uv sync                                                # install dependencies
uv run alembic upgrade head                            # create / update the database tables
uv run uvicorn mini_dify.main:app --reload --app-dir src
```

Then:

- http://localhost:8000/health: liveness check
- http://localhost:8000/ready: readiness check (Postgres, pgvector, Redis)
- http://localhost:8000/docs: interactive API docs

## Try it

Whole reply at once:

```bash
curl -s localhost:8000/v1/chat -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is Python?"}]}'
```

Streamed token by token (Server-Sent Events):

```bash
curl -N localhost:8000/v1/chat -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Write a haiku about GPUs"}],"stream":true}'
```

Optional fields: `model` (any model you've pulled in Ollama), `temperature` (0 to 2), `max_tokens`.

## Chat apps

An app saves a system prompt (with `{{variables}}`), a model, a temperature and a memory window. Conversations with it are stored in Postgres, and the last N messages are sent back to the model as memory.

```bash
# Create an app
curl -s localhost:8000/v1/apps -H 'content-type: application/json' -d '{
  "name": "Python tutor",
  "system_prompt": "You are a patient Python tutor for {{name}}, a {{level}} programmer.",
  "memory_window": 6
}'

# Start a conversation (inputs fill the variables)
curl -s localhost:8000/v1/apps/<app_id>/chat-messages -H 'content-type: application/json' \
  -d '{"query": "What is a list?", "inputs": {"name": "Abhi", "level": "beginner"}}'

# Continue it (add "stream": true to stream)
curl -s localhost:8000/v1/apps/<app_id>/chat-messages -H 'content-type: application/json' \
  -d '{"query": "And a tuple?", "conversation_id": "<conversation_id>"}'

# Read the history
curl -s localhost:8000/v1/conversations/<conversation_id>/messages
```

## Knowledge bases (RAG)

A dataset holds documents (`.txt`, `.md`, `.pdf`). Each upload is split into overlapping chunks, embedded with `nomic-embed-text`, and stored in Postgres with pgvector. Retrieval can be `vector` (meaning), `keyword` (full-text search) or `hybrid` (both, merged with Reciprocal Rank Fusion). An app linked to datasets answers from them and returns numbered citations.

```bash
# Create a dataset
curl -s localhost:8000/v1/datasets -H 'content-type: application/json' \
  -d '{"name": "Handbook", "chunk_size": 500, "chunk_overlap": 80}'

# Upload a document (indexing runs in the background)
curl -s localhost:8000/v1/datasets/<dataset_id>/documents -F file=@handbook.pdf

# Check its status, then look at the chunks
curl -s localhost:8000/v1/datasets/<dataset_id>/documents
curl -s localhost:8000/v1/documents/<document_id>/chunks

# Search the dataset directly
curl -s localhost:8000/v1/datasets/<dataset_id>/retrieve -H 'content-type: application/json' \
  -d '{"query": "How many vacation days do I get?", "top_k": 4, "mode": "hybrid"}'

# Link it to an app: answers now include "citations"
curl -s localhost:8000/v1/apps -H 'content-type: application/json' \
  -d '{"name": "HR bot", "system_prompt": "You answer HR questions.", "dataset_ids": ["<dataset_id>"]}'
```

## Database migrations

Tables are managed with [Alembic](https://alembic.sqlalchemy.org). After changing `src/mini_dify/models.py`:

```bash
uv run alembic revision --autogenerate -m "describe the change"   # then read the generated file
uv run alembic upgrade head
```

## Tests and lint

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest -v
```

Tests use a separate `minidify_test` database (created by `docker/init.sql` on a fresh volume, or once by hand with `docker compose exec db createdb -U minidify minidify_test`).

CI runs the same checks on every pull request, against real Postgres and Redis service containers. Tests never call a real LLM: they use a fake provider and a mocked HTTP transport, so CI needs no GPU.

## Roadmap

| Phase | What | Status |
|---|---|---|
| 0 | Project skeleton, local stack, CI | ✅ |
| 1 | Model provider layer, streaming, token and latency tracking | ✅ |
| 2 | Chat apps, prompt templates, conversation memory | ✅ |
| 3 | RAG: ingestion, chunking, embeddings, hybrid retrieval, citations | ✅ |
| 4 | Tools and agents (function calling, ReAct) | ⏳ |
| 5 | Workflow engine (graph of LLM, retrieval, branching, code and HTTP nodes) | |
| 6 | MCP client and server | |
| 7 | Tracing and evals | |
| 8 | Public API with keys and rate limits, minimal UI | |
