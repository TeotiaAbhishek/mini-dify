import uuid
from datetime import UTC, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    Boolean,
    Computed,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column

from mini_dify.db import Base
from mini_dify.llm.embeddings import EMBEDDING_DIM


def utcnow() -> datetime:
    return datetime.now(UTC)


class App(Base):
    """A configured chat app: which model, which system prompt, how much memory."""

    __tablename__ = "apps"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(100))
    system_prompt: Mapped[str] = mapped_column(Text, default="")
    variables: Mapped[list[str]] = mapped_column(JSON, default=list)
    model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    temperature: Mapped[float] = mapped_column(Float, default=0.7)
    memory_window: Mapped[int] = mapped_column(Integer, default=10)
    # Knowledge bases this app answers from. server_default fills existing rows
    # when the migration adds these columns to a table that already has data.
    dataset_ids: Mapped[list[str]] = mapped_column(JSON, default=list, server_default=text("'[]'"))
    retrieval_top_k: Mapped[int] = mapped_column(Integer, default=4, server_default=text("4"))
    # Agent settings. An app with at least one tool is an agent app.
    tools: Mapped[list[str]] = mapped_column(JSON, default=list, server_default=text("'[]'"))
    agent_mode: Mapped[str] = mapped_column(
        String(20), default="function_calling", server_default=text("'function_calling'")
    )
    max_iterations: Mapped[int] = mapped_column(Integer, default=5, server_default=text("5"))
    # MCP servers whose tools this app's agent can use (ids from the mcp_servers table).
    mcp_server_ids: Mapped[list[str]] = mapped_column(
        JSON, default=list, server_default=text("'[]'")
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Conversation(Base):
    """One chat session with an app. Holds the variable values for its system prompt."""

    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    app_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("apps.id", ondelete="CASCADE"), index=True)
    inputs: Mapped[dict[str, str]] = mapped_column(JSON, default=dict)
    # Who this conversation belongs to, for conversations started through the public
    # API: the caller's own id for its user. None for conversations from the console.
    end_user: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Message(Base):
    """One message in a conversation, from the user or the assistant."""

    __tablename__ = "messages"
    __table_args__ = (Index("ix_messages_conversation_created", "conversation_id", "created_at"),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE")
    )
    role: Mapped[str] = mapped_column(String(20))
    content: Mapped[str] = mapped_column(Text)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    # For agent answers: the tool calls made on the way to the answer.
    agent_steps: Mapped[list[dict] | None] = mapped_column(JSON, nullable=True)
    # The trace of how this answer was made (for assistant messages). Not a foreign key:
    # spans are written after the message, and tracing can be switched off.
    trace_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Dataset(Base):
    """A knowledge base: a named collection of documents that apps can search."""

    __tablename__ = "datasets"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(100))
    description: Mapped[str] = mapped_column(Text, default="")
    chunk_size: Mapped[int] = mapped_column(Integer, default=1000)
    chunk_overlap: Mapped[int] = mapped_column(Integer, default=150)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Document(Base):
    """One uploaded file. `status` tracks indexing: pending -> indexing -> completed / error."""

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), index=True
    )
    filename: Mapped[str] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(20), default="pending")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    char_count: Mapped[int] = mapped_column(Integer, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Chunk(Base):
    """A piece of a document, with its embedding (for meaning) and tsvector (for keywords)."""

    __tablename__ = "chunks"
    __table_args__ = (
        # Approximate nearest-neighbour index for fast vector search by cosine distance.
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        # Inverted index for fast full-text (keyword) search.
        Index("ix_chunks_content_tsv", "content_tsv", postgresql_using="gin"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), index=True
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    position: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))
    # Postgres computes this from `content` on every insert/update.
    content_tsv: Mapped[str] = mapped_column(
        TSVECTOR, Computed("to_tsvector('english', content)", persisted=True)
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Workflow(Base):
    """A saved workflow: a graph of nodes and edges, stored as JSON."""

    __tablename__ = "workflows"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(100))
    description: Mapped[str] = mapped_column(Text, default="")
    graph: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class WorkflowRun(Base):
    """One run of a workflow: its inputs, outputs, status and how long it took."""

    __tablename__ = "workflow_runs"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    workflow_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("workflows.id", ondelete="CASCADE"), index=True
    )
    # A copy of the graph as it was when this run started, so editing the workflow
    # later doesn't change what an old run's record means.
    graph: Mapped[dict] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(20), default="running")
    inputs: Mapped[dict] = mapped_column(JSON, default=dict)
    outputs: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    elapsed_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    trace_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class NodeRun(Base):
    """What one node did in one run: the values it read and wrote, or its error."""

    __tablename__ = "node_runs"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("workflow_runs.id", ondelete="CASCADE"), index=True
    )
    node_id: Mapped[str] = mapped_column(String(50))
    node_type: Mapped[str] = mapped_column(String(30))
    title: Mapped[str] = mapped_column(String(100))
    status: Mapped[str] = mapped_column(String(20))
    inputs: Mapped[dict] = mapped_column(JSON, default=dict)
    outputs: Mapped[dict] = mapped_column(JSON, default=dict)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    elapsed_ms: Mapped[float] = mapped_column(Float, default=0.0)
    # The order nodes finished in, so the run can be replayed step by step.
    position: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class MCPServer(Base):
    """An outside MCP server registered by URL. Its tools are cached in `tools`."""

    __tablename__ = "mcp_servers"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    # Short and unique: it becomes the prefix of each tool's name (deepwiki__ask_question).
    name: Mapped[str] = mapped_column(String(30), unique=True)
    url: Mapped[str] = mapped_column(String(500))
    # Sent with every request, e.g. {"Authorization": "Bearer ..."}. Never returned by the API.
    headers: Mapped[dict[str, str]] = mapped_column(JSON, default=dict)
    # The tools from the server's last tools/list: [{name, description, input_schema}].
    tools: Mapped[list[dict]] = mapped_column(JSON, default=list)
    server_info: Mapped[dict] = mapped_column(JSON, default=dict)
    protocol_version: Mapped[str] = mapped_column(String(20))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class Span(Base):
    """One timed step of a trace: an LLM call, a search, a tool call, a workflow node...

    Spans form a tree through parent_id. The root span has no parent, and its id is
    the trace's id.
    """

    __tablename__ = "spans"
    __table_args__ = (
        Index("ix_spans_trace_position", "trace_id", "position"),
        # Lists the newest traces fast: only root spans are in this index.
        Index(
            "ix_spans_roots_started",
            "started_at",
            postgresql_where=text("parent_id IS NULL"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    trace_id: Mapped[uuid.UUID]
    parent_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)
    name: Mapped[str] = mapped_column(String(100))
    kind: Mapped[str] = mapped_column(String(20))
    status: Mapped[str] = mapped_column(String(10))
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    inputs: Mapped[dict] = mapped_column(JSON, default=dict)
    outputs: Mapped[dict] = mapped_column(JSON, default=dict)
    # Numbers worth adding up or filtering on: model, tokens, time to first token...
    attributes: Mapped[dict] = mapped_column(JSON, default=dict)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    duration_ms: Mapped[float] = mapped_column(Float)
    position: Mapped[int] = mapped_column(Integer)


class EvalSet(Base):
    """A named list of test questions (cases) to run against an app."""

    __tablename__ = "eval_sets"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(100))
    description: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class EvalCase(Base):
    """One test question, and what a good answer looks like."""

    __tablename__ = "eval_cases"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    eval_set_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("eval_sets.id", ondelete="CASCADE"), index=True
    )
    position: Mapped[int] = mapped_column(Integer)
    query: Mapped[str] = mapped_column(Text)
    # Values for the app's {{variables}}, if it has any.
    inputs: Mapped[dict] = mapped_column(JSON, default=dict)
    # A reference answer for the LLM judge to compare against (optional).
    expected_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Words or phrases the answer must contain (optional).
    must_include: Mapped[list[str]] = mapped_column(JSON, default=list)
    # Filenames the answer should cite (optional).
    expected_sources: Mapped[list[str]] = mapped_column(JSON, default=list)


class EvalRun(Base):
    """One run of an eval set against one app, with its summary once it's done."""

    __tablename__ = "eval_runs"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    eval_set_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("eval_sets.id", ondelete="CASCADE"), index=True
    )
    app_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("apps.id", ondelete="CASCADE"))
    # A copy of the app's settings when the run started, so you can tell later
    # what changed between two runs.
    app_config: Mapped[dict] = mapped_column(JSON)
    judge_model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="running")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class EvalResult(Base):
    """How the app did on one case in one run."""

    __tablename__ = "eval_results"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("eval_runs.id", ondelete="CASCADE"), index=True
    )
    case_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("eval_cases.id", ondelete="CASCADE"))
    position: Mapped[int] = mapped_column(Integer)
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Each metric's value, e.g. {"contains": 1.0, "source_recall": 0.5, "judge": 4}.
    scores: Mapped[dict] = mapped_column(JSON, default=dict)
    judge_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    passed: Mapped[bool] = mapped_column(Boolean)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    # The answer's trace: open it to see why a case failed.
    trace_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)


class ApiKey(Base):
    """A secret key that lets another program use one app through the public API.

    Only a hash of the key is stored. The key itself is shown once, when it's created.
    """

    __tablename__ = "api_keys"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    app_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("apps.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(100))
    # The first characters of the key, so people can tell their keys apart.
    prefix: Mapped[str] = mapped_column(String(20))
    key_hash: Mapped[str] = mapped_column(String(64), unique=True)
    # Requests per minute. None means the RATE_LIMIT_PER_MINUTE setting.
    rate_limit_per_minute: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
