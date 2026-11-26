from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """App settings, read from environment variables or a .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://minidify:minidify@localhost:5432/minidify"
    redis_url: str = "redis://localhost:6379/0"

    llm_provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    default_model: str = "llama3.2:3b"

    embedding_provider: str = "ollama"
    embedding_model: str = "nomic-embed-text"
    # nomic-embed-text was trained with these task prefixes; other models may use none.
    embedding_document_prefix: str = "search_document: "
    embedding_query_prefix: str = "search_query: "

    # Workflow HTTP nodes may not call private addresses (localhost, 10.x, 192.168.x...)
    # unless this is on. Leave it off unless you need a workflow to call a local service.
    workflow_http_allow_private: bool = False

    # The same rule for MCP server URLs. Turn it on to use an MCP server on your own
    # machine, like mini-dify's own /mcp endpoint.
    mcp_allow_private_urls: bool = False

    # Record a trace (a tree of timed spans) for every chat message and workflow run.
    tracing_enabled: bool = True

    # Public API: requests per minute for each API key, unless the key sets its own.
    rate_limit_per_minute: int = 60


settings = Settings()
