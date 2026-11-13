from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """App settings, read from environment variables or a .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://minidify:minidify@localhost:5432/minidify"
    redis_url: str = "redis://localhost:6379/0"

    llm_provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    default_model: str = "llama3.2:3b"


settings = Settings()
