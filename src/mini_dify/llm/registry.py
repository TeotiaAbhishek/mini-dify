from functools import lru_cache

from mini_dify.config import settings
from mini_dify.llm.base import LLMProvider
from mini_dify.llm.fake import FakeProvider
from mini_dify.llm.ollama import OllamaProvider


@lru_cache
def get_provider() -> LLMProvider:
    """Return the configured provider. Cached, so there is one per process."""
    if settings.llm_provider == "ollama":
        return OllamaProvider(settings.ollama_base_url)
    if settings.llm_provider == "fake":
        return FakeProvider()
    raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")
