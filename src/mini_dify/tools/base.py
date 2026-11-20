import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from mini_dify.llm.embeddings import EmbeddingProvider
from mini_dify.rag.retrieval import RetrievedChunk


class ToolError(Exception):
    """A tool failed in a way the model should hear about (bad input, not found...)."""


@dataclass
class ToolContext:
    """What tools may need from the outside world while one agent run is going on."""

    session: AsyncSession
    embedder: EmbeddingProvider
    dataset_ids: list[uuid.UUID] = field(default_factory=list)
    retrieval_top_k: int = 4
    # Chunks found by knowledge_search during this run. They become the citations.
    retrieved: list[RetrievedChunk] = field(default_factory=list)


class Tool(ABC):
    """Something the model can ask us to do: a name, a description and typed arguments."""

    name: str
    description: str
    Args: type[BaseModel]

    def spec(self) -> dict[str, Any]:
        """The tool definition sent to the model (OpenAI/Ollama "function" format)."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.Args.model_json_schema(),
            },
        }

    @abstractmethod
    async def run(self, args: Any, ctx: ToolContext) -> str:
        """Do the work and return the result as text for the model to read."""

    async def invoke(self, arguments: dict[str, Any], ctx: ToolContext) -> str:
        """Validate the model's arguments, run the tool, and turn failures into text.

        Errors are returned (not raised) so the model can see what went wrong and try again.
        """
        try:
            args = self.Args.model_validate(arguments)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in error['loc']) or 'arguments'}: {error['msg']}"
                for error in exc.errors()
            )
            return f"Error: invalid arguments for {self.name}: {problems}"
        try:
            return await self.run(args, ctx)
        except ToolError as exc:
            return f"Error: {exc}"
