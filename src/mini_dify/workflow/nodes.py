import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

import httpx
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from mini_dify.llm.base import LLMProvider
from mini_dify.llm.embeddings import EmbeddingProvider
from mini_dify.llm.types import Message
from mini_dify.rag.retrieval import SearchMode, retrieve
from mini_dify.security import UnsafeURLError, check_url_is_public
from mini_dify.workflow.graph import Graph, GraphError, NodeConfig
from mini_dify.workflow.variables import VariablePool, to_text


class NodeError(Exception):
    """A node couldn't do its job. The run stops and reports this message."""


@dataclass
class NodeContext:
    """What nodes may need from the outside world during one run."""

    provider: LLMProvider
    embedder: EmbeddingProvider
    sessionmaker: async_sessionmaker[AsyncSession]
    default_model: str
    inputs: dict[str, Any] = field(default_factory=dict)
    # Tests pass an httpx.MockTransport here, so no real requests are made.
    http_transport: httpx.AsyncBaseTransport | None = None
    allow_private_http: bool = False


@dataclass
class NodeResult:
    """What a node produced. `handle` picks which outgoing edges are followed."""

    inputs: dict[str, Any]
    outputs: dict[str, Any]
    handle: str = "source"


class Node(ABC):
    """One step of a workflow. Each type has a `Data` model for its settings."""

    type: ClassVar[str]
    Data: ClassVar[type[BaseModel]]

    def __init__(self, config: NodeConfig):
        self.id = config.id
        self.title = config.title or config.id
        self.data = self.Data.model_validate(config.data)

    @abstractmethod
    async def run(self, pool: VariablePool, ctx: NodeContext) -> NodeResult:
        """Read what it needs from the pool, do the work, return inputs and outputs."""


# ---------- start and end ----------


class StartVariable(BaseModel):
    name: str = Field(pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$")
    required: bool = True
    default: Any = None


class StartNode(Node):
    """The entry point. Its outputs are the run's inputs, so others can use {{#start.x#}}."""

    type = "start"

    class Data(BaseModel):
        variables: list[StartVariable] = []

    def missing_inputs(self, inputs: dict[str, Any]) -> list[str]:
        return [v.name for v in self.data.variables if v.required and v.name not in inputs]

    async def run(self, pool: VariablePool, ctx: NodeContext) -> NodeResult:
        missing = self.missing_inputs(ctx.inputs)
        if missing:
            raise NodeError(f"Missing inputs: {', '.join(missing)}")
        values = {v.name: ctx.inputs.get(v.name, v.default) for v in self.data.variables}
        return NodeResult(inputs=dict(ctx.inputs), outputs=values)


class OutputMapping(BaseModel):
    name: str = Field(pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$")
    value: str


class EndNode(Node):
    """The exit. Picks which values become the workflow's outputs."""

    type = "end"

    class Data(BaseModel):
        outputs: list[OutputMapping] = []

    async def run(self, pool: VariablePool, ctx: NodeContext) -> NodeResult:
        outputs = {item.name: pool.resolve(item.value) for item in self.data.outputs}
        return NodeResult(inputs={}, outputs=outputs)


# ---------- LLM, knowledge and template ----------


class LLMNode(Node):
    """Ask the model. The prompts can use {{#node.var#}} references."""

    type = "llm"

    class Data(BaseModel):
        model: str | None = None
        system_prompt: str = ""
        prompt: str = Field(min_length=1)
        temperature: float = Field(default=0.7, ge=0.0, le=2.0)
        max_tokens: int | None = Field(default=None, gt=0)

    async def run(self, pool: VariablePool, ctx: NodeContext) -> NodeResult:
        system = pool.render(self.data.system_prompt)
        prompt = pool.render(self.data.prompt)
        messages = [Message(role="system", content=system)] if system else []
        messages.append(Message(role="user", content=prompt))
        model = self.data.model or ctx.default_model

        result = await ctx.provider.chat(
            model, messages, self.data.temperature, self.data.max_tokens
        )
        return NodeResult(
            inputs={"model": model, "system_prompt": system, "prompt": prompt},
            outputs={
                "text": result.content,
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
            },
        )


class KnowledgeRetrievalNode(Node):
    """Search knowledge bases. Outputs the chunks, and `text` ready to put in a prompt."""

    type = "knowledge_retrieval"

    class Data(BaseModel):
        dataset_ids: list[uuid.UUID] = Field(min_length=1)
        query: str = Field(min_length=1)
        top_k: int = Field(default=4, ge=1, le=20)
        mode: SearchMode = "hybrid"

    async def run(self, pool: VariablePool, ctx: NodeContext) -> NodeResult:
        query = to_text(pool.resolve(self.data.query))
        # Nodes can run at the same time, and one AsyncSession must not be shared
        # between tasks, so each retrieval opens its own session.
        async with ctx.sessionmaker() as session:
            chunks = await retrieve(
                session, ctx.embedder, self.data.dataset_ids, query, self.data.top_k, self.data.mode
            )
        result = [
            {
                "content": chunk.content,
                "filename": chunk.filename,
                "document_id": str(chunk.document_id),
                "score": chunk.score,
            }
            for chunk in chunks
        ]
        text = "\n\n".join(
            f"[{number}] ({chunk.filename}) {chunk.content}"
            for number, chunk in enumerate(chunks, start=1)
        )
        return NodeResult(inputs={"query": query}, outputs={"result": result, "text": text})


class TemplateNode(Node):
    """Join values into one text, e.g. to build a report from earlier nodes."""

    type = "template"

    class Data(BaseModel):
        template: str = Field(min_length=1)

    async def run(self, pool: VariablePool, ctx: NodeContext) -> NodeResult:
        return NodeResult(inputs={}, outputs={"output": pool.render(self.data.template)})


# ---------- if / else ----------

Operator = Literal[
    "contains",
    "not_contains",
    "equals",
    "not_equals",
    "starts_with",
    "ends_with",
    "is_empty",
    "is_not_empty",
    "greater_than",
    "less_than",
]


class Condition(BaseModel):
    variable: str = Field(min_length=1)  # usually one reference, like {{#llm.text#}}
    operator: Operator
    value: str = ""


def to_number(value: Any, what: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        raise NodeError(f"{what} is not a number: {value!r}") from None


def check_condition(actual: Any, operator: Operator, expected: str) -> bool:
    if operator in ("greater_than", "less_than"):
        left, right = to_number(actual, "The value"), to_number(expected, "The comparison value")
        return left > right if operator == "greater_than" else left < right
    if operator == "is_empty":
        return actual is None or actual == "" or actual == [] or actual == {}
    if operator == "is_not_empty":
        return not check_condition(actual, "is_empty", expected)

    text = to_text(actual)
    match operator:
        case "contains":
            return expected in text
        case "not_contains":
            return expected not in text
        case "equals":
            return text == expected
        case "not_equals":
            return text != expected
        case "starts_with":
            return text.startswith(expected)
        case "ends_with":
            return text.endswith(expected)
    raise NodeError(f"Unknown operator {operator!r}")


class IfElseNode(Node):
    """Branch: follow the "true" edge if the conditions hold, otherwise the "false" edge."""

    type = "if_else"

    class Data(BaseModel):
        conditions: list[Condition] = Field(min_length=1)
        logical_operator: Literal["and", "or"] = "and"

    async def run(self, pool: VariablePool, ctx: NodeContext) -> NodeResult:
        checks = []
        for condition in self.data.conditions:
            actual = pool.resolve(condition.variable)
            checks.append(
                {
                    "actual": actual,
                    "operator": condition.operator,
                    "value": condition.value,
                    "passed": check_condition(actual, condition.operator, condition.value),
                }
            )
        passed = [check["passed"] for check in checks]
        result = all(passed) if self.data.logical_operator == "and" else any(passed)
        return NodeResult(
            inputs={"conditions": checks},
            outputs={"result": result},
            handle="true" if result else "false",
        )


# ---------- HTTP request ----------


MAX_BODY_CHARS = 100_000


class HttpRequestNode(Node):
    """Call an HTTP API. The URL, headers and body can use references."""

    type = "http_request"

    class Data(BaseModel):
        method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"] = "GET"
        url: str = Field(min_length=1)
        headers: dict[str, str] = {}
        body: str | None = None
        timeout: float = Field(default=10.0, gt=0, le=60)

    async def run(self, pool: VariablePool, ctx: NodeContext) -> NodeResult:
        try:
            url = httpx.URL(pool.render(self.data.url))
        except httpx.InvalidURL as exc:
            raise NodeError(f"Invalid URL: {exc}") from exc
        headers = {name: pool.render(value) for name, value in self.data.headers.items()}
        body = pool.render(self.data.body) if self.data.body is not None else None
        if not ctx.allow_private_http:
            try:
                await check_url_is_public(url)
            except UnsafeURLError as exc:
                raise NodeError(str(exc)) from exc

        # follow_redirects=False: a redirect could point at a private address we never checked.
        async with httpx.AsyncClient(
            transport=ctx.http_transport, timeout=self.data.timeout, follow_redirects=False
        ) as client:
            try:
                response = await client.request(
                    self.data.method, url, headers=headers, content=body
                )
            except httpx.HTTPError as exc:
                raise NodeError(f"HTTP request failed: {exc}") from exc

        return NodeResult(
            inputs={"method": self.data.method, "url": str(url), "headers": headers, "body": body},
            outputs={
                "status_code": response.status_code,
                "body": response.text[:MAX_BODY_CHARS],
                "headers": dict(response.headers),
            },
        )


# ---------- building nodes from a graph ----------

NODE_TYPES: dict[str, type[Node]] = {
    node_class.type: node_class
    for node_class in (
        StartNode,
        LLMNode,
        KnowledgeRetrievalNode,
        IfElseNode,
        TemplateNode,
        HttpRequestNode,
        EndNode,
    )
}


def build_nodes(graph: Graph) -> dict[str, Node]:
    """Turn every node's config into a Node, checking its settings."""
    nodes: dict[str, Node] = {}
    for config in graph.nodes.values():
        try:
            nodes[config.id] = NODE_TYPES[config.type](config)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in error['loc']) or 'data'}: {error['msg']}"
                for error in exc.errors()
            )
            raise GraphError(f"Node {config.id!r} has invalid settings: {problems}") from None
    return nodes
