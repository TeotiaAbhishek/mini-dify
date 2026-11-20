import ast
import operator
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
from pydantic import BaseModel, Field

from mini_dify.rag.retrieval import retrieve
from mini_dify.tools.base import Tool, ToolContext, ToolError

# ---------- calculator ----------

OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def safe_eval(expression: str) -> float | int:
    """Evaluate +, -, *, /, //, %, ** and brackets on numbers. Nothing else is allowed."""

    def visit(node: ast.AST) -> float | int:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ToolError("exponent too large (max 100)")
            return OPERATORS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in OPERATORS:
            return OPERATORS[type(node.op)](visit(node.operand))
        raise ToolError(f"unsupported expression: {ast.dump(node)[:60]}")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"not a valid math expression: {expression!r}") from exc
    try:
        return visit(tree)
    except ZeroDivisionError as exc:
        raise ToolError("division by zero") from exc


class CalculatorTool(Tool):
    name = "calculator"
    description = (
        "Evaluate a math expression exactly. Supports + - * / // % ** and brackets. "
        "Use it for any arithmetic instead of calculating in your head."
    )

    class Args(BaseModel):
        expression: str = Field(description="The expression, for example '(1234 * 5678) / 3'")

    async def run(self, args: Args, ctx: ToolContext) -> str:
        result = safe_eval(args.expression)
        # 12 significant digits hides float noise like 0.30000000000000004.
        text = f"{result:.12g}" if isinstance(result, float) else str(result)
        return f"{args.expression} = {text}"


# ---------- current_time ----------


class CurrentTimeTool(Tool):
    name = "current_time"
    description = "Get the current date, time and weekday in a time zone."

    class Args(BaseModel):
        timezone: str = Field(
            default="UTC", description="An IANA time zone, for example 'Australia/Sydney'"
        )

    async def run(self, args: Args, ctx: ToolContext) -> str:
        try:
            zone = ZoneInfo(args.timezone)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ToolError(f"unknown time zone {args.timezone!r}") from exc
        now = datetime.now(zone)
        return now.strftime(f"%A %d %B %Y, %H:%M ({args.timezone}, UTC%z)")


# ---------- wikipedia ----------


class WikipediaTool(Tool):
    name = "wikipedia"
    description = (
        "Look up a topic on English Wikipedia and get a short summary. "
        "Use the exact article title, for example 'Alan Turing'."
    )
    API = "https://en.wikipedia.org/api/rest_v1/page/summary/"

    class Args(BaseModel):
        title: str = Field(description="The Wikipedia article title")

    def __init__(self, client: httpx.AsyncClient | None = None):
        self.client = client

    async def run(self, args: Args, ctx: ToolContext) -> str:
        url = self.API + args.title.strip().replace(" ", "_")
        # Wikipedia asks API clients to identify themselves with a User-Agent.
        headers = {"User-Agent": "mini-dify/0.1 (learning project)"}
        try:
            if self.client is not None:
                response = await self.client.get(url, headers=headers)
            else:
                async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
                    response = await client.get(url, headers=headers)
        except httpx.HTTPError as exc:
            raise ToolError(f"could not reach Wikipedia: {exc}") from exc
        if response.status_code == 404:
            raise ToolError(f"no Wikipedia article called {args.title!r}")
        if response.status_code != 200:
            raise ToolError(f"Wikipedia returned {response.status_code}")
        data = response.json()
        return f"{data.get('title', args.title)}: {data.get('extract', '(no summary)')}"


# ---------- knowledge_search ----------


class KnowledgeSearchTool(Tool):
    """Searches the app's datasets. Added automatically to agent apps that have datasets."""

    name = "knowledge_search"
    description = (
        "Search the company knowledge base (uploaded documents). Use it for any question "
        "about the documents. Cite results by their number, like [1]."
    )

    class Args(BaseModel):
        query: str = Field(description="What to search for, as a short question or keywords")

    async def run(self, args: Args, ctx: ToolContext) -> str:
        chunks = await retrieve(
            ctx.session, ctx.embedder, ctx.dataset_ids, args.query, ctx.retrieval_top_k
        )
        if not chunks:
            return "No matching documents found."
        # Number the results after the ones found earlier in this run, so that
        # [n] in the answer always points at the same citation.
        start = len(ctx.retrieved) + 1
        ctx.retrieved.extend(chunks)
        return "\n\n".join(
            f"[{number}] (from {chunk.filename})\n{chunk.content}"
            for number, chunk in enumerate(chunks, start=start)
        )
