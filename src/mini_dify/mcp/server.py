import json
import re
from typing import Annotated, Any

import httpx
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from mini_dify.api.workflows import get_http_transport
from mini_dify.config import settings
from mini_dify.db import get_session
from mini_dify.llm.base import LLMProvider
from mini_dify.llm.embeddings import EmbeddingProvider
from mini_dify.llm.registry import get_embedder, get_provider
from mini_dify.mcp import protocol
from mini_dify.mcp.protocol import JsonRpcError
from mini_dify.models import Workflow
from mini_dify.services.workflow_service import compile_workflow, missing_inputs, run_workflow
from mini_dify.tracing.llm import TracedProvider
from mini_dify.workflow.engine import WorkflowFinished
from mini_dify.workflow.graph import GraphError
from mini_dify.workflow.nodes import NodeContext, StartNode

router = APIRouter(tags=["mcp"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]
ProviderDep = Annotated[LLMProvider, Depends(get_provider)]
EmbedderDep = Annotated[EmbeddingProvider, Depends(get_embedder)]
TransportDep = Annotated[httpx.AsyncBaseTransport | None, Depends(get_http_transport)]

LOCAL_HOSTS = {"localhost", "127.0.0.1", "[::1]"}


# ---------- workflows as tools ----------


def tool_name_for(name: str) -> str:
    """A workflow name as a tool name: lowercase letters, digits and underscores."""
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return (slug or "workflow")[:60]


async def workflow_tools(session: AsyncSession) -> dict[str, tuple[Workflow, dict[str, Any]]]:
    """Every runnable workflow as an MCP tool: tool name -> (workflow, tool definition)."""
    workflows = await session.scalars(select(Workflow).order_by(Workflow.created_at))
    tools: dict[str, tuple[Workflow, dict[str, Any]]] = {}
    for workflow in workflows:
        try:
            graph, nodes = compile_workflow(workflow.graph)
        except GraphError:
            continue  # saved before a rule was added and no longer valid: don't offer it
        start = nodes[graph.start_id]
        assert isinstance(start, StartNode)

        name = base = tool_name_for(workflow.name)
        suffix = 2
        while name in tools:  # two workflows with the same name get _2, _3...
            name, suffix = f"{base}_{suffix}", suffix + 1

        definition = {
            "name": name,
            "title": workflow.name,
            "description": workflow.description or f"Run the mini-dify workflow {workflow.name!r}.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    variable.name: {"description": f"The workflow input {variable.name!r}"}
                    for variable in start.data.variables
                },
                "required": [v.name for v in start.data.variables if v.required],
            },
        }
        tools[name] = (workflow, definition)
    return tools


# ---------- the JSON-RPC methods ----------


def initialize(params: dict[str, Any]) -> dict[str, Any]:
    requested = params.get("protocolVersion")
    # Use the client's version if we speak it, otherwise offer our newest one.
    # The client then decides whether it can work with that.
    version = (
        requested if requested in protocol.PROTOCOL_VERSIONS else protocol.LATEST_PROTOCOL_VERSION
    )
    return {
        "protocolVersion": version,
        "capabilities": {"tools": {"listChanged": False}},
        "serverInfo": {"name": "mini-dify", "version": "0.1.0"},
        "instructions": "Each tool runs one mini-dify workflow and returns its outputs.",
    }


async def call_tool(
    params: dict[str, Any],
    request: Request,
    session: AsyncSession,
    provider: LLMProvider,
    embedder: EmbeddingProvider,
    transport: httpx.AsyncBaseTransport | None,
) -> dict[str, Any]:
    name = params.get("name")
    arguments = params.get("arguments") or {}
    if not isinstance(arguments, dict):
        raise JsonRpcError(protocol.INVALID_PARAMS, "arguments must be an object")
    tools = await workflow_tools(session)
    if name not in tools:
        raise JsonRpcError(protocol.INVALID_PARAMS, f"Unknown tool: {name}")
    workflow, _ = tools[name]

    # From here on, problems are reported as a tool result with isError, not a
    # protocol error, so the model calling the tool can read them and try again.
    graph, nodes = compile_workflow(workflow.graph)
    missing = missing_inputs(graph, nodes, arguments)
    if missing:
        return tool_result(f"Missing inputs: {', '.join(missing)}", is_error=True)

    sessionmaker = request.app.state.sessionmaker
    ctx = NodeContext(
        provider=TracedProvider(provider),
        embedder=embedder,
        sessionmaker=sessionmaker,
        default_model=settings.default_model,
        inputs=arguments,
        http_transport=transport,
        allow_private_http=settings.workflow_http_allow_private,
    )
    finished = None
    events = run_workflow(sessionmaker, workflow.id, workflow.graph, ctx, workflow.name)
    async for event in events:
        if isinstance(event, WorkflowFinished):
            finished = event
    assert finished is not None
    if finished.status == "failed":
        return tool_result(f"The workflow failed: {finished.error}", is_error=True)
    outputs = finished.model_dump(mode="json")["outputs"]
    return tool_result(json.dumps(outputs, ensure_ascii=False), structured=outputs)


def tool_result(text: str, is_error: bool = False, structured: dict | None = None) -> dict:
    value: dict[str, Any] = {"content": [{"type": "text", "text": text}], "isError": is_error}
    if structured is not None:
        value["structuredContent"] = structured
    return value


# ---------- the endpoint ----------


def origin_allowed(origin: str | None) -> bool:
    """Only local web pages may call us from a browser (see "DNS rebinding" in the guide).
    Requests without an Origin header come from programs, not browsers, and are fine."""
    if origin is None:
        return True
    return httpx.URL(origin).netloc.decode().rsplit(":", 1)[0] in LOCAL_HOSTS


@router.post("/mcp")
async def mcp_endpoint(
    request: Request,
    session: SessionDep,
    provider: ProviderDep,
    embedder: EmbedderDep,
    transport: TransportDep,
):
    """The MCP server: one URL, one JSON-RPC message per POST."""
    if not origin_allowed(request.headers.get("origin")):
        return JSONResponse(
            protocol.error(None, protocol.INVALID_REQUEST, "Origin not allowed"), status_code=403
        )
    try:
        message = await request.json()
    except ValueError:
        return JSONResponse(
            protocol.error(None, protocol.PARSE_ERROR, "Invalid JSON"), status_code=400
        )
    if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
        return JSONResponse(
            protocol.error(None, protocol.INVALID_REQUEST, "Expected one JSON-RPC 2.0 message"),
            status_code=400,
        )

    # Notifications (no id) and replies (no method) need no answer: just "accepted".
    if "id" not in message or "method" not in message:
        return Response(status_code=202)

    request_id, method = message["id"], message["method"]
    params = message.get("params") or {}

    # After the handshake, clients send the agreed version in a header. Refuse one we
    # don't know. (No header is allowed, for older clients.)
    version = request.headers.get("mcp-protocol-version")
    if method != "initialize" and version and version not in protocol.PROTOCOL_VERSIONS:
        return JSONResponse(
            protocol.error(
                request_id, protocol.INVALID_REQUEST, f"Unsupported protocol version {version}"
            ),
            status_code=400,
        )

    try:
        match method:
            case "initialize":
                value = initialize(params)
            case "ping":
                value = {}
            case "tools/list":
                tools = await workflow_tools(session)
                value = {"tools": [definition for _, definition in tools.values()]}
            case "tools/call":
                value = await call_tool(params, request, session, provider, embedder, transport)
            case _:
                raise JsonRpcError(protocol.METHOD_NOT_FOUND, f"Method not found: {method}")
    except JsonRpcError as exc:
        return JSONResponse(protocol.error(request_id, exc.code, exc.message))
    return JSONResponse(protocol.result(request_id, value))


@router.get("/mcp")
async def mcp_stream():
    """Servers may offer a GET stream for messages they start themselves. We have none."""
    return Response(status_code=405, headers={"Allow": "POST"})
