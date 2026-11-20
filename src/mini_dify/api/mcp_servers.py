import uuid
from datetime import datetime
from typing import Annotated, Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from mini_dify.config import settings
from mini_dify.db import get_session
from mini_dify.mcp.client import MCPClient, MCPError
from mini_dify.models import MCPServer

router = APIRouter(prefix="/v1", tags=["mcp servers"])


def get_mcp_transport() -> httpx.AsyncBaseTransport | None:
    """The transport the MCP client uses. None means the real network; tests override it."""
    return None


SessionDep = Annotated[AsyncSession, Depends(get_session)]
MCPTransportDep = Annotated[httpx.AsyncBaseTransport | None, Depends(get_mcp_transport)]


class MCPServerCreate(BaseModel):
    # Becomes the prefix of every tool name, so keep it short and simple.
    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,29}$")
    url: str = Field(min_length=1, max_length=500)
    headers: dict[str, str] = {}


class RemoteToolOut(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]


class MCPServerOut(BaseModel):
    id: uuid.UUID
    name: str
    url: str
    # Only the header names: values are often secrets (API keys), so they're never sent back.
    header_names: list[str]
    tools: list[RemoteToolOut]
    server_info: dict[str, Any]
    protocol_version: str
    created_at: datetime
    updated_at: datetime


def to_out(server: MCPServer) -> MCPServerOut:
    return MCPServerOut(
        id=server.id,
        name=server.name,
        url=server.url,
        header_names=sorted(server.headers),
        tools=[RemoteToolOut(**tool) for tool in server.tools],
        server_info=server.server_info,
        protocol_version=server.protocol_version,
        created_at=server.created_at,
        updated_at=server.updated_at,
    )


async def fetch_tools(
    url: str, headers: dict[str, str], transport: httpx.AsyncBaseTransport | None
) -> tuple[list[dict], dict, str]:
    """Connect to the server and list its tools. A failure becomes a 502 with the reason."""
    try:
        async with MCPClient(
            url, headers, transport=transport, allow_private=settings.mcp_allow_private_urls
        ) as client:
            tools = await client.list_tools()
            return (
                [
                    {"name": t.name, "description": t.description, "input_schema": t.input_schema}
                    for t in tools
                ],
                client.server_info,
                client.protocol_version or "",
            )
    except MCPError as exc:
        raise HTTPException(status_code=502, detail=f"MCP server error: {exc}") from exc


async def get_server_or_404(session: AsyncSession, server_id: uuid.UUID) -> MCPServer:
    server = await session.get(MCPServer, server_id)
    if server is None:
        raise HTTPException(status_code=404, detail="MCP server not found")
    return server


@router.post("/mcp-servers", response_model=MCPServerOut, status_code=201)
async def create_mcp_server(body: MCPServerCreate, session: SessionDep, transport: MCPTransportDep):
    """Register an MCP server: connect, list its tools, and save them."""
    if await session.scalar(select(MCPServer).where(MCPServer.name == body.name)):
        raise HTTPException(status_code=409, detail=f"An MCP server named {body.name!r} exists")
    tools, server_info, version = await fetch_tools(body.url, body.headers, transport)
    server = MCPServer(
        name=body.name,
        url=body.url,
        headers=body.headers,
        tools=tools,
        server_info=server_info,
        protocol_version=version,
    )
    session.add(server)
    await session.commit()
    return to_out(server)


@router.get("/mcp-servers", response_model=list[MCPServerOut])
async def list_mcp_servers(session: SessionDep):
    servers = await session.scalars(select(MCPServer).order_by(MCPServer.created_at))
    return [to_out(server) for server in servers]


@router.get("/mcp-servers/{server_id}", response_model=MCPServerOut)
async def get_mcp_server(server_id: uuid.UUID, session: SessionDep):
    return to_out(await get_server_or_404(session, server_id))


@router.post("/mcp-servers/{server_id}/refresh", response_model=MCPServerOut)
async def refresh_mcp_server(server_id: uuid.UUID, session: SessionDep, transport: MCPTransportDep):
    """List the server's tools again, e.g. after it added a new one."""
    server = await get_server_or_404(session, server_id)
    server.tools, server.server_info, server.protocol_version = await fetch_tools(
        server.url, server.headers, transport
    )
    await session.commit()
    return to_out(server)


@router.delete("/mcp-servers/{server_id}", status_code=204)
async def delete_mcp_server(server_id: uuid.UUID, session: SessionDep):
    await session.delete(await get_server_or_404(session, server_id))
    await session.commit()
    return Response(status_code=204)
