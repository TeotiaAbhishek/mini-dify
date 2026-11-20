"""The small part of the Model Context Protocol (MCP) that mini-dify speaks.

MCP messages are JSON-RPC 2.0: a request has an `id`, a `method` and `params`; the reply
has the same `id` and either a `result` or an `error`. A notification has no `id` and gets
no reply. We use the Streamable HTTP transport: each message is POSTed to one URL.
"""

from typing import Any

# Protocol revisions that start with an `initialize` handshake, oldest to newest.
# (2026-07-28 drops the handshake; servers that speak it still accept these.)
PROTOCOL_VERSIONS = ("2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25")
LATEST_PROTOCOL_VERSION = PROTOCOL_VERSIONS[-1]

# Standard JSON-RPC error codes.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


class JsonRpcError(Exception):
    """An error to send back as a JSON-RPC `error` object."""

    def __init__(self, code: int, message: str):
        self.code, self.message = code, message
        super().__init__(message)


def request(request_id: int | str, method: str, params: dict[str, Any] | None = None) -> dict:
    message: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        message["params"] = params
    return message


def notification(method: str, params: dict[str, Any] | None = None) -> dict:
    message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        message["params"] = params
    return message


def result(request_id: int | str | None, value: dict[str, Any]) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "result": value}


def error(request_id: int | str | None, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}
