import asyncio
import ipaddress

import httpx


class UnsafeURLError(ValueError):
    """The URL points somewhere our server must not call."""


async def check_url_is_public(url: httpx.URL) -> None:
    """Stop SSRF: user-supplied URLs must not reach our own network.

    Without this, anyone who can create a workflow or register an MCP server could
    make the server call http://localhost:5432, a cloud metadata address, or another
    internal service. Used by the workflow HTTP node and the MCP client.
    """
    if url.scheme not in ("http", "https"):
        raise UnsafeURLError(f"Only http and https URLs are allowed, not {url.scheme!r}")
    if not url.host:
        raise UnsafeURLError("The URL has no host")
    port = url.port or (443 if url.scheme == "https" else 80)
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(url.host, port)
    except OSError as exc:
        raise UnsafeURLError(f"Can't resolve {url.host}: {exc}") from exc
    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_reserved
            or address.is_multicast
            or address.is_unspecified
        ):
            raise UnsafeURLError(f"{url.host} resolves to a private address ({address}): blocked")
