import secrets
from typing import Annotated

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from mini_dify.config import settings

bearer = HTTPBearer(auto_error=False)


def console_headers() -> dict[str, str]:
    """Headers for mini-dify calling its own console API (the eval runner does)."""
    return {"Authorization": f"Bearer {settings.console_token}"} if settings.console_token else {}


async def require_console_token(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
) -> None:
    """Guard for the console API. Does nothing unless CONSOLE_TOKEN is set; then every
    request must send it as 'Authorization: Bearer <token>'."""
    expected = settings.console_token
    if not expected:
        return
    # compare_digest takes the same time however many characters match, so the
    # response time doesn't hint at how close a guess was.
    if credentials is None or not secrets.compare_digest(
        credentials.credentials.encode(), expected.encode()
    ):
        raise HTTPException(
            status_code=401,
            detail="Console token required",
            headers={"WWW-Authenticate": "Bearer"},
        )
