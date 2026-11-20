from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from mini_dify.tools.registry import BUILTIN_TOOLS

router = APIRouter(prefix="/v1", tags=["tools"])


class ToolOut(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]


@router.get("/tools", response_model=list[ToolOut])
async def list_tools():
    """The tools an app can use. Put their names in an app's `tools` list."""
    return [
        ToolOut(
            name=tool.name,
            description=tool.description,
            parameters=tool.Args.model_json_schema(),
        )
        for tool in BUILTIN_TOOLS.values()
    ]
