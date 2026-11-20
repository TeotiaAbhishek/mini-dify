import httpx
import pytest

from mini_dify.tools.base import ToolError
from mini_dify.tools.builtin import CalculatorTool, CurrentTimeTool, WikipediaTool, safe_eval
from mini_dify.tools.registry import BUILTIN_TOOLS, get_tools

# These tools don't touch the database or embeddings, so they need no context.
NO_CONTEXT = None


def test_safe_eval_does_arithmetic():
    assert safe_eval("2 + 3 * 4") == 14
    assert safe_eval("(2 + 3) * 4") == 20
    assert safe_eval("-2 ** 2") == -4
    assert safe_eval("7 // 2 + 7 % 2") == 4


@pytest.mark.parametrize(
    "expression", ["__import__('os').system('ls')", "x + 1", "'a' * 3", "[1, 2]", "2 ** 1000"]
)
def test_safe_eval_rejects_anything_else(expression):
    with pytest.raises(ToolError):
        safe_eval(expression)


@pytest.mark.anyio
async def test_calculator_returns_errors_as_text():
    calculator = CalculatorTool()
    assert await calculator.invoke({"expression": "0.1 + 0.2"}, NO_CONTEXT) == "0.1 + 0.2 = 0.3"
    assert await calculator.invoke({"expression": "1 / 0"}, NO_CONTEXT) == "Error: division by zero"
    missing = await calculator.invoke({}, NO_CONTEXT)
    assert missing.startswith("Error: invalid arguments for calculator: expression")


@pytest.mark.anyio
async def test_current_time():
    result = await CurrentTimeTool().invoke({"timezone": "Asia/Kolkata"}, NO_CONTEXT)
    assert "(Asia/Kolkata, UTC+0530)" in result
    bad = await CurrentTimeTool().invoke({"timezone": "Mars/Base"}, NO_CONTEXT)
    assert bad == "Error: unknown time zone 'Mars/Base'"


@pytest.mark.anyio
async def test_wikipedia_summary_and_not_found():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["User-Agent"].startswith("mini-dify")
        if request.url.path.endswith("/Alan_Turing"):
            return httpx.Response(200, json={"title": "Alan Turing", "extract": "A scientist."})
        return httpx.Response(404, json={})

    tool = WikipediaTool(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    assert await tool.invoke({"title": "Alan Turing"}, NO_CONTEXT) == "Alan Turing: A scientist."
    missing = await tool.invoke({"title": "Nope"}, NO_CONTEXT)
    assert missing == "Error: no Wikipedia article called 'Nope'"


def test_tool_spec_is_json_schema():
    spec = CalculatorTool().spec()
    assert spec["type"] == "function"
    assert spec["function"]["name"] == "calculator"
    assert spec["function"]["parameters"]["required"] == ["expression"]


def test_registry():
    assert set(BUILTIN_TOOLS) == {"calculator", "current_time", "wikipedia"}
    assert [tool.name for tool in get_tools(["wikipedia"])] == ["wikipedia"]
    with pytest.raises(KeyError):
        get_tools(["rm_rf"])
