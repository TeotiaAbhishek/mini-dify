from mini_dify.tools.base import Tool
from mini_dify.tools.builtin import CalculatorTool, CurrentTimeTool, WikipediaTool

# Tools an app builder can switch on by name. knowledge_search is not listed here:
# it's added automatically when an agent app has datasets.
BUILTIN_TOOLS: dict[str, Tool] = {
    tool.name: tool for tool in [CalculatorTool(), CurrentTimeTool(), WikipediaTool()]
}


def get_tools(names: list[str]) -> list[Tool]:
    """Look up tools by name. Raises KeyError for an unknown name."""
    return [BUILTIN_TOOLS[name] for name in names]
