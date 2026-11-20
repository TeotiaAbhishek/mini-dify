import json
import re
from typing import Any

# A reference to another node's output: {{#node_id.variable#}}, the same syntax Dify uses.
REFERENCE_PATTERN = re.compile(
    r"\{\{#\s*([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)\s*#\}\}"
)


class VariableNotFoundError(KeyError):
    def __init__(self, node_id: str, name: str):
        self.node_id, self.name = node_id, name
        super().__init__(f"{{{{#{node_id}.{name}#}}}} has no value")

    def __str__(self) -> str:
        return self.args[0]


def find_references(text: str) -> list[tuple[str, str]]:
    """Every (node_id, variable) a text refers to, in order."""
    return REFERENCE_PATTERN.findall(text)


def to_text(value: Any) -> str:
    """How a value looks when it's put into a prompt or a URL."""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


class VariablePool:
    """Every node's outputs during one run, looked up by (node id, variable name)."""

    def __init__(self) -> None:
        self._values: dict[str, dict[str, Any]] = {}

    def set_outputs(self, node_id: str, outputs: dict[str, Any]) -> None:
        self._values[node_id] = dict(outputs)

    def get_outputs(self, node_id: str) -> dict[str, Any]:
        return dict(self._values.get(node_id, {}))

    def get(self, node_id: str, name: str) -> Any:
        try:
            return self._values[node_id][name]
        except KeyError:
            raise VariableNotFoundError(node_id, name) from None

    def render(self, template: str) -> str:
        """Replace every {{#node.var#}} in a text with that variable's value."""
        return REFERENCE_PATTERN.sub(lambda m: to_text(self.get(m.group(1), m.group(2))), template)

    def resolve(self, text: str) -> Any:
        """A text that is exactly one reference gives the raw value (a number stays a number).
        Anything else is rendered as text."""
        match = REFERENCE_PATTERN.fullmatch(text.strip())
        if match:
            return self.get(match.group(1), match.group(2))
        return self.render(text)
