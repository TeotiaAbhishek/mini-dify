import re

# Matches {{name}} or {{ name }}. The name must look like a Python variable name.
VARIABLE_PATTERN = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")


class MissingVariablesError(ValueError):
    def __init__(self, missing: list[str]):
        self.missing = missing
        super().__init__(f"Missing values for: {', '.join(missing)}")


def extract_variables(template: str) -> list[str]:
    """Return the variable names used in a template, in order, without duplicates."""
    return list(dict.fromkeys(VARIABLE_PATTERN.findall(template)))


def render_template(template: str, inputs: dict[str, str]) -> str:
    """Replace every {{name}} with inputs[name]. Raise if any value is missing."""
    missing = [name for name in extract_variables(template) if name not in inputs]
    if missing:
        raise MissingVariablesError(missing)
    return VARIABLE_PATTERN.sub(lambda match: str(inputs[match.group(1)]), template)
