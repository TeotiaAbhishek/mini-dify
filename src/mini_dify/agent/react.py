import json
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from mini_dify.agent.base import OUT_OF_STEPS, AgentAnswer, AgentEvent, AgentRunner, AgentStep
from mini_dify.llm.types import Message
from mini_dify.tools.base import Tool

REACT_INSTRUCTIONS = """You can use tools to answer. The tools are:

{tools}

To use a tool, reply in exactly this format and then stop:

Thought: what you need to do next
Action: the tool name, one of [{names}]
Action Input: the arguments as a JSON object

The system will reply with "Observation: <the tool result>". Repeat as many times as needed.
When you know the answer, reply in this format:

Thought: I now know the final answer
Final Answer: your answer to the user

Never write the Observation yourself."""


def describe_tools(tools: list[Tool]) -> str:
    lines = []
    for tool in tools:
        properties = tool.Args.model_json_schema().get("properties", {})
        arguments = {
            name: f"{schema.get('type', 'any')}: {schema.get('description', '')}"
            for name, schema in properties.items()
        }
        lines.append(f"- {tool.name}: {tool.description} Arguments: {json.dumps(arguments)}")
    return "\n".join(lines)


@dataclass
class ReActAction:
    thought: str
    tool: str
    arguments: dict[str, Any]
    text: str  # the model's output up to the end of Action Input
    error: str | None = None  # set when the Action Input was not valid JSON


@dataclass
class ReActFinal:
    thought: str
    answer: str


ACTION_PATTERN = re.compile(r"Action\s*:\s*(.+?)\s*\n+\s*Action\s*Input\s*:\s*", re.IGNORECASE)
FINAL_PATTERN = re.compile(r"Final\s*Answer\s*:\s*", re.IGNORECASE)
THOUGHT_PATTERN = re.compile(r"Thought\s*:\s*(.*?)\s*(?=\n\s*(Action|Final\s*Answer)\s*:|$)", re.S)


def parse_react_output(text: str) -> ReActAction | ReActFinal:
    """Work out whether the model wants a tool or is giving the final answer."""
    thought_match = THOUGHT_PATTERN.search(text)
    thought = thought_match.group(1).strip() if thought_match else ""

    action = ACTION_PATTERN.search(text)
    final = FINAL_PATTERN.search(text)

    # Whichever comes first wins. Models sometimes write an Action and then
    # invent its Observation and a Final Answer; we only trust the Action.
    if action and (not final or action.start() < final.start()):
        tool = action.group(1).strip().strip("`\"'")
        rest = text[action.end() :]
        try:
            start = rest.index("{")  # skips anything before the JSON, like ```json
            arguments, end = json.JSONDecoder().raw_decode(rest, start)
            if not isinstance(arguments, dict):
                raise ValueError("not an object")
        except ValueError:  # also catches json.JSONDecodeError, which is a ValueError
            error = 'Action Input must be a JSON object, like {"name": "value"}'
            return ReActAction(thought, tool, {}, text[: action.end()], error)
        # Cut the text right after the JSON, dropping anything the model made up after it.
        return ReActAction(thought, tool, arguments, text[: action.end() + end])

    if final:
        return ReActFinal(thought, text[final.end() :].strip())

    # The model ignored the format and just answered. Accept that as the answer.
    return ReActFinal(thought, text.strip())


class ReActAgent(AgentRunner):
    """Tool use through plain text (Reason + Act). Works with models that have no
    tool-calling support: the tools are described in the prompt and we parse the reply."""

    async def run(self, messages: list[Message]) -> AsyncIterator[AgentEvent]:
        instructions = REACT_INSTRUCTIONS.format(
            tools=describe_tools(list(self.tools.values())), names=", ".join(self.tools)
        )
        messages = list(messages)
        if messages and messages[0].role == "system":
            messages[0] = Message(role="system", content=f"{messages[0].content}\n\n{instructions}")
        else:
            messages.insert(0, Message(role="system", content=instructions))

        for iteration in range(1, self.max_iterations + 1):
            result = await self.provider.chat(self.model, messages, self.temperature)
            self.add_usage(result.usage)
            parsed = parse_react_output(result.content)

            if isinstance(parsed, ReActFinal):
                yield AgentAnswer(content=parsed.answer, iterations=iteration, usage=self.usage)
                return

            if parsed.error:
                observation = f"Error: {parsed.error}"
            else:
                observation = await self.call_tool(parsed.tool, parsed.arguments)
            # The model's text goes back as its own turn, the observation as the next user turn.
            messages.append(Message(role="assistant", content=parsed.text))
            messages.append(Message(role="user", content=f"Observation: {observation}"))
            yield AgentStep(
                iteration=iteration,
                thought=parsed.thought,
                tool=parsed.tool,
                arguments=parsed.arguments,
                observation=observation,
            )

        messages.append(
            Message(role="user", content=f"{OUT_OF_STEPS} Reply with 'Final Answer: ...'.")
        )
        result = await self.provider.chat(self.model, messages, self.temperature)
        self.add_usage(result.usage)
        parsed = parse_react_output(result.content)
        answer = parsed.answer if isinstance(parsed, ReActFinal) else result.content.strip()
        yield AgentAnswer(
            content=answer, iterations=self.max_iterations, usage=self.usage, stopped_early=True
        )
