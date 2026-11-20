from collections.abc import AsyncIterator

from mini_dify.agent.base import OUT_OF_STEPS, AgentAnswer, AgentEvent, AgentRunner, AgentStep
from mini_dify.llm.types import Message


class FunctionCallingAgent(AgentRunner):
    """Uses the model's built-in tool calling: tools go in a separate `tools` field,
    and the model replies with structured tool calls instead of text."""

    async def run(self, messages: list[Message]) -> AsyncIterator[AgentEvent]:
        messages = list(messages)  # don't change the caller's list
        specs = [tool.spec() for tool in self.tools.values()]

        for iteration in range(1, self.max_iterations + 1):
            result = await self.provider.chat(self.model, messages, self.temperature, tools=specs)
            self.add_usage(result.usage)

            if not result.tool_calls:
                yield AgentAnswer(content=result.content, iterations=iteration, usage=self.usage)
                return

            # Keep the model's request in the history, then answer each call with a tool message.
            messages.append(
                Message(role="assistant", content=result.content, tool_calls=result.tool_calls)
            )
            for call in result.tool_calls:
                observation = await self.call_tool(call.name, call.arguments)
                messages.append(Message(role="tool", content=observation, tool_name=call.name))
                yield AgentStep(
                    iteration=iteration,
                    thought=result.content,
                    tool=call.name,
                    arguments=call.arguments,
                    observation=observation,
                )

        # Out of iterations: one last call without tools, so the model has to answer in text.
        messages.append(Message(role="user", content=OUT_OF_STEPS))
        result = await self.provider.chat(self.model, messages, self.temperature)
        self.add_usage(result.usage)
        yield AgentAnswer(
            content=result.content,
            iterations=self.max_iterations,
            usage=self.usage,
            stopped_early=True,
        )
