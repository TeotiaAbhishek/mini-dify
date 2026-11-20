import pytest

from mini_dify.agent.base import AgentAnswer, AgentStep
from mini_dify.agent.function_calling import FunctionCallingAgent
from mini_dify.agent.react import ReActAction, ReActAgent, ReActFinal, parse_react_output
from mini_dify.llm.fake import ScriptedProvider
from mini_dify.llm.types import Message, ToolCall
from mini_dify.tools.builtin import CalculatorTool

QUESTION = [
    Message(role="system", content="You are helpful."),
    Message(role="user", content="What is 6 * 7?"),
]


async def run(agent, messages=QUESTION):
    events = [event async for event in agent.run(messages)]
    steps = [event for event in events if isinstance(event, AgentStep)]
    answers = [event for event in events if isinstance(event, AgentAnswer)]
    assert len(answers) == 1, "an agent must finish with exactly one answer"
    return steps, answers[0]


def function_calling_agent(replies, max_iterations=5):
    provider = ScriptedProvider(replies)
    agent = FunctionCallingAgent(
        provider, "test-model", [CalculatorTool()], ctx=None, max_iterations=max_iterations
    )
    return agent, provider


# ---------- Function calling ----------


@pytest.mark.anyio
async def test_function_calling_uses_tool_then_answers():
    agent, provider = function_calling_agent(
        [[ToolCall(name="calculator", arguments={"expression": "6 * 7"})], "It is 42."]
    )
    steps, answer = await run(agent)

    assert [(s.tool, s.observation) for s in steps] == [("calculator", "6 * 7 = 42")]
    assert answer.content == "It is 42."
    assert answer.iterations == 2
    assert answer.usage.prompt_tokens == 20  # 10 per call, two calls

    # The model was offered the tool, and on the 2nd call it saw its own call + the result.
    assert provider.calls[0]["tools"][0]["function"]["name"] == "calculator"
    second = provider.calls[1]["messages"]
    assert second[-2].role == "assistant" and second[-2].tool_calls[0].name == "calculator"
    assert second[-1].role == "tool" and second[-1].content == "6 * 7 = 42"


@pytest.mark.anyio
async def test_unknown_tool_is_reported_back_to_the_model():
    agent, provider = function_calling_agent(
        [[ToolCall(name="weather", arguments={})], "Sorry, I can't check the weather."]
    )
    steps, answer = await run(agent)

    assert steps[0].observation.startswith("Error: unknown tool 'weather'")
    assert answer.content == "Sorry, I can't check the weather."


@pytest.mark.anyio
async def test_agent_stops_after_max_iterations():
    call = [ToolCall(name="calculator", arguments={"expression": "1 + 1"})]
    agent, provider = function_calling_agent([call, call, "Best guess: 2."], max_iterations=2)
    steps, answer = await run(agent)

    assert len(steps) == 2
    assert answer.stopped_early is True
    assert answer.content == "Best guess: 2."
    assert provider.calls[-1]["tools"] is None  # the last call offers no tools


# ---------- ReAct ----------


def test_parse_action_ignores_invented_observation():
    text = (
        "Thought: I should multiply.\nAction: calculator\n"
        'Action Input: {"expression": "6 * 7"}\nObservation: 41\nFinal Answer: 41'
    )
    parsed = parse_react_output(text)
    assert isinstance(parsed, ReActAction)
    assert parsed.thought == "I should multiply."
    assert parsed.tool == "calculator"
    assert parsed.arguments == {"expression": "6 * 7"}
    assert parsed.text.endswith('{"expression": "6 * 7"}')


def test_parse_final_answer_and_plain_text():
    final = parse_react_output("Thought: I now know the final answer\nFinal Answer: 42.")
    assert isinstance(final, ReActFinal) and final.answer == "42."
    plain = parse_react_output("It's 42.")
    assert isinstance(plain, ReActFinal) and plain.answer == "It's 42."


def test_parse_bad_action_input():
    parsed = parse_react_output("Action: calculator\nAction Input: 6 * 7")
    assert isinstance(parsed, ReActAction)
    assert parsed.error is not None


@pytest.mark.anyio
async def test_react_agent_loop():
    provider = ScriptedProvider(
        [
            'Thought: multiply\nAction: calculator\nAction Input: {"expression": "6 * 7"}',
            "Thought: I now know the final answer\nFinal Answer: 42",
        ]
    )
    agent = ReActAgent(provider, "test-model", [CalculatorTool()], ctx=None)
    steps, answer = await run(agent)

    assert [(s.thought, s.observation) for s in steps] == [("multiply", "6 * 7 = 42")]
    assert answer.content == "42"

    first = provider.calls[0]
    assert first["tools"] is None  # ReAct never uses the tools field
    assert "Action Input" in first["messages"][0].content  # instructions in the system prompt
    assert first["messages"][0].content.startswith("You are helpful.")
    assert provider.calls[1]["messages"][-1].content == "Observation: 6 * 7 = 42"
