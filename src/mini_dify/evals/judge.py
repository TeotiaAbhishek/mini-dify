import json
import re
from dataclasses import dataclass

from mini_dify.llm.base import LLMProvider
from mini_dify.llm.types import Message

JUDGE_SYSTEM_PROMPT = """You grade answers given by an AI assistant.
You get a question, a reference answer that is known to be correct, and the assistant's answer.
Judge only whether the assistant's answer is correct and complete compared with the reference.
Ignore style and length. Extra details are fine if they don't contradict the reference.

Score from 1 to 5:
5 = correct and complete
4 = correct, a minor detail missing
3 = partly correct, or an important detail missing
2 = mostly wrong
1 = wrong, or doesn't answer the question

Reply with JSON only, like {"reason": "one short sentence", "score": 4}"""


@dataclass
class Verdict:
    score: int | None  # None when the judge's reply couldn't be read
    reason: str


def build_judge_messages(question: str, reference: str, answer: str) -> list[Message]:
    user = (
        f"Question:\n{question}\n\nReference answer:\n{reference}\n\nAssistant's answer:\n{answer}"
    )
    return [Message(role="system", content=JUDGE_SYSTEM_PROMPT), Message(role="user", content=user)]


def parse_verdict(reply: str) -> Verdict:
    """Read the judge's reply. Small models often wrap the JSON in text or code fences,
    so find the first {...} block; failing that, look for "score: N"."""
    match = re.search(r"\{.*\}", reply, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            score = int(data["score"])
            if 1 <= score <= 5:
                return Verdict(score=score, reason=str(data.get("reason", "")))
        except (ValueError, KeyError, TypeError):
            pass
    match = re.search(r"score\D{0,5}([1-5])\b", reply, re.IGNORECASE)
    if match:
        return Verdict(score=int(match.group(1)), reason=reply.strip()[:500])
    return Verdict(score=None, reason=f"Could not read the judge's reply: {reply.strip()[:200]}")


async def judge_answer(
    provider: LLMProvider, model: str, question: str, reference: str, answer: str
) -> Verdict:
    """Ask the judge model to grade one answer. Temperature 0, so grading is repeatable."""
    messages = build_judge_messages(question, reference, answer)
    result = await provider.chat(model, messages, temperature=0.0)
    return parse_verdict(result.content)
