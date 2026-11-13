import uuid

import pytest

from mini_dify.models import App, Conversation, Message
from mini_dify.prompts import MissingVariablesError, extract_variables, render_template
from mini_dify.services.chat_service import build_prompt


def test_extract_variables_in_order_without_duplicates():
    template = "You help {{ name }} with {{topic}}. Be kind to {{name}}."
    assert extract_variables(template) == ["name", "topic"]


def test_render_template_fills_values():
    assert render_template("Hi {{name}}!", {"name": "Abhi"}) == "Hi Abhi!"


def test_render_template_reports_every_missing_variable():
    with pytest.raises(MissingVariablesError) as exc_info:
        render_template("{{a}} {{b}} {{c}}", {"b": "x"})
    assert exc_info.value.missing == ["a", "c"]


def test_render_template_leaves_other_braces_alone():
    assert render_template('Reply as JSON: {"ok": true}', {}) == 'Reply as JSON: {"ok": true}'


def test_build_prompt_order_is_system_then_history_then_question():
    app = App(name="tutor", system_prompt="You teach {{subject}}.", memory_window=10)
    conversation = Conversation(app_id=uuid.uuid4(), inputs={"subject": "maths"})
    history = [
        Message(role="user", content="What is 2+2?"),
        Message(role="assistant", content="4"),
    ]

    prompt = build_prompt(app, conversation, history, "And 3+3?")

    assert [(m.role, m.content) for m in prompt] == [
        ("system", "You teach maths."),
        ("user", "What is 2+2?"),
        ("assistant", "4"),
        ("user", "And 3+3?"),
    ]


def test_build_prompt_skips_empty_system_prompt():
    app = App(name="plain", system_prompt="", memory_window=10)
    conversation = Conversation(app_id=uuid.uuid4(), inputs={})
    prompt = build_prompt(app, conversation, [], "hi")
    assert [m.role for m in prompt] == ["user"]
