"""Role-templated paraphrase prompts (Razavi 2025 pattern)."""

from __future__ import annotations

import pytest

from prompt_sensitivity.paraphrases.prompts import (
    ROLE_NAMES,
    build_paraphrase_messages,
    list_persona_descriptions,
)


def test_all_four_roles_present():
    assert set(ROLE_NAMES) == {"neutral", "journalist", "casual_user", "domain_expert"}


def test_build_returns_two_messages_with_system_and_user():
    msgs = build_paraphrase_messages("What is the capital of France?", "neutral")
    assert len(msgs) == 2
    assert msgs[0].role == "system"
    assert msgs[1].role == "user"
    assert "What is the capital of France?" in msgs[1].content


def test_persona_text_differs_across_roles():
    """User prompts must encode the role; otherwise paraphrases all look the same."""
    contents = {
        role: build_paraphrase_messages("Q.", role)[1].content for role in ROLE_NAMES
    }
    # All four must be pairwise different.
    assert len(set(contents.values())) == 4


def test_unknown_role_raises():
    with pytest.raises(ValueError):
        build_paraphrase_messages("Q.", "shakespearean")  # type: ignore[arg-type]


def test_system_prompt_forbids_answering():
    """Avoid the generator accidentally answering the question; this regressed once in PromptSET."""
    msgs = build_paraphrase_messages("Q.", "neutral")
    sys_text = msgs[0].content.lower()
    assert "must not answer" in sys_text


def test_personas_documented_for_writeup():
    """Sprint 2 writeup needs the persona definitions; keep them queryable."""
    personas = list_persona_descriptions()
    assert set(personas) == set(ROLE_NAMES)
    for p in personas.values():
        assert len(p) > 10
