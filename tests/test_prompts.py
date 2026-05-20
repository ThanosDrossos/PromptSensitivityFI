"""QA prompt assembler — pure-string tests (no LLM)."""

from __future__ import annotations

from prompt_sensitivity.data.schemas import HotpotParagraph
from prompt_sensitivity.prompts import (
    QA_SYSTEM_PROMPT,
    QA_USER_TEMPLATE,
    assemble_qa_messages,
)


def test_assemble_with_context_emits_two_messages():
    paragraphs = [
        HotpotParagraph(title="Paris", sentences=["Paris is the capital of France. "]),
        HotpotParagraph(title="France", sentences=["France is in Europe. "]),
    ]
    msgs = assemble_qa_messages("What is the capital of France?", paragraphs)
    assert len(msgs) == 2
    assert msgs[0].role == "system"
    assert msgs[0].content == QA_SYSTEM_PROMPT
    assert msgs[1].role == "user"
    assert "Paris is the capital of France." in msgs[1].content
    assert "What is the capital of France?" in msgs[1].content
    assert msgs[1].content.endswith("Answer:")


def test_assemble_without_context_omits_context_block():
    """Level 0: no paragraphs -> closed-book prompt with just the question."""
    msgs = assemble_qa_messages("What is the capital of France?", [])
    assert "Context:" not in msgs[1].content
    assert "What is the capital of France?" in msgs[1].content


def test_system_prompt_forbids_extra_prose():
    """Anti-pattern: model must not add reasoning chains."""
    sp = QA_SYSTEM_PROMPT.lower()
    assert "reasoning" in sp or "do not add" in sp
    assert "context" in sp


def test_unknown_paragraph_with_no_sentences_skipped():
    """Empty-body paragraphs don't pollute the context block."""
    paragraphs = [
        HotpotParagraph(title="Empty", sentences=[]),
        HotpotParagraph(title="Full", sentences=["Real content here."]),
    ]
    msgs = assemble_qa_messages("Q?", paragraphs)
    assert "Real content here." in msgs[1].content
    # Empty title should NOT appear as a header.
    assert "Empty:" not in msgs[1].content


def test_template_constants_exported():
    """The verbatim template strings must be importable for the writeup."""
    assert QA_SYSTEM_PROMPT.strip()
    assert "{question}" in QA_USER_TEMPLATE
    assert "{context_block}" in QA_USER_TEMPLATE
