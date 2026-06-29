"""QA prompt assembler — pure-string tests (no LLM)."""

from __future__ import annotations

import pytest

from prompt_sensitivity.data.schemas import HotpotParagraph
from prompt_sensitivity.prompts import (
    QA_SYSTEM_PROMPT,
    QA_USER_TEMPLATE,
    assemble_qa_messages,
    parse_answer_line,
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


# --------------------------------------------------------------------------- #
# parse_answer_line — final-answer extraction from CoT responses.             #
#                                                                             #
# Regression for the MuSiQue cluster pilot: the local 7-8B models recovered   #
# the reasoning chain (chain-F high) but final_answer_f_mean read ~0 because   #
# they wrote the answer label in formats the old `^answer:` regex missed, so   #
# the parser handed the whole essay to the gold->answer NLI scorer (F=0).     #
# Each `_CHAIN` below is a realistic 2-step CoT; only the final line varies.   #
# --------------------------------------------------------------------------- #

_CHAIN = (
    "Step 1: The nation that set up the Commission of Truth and Friendship is "
    "Timor-Leste.\n"
    "Step 2: Lion Air serves the airport in Dili, the capital of Timor-Leste.\n"
)
_GOLD = "Jose Ramos-Horta"


@pytest.mark.parametrize(
    "final_line",
    [
        "Answer: Jose Ramos-Horta",          # canonical (the old parser handled this)
        "**Answer:** Jose Ramos-Horta",      # markdown-bold label + value
        "**Answer**: Jose Ramos-Horta",      # bold label, plain colon
        "Final Answer: Jose Ramos-Horta",    # "Final Answer:"
        "FINAL ANSWER: Jose Ramos-Horta",    # upper-case
        "Answer - Jose Ramos-Horta",         # dash separator
        "> Answer: Jose Ramos-Horta",        # blockquote glyph
        "Answer: **Jose Ramos-Horta**",      # value wrapped in emphasis
        'Answer: "Jose Ramos-Horta"',        # value quoted
    ],
)
def test_parse_answer_line_tolerates_label_formats(final_line):
    """The answer span is recovered regardless of label dressing / emphasis."""
    assert parse_answer_line(_CHAIN + final_line) == _GOLD


def test_parse_answer_line_value_on_next_line():
    """Bare label line -> value is read off the following non-empty line."""
    assert parse_answer_line(_CHAIN + "Answer:\nJose Ramos-Horta") == _GOLD
    assert parse_answer_line(_CHAIN + "### Answer\nJose Ramos-Horta") == _GOLD


def test_parse_answer_line_prose_statement():
    """No label, but a 'the answer is X' conclusion."""
    assert parse_answer_line(_CHAIN + "The answer is Jose Ramos-Horta.") == _GOLD
    assert parse_answer_line(_CHAIN + "So the final answer would be Denver.") == "Denver"


def test_parse_answer_line_rejects_bare_label():
    """2026-06-29: a refusal ending in a bare 'Answer:' must NOT extract the literal
    label word — it falls back to the last meaningful line, or "" if there is none."""
    assert parse_answer_line("Answer:") == ""
    assert parse_answer_line("**Final Answer:**") == ""
    refusal = "No context was provided, so I cannot determine the president.\n\nAnswer:"
    got = parse_answer_line(refusal)
    assert got != "Answer" and "cannot determine" in got.lower()


def test_parse_answer_line_last_marker_wins():
    """Multiple label lines -> the last one is the final answer."""
    resp = "Answer: Paris\nOn reflection that is wrong.\nAnswer: London"
    assert parse_answer_line(resp) == "London"


def test_parse_answer_line_fallback_is_last_line_not_whole_essay():
    """No marker at all -> last non-empty line, never the multi-line essay.

    The whole essay systematically scored 0 under the gold->answer NLI scorer;
    the last line at least carries the model's conclusion.
    """
    parsed = parse_answer_line(_CHAIN + "Trey Parker was born in Denver.")
    assert parsed == "Trey Parker was born in Denver"
    assert "Step 1" not in parsed  # the reasoning prefix must be dropped


def test_parse_answer_line_empty_input():
    assert parse_answer_line("") == ""
    assert parse_answer_line("   \n  \n") == ""
