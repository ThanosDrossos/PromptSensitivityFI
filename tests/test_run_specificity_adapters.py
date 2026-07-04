"""Specificity driver adapters: closed-book prompt via the REAL assembler."""

from __future__ import annotations

from prompt_sensitivity.scripts.e2e_smoke import _assemble_messages
from prompt_sensitivity.scripts.run_specificity import (
    _ladder_row_for,
    _SpecQuestionView,
)
from prompt_sensitivity.specificity.build_levels import SpecRow


def _row(level: int = 0) -> SpecRow:
    return SpecRow(
        question_id="aq1",
        spec_level=level,
        question_text="Who plays the doctor in dexter season 1?",
        target_answers=["Tony Goldwyn", "Goldwyn"],
        m_valid=3 if level == 0 else 1,
        m0=3,
        target_idx=1,
    )


def test_view_is_closed_book_binary_path():
    view = _SpecQuestionView(_row())
    assert view.paragraphs == []
    assert view.has_decomposition() is False      # -> use_cot=False, binary NLI path
    assert view.answer == "Tony Goldwyn"          # primary gold; OR-variants in scoring
    assert view.dataset == "ambigqa"


def test_ladder_row_yields_no_context_prompt():
    row = _row(level=1)
    view = _SpecQuestionView(row)
    lrow = _ladder_row_for(row)
    assert lrow.paragraph_indices == [] and lrow.gold_count == 0
    msgs = _assemble_messages(view, "Some paraphrase of the question?", lrow, use_cot=False)
    assert [m.role for m in msgs] == ["system", "user"]
    user = msgs[1].content
    assert "Some paraphrase of the question?" in user
    # closed book: no context block sneaks in
    assert "Context" not in user and "Paragraph" not in user
