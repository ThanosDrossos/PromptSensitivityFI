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


def test_main_frees_generator_vram_between_prep_and_eval(monkeypatch, tmp_path):
    """Regression for cluster job 5762430: Phi-4 (28 GiB) stayed resident after
    paraphrase prep and OOM'd the Qwen load on the 40 GB gpu_a100_short card.
    main() must call _free_vram AFTER the prep and BEFORE the first cell."""
    import sys

    from prompt_sensitivity.data.ambigqa_schemas import AmbigInterpretation, AmbigQuestion
    from prompt_sensitivity.scripts import run_specificity as rs

    calls: list[str] = []
    q = AmbigQuestion(
        id="aq-oom", question="Ambiguous?",
        interpretations=[
            AmbigInterpretation(disambiguated_question=f"Variant {i}?", answers=[f"a{i}"])
            for i in range(2)
        ],
    )
    monkeypatch.setattr(rs, "load_ambigqa", lambda **kw: [q])
    monkeypatch.setattr(
        rs, "_generate_spec_paraphrases",
        lambda cfg, rows, mp: (calls.append("prep"),
                               {(r.question_id, r.spec_level): ["p"] for r in rows})[1],
    )
    monkeypatch.setattr(rs, "_free_vram", lambda: calls.append("free"))
    monkeypatch.setattr(
        rs, "_run_spec_cell",
        lambda config, row, model_key, paras, **kw: (calls.append("cell"), {
            "question_id": row.question_id, "spec_level": row.spec_level,
            "model_key": model_key, "f_mean": 1.0, "fi_spec": 0.0,
        })[1],
    )
    monkeypatch.setattr(sys, "argv", [
        "run_specificity", "--n-questions", "1", "--models", "qwen_2_5_7b",
        "--out", str(tmp_path / "spec.parquet"),
    ])
    assert rs.main() == 0
    assert calls[0] == "prep" and calls[1] == "free", f"order was {calls}"
    assert calls[2:] == ["cell", "cell"]          # 2 levels x 1 model
