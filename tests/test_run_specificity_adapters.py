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

    from prompt_sensitivity.data.ambigqa_schemas import (
        AmbigInterpretation,
        AmbigQuestion,
        EvidenceSnippet,
    )
    from prompt_sensitivity.scripts import run_specificity as rs

    calls: list[str] = []
    q = AmbigQuestion(
        id="aq-oom", question="Ambiguous?",
        interpretations=[
            AmbigInterpretation(disambiguated_question=f"Variant {i}?", answers=[f"a{i}"])
            for i in range(2)
        ],
        # v2: the default config requires the target answer in the evidence
        # bundle; cover both possible targets so the question survives the filter.
        evidence=[EvidenceSnippet(title="t", snippet="a0 and a1 are both discussed here")],
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
        lambda config, row, model_key, paras, **kw: (calls.append("cell"), ({
            "question_id": row.question_id, "spec_level": row.spec_level,
            "model_key": model_key, "f_mean": 1.0, "fi_spec": 0.0,
        }, None))[1],
    )
    monkeypatch.setattr(sys, "argv", [
        "run_specificity", "--n-questions", "1", "--models", "qwen_2_5_7b",
        "--out", str(tmp_path / "spec.parquet"),
    ])
    assert rs.main() == 0
    assert calls[0] == "prep" and calls[1] == "free", f"order was {calls}"
    assert calls[2:] == ["cell", "cell"]          # 2 levels x 1 model


def test_paraphrase_universes_persist_incrementally(monkeypatch, tmp_path):
    """30-min singleton-chain requirement: each universe lands in the cache
    parquet AS SOON as it is generated (walltime kill loses at most one), and
    singleton fallbacks are cached too so they are never re-attempted."""
    from prompt_sensitivity.config import load_config
    from prompt_sensitivity.scripts import run_specificity as rs
    from prompt_sensitivity.specificity.build_levels import SpecRow
    import pandas as pd
    import prompt_sensitivity.paraphrases.pipeline as pipeline_mod

    cache = tmp_path / "para.parquet"
    monkeypatch.setattr(rs, "_AMBIGQA_PARAPHRASE_PARQUET", str(cache))
    rows = [
        SpecRow(question_id="qA", spec_level=0, question_text="A?", target_answers=["a"],
                m_valid=2, m0=2, target_idx=0),
        SpecRow(question_id="qB", spec_level=0, question_text="B?", target_answers=["b"],
                m_valid=2, m0=2, target_idx=0),
    ]

    class _PSet:
        def __init__(self, texts):
            self.accepted = [type("AP", (), {"text": t})() for t in texts]

    snapshots: list[int] = []

    def fake_build(qid, text, *, config=None, gold_answer=None, gold_answers=None):
        # capture how many rows were ALREADY persisted when this universe starts
        snapshots.append(len(pd.read_parquet(cache)) if cache.exists() else 0)
        if text == "A?":
            return _PSet(["A one?", "A two?"])
        raise RuntimeError("generator exploded")      # qB -> singleton fallback

    monkeypatch.setattr(pipeline_mod, "build_paraphrase_set", fake_build)
    out = rs._generate_spec_paraphrases(load_config(), rows, max_paraphrases=10)

    assert snapshots == [0, 2]                 # qA's 2 rows were on disk BEFORE qB ran
    df = pd.read_parquet(cache)
    assert len(df) == 3                        # 2 accepted + 1 singleton_fallback
    assert set(df["outcome"]) == {"accepted", "singleton_fallback"}
    assert out[("qB", 0)] == ["B?"]

    # Second call: everything (incl. the fallback) served from cache — the
    # generator must NOT run again.
    def exploding_build(*a, **k):
        raise AssertionError("generator must not be re-invoked on resume")

    monkeypatch.setattr(pipeline_mod, "build_paraphrase_set", exploding_build)
    out2 = rs._generate_spec_paraphrases(load_config(), rows, max_paraphrases=10)
    assert out2[("qA", 0)] == ["A one?", "A two?"] and out2[("qB", 0)] == ["B?"]


def test_generation_uses_multi_gold_at_L0_and_target_at_L1(tmp_path, monkeypatch):
    """The fix: L0 (ambiguous) paraphrases are constrained against the WHOLE
    interpretation set (all_answers); L1 (disambiguated) against the target's
    variants only. Single-gold at L0 rejected 100% of NLI-valid paraphrases."""
    from prompt_sensitivity.config import load_config
    from prompt_sensitivity.scripts import run_specificity as rs
    from prompt_sensitivity.specificity.build_levels import SpecRow
    import prompt_sensitivity.paraphrases.pipeline as pipeline_mod

    monkeypatch.setattr(rs, "_AMBIGQA_PARAPHRASE_PARQUET", str(tmp_path / "p.parquet"))
    rows = [
        SpecRow(question_id="q", spec_level=0, question_text="Ambig?",
                target_answers=["A1", "A1-alias"], all_answers=["A1", "A1-alias", "A2", "A3"],
                m_valid=3, m0=3, target_idx=0),
        SpecRow(question_id="q", spec_level=1, question_text="Disambig?",
                target_answers=["A1", "A1-alias"], all_answers=["A1", "A1-alias", "A2", "A3"],
                m_valid=1, m0=3, target_idx=0),
    ]
    seen: dict[int, list[str]] = {}

    def capture_build(qid, text, *, config=None, gold_answer=None, gold_answers=None):
        lvl = 0 if qid.endswith("L0") else 1
        seen[lvl] = list(gold_answers) if gold_answers is not None else None
        return type("PS", (), {"accepted": [type("AP", (), {"text": text})()]})()

    monkeypatch.setattr(pipeline_mod, "build_paraphrase_set", capture_build)
    rs._generate_spec_paraphrases(load_config(), rows, max_paraphrases=10)

    assert seen[0] == ["A1", "A1-alias", "A2", "A3"]   # L0 -> full interpretation set
    assert seen[1] == ["A1", "A1-alias"]               # L1 -> target variants only


def test_spec_inspection_renderer_contains_every_step():
    from prompt_sensitivity.scripts.run_specificity import _render_spec_inspection_md

    rec = {
        "question_id": "q1", "spec_level": 1, "question_text": "What party took control?",
        "target_answers": ["National Fascist Party", "Fascists"],
        "m0": 3, "m_valid": 1, "target_idx": 0, "model_key": "qwen_2_5_7b",
        "paraphrases": [{"idx": 0, "paraphrase": "Which party seized power?",
                         "answer_t0": "The National Fascist Party", "f": 1.0}],
        "hsem": {"k": 10, "n_clusters": 1, "representatives": {0: "The National Fascist Party"}},
        "metrics": {"f_mean": 1.0, "aufi_in": 0.0, "fi_out_mean": 0.0,
                    "h_sem_mean": 0.0, "a_q": 1, "fi_spec": 1.585},
    }
    md = _render_spec_inspection_md([rec])
    for needle in ["inspection bundle", "Fixed gold", "National Fascist Party",
                   "Which party seized power?", "FI_spec=1.585", "H_sem (k=10): 1 clusters"]:
        assert needle in md, f"missing {needle!r}"
