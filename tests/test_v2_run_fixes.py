"""Fixes from the v2 full-run post-mortem: windowed inspection persistence,
N-mismatch exclusion in paired deltas, and the universe repair planner."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from prompt_sensitivity.scripts.repair_spec_universes import plan_repair
from prompt_sensitivity.scripts.run_specificity import _persist_and_render_inspection
from prompt_sensitivity.scripts.show_specificity import add_fi_out_fixed, paired_deltas


def _rec(qid="q1", level=0):
    return {
        "question_id": qid, "spec_level": level, "question_text": "Q?",
        "target_answers": ["a"], "m0": 2, "m_valid": 2 if level == 0 else 1,
        "target_idx": 0, "model_key": "m", "context_mode": "uniform_evidence",
        "evidence": ["T: snip"], "n_evidence": 1,
        "paraphrases": [{"idx": 0, "paraphrase": "P?", "answer_t0": "a", "f": 1.0}],
        "hsem": None,
        "metrics": {"f_mean": 1.0, "aufi_in": 0.0, "fi_out_mean": 0.0,
                    "h_sem_mean": 0.0, "a_q": 1, "fi_spec": 0.0},
    }


def test_inspection_survives_windowed_resume(tmp_path):
    """Window A writes records + dies; window B computes nothing (all resumed)
    but must still (re)render the FULL md from the jsonl."""
    out = tmp_path / "specificity_v2_metrics.parquet"
    _persist_and_render_inspection(out, [_rec("q1", 0), _rec("q1", 1)])   # window A
    md = tmp_path / "inspect_specificity_v2_metrics.md"
    assert md.exists() and "q1" in md.read_text(encoding="utf-8")

    md.unlink()                                              # simulate md lost
    _persist_and_render_inspection(out, [])                  # window B: no new recs
    assert md.exists(), "no-op window must regenerate the md from the jsonl"
    text = md.read_text(encoding="utf-8")
    assert "q1" in text and "Level 0" in text and "Level 1" in text

    # dedup: re-running a cell overwrites its record instead of duplicating
    _persist_and_render_inspection(out, [_rec("q1", 0)])
    jsonl = tmp_path / "inspect_specificity_v2_metrics.jsonl"
    assert len(jsonl.read_text(encoding="utf-8").splitlines()) == 3   # appends kept
    assert md.read_text(encoding="utf-8").count("## 1.") == 1          # rendered once


def test_inspection_tolerates_torn_jsonl_line(tmp_path):
    out = tmp_path / "m.parquet"
    jsonl = tmp_path / "inspect_m.jsonl"
    jsonl.write_text(json.dumps(_rec()) + '\n{"question_id": "torn', encoding="utf-8")
    _persist_and_render_inspection(out, [])                  # must not raise
    assert (tmp_path / "inspect_m.md").exists()


def _cells(qid, n0, n1, aufi0, aufi1):
    base = dict(model_key="m", f_mean=0.0, h_sem_mean=0.1, m0=2, fi_spec=0.0)
    return [
        {**base, "question_id": qid, "spec_level": 0, "n_paraphrases": n0, "aufi_in": aufi0},
        {**base, "question_id": qid, "spec_level": 1, "n_paraphrases": n1, "aufi_in": aufi1},
    ]


def test_paired_deltas_excludes_n_mismatched_pairs():
    df = pd.DataFrame(
        _cells("ok", 10, 10, 3.0, 1.0)          # legit: delta -2
        + _cells("artifact", 1, 10, 0.98, 3.37)  # the v2 outlier pattern: +2.4 "increase"
    )
    deltas, n_excluded = paired_deltas(df, ["aufi_in"])
    assert n_excluded == 1
    assert len(deltas) == 1
    assert float(deltas["aufi_in"].iloc[0]) == -2.0
    # kept when exclusion is off
    d2, n2 = paired_deltas(df, ["aufi_in"], exclude_mismatched=False)
    assert n2 == 0 and len(d2) == 2


def test_add_fi_out_fixed_is_log2_m0_minus_hsem():
    df = pd.DataFrame([{"m0": 4, "h_sem_mean": 0.5}, {"m0": 1, "h_sem_mean": 0.0}])
    out = add_fi_out_fixed(df)
    assert np.isclose(out["fi_out_fixed"].iloc[0], 2.0 - 0.5)
    assert np.isclose(out["fi_out_fixed"].iloc[1], 0.0)


def test_add_fi_out_fixed_respects_pipeline_column():
    """Mixed resume parquet: rows written by the new driver carry fi_out_fixed;
    older rows are NaN and get the identical derivation filled in."""
    df = pd.DataFrame([
        {"m0": 4, "h_sem_mean": 0.5, "fi_out_fixed": 1.5},    # pipeline row kept
        {"m0": 4, "h_sem_mean": 1.0, "fi_out_fixed": np.nan},  # old row -> derived
    ])
    out = add_fi_out_fixed(df)
    assert np.isclose(out["fi_out_fixed"].iloc[0], 1.5)
    assert np.isclose(out["fi_out_fixed"].iloc[1], 2.0 - 1.0)


def test_graded_f_scores_means_per_paraphrase(monkeypatch):
    """F_graded(x) = fraction of that paraphrase's k samples hitting any gold,
    computed in ONE flattened multi-gold batch, ordered by paraphrase index."""
    from prompt_sensitivity.scripts import run_specificity as rs

    captured: dict = {}

    def fake_multi(golds, answers, *, config=None, permissive=False):
        captured["golds"] = list(golds)
        captured["answers"] = list(answers)
        # paraphrase 0: [1,0], paraphrase 1: [1,1]
        return [1.0, 0.0, 1.0, 1.0]

    monkeypatch.setattr(rs, "f_score_batch_multi_gold", fake_multi)
    out = rs._graded_f_scores(
        ["g1", "g2"],
        {1: ["s1c", "s1d"], 0: ["s0a", "s0b"]},   # dict order shuffled on purpose
        config=None,
    )
    assert out == [0.5, 1.0]                       # sorted by paraphrase index
    assert captured["answers"] == ["s0a", "s0b", "s1c", "s1d"]  # one flat batch
    assert captured["golds"] == ["g1", "g2"]
    assert rs._graded_f_scores(["g"], {}, config=None) == []


def test_repair_planner_pair_consistent_and_retries_fallbacks():
    para = pd.DataFrame(
        # qA: complete (10+10) -> untouched
        [{"question_id": "qA", "spec_level": lvl, "outcome": "accepted",
          "paraphrase_idx": i, "text": f"t{i}"} for lvl in (0, 1) for i in range(10)]
        # qB: L0 singleton fallback, L1 full -> BOTH levels dropped (pair-consistent)
        + [{"question_id": "qB", "spec_level": 0, "outcome": "singleton_fallback",
            "paraphrase_idx": 0, "text": "orig"}]
        + [{"question_id": "qB", "spec_level": 1, "outcome": "accepted",
            "paraphrase_idx": i, "text": f"b{i}"} for i in range(10)]
    )
    metrics = pd.DataFrame([
        {"question_id": q, "spec_level": lvl, "model_key": "m"}
        for q in ("qA", "qB") for lvl in (0, 1)
    ])
    affected, para_keep, metrics_keep = plan_repair(para, metrics, min_n=10)
    assert affected == {"qB"}
    kept = para[para_keep]
    assert set(kept.question_id) == {"qA"}          # qB rows (incl. fallback) dropped
    assert len(metrics[metrics_keep]) == 2          # only qA cells survive
    # nothing to do when everything is at target
    aff2, keep2, _ = plan_repair(para[para.question_id == "qA"], None, min_n=10)
    assert aff2 == set() and keep2.all()
