"""Output-space clustering collapse fix (2026-06-28): label criterion (T2),
answer-level clustering (T1), and a_q surfacing (T4).

The NLI forward pass (`h_sem._nli_prob_vectors`) is patched with deterministic
fakes so the criterion + union-find logic is exercised WITHOUT loading DeBERTa,
mirroring the existing test_h_sem_pure convention.
"""

from __future__ import annotations

import sys

import numpy as np

from prompt_sensitivity.config import load_config
from prompt_sensitivity.metrics import build_metric_tuple
from prompt_sensitivity.scripts.e2e_smoke import _clustering_inputs

# id2label is {0: entailment, 1: neutral, 2: contradiction}; entail_idx = 0.
_ENTAIL = np.array([0.90, 0.05, 0.05])   # argmax -> entailment
_CONTRA = np.array([0.05, 0.05, 0.90])   # argmax -> contradiction
_H = sys.modules["prompt_sensitivity.metrics.h_sem"]


def _fake_by_equivalence(equiv: set[frozenset[str]]):
    """Pairs in `equiv` (or identical strings) entail; everything else contradicts."""
    def fake(premises, hypotheses, *, model_name=None):
        out = []
        for p, h in zip(premises, hypotheses):
            same = p == h or frozenset({p, h}) in equiv
            out.append(_ENTAIL.copy() if same else _CONTRA.copy())
        return out, 0
    return fake


def _fake_fixed_vector(vec: np.ndarray):
    def fake(premises, hypotheses, *, model_name=None):
        return [vec.copy() for _ in premises], 0
    return fake


# --------------------------------------------------------------------------- #
# T2 — label criterion                                                        #
# --------------------------------------------------------------------------- #


def test_label_criterion_merges_equivalent_splits_distinct(monkeypatch):
    cfg = load_config()
    monkeypatch.setattr(
        _H, "_nli_prob_vectors",
        _fake_by_equivalence({frozenset({"Paris", "the city of Paris"})}),
    )
    one = _H.cluster_responses(["Paris", "the city of Paris"], config=cfg, criterion="label")
    assert len(set(one)) == 1

    monkeypatch.setattr(_H, "_nli_prob_vectors", _fake_by_equivalence(set()))
    two = _H.cluster_responses(["Paris", "Berlin"], config=cfg, criterion="label")
    assert len(set(two)) == 2


def test_label_criterion_is_categorical_not_a_05_cutoff(monkeypatch):
    """argmax==neutral blocks a merge even when P(entail)=0.45 — the over-merge
    regime the fix targets (prob@0.5 would also not merge here, but label would
    also refuse at 0.55 where prob would merge)."""
    cfg = load_config()
    neutral_top = np.array([0.45, 0.50, 0.05])   # argmax -> neutral
    monkeypatch.setattr(_H, "_nli_prob_vectors", _fake_fixed_vector(neutral_top))
    assert len(set(_H.cluster_responses(["a", "b"], config=cfg, criterion="label"))) == 2


# --------------------------------------------------------------------------- #
# T2 — prob criterion reproduces the legacy threshold merge                    #
# --------------------------------------------------------------------------- #


def test_clustering_dedups_identical_strings_before_nli(monkeypatch):
    """Perf: exact-duplicate responses are collapsed before the O(u^2) NLI pass,
    and still land in the same cluster."""
    cfg = load_config()
    seen = {}

    def fake(premises, hypotheses, *, model_name=None):
        seen["n_premises"] = len(premises)   # how many NLI inputs were built
        return [_CONTRA.copy() for _ in premises], 0   # A vs B -> distinct

    monkeypatch.setattr(_H, "_nli_prob_vectors", fake)
    a = _H.cluster_responses(["A", "A", "A", "B"], config=cfg, criterion="label")
    assert a[0] == a[1] == a[2] and a[3] != a[0]   # the three A's share a cluster
    assert seen["n_premises"] == 2                  # 1 unique pair x 2 directions (not 12)


def test_prob_criterion_reproduces_threshold(monkeypatch):
    cfg = load_config()
    # entail 0.6 >= 0.5 both ways -> merge -> 1 cluster.
    monkeypatch.setattr(_H, "_nli_prob_vectors", _fake_fixed_vector(np.array([0.6, 0.2, 0.2])))
    assert len(set(_H.cluster_responses(["a", "b"], config=cfg, criterion="prob", threshold=0.5))) == 1
    # entail 0.4 < 0.5 -> no merge -> 2 clusters.
    monkeypatch.setattr(_H, "_nli_prob_vectors", _fake_fixed_vector(np.array([0.4, 0.3, 0.3])))
    assert len(set(_H.cluster_responses(["a", "b"], config=cfg, criterion="prob", threshold=0.5))) == 2


# --------------------------------------------------------------------------- #
# T1 — answer-level clustering inputs                                          #
# --------------------------------------------------------------------------- #


def test_clustering_inputs_answer_collapses_prose_keeps_distinct():
    resp = {
        0: ["Long reasoning...\nAnswer: Paris", "Totally different prose.\nFinal Answer: Paris"],
        1: ["Step 1: ...\nAnswer: Berlin"],
    }
    out = _clustering_inputs(resp, "answer")
    assert out[0][0] == out[0][1] == "Paris"   # same answer, different prose -> one string
    assert out[1][0] == "Berlin"
    assert out[0][0] != out[1][0]


def test_clustering_inputs_response_mode_passes_raw_through():
    resp = {0: ["a\nAnswer: X", "b\nAnswer: X"]}
    assert _clustering_inputs(resp, "response") == resp


def test_clustering_inputs_empty_extraction_falls_back_to_raw():
    resp = {0: ["   "]}                         # nothing extractable
    out = _clustering_inputs(resp, "answer")
    assert out[0][0] == "   "                   # raw kept, sample not dropped


# --------------------------------------------------------------------------- #
# T4 — a_q surfaced; a_q == 1 implies floored output metrics                   #
# --------------------------------------------------------------------------- #


def test_a_q_one_implies_floored_output_metrics():
    cluster_assignments = {0: [0, 0], 1: [0, 0], 2: [0, 0]}   # single pooled cluster
    rng = np.random.default_rng(0)
    tup = build_metric_tuple(
        question_id="q", ladder_type="random", level=0, model_key="llama_3_1_8b",
        scores=[1.0, 0.5, 0.0], cluster_assignments=cluster_assignments,
        prompt_embeddings=rng.normal(size=(3, 8)),
        response_embeddings={i: rng.normal(size=(2, 8)) for i in range(3)},
    )
    assert tup.a_q == 1
    assert tup.h_sem_mean == 0.0
    assert tup.fi_out_mean == 0.0
    assert tup.variation_ratio == 0.0


def test_a_q_gt_one_gives_positive_h_sem():
    cluster_assignments = {0: [0, 1], 1: [0, 1]}              # 2 balanced clusters
    rng = np.random.default_rng(1)
    tup = build_metric_tuple(
        question_id="q", ladder_type="random", level=0, model_key="llama_3_1_8b",
        scores=[1.0, 0.0], cluster_assignments=cluster_assignments,
        prompt_embeddings=rng.normal(size=(2, 8)),
        response_embeddings={i: rng.normal(size=(2, 8)) for i in range(2)},
    )
    assert tup.a_q >= 2
    assert tup.h_sem_mean > 0.0
