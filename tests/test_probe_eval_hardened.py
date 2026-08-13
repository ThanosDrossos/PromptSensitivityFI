"""R7: the hardened probe-evaluation helpers.

What matters: (a) the grouped OOF heads recover real signal and stay at chance
under a null, (b) the permutation/flip nulls are centred at 0.5, (c) nested
layer selection picks the informative layer without touching the outer test
fold, (d) baselines behave (length orientation, first-word fallback).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from prompt_sensitivity.scripts.probe_eval_hardened import (
    _auc_best_orientation,
    flip_control,
    logistic_oof,
    massmean_oof,
    nested_layer_selection,
    permutation_null,
    text_baseline_oof,
)


def _cell_data(n_q=40, per_q=4, dim=30, signal=1.0, seed=0):
    """Question-blocked binary data: label is a QUESTION property (like
    dispersion/fragility), features carry `signal` times the label direction."""
    rng = np.random.default_rng(seed)
    qids, y, X = [], [], []
    w = rng.normal(size=dim)
    for q in range(n_q):
        lab = q % 2
        for _ in range(per_q):
            qids.append(f"q{q}")
            y.append(lab)
            X.append(rng.normal(size=dim) + signal * lab * w / np.linalg.norm(w))
    return np.array(X), np.array(y), pd.Series(qids)


def test_massmean_and_logistic_recover_signal_and_stay_at_chance_on_noise():
    X, y, q = _cell_data(signal=3.0)
    for fn in (massmean_oof, logistic_oof):
        assert roc_auc_score(y, fn(X, y, q)) > 0.9
    # On pure noise, cross-validated mean-based heads score systematically BELOW
    # 0.5 (leave-out anticorrelation: a held-out point was excluded from its own
    # class mean, which pushes that mean away from it). This is exactly why the
    # evaluation compares against REFITTED permutation nulls — which inherit the
    # same bias — and never against a nominal 0.5. The test encodes that: the
    # noise AUROC must sit inside its own permutation null band, wherever that is.
    X0, y0, q0 = _cell_data(signal=0.0, seed=1)
    null = permutation_null(lambda X_, y_, q_, seed: massmean_oof(X_, y_, q_, seed=seed),
                            X0, y0, q0, n_perm=30, seed=0)
    real = roc_auc_score(y0, massmean_oof(X0, y0, q0))
    lo, hi = np.percentile(null, [2.5, 97.5])
    assert lo - 0.05 < real < hi + 0.05, (
        f"noise AUROC {real:.3f} outside its own null band [{lo:.3f},{hi:.3f}]")


def test_permutation_null_is_centred_at_half_even_with_real_signal():
    """The null must break the label-feature link while preserving the block
    structure — its mean must sit at 0.5 regardless of how strong the true
    signal is (V10: a single draw ranged .37–.58; a distribution is the fix)."""
    X, y, q = _cell_data(signal=3.0)
    null = permutation_null(lambda X_, y_, q_, seed: massmean_oof(X_, y_, q_, seed=seed),
                            X, y, q, n_perm=30, seed=0)
    assert len(null) >= 25
    assert abs(null.mean() - 0.5) < 0.05
    assert null.std() > 0.01, "must be a distribution, not a constant"
    # and the real signal must clear it
    real = roc_auc_score(y, massmean_oof(X, y, q))
    assert real > null.max()


def test_flip_control_breaks_within_question_signal():
    """For a within-question label (vagueness), permutation between questions is
    a no-op; the flip control is the meaningful null and must sit at ~0.5."""
    rng = np.random.default_rng(2)
    n_q, dim = 60, 20
    qids, y, X = [], [], []
    w = rng.normal(size=dim)
    for qi in range(n_q):
        for lab in (0, 1):                       # every question has both labels
            qids.append(f"q{qi}")
            y.append(lab)
            X.append(rng.normal(size=dim) + 3.0 * lab * w / np.linalg.norm(w))
    X, y, qids = np.array(X), np.array(y), pd.Series(qids)
    oof = massmean_oof(X, y, qids)
    assert roc_auc_score(y, oof) > 0.85
    null = flip_control(oof, y, qids, n_draws=60, seed=0)
    assert abs(null.mean() - 0.5) < 0.06


def test_nested_layer_selection_finds_the_informative_layer():
    Xs_good, y, q = _cell_data(signal=3.0, seed=3)
    Xs_noise, _, _ = _cell_data(signal=0.0, seed=4)
    Xs = {0.25: Xs_noise, 0.75: Xs_good}
    res = nested_layer_selection(Xs, y, q, binary=True, seed=0)
    assert res["selected_mode"] == 0.75
    assert res["score"] > 0.8


def test_nested_selection_does_not_leak_on_pure_noise():
    """With two noise layers, selecting the better-looking one inside CV must
    NOT produce OPTIMISTIC outer scores — the whole point of nesting. (Below
    0.5 is fine: the leave-out anticorrelation bias pushes null OOF scores
    down, never up, so only the upper side indicates leakage.)"""
    Xa, y, q = _cell_data(signal=0.0, seed=5)
    Xb, _, _ = _cell_data(signal=0.0, seed=6)
    res = nested_layer_selection({0.25: Xa, 0.75: Xb}, y, q, binary=True, seed=0)
    assert res["score"] < 0.62, f"optimistic leak: {res['score']:.3f}"


def test_length_baseline_orientation_is_resolved_by_max():
    y = np.array([0, 0, 0, 1, 1, 1])
    short_is_pos = np.array([9.0, 8.0, 7.0, 2.0, 1.0, 3.0])
    assert _auc_best_orientation(y, short_is_pos) == pytest.approx(1.0)
    assert _auc_best_orientation(y, -short_is_pos) == pytest.approx(1.0)


def test_first_word_baseline_uses_train_rates_and_falls_back_to_prevalence():
    texts = ["who won", "who lost", "when did", "when was",
             "who is", "who are", "when will", "zebra question"]
    y = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    q = pd.Series([f"q{i}" for i in range(len(y))])
    s = text_baseline_oof(texts, y, q, "first_word", n_splits=2, seed=0)
    # 'who' rows must score higher than 'when' rows wherever both were seen in train
    assert np.nanmean(s[[0, 1, 4, 5]]) > np.nanmean(s[[2, 3, 6]])
    assert np.all(np.isfinite(s)), "unseen first words must fall back, not NaN"
