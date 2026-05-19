"""Errica two-number axis: S_τ + 1-TVD."""

from __future__ import annotations

import math

import pytest

from prompt_sensitivity.metrics.errica import (
    s_tau_freeform,
    s_tau_multiple_choice,
    tvd,
    tvd_consistency,
)


# --- S_tau (free-form) ----------------------------------------------------


def test_s_tau_freeform_all_same_cluster_is_zero():
    """No within-prompt sensitivity if every response is the same cluster."""
    assert s_tau_freeform([0, 0, 0, 0], a_q_size=4) == 0.0


def test_s_tau_freeform_uniform_is_one():
    """Uniform spread over |A_q| clusters -> H = log2|A_q| -> S_τ = 1."""
    assert s_tau_freeform([0, 1, 2, 3], a_q_size=4) == pytest.approx(1.0)


def test_s_tau_freeform_returns_zero_when_a_q_singleton():
    """If |A_q| = 1 the metric is undefined; return 0 conservatively."""
    assert s_tau_freeform([0, 0, 0], a_q_size=1) == 0.0


# --- S_tau (multiple choice) ---------------------------------------------


def test_s_tau_mc_matches_errica_eq3_on_two_class():
    """S_τ on binary class with 50/50 split is exactly 1."""
    assert s_tau_multiple_choice([5, 5], n_classes=2) == pytest.approx(1.0)


def test_s_tau_mc_all_one_class_is_zero():
    assert s_tau_multiple_choice([10, 0, 0], n_classes=3) == 0.0


def test_s_tau_mc_zero_samples_is_zero():
    assert s_tau_multiple_choice([0, 0, 0], n_classes=3) == 0.0


# --- 1 - TVD --------------------------------------------------------------


def test_tvd_identical_distributions_is_zero():
    import numpy as np
    p = np.array([0.5, 0.5])
    assert tvd(p, p) == 0.0


def test_tvd_disjoint_supports_is_one():
    import numpy as np
    p = np.array([1.0, 0.0])
    q = np.array([0.0, 1.0])
    assert tvd(p, q) == pytest.approx(1.0)


def test_tvd_consistency_identical_assignments_is_one():
    """Two paraphrases with same cluster distribution -> 1 - TVD = 1."""
    assignments = {0: [0, 0, 1, 1], 1: [0, 0, 1, 1]}
    assert tvd_consistency(assignments) == pytest.approx(1.0)


def test_tvd_consistency_orthogonal_assignments_is_zero():
    """Paraphrase 0 always cluster 0; paraphrase 1 always cluster 1 -> 1 - TVD = 0."""
    assignments = {0: [0, 0, 0, 0], 1: [1, 1, 1, 1]}
    assert tvd_consistency(assignments) == pytest.approx(0.0)


def test_tvd_consistency_single_paraphrase_is_one():
    """No pairs to compare -> trivially consistent."""
    assert tvd_consistency({0: [0, 1, 2]}) == 1.0


def test_tvd_consistency_three_way_pairs():
    """Mean across (N choose 2) pairs, not just adjacent."""
    assignments = {0: [0, 0], 1: [0, 0], 2: [1, 1]}
    # pairs: (0,1) TVD=0 -> 1.0; (0,2) TVD=1 -> 0.0; (1,2) TVD=1 -> 0.0
    # mean = 1/3
    assert tvd_consistency(assignments) == pytest.approx(1 / 3)
