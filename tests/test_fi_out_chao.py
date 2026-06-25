"""P2-4: Chao-1987 rarefaction correction in estimate_a_q."""

from __future__ import annotations

from prompt_sensitivity.metrics.fi_out import estimate_a_q


def test_chao_bumps_richness_when_singletons_present():
    assignments = {0: [0, 0, 1], 1: [2, 3, 3], 2: [4, 5, 6]}
    observed = len({c for a in assignments.values() for c in a})
    assert observed == 7
    est = estimate_a_q(assignments)
    assert est >= observed          # never below the observed count
    assert est > observed           # singletons present -> non-trivial correction


def test_chao_no_singletons_equals_observed():
    # every cluster seen >= 2x -> f1 = 0 -> n_chao = observed
    assignments = {0: [0, 0], 1: [1, 1], 2: [2, 2]}
    assert estimate_a_q(assignments) == 3


def test_chao_floor_one_on_empty():
    assert estimate_a_q({}) == 1
