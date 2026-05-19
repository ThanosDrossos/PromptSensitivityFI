"""Bit-cost estimators. Research_Design_v3 §4.4.

Brief invariants:
  1. Closed-form b_theo must match scipy.stats.hypergeom analytical solution
     for every l in {1, 2, 4, 6, 8}.
  2. b_theo(N=10, K=2, l=10) == 0 (gold is guaranteed at full coverage).
  3. b_theo(N, K, 0) == +inf (no paragraphs => no gold => surprise infinite).
  4. b_emp recovers the log2 ratio and handles zero edge cases.
"""

from __future__ import annotations

import math

import pytest

from prompt_sensitivity.ladders.bit_cost import (
    _scipy_prob_at_least_one_gold,
    b_emp,
    b_theo,
    b_theo_table,
    prob_at_least_one_gold,
)


# --- invariant 1 ----------------------------------------------------------


@pytest.mark.parametrize("l", [1, 2, 4, 6, 8])
def test_b_theo_matches_scipy_hypergeom(l: int):
    closed_form = prob_at_least_one_gold(10, 2, l)
    reference = _scipy_prob_at_least_one_gold(10, 2, l)
    assert closed_form == pytest.approx(reference, rel=1e-12)


# --- invariant 2 ----------------------------------------------------------


def test_b_theo_zero_when_gold_is_guaranteed():
    """Pigeonhole: with K=2 gold in N=10, any sample of l>=N-K+1=9 forces gold inclusion."""
    assert b_theo(10, 2, 10) == 0.0
    assert b_theo(10, 2, 9) == 0.0
    # At l = N-K = 8, gold is NOT yet guaranteed: a sample could still pick
    # all 8 distractors. P(no gold) = C(8,8)/C(10,8) = 1/45 -> b_theo ~ 0.032.
    expected = -math.log2(1 - 1 / 45)
    assert b_theo(10, 2, 8) == pytest.approx(expected, rel=1e-9)


# --- invariant 3 ----------------------------------------------------------


def test_b_theo_at_zero_paragraphs_is_infinite():
    assert math.isinf(b_theo(10, 2, 0))


# --- table consistency ----------------------------------------------------


def test_b_theo_table_levels_match_design_doc():
    table = b_theo_table(10, 2, [0, 2, 4, 6, 8, 10])
    levels = [r["level"] for r in table]
    assert levels == [0, 2, 4, 6, 8, 10]
    # As level grows, P(>=1 gold) is monotonically non-decreasing.
    probs = [r["P(>=1 gold)"] for r in table]
    for a, b in zip(probs, probs[1:]):
        assert b >= a


# --- monotonicity ----------------------------------------------------------


def test_b_theo_decreases_in_level():
    """b_theo(l) is monotonically non-increasing in l (after level 0)."""
    prev = math.inf
    for l in [2, 4, 6, 8, 10]:
        cur = b_theo(10, 2, l)
        assert cur <= prev + 1e-12
        prev = cur


# --- invariant 4: b_emp edge cases ----------------------------------------


def test_b_emp_basic_ratio():
    # |U_above|/|U_below| = 8/4 = 2 -> log2 = 1
    assert b_emp(8, 4) == pytest.approx(1.0)


def test_b_emp_zero_below_is_inf():
    assert math.isinf(b_emp(5, 0))
    assert b_emp(5, 0) > 0


def test_b_emp_zero_above_is_neg_inf():
    assert math.isinf(b_emp(0, 5))
    assert b_emp(0, 5) < 0


# --- degenerate K=0 (no gold) --------------------------------------------


def test_no_gold_means_zero_prob_and_inf_bit_cost():
    """K=0: no golds exist, so prob = 0 for any l."""
    for l in [0, 2, 5, 10]:
        assert prob_at_least_one_gold(10, 0, l) == 0.0
        assert math.isinf(b_theo(10, 0, l))
