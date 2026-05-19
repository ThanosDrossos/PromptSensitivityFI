"""FI_out tests + the §7.4.1 identity check FI_out + H_sem = log2|A_q|."""

from __future__ import annotations

import math

import pytest

from prompt_sensitivity.metrics.fi_out import (
    estimate_a_q,
    fi_out,
    fi_out_per_prompt,
    fi_out_summary,
)
from prompt_sensitivity.metrics.h_sem import entropy_from_assignment


def test_fi_out_identity():
    """Section_7 §7.4.1: FI_out(x) + H_sem(Y|X=x) = log2|A_q|."""
    cluster_assignment = [0, 0, 1, 1, 2]
    a_q = 4  # pretend |A_q| = 4
    h = entropy_from_assignment(cluster_assignment)
    fi = fi_out_per_prompt(cluster_assignment, a_q)
    assert h + fi == pytest.approx(math.log2(a_q))


def test_fi_out_maximally_restrictive_prompt():
    """All responses in one cluster -> FI_out = log2|A_q| (max restrictiveness)."""
    cluster_assignment = [0, 0, 0, 0]
    a_q = 4
    assert fi_out_per_prompt(cluster_assignment, a_q) == pytest.approx(math.log2(a_q))


def test_fi_out_uniform_over_a_q_is_zero():
    """Uniform over |A_q| clusters -> H = log2|A_q|, FI_out = 0."""
    cluster_assignment = [0, 1, 2, 3]
    a_q = 4
    assert fi_out_per_prompt(cluster_assignment, a_q) == pytest.approx(0.0, abs=1e-9)


def test_fi_out_clamps_negative_to_zero():
    """If |A_q| is under-estimated, FI_out could go negative; we clamp."""
    cluster_assignment = [0, 1, 2, 3, 4]  # 5 unique clusters in one prompt
    a_q = 2  # severely under-estimated
    val = fi_out_per_prompt(cluster_assignment, a_q)
    assert val == 0.0


def test_estimate_a_q_unions_across_prompts():
    """|A_q| is the count of unique cluster IDs pooled across paraphrases."""
    assignments = {0: [0, 1], 1: [1, 2], 2: [3]}
    assert estimate_a_q(assignments) == 4


def test_fi_out_returns_one_per_prompt_dict():
    assignments = {0: [0, 0, 1], 1: [2, 2, 2]}
    per_prompt = fi_out(assignments)
    assert set(per_prompt.keys()) == {0, 1}
    # Prompt 1 has zero entropy -> maximally restrictive at this |A_q|.
    a_q = estimate_a_q(assignments)
    assert per_prompt[1] == pytest.approx(math.log2(a_q))


def test_fi_out_summary_returns_mean_var():
    per_prompt = {0: 0.5, 1: 1.5}
    mean, var = fi_out_summary(per_prompt)
    assert mean == pytest.approx(1.0)
    assert var == pytest.approx(0.25)
