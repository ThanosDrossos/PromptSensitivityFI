"""Cohen's κ — gate computation."""

import pytest

from prompt_sensitivity.scripts.compute_kappa import _parse_label, cohens_kappa


def test_perfect_agreement_is_one():
    r = cohens_kappa([1, 0, 1, 0, 1], [1, 0, 1, 0, 1])
    assert r["kappa"] == pytest.approx(1.0)
    assert r["p_observed"] == pytest.approx(1.0)


def test_chance_agreement_is_zero():
    """Construction with p_obs == p_exp -> κ = 0 exactly.

    Each rater uses prevalence 0.5 (balanced). Exactly half of the four pairs
    match -> p_obs = 0.5 = p_exp.
    """
    a = [1, 1, 0, 0]
    b = [1, 0, 1, 0]
    r = cohens_kappa(a, b)
    assert r["kappa"] == pytest.approx(0.0)
    assert r["p_observed"] == pytest.approx(0.5)
    assert r["p_expected"] == pytest.approx(0.5)


def test_total_disagreement_is_minus_one():
    r = cohens_kappa([1, 0, 1, 0], [0, 1, 0, 1])
    assert r["kappa"] == pytest.approx(-1.0)


def test_parse_label_handles_variants():
    assert _parse_label("1") == 1
    assert _parse_label(0) == 0
    assert _parse_label("yes") == 1
    assert _parse_label("No") == 0
    assert _parse_label("") is None
    assert _parse_label(None) is None
    assert _parse_label("maybe") is None


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        cohens_kappa([1, 0], [1, 0, 1])
