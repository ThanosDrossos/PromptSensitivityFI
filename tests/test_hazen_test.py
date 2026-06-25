"""P2-2: Hazen stepped-behavior detection on FI_in(k) curves."""

from __future__ import annotations

from prompt_sensitivity.metrics.hazen_test import detect_steps


def test_one_clean_step_two_plateaus():
    # [0,0,0,1,1,1] over 6 k-bins with tight (zero-width) CIs: one step at i=2.
    ks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    vals = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    lo = list(vals)
    hi = list(vals)
    out = detect_steps(ks, vals, lo, hi)
    assert out["n_plateaus"] == 2
    assert out["n_steps"] == 1
    assert out["step_locations"] == [2]
    assert out["fits_hazen_pattern"] is True


def test_flat_curve_no_steps():
    ks = [0.0, 0.5, 1.0]
    vals = [0.3, 0.3, 0.3]
    out = detect_steps(ks, vals, list(vals), list(vals))
    assert out["n_steps"] == 0
    assert out["fits_hazen_pattern"] is False


def test_overlapping_cis_suppress_step():
    # Big value jump but wide overlapping CIs -> not a (statistical) step.
    ks = [0.0, 1.0]
    vals = [0.0, 1.0]
    lo = [0.0, 0.0]
    hi = [1.0, 1.0]
    out = detect_steps(ks, vals, lo, hi)
    assert out["n_steps"] == 0
    assert out["fits_hazen_pattern"] is False
