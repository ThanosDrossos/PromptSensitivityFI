"""Tests for the A5 recovery-simulation helpers (pure functions only)."""

import numpy as np
import pytest

from prompt_sensitivity.scripts.rho_f_recovery_sim import (
    paired_mean_delta,
    rubin_paired_test,
    simulate_cells,
)


def test_simulate_cells_shapes_and_determinism():
    mus = np.array([0.5, 0.9, 0.05])
    sizes = np.array([10, 7, 4])
    rhos = np.array([0.3, 0.3, 0.3])
    a = simulate_cells(mus, sizes, rhos, k=10, rng=np.random.default_rng(0))
    b = simulate_cells(mus, sizes, rhos, k=10, rng=np.random.default_rng(0))
    assert [len(c) for c in a] == [10, 7, 4]
    assert a == b  # same seed -> identical draw
    flat = [v for cell in a for v in cell]
    assert all(0.0 <= v <= 1.0 for v in flat)
    assert all(round(v * 10) == pytest.approx(v * 10) for v in flat)  # multiples of 1/k


def test_simulate_cells_high_rho_produces_more_between_spread():
    rng_lo, rng_hi = np.random.default_rng(1), np.random.default_rng(1)
    mus = np.full(200, 0.5)
    sizes = np.full(200, 10)
    lo = simulate_cells(mus, sizes, np.full(200, 0.01), k=10, rng=rng_lo)
    hi = simulate_cells(mus, sizes, np.full(200, 0.8), k=10, rng=rng_hi)

    def spread(cells: list[list[float]]) -> float:
        return float(np.mean([np.var(c) for c in cells]))

    assert spread(hi) > 2 * spread(lo)


def test_paired_mean_delta_simple():
    vals = np.array([0.1, 0.4, 0.2, 0.2])
    qids = np.array(["a", "a", "b", "b"])
    lvls = np.array([0, 1, 0, 1])
    assert paired_mean_delta(vals, qids, lvls) == pytest.approx(0.15)


def test_rubin_paired_test_detects_a_shift_and_accepts_a_null():
    rng = np.random.default_rng(7)
    n_q, m = 80, 50
    qids = np.repeat([f"q{i}" for i in range(n_q)], 2)
    lvls = np.tile([0, 1], n_q)
    base = rng.normal(0.3, 0.05, size=2 * n_q)
    draws_null = base[None, :] + rng.normal(0, 0.01, size=(m, 2 * n_q))
    res_null = rubin_paired_test(draws_null, qids, lvls)
    assert res_null["p"] > 0.05
    shifted = base.copy()
    shifted[lvls == 1] += 0.1
    draws_eff = shifted[None, :] + rng.normal(0, 0.01, size=(m, 2 * n_q))
    res_eff = rubin_paired_test(draws_eff, qids, lvls)
    assert res_eff["p"] < 0.01
    assert res_eff["delta"] == pytest.approx(0.1, abs=0.02)
    assert res_eff["ci_lo"] < res_eff["delta"] < res_eff["ci_hi"]
