"""R2/R3: the hierarchical rho_F estimator and the independence-analysis helpers.

The properties that matter are (i) it recovers a known rho, (ii) it is defined on
the degenerate cells the MoM estimator drops, (iii) it does NOT score those
degenerate cells as maximally phrasing-sensitive (the flat-mu-prior artifact), and
(iv) the equivalence machinery reports what the data support rather than a point
estimate.
"""

from __future__ import annotations

import numpy as np
import pytest

from prompt_sensitivity.analysis.rho_f_hierarchical import (
    fit_hierarchical_rho_f,
    sigma2_between,
)
from prompt_sensitivity.metrics.sensitivity_v2 import rho_f as rho_f_mom
from prompt_sensitivity.scripts.independence_analysis import (
    dispersion_factor,
    pooled_correlation_mi,
    tost_equivalence,
)

K = 10
N = 10


def _simulate(true_rho: float, n_cells: int, rng, mu_lo=0.2, mu_hi=0.8):
    cells = []
    for _ in range(n_cells):
        mu = rng.uniform(mu_lo, mu_hi)
        if true_rho <= 1e-9:
            p = np.full(N, mu)
        else:
            s = (1 - true_rho) / true_rho
            p = rng.beta(mu * s, (1 - mu) * s, size=N)
        cells.append((rng.binomial(K, p) / K).tolist())
    return cells


# ------------------------------------------------------------------ recovery


@pytest.mark.parametrize("true_rho", [0.05, 0.2, 0.5])
def test_recovers_known_rho(true_rho):
    rng = np.random.default_rng(0)
    fit = fit_hierarchical_rho_f(_simulate(true_rho, 250, rng), K)
    assert abs(fit.rho_mean.mean() - true_rho) < 0.06


def test_defined_on_degenerate_cells_where_mom_is_nan():
    """The coverage fix: MoM returns NaN, the hierarchical posterior does not."""
    rng = np.random.default_rng(1)
    cells = _simulate(0.2, 100, rng) + [[0.0] * N] * 40 + [[1.0] * N] * 10
    assert np.isnan(rho_f_mom([0.0] * N, K))
    assert np.isnan(rho_f_mom([1.0] * N, K))
    fit = fit_hierarchical_rho_f(cells, K)
    assert np.all(np.isfinite(fit.rho_mean)), "must be defined for every cell"
    assert np.all((fit.rho_mean >= 0) & (fit.rho_mean <= 1))


def test_degenerate_cells_are_not_scored_as_maximally_sensitive():
    """Guards the hierarchical-mu-prior fix.

    Under a FLAT mu prior an all-wrong cell is ~9x more likely at rho=1 than at
    rho=0 (1/(N+1) vs 1/(Nk+1)), which would score uninformative cells as the MOST
    phrasing-sensitive in the dataset. With mu's prior fitted to the data they
    must instead fall back near the informative cells' level.
    """
    rng = np.random.default_rng(2)
    informative = _simulate(0.2, 200, rng)
    degenerate = [[0.0] * N] * 100
    fit = fit_hierarchical_rho_f(informative + degenerate, K)
    inf_mean = fit.rho_mean[:200].mean()
    deg_mean = fit.rho_mean[200:].mean()
    assert deg_mean < inf_mean + 0.15, (
        f"degenerate cells inflated: {deg_mean:.3f} vs informative {inf_mean:.3f}")
    # and their uncertainty must be honest (wide), not falsely precise
    assert fit.rho_sd[200:].mean() > 0.05


def test_agrees_with_mom_on_informative_cells():
    from scipy import stats
    rng = np.random.default_rng(3)
    cells = _simulate(0.3, 250, rng)
    fit = fit_hierarchical_rho_f(cells, K)
    mom = np.array([rho_f_mom(c, K) for c in cells])
    ok = np.isfinite(mom)
    r = stats.spearmanr(fit.rho_mean[ok], mom[ok])[0]
    assert r > 0.5, f"should track the MoM estimator on cells where it exists (got {r:.2f})"


def test_empty_cell_gets_prior_not_a_crash():
    rng = np.random.default_rng(4)
    cells = _simulate(0.2, 60, rng) + [None, []]
    fit = fit_hierarchical_rho_f(cells, K)
    assert np.all(np.isfinite(fit.rho_mean))


def test_posterior_draws_have_right_shape_and_support():
    rng = np.random.default_rng(5)
    fit = fit_hierarchical_rho_f(_simulate(0.2, 40, rng), K)
    draws = fit.sample(50, seed=7)
    assert draws.shape == (50, 40)
    assert draws.min() >= 0.0 and draws.max() <= 1.0
    # draws must vary (otherwise multiple imputation is pointless)
    assert draws.std() > 0.01


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        fit_hierarchical_rho_f([[0.5] * N], 1)
    with pytest.raises(ValueError):
        fit_hierarchical_rho_f([[1.5] * N], K)


# ------------------------------------------------------------ sigma2_between


def test_sigma2_between_is_zero_without_phrasing_effect_and_positive_with_one():
    assert sigma2_between([0.5] * N, K) == 0.0
    spread = sigma2_between([0.0] * 5 + [1.0] * 5, K)
    assert spread > 0.2
    assert np.isnan(sigma2_between([0.5], K))


def test_sigma2_between_is_not_deflated_by_decoding_noise_the_way_the_share_is():
    """The point of reporting it: a noisier model gets a smaller rho_F share for
    the same absolute phrasing effect, but sigma2_B is unaffected."""
    quiet = [0.0, 0.0, 1.0, 1.0] * 2 + [0.0, 1.0]          # deterministic paraphrases
    noisy = [0.2, 0.2, 0.8, 0.8] * 2 + [0.2, 0.8]          # same spread, more within-noise
    assert rho_f_mom(quiet, K) > rho_f_mom(noisy, K)
    assert sigma2_between(quiet, K) > sigma2_between(noisy, K) > 0


# ------------------------------------------------------------ R3 helpers


def test_tost_reports_what_the_data_support():
    tight = tost_equivalence(0.0, 5000, bound=0.2)
    assert tight["equivalent"] is True
    assert tight["smallest_supported"] < 0.1

    thin = tost_equivalence(0.0, 40, bound=0.2)
    assert thin["equivalent"] is False, "n=40 cannot establish |rho| < 0.2"
    assert thin["smallest_supported"] > 0.2

    big = tost_equivalence(0.6, 500, bound=0.2)
    assert big["equivalent"] is False


def test_pooled_correlation_mi_widens_ci_relative_to_a_single_draw():
    rng = np.random.default_rng(11)
    n = 150
    y = rng.normal(size=n)
    # draws that all agree -> narrow; draws that disagree -> wide
    agree = np.tile(y + rng.normal(scale=0.1, size=n), (60, 1))
    disagree = np.stack([rng.normal(size=n) for _ in range(60)])
    a = pooled_correlation_mi(agree, y)
    d = pooled_correlation_mi(disagree, y)
    assert (a["ci_hi"] - a["ci_lo"]) < (d["ci_hi"] - d["ci_lo"])
    assert abs(d["r"]) < 0.3


def test_dispersion_factor_is_oriented_with_h_sem():
    import pandas as pd
    rng = np.random.default_rng(12)
    h = rng.normal(size=200)
    df = pd.DataFrame({
        "h_sem_mean": h,
        "s_tau_mean": h + rng.normal(scale=0.2, size=200),
        "variation_ratio": h + rng.normal(scale=0.3, size=200),
        "fi_out_var": h + rng.normal(scale=0.3, size=200),
        "a_q": h + rng.normal(scale=0.3, size=200),
        "consistency_mean": -h + rng.normal(scale=0.3, size=200),
    })
    from scipy import stats
    pc1 = dispersion_factor(df)
    assert stats.spearmanr(pc1, df.h_sem_mean)[0] > 0.7
