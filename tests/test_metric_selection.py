"""Metric selection: the supplementary projection and the retention rule must be exact.

What matters: (a) projecting an active variable returns its own rotated loading, so held-out
loadings live on the same scale as active ones; (b) the projection equals the empirical
correlation with the component scores; (c) an isolated variable can never be retained by
parallel analysis, which is why a factor needs several indicators; (d) the derived extremes
bound the mean; (e) the committed artifact reflects the two-stage result the paper quotes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prompt_sensitivity.scripts.metric_selection import (
    CONSTRUCTED,
    FAMILY_ORDER,
    PUBLISHED,
    RHO_F,
    component_families,
    component_titles,
    decompose,
    derive_extremes,
    project,
)

REPO = Path(__file__).resolve().parents[1]


def _factor_data(seed: int, p: int = 7, n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(n, 3))
    weights = rng.normal(size=(3, p)) * (rng.random((3, p)) > 0.4)
    X = factors @ weights + 0.6 * rng.normal(size=(n, p))
    return np.corrcoef(X, rowvar=False), X


def _names(p: int) -> list[str]:
    return [f"v{i}" for i in range(p)]


def test_project_returns_the_active_loading_for_an_active_variable():
    C, _ = _factor_data(0)
    dec = decompose(C, _names(C.shape[0]), n_obs=400, k=3)
    for i in range(C.shape[0]):
        assert np.allclose(project(dec, C[i]), dec.loadings[i], atol=1e-10)


def test_rotation_is_orthogonal_and_reproduces_the_rotated_loadings():
    C, _ = _factor_data(1)
    dec = decompose(C, _names(C.shape[0]), n_obs=400, k=3)
    assert np.allclose(dec.rotation.T @ dec.rotation, np.eye(3), atol=1e-8)
    unrotated = dec.evecs[:, :3] * np.sqrt(dec.evals[:3])
    assert np.allclose(unrotated @ dec.rotation, dec.loadings, atol=1e-8)
    for j in range(3):  # sign convention: the largest |loading| of every component is positive
        assert dec.loadings[np.abs(dec.loadings[:, j]).argmax(), j] > 0


def test_projection_equals_empirical_correlation_with_component_scores():
    C, X = _factor_data(2, p=8)
    active, held = list(range(7)), 7
    dec = decompose(C[np.ix_(active, active)], _names(7), n_obs=400, k=3)
    Z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    scores = Z[:, active] @ ((dec.evecs[:, :3] / np.sqrt(dec.evals[:3])) @ dec.rotation)
    empirical = [np.corrcoef(Z[:, held], scores[:, j])[0, 1] for j in range(3)]
    assert np.allclose(empirical, project(dec, C[held, active]), atol=1e-8)


@pytest.mark.parametrize("n_obs", [50, 150, 5000])
def test_an_isolated_variable_is_never_retained(n_obs: int):
    # one dense block of five plus one variable uncorrelated with everything:
    # its eigenvalue is exactly 1, and every parallel threshold above rank 1 exceeds 1.
    C = np.full((6, 6), 0.8)
    np.fill_diagonal(C, 1.0)
    C[5, :5] = C[:5, 5] = 0.0
    dec = decompose(C, _names(6), n_obs=n_obs, k=2)
    assert dec.evals[1] == pytest.approx(1.0)
    assert dec.n_retained == 1


def test_component_families_and_titles_on_a_clean_three_block_structure():
    C = np.eye(9)
    for lo, hi in ((0, 3), (3, 6), (6, 9)):
        C[lo:hi, lo:hi] = 0.7
    np.fill_diagonal(C, 1.0)
    names = _names(9)
    family_of = {n: FAMILY_ORDER[i // 3] for i, n in enumerate(names)}
    dec = decompose(C, names, n_obs=200, k=3)
    assert sorted(component_families(dec, family_of)) == sorted(FAMILY_ORDER)
    assert not any(t.startswith("mixed") for t in component_titles(dec, family_of))


def test_derive_extremes_bounds_the_graded_mean():
    df = pd.DataFrame(
        {
            "f_graded_per_paraphrase": [
                np.array([0.2, 0.9, 0.5]),
                np.array([0.4]),
                np.array([1.0, 0.0]),
            ],
            "f_graded_mean": [1.6 / 3, 0.4, 0.5],
        }
    )
    out = derive_extremes(df)
    assert (out["f_max"] >= out["f_graded_mean"] - 1e-12).all()
    assert (out["f_min"] <= out["f_graded_mean"] + 1e-12).all()
    assert out.loc[1, "f_max"] == out.loc[1, "f_min"] == 0.4  # a single formulation


def test_published_and_constructed_sets_extend_the_committed_matrix():
    meta = json.loads((REPO / "figures/v3_metric_corr_labels.json").read_text(encoding="utf-8"))
    pub = {lab for _, lab, _, _ in PUBLISHED}
    con = {lab for _, lab, _ in CONSTRUCTED}
    assert not pub & con
    assert len(pub) == 10 and len(con) == 6
    derived = {"F_max (best formulation)", "F_min (worst formulation)"}
    assert (pub | con) - derived == set(meta["labels"])


def test_committed_artifact_matches_the_two_stage_result():
    """The paper quotes data/metric_selection.json; regenerate it when this fails."""
    art = json.loads((REPO / "data/metric_selection.json").read_text(encoding="utf-8"))
    stage1, stage2 = art["stage1_three_components"], art["stage2_three_components"]
    assert stage1["n_retained"] == 2 and art["stage1_counts_over_n_and_seeds"] == [2]
    assert stage2["n_retained"] == 3 and art["stage2_counts_over_n_and_seeds"] == [3]
    assert sorted(stage1["component_families"]) == sorted(FAMILY_ORDER)
    j_dep = stage1["component_families"].index("dependence")
    assert int(np.abs(stage1["supplementary"][RHO_F]).argmax()) == j_dep
    j_disp = stage1["component_families"].index("dispersion")
    disp = {n: v[j_disp] for n, v in stage1["loadings"].items()}
    assert max(disp, key=disp.get) == "H_sem"
    j_succ = stage1["component_families"].index("success")
    succ = {n: v[j_succ] for n, v in stage1["loadings"].items()}
    assert max(succ, key=succ.get) == "accuracy"
