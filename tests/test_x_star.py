"""P2-3: x* selection + distance-rho pure core."""

from __future__ import annotations

import numpy as np

from prompt_sensitivity.analysis.x_star import aggregate, cell_x_star_rho, pick_x_star


def test_pick_x_star_tiebreak_shortest_then_lex():
    assert pick_x_star([1.0, 1.0, 0.5], ["bb", "a", "ccc"]) == 1   # F tie -> shortest "a"
    assert pick_x_star([1.0, 1.0], ["b", "a"]) == 1                # same len -> lex "a"


def test_cell_rho_negative_when_farther_is_worse():
    f = [1.0, 0.8, 0.2]
    emb = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([3.0, 0.0])]
    rho = cell_x_star_rho(f, emb, ["x", "y", "z"])
    assert rho is not None and rho < 0


def test_cell_rho_none_when_degenerate():
    assert cell_x_star_rho([1.0, 1.0], [np.zeros(2), np.zeros(2)], ["a", "b"]) is None  # n<3
    f = [1.0, 0.5, 0.5]  # no F variance among the non-x* paraphrases
    emb = [np.zeros(2), np.array([1.0, 0.0]), np.array([2.0, 0.0])]
    assert cell_x_star_rho(f, emb, ["a", "b", "c"]) is None


def test_aggregate_overall_mean_ignores_none():
    per = [
        {"model_key": "m", "ladder_family": "context", "level": 0, "rho": -0.5},
        {"model_key": "m", "ladder_family": "context", "level": 0, "rho": -0.3},
        {"model_key": "m", "ladder_family": "context", "level": 0, "rho": None},
    ]
    agg = aggregate(per)
    assert abs(agg["overall_mean_rho"] - (-0.4)) < 1e-9
    assert agg["n_cells"] == 2
