"""Hazen stepped-behavior test on FI_in(k) curves. Section_7 §7.9 C2.

Hazen's "islands of function" predicts that functional-information curves rise in
PLATEAUS separated by STEPS rather than smoothly. We quantify this on the
persisted FI_in(k) curve (P0-3) + its bootstrap CI (P1-3):

  - a k->k+1 transition is a STEP iff the two bins' CIs do NOT overlap AND the
    value jump exceeds `jump_bits` (default 0.1 bit);
  - otherwise it is FLAT;
  - plateaus = the contiguous runs of bins separated by steps (= n_steps + 1);
  - a cell "fits Hazen" iff it has >= 2 plateaus separated by >= 1 step.

This module reads inputs already on the MetricTuple; it does NOT recompute FI_in.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def detect_steps(
    curve_ks: Sequence[float],
    curve_vals: Sequence[float],
    ci_lower: Sequence[float],
    ci_upper: Sequence[float],
    *,
    jump_bits: float = 0.1,
) -> dict[str, Any]:
    """Classify FI_in(k) bin transitions into flat vs step; summarise the pattern."""
    vals = list(curve_vals)
    n = len(vals)
    if n < 2:
        return {
            "n_plateaus": 1 if n == 1 else 0, "n_steps": 0, "step_locations": [],
            "monotone_decreasing": True, "fits_hazen_pattern": False,
        }
    lo, hi = list(ci_lower), list(ci_upper)
    is_step: list[bool] = []
    for i in range(n - 1):
        # CIs overlap unless one lies entirely above the other.
        disjoint = (hi[i] < lo[i + 1]) or (hi[i + 1] < lo[i])
        jump = abs(vals[i + 1] - vals[i]) > jump_bits
        is_step.append(disjoint and jump)
    n_steps = sum(is_step)
    step_locations = [i for i, s in enumerate(is_step) if s]
    n_plateaus = n_steps + 1            # steps partition the bins into n_steps+1 runs
    monotone_decreasing = all(vals[i + 1] <= vals[i] + 1e-9 for i in range(n - 1))
    fits = n_plateaus >= 2 and n_steps >= 1
    return {
        "n_plateaus": n_plateaus, "n_steps": n_steps, "step_locations": step_locations,
        "monotone_decreasing": monotone_decreasing, "fits_hazen_pattern": fits,
    }


def fits_hazen_row(row: Mapping[str, Any], *, jump_bits: float = 0.1) -> bool | None:
    """Apply detect_steps to one parquet row; None if the curve/CI columns are absent."""
    ks = row.get("fi_in_curve_ks")
    vals = row.get("fi_in_curve_vals")
    lo = row.get("fi_in_ci_lower")
    hi = row.get("fi_in_ci_upper")
    if ks is None or vals is None or lo is None or hi is None or len(vals) < 2:
        return None
    return detect_steps(ks, vals, lo, hi, jump_bits=jump_bits)["fits_hazen_pattern"]


def hazen_fraction_by_model(df, *, jump_bits: float = 0.1) -> dict[str, float]:
    """Per-model fraction of cells that fit the Hazen plateau pattern."""
    out: dict[str, float] = {}
    if "model_key" not in df.columns:
        return out
    for model, sub in df.groupby("model_key"):
        fits = [fits_hazen_row(r, jump_bits=jump_bits) for _, r in sub.iterrows()]
        fits = [f for f in fits if f is not None]
        out[str(model)] = (sum(fits) / len(fits)) if fits else float("nan")
    return out
