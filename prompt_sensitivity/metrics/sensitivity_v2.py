"""Sensitivity metrics v2 (METRIC_PROPOSALS.md M1+M2) — beyond AUFI.

Motivation: AUFI_in is a monotone transform of accuracy (Spearman −1.000 exact
under binary F, −0.999 graded; v3 audit 2026-07-24). These two scalars measure
formulation-sensitivity WITHOUT re-measuring ability.

M1  rho_F — functional ICC ("formulation share of functional variance").
    One-way random-effects ICC(1) on the N×k correctness outcomes, groups =
    paraphrases. With p̂_i the per-paraphrase success rate over k samples
    (exactly what f_graded_per_paraphrase stores) and p̄ their mean:

        SS_between = k · Σ_i (p̂_i − p̄)²             df_b = N − 1
        SS_within  = Σ_i k · p̂_i (1 − p̂_i)          df_w = N (k − 1)
            [exact for binary outcomes: Σ_j (y_ij − p̂_i)² = k p̂_i (1 − p̂_i)]
        ICC(1) = (MSB − MSW) / (MSB + (k − 1) · MSW)

    Reading: fraction of success variability attributable to PHRASING CHOICE
    rather than decoding noise. 0 = rephrasing irrelevant; 1 = success fully
    determined by which paraphrase was picked. Negative estimates clamp to 0
    (standard ICC practice); a cell with NO variance anywhere (all-0/all-1)
    returns NaN — sensitivity is unmeasurable there, not zero, and coverage
    must be reported alongside.

    Placement: the F-space analogue of Cox's rho_u (same ANOVA, embeddings →
    correctness); the noise-corrected version of raw performance dispersion
    across variants (ProSA's PSS et al. conflate formulation variance with
    finite-k sampling noise).

M2  fi_premium — reliability premium ΔFI(k_lo → k_hi).
    Two-threshold Szostak contrast on the graded FI_in curve (§7.3.2):

        ΔFI = FI_in(k_hi) − FI_in(k_lo) = log₂( N_{F ≥ k_lo} / N_{F ≥ k_hi} )

    with each FI clamped at log₂(N+1) as everywhere else. Defaults k_lo=0.5
    ("usable"), k_hi=1.0 ("perfect"): the extra bits of phrasing-rarity that
    perfect reliability demands over usable reliability. Identically 0 under
    binary F — it isolates exactly the information the graded track added.
    Floor cells (no paraphrase reaches k_lo) return 0: the curve has no shape
    information there.

Both are computable from persisted `f_graded_per_paraphrase` — historic
parquets are backfillable without a cluster re-run
(scripts/backfill_sensitivity_v2.py).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

_EPS = 1e-9   # F values are means of k binaries; guard threshold comparisons


def rho_f(scores: Sequence[float], k: int) -> float:
    """M1: functional ICC(1) from per-paraphrase success rates over k samples."""
    if k < 2:
        raise ValueError("rho_f needs k >= 2 samples per paraphrase")
    p = [float(s) for s in scores]
    n = len(p)
    if n < 2:
        return math.nan
    mean = sum(p) / n
    ssb = k * sum((x - mean) ** 2 for x in p)
    ssw = sum(k * x * (1.0 - x) for x in p)
    msb = ssb / (n - 1)
    msw = ssw / (n * (k - 1))
    denom = msb + (k - 1) * msw
    if denom <= 0.0:
        return math.nan            # all-0 / all-1 cell: nothing to attribute
    return min(1.0, max(0.0, (msb - msw) / denom))


def _fi_at(scores: Sequence[float], k: float) -> float:
    n = len(scores)
    n_pass = sum(1 for s in scores if s >= k - _EPS)
    cap = math.log2(n + 1)
    return cap if n_pass == 0 else min(cap, -math.log2(n_pass / n))


def fi_premium(
    scores: Sequence[float], *, k_lo: float = 0.5, k_hi: float = 1.0
) -> float:
    """M2: ΔFI(k_lo → k_hi), the reliability premium in bits (>= 0)."""
    if not scores:
        return math.nan
    if not k_lo < k_hi:
        raise ValueError("need k_lo < k_hi")
    return max(0.0, _fi_at(scores, k_hi) - _fi_at(scores, k_lo))


def compute_row_metrics(
    f_graded: Sequence[float] | None, k_samples: int
) -> dict[str, float]:
    """Single shared entry point for the run driver AND the backfill script,
    so both paths stay identical by construction. NaNs where undefined."""
    if not f_graded:
        return {"rho_f": math.nan, "fi_premium": math.nan}
    return {
        "rho_f": rho_f(f_graded, k_samples),
        "fi_premium": fi_premium(f_graded),
    }
