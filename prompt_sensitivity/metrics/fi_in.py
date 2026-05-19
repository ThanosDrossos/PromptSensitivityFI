"""FI_in — Functional Information in prompt space. PRIMARY contribution.

Direct Szostak-Hazen translation per Section_7 §7.3.1:

    FI_in(q, k) = -log2( N_k(q) / |U_q| )

where N_k(q) = |{x in U_q : F(x) >= k}|. Reference Python in Section_7 §7.6.3
is mirrored below with three additions:

  - `aufi_in` clamps +inf at `log2(N + 1)` before trapezoidal integration,
    so a single-magic-phrasing case (only one paraphrase works) does not
    blow up to infinity in the area summary.
  - `fi_in_bootstrap` returns the per-k percentile-bootstrap confidence band.
  - `fi_in_curve` defaults `ks` to a 21-point linspace; the design doc's
    Tier-A reporting takes the integral over k in [0, 1].

Caller responsibility: provide `scores: list[float]` already aligned with the
paraphrase set U_q,l at one (question, ladder, level) cell. The metric itself
is pure.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np


def fi_in(scores: Sequence[float], k: float) -> float:
    """Hazen-Szostak functional information at threshold k.

    Edge cases (Section_7 §7.3.1 "Reading of the formula"):
      - All paraphrases achieve k       -> FI_in = 0 bits
      - Only one paraphrase achieves k  -> FI_in = log2(|U_q|)
      - No paraphrase achieves k        -> +inf (undefined; caller clamps if needed)
    """
    n = len(scores)
    if n == 0:
        raise ValueError("scores must be non-empty")
    n_pass = sum(1 for s in scores if s >= k)
    if n_pass == 0:
        return math.inf
    return -math.log2(n_pass / n)


def fi_in_curve(
    scores: Sequence[float],
    ks: Sequence[float] | None = None,
) -> dict[float, float]:
    """Curve of FI_in over a grid of thresholds. Returned as {k -> FI_in(k)}."""
    if ks is None:
        ks = np.linspace(0.0, 1.0, 21).tolist()
    return {float(k): fi_in(scores, float(k)) for k in ks}


def aufi_in(curve: dict[float, float], n: int) -> float:
    """Area under the FI_in(k) curve via trapezoid rule.

    `+inf` values are clamped to `log2(n + 1)` (the +1 avoids 0 when n=1)
    so the integral remains finite when some threshold is unreachable for
    every paraphrase. This is the same convention Section_7 §7.6.3 codes up.
    """
    if not curve:
        return 0.0
    ks = sorted(curve.keys())
    cap = math.log2(max(n, 1) + 1)
    vals = [min(curve[k], cap) for k in ks]
    return float(np.trapezoid(vals, ks))


def fi_in_bootstrap(
    scores: Sequence[float],
    ks: Sequence[float] | None = None,
    *,
    n_iterations: int = 1000,
    confidence: float = 0.95,
    seed: int | None = 42,
) -> dict[float, tuple[float, float]]:
    """Percentile bootstrap CI per threshold k, computed over paraphrase resamples.

    Returns `{k -> (lower, upper)}` for the requested confidence interval.
    """
    if ks is None:
        ks = np.linspace(0.0, 1.0, 21).tolist()
    n = len(scores)
    if n == 0:
        raise ValueError("scores must be non-empty")
    rng = np.random.default_rng(seed)
    scores_arr = np.asarray(scores, dtype=float)

    # iterations × len(ks) matrix of FI_in samples, then percentile across iterations.
    samples = np.empty((n_iterations, len(ks)), dtype=float)
    for b in range(n_iterations):
        resample_idx = rng.integers(0, n, size=n)
        resample = scores_arr[resample_idx]
        for j, k in enumerate(ks):
            n_pass = int((resample >= k).sum())
            samples[b, j] = math.inf if n_pass == 0 else -math.log2(n_pass / n)

    # Clamp infinities for the percentile computation, matching aufi_in's convention.
    cap = math.log2(n + 1)
    samples = np.where(np.isinf(samples), cap, samples)

    alpha = (1.0 - confidence) / 2.0
    lower = np.quantile(samples, alpha, axis=0)
    upper = np.quantile(samples, 1.0 - alpha, axis=0)
    return {float(k): (float(lower[j]), float(upper[j])) for j, k in enumerate(ks)}


# --- convenience for the orchestrator -------------------------------------


def aufi_in_from_scores(scores: Iterable[float], ks: Sequence[float] | None = None) -> float:
    """One-shot helper: curve + integral."""
    scores = list(scores)
    if not scores:
        return 0.0
    curve = fi_in_curve(scores, ks)
    return aufi_in(curve, n=len(scores))
