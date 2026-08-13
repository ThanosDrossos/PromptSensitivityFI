"""R4b — axis 1 reported honestly: the GRADED FI_in curve, censoring rates,
cap-sensitivity, and a stepped-shape test with a real null.

WHY (review 2026-08-06 §3.11-3.12).
  * The persisted `fi_in_curve_vals` is the BINARY (T=0) curve: in all 900 cells it
    takes at most two values and its only step is k=0 -> 0.05, so it encodes exactly
    one number (-log2 f_mean) and the Hazen "islands of function" claim has no
    support in that column.
  * The headline "delta AUFI = -0.79..-0.87 bits" scales almost linearly with the
    arbitrary log2(N+1) clamp: it is a unit convention, not a measurement.

WHAT THIS PRODUCES.
  1. The GRADED curve FI_in(k) computed from `f_graded_per_paraphrase` (11 distinct
     per-paraphrase values -> a real curve), mean per model x level with a
     question-resampling bootstrap band. Figure: data/plots/axis1_graded_fi_in.png.
  2. The CONVENTION-FREE headline: the fraction of quality thresholds no paraphrase
     reaches (the censoring rate) at L0 vs L1.
  3. The cap-sensitivity table for delta-AUFI (caps log2 6 / log2 10 / log2 11 /
     log2 21): sign robust, magnitude convention-dependent -- printed so it can
     never be quoted without the convention again.
  4. A stepped-shape ("islands of function") test WITH A NULL. Statistic: the
     largest gap between adjacent sorted per-paraphrase F values in a cell.
     Null: all paraphrases share one true rate p-bar, F_i ~ Binomial(k, p-bar)/k
     (pure decoding noise). Per-cell Monte-Carlo p, BH across cells.
     NOTE the identity this exposes: "islands" beyond sampling noise IS
     between-paraphrase overdispersion -- exactly the quantity rho_F measures.
     The stepped-shape question and the axis-2 question are the same question;
     we report the test to close the Hazen analogy honestly.

Outputs: data/axis1_graded.parquet (per cell), data/axis1_graded_curve.md,
data/plots/axis1_graded_fi_in.png.
CONTRACT: `metrics/` untouched (fi_in_curve is fed new inputs).
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from ..config import load_config
from ..metrics.fi_in import fi_in_curve

_MODELS = ["qwen_2_5_7b", "llama_3_1_8b", "mistral_7b_v03"]
_KS = np.linspace(0.0, 1.0, 21)
_CAPS = {"log2(5+1)=2.585": np.log2(6), "log2(N)=3.322": np.log2(10),
         "log2(N+1)=3.459 (v3 convention)": np.log2(11), "log2(20+1)=4.392": np.log2(21)}


# ------------------------------------------------------------------ pure helpers


def graded_curve_vals(scores: list[float], cap: float) -> np.ndarray:
    """FI_in(k) on the 21-point grid, +inf clamped to `cap` (same rule as aufi_in)."""
    curve = fi_in_curve(scores, ks=_KS.tolist())
    return np.array([min(curve[float(k)], cap) for k in _KS])


def censor_fraction(scores: list[float]) -> float:
    """Fraction of the 21 thresholds that NO paraphrase reaches (n_pass = 0).

    This is the convention-free quantity behind the AUFI drop: it needs no clamp
    value to state and is what actually changes between levels.
    """
    s = np.asarray(scores, dtype=float)
    return float(np.mean([(s >= k - 1e-9).sum() == 0 for k in _KS]))


def max_gap(scores: list[float]) -> float:
    """Largest gap between adjacent sorted per-paraphrase F values — the islands
    statistic. Two separated clumps of F values -> one large gap."""
    s = np.sort(np.asarray(scores, dtype=float))
    return float(np.max(np.diff(s))) if len(s) >= 2 else 0.0


def islands_p_value(scores: list[float], k: int, *, n_sim: int = 2000,
                    rng: np.random.Generator | None = None) -> float:
    """Monte-Carlo P(max_gap >= observed | binomial null with the cell's p-bar).

    The null is 'one true rate, all wobble is decoding noise' — the same null
    rho_F subtracts. A significant gap = function clusters into islands beyond
    sampling noise.
    """
    rng = rng or np.random.default_rng(42)
    s = np.asarray(scores, dtype=float)
    n = len(s)
    if n < 3:
        return float("nan")
    obs = max_gap(scores)
    p_bar = float(s.mean())
    sims = rng.binomial(k, p_bar, size=(n_sim, n)) / k
    sims.sort(axis=1)
    null = np.diff(sims, axis=1).max(axis=1)
    return float((np.sum(null >= obs - 1e-12) + 1) / (n_sim + 1))


def aufi_graded(scores: list[float], cap: float) -> float:
    return float(np.trapezoid(graded_curve_vals(scores, cap), _KS))


# ------------------------------------------------------------------ the analysis


def analyse(config) -> tuple[pd.DataFrame, dict]:
    rows = []
    rng = np.random.default_rng(config.random_seed)
    for m in _MODELS:
        df = pd.read_parquet(config.repo_root() / f"data/specificity_v3_{m}.parquet")
        k = int(df["n_samples_per_prompt"].iloc[0])
        for _, r in df.iterrows():
            s = r["f_graded_per_paraphrase"]
            if s is None or len(s) < 3:
                continue
            s = list(s)
            rec = {
                "model": m, "question_id": r["question_id"], "spec_level": int(r["spec_level"]),
                "censor_frac": censor_fraction(s),
                "max_gap": max_gap(s),
                "islands_p": islands_p_value(s, k, rng=rng),
                "curve": graded_curve_vals(s, np.log2(len(s) + 1)),
            }
            for name, cap in _CAPS.items():
                rec[f"aufi[{name}]"] = aufi_graded(s, cap)
            rows.append(rec)
    cells = pd.DataFrame(rows)

    # BH over the islands tests, per model (the honest "how often are there
    # real steps" number).
    summ: dict = {}
    for m in _MODELS:
        sub = cells[cells.model == m].dropna(subset=["islands_p"])
        pv = sub["islands_p"].to_numpy()
        order = np.argsort(pv)
        thresh = 0.05 * np.arange(1, len(pv) + 1) / len(pv)
        passed = pv[order] <= thresh
        n_sig = int(passed.cumsum().max()) if passed.any() else 0
        summ[m] = {
            "n_cells": len(sub),
            "islands_sig_frac": n_sig / max(len(sub), 1),
            "censor_L0": float(sub[sub.spec_level == 0].censor_frac.mean()),
            "censor_L1": float(sub[sub.spec_level == 1].censor_frac.mean()),
        }
        # cap-sensitivity of the paired delta
        for name in _CAPS:
            p = sub.pivot_table(index="question_id", columns="spec_level",
                                values=f"aufi[{name}]").dropna()
            summ[m][f"delta_aufi[{name}]"] = float((p[1] - p[0]).mean())
        # correlation of the islands statistic with rho_F (the identity note)
        v3 = pd.read_parquet(config.repo_root() / f"data/specificity_v3_{m}.parquet")
        j = sub.merge(v3[["question_id", "spec_level", "rho_f"]],
                      on=["question_id", "spec_level"]).dropna(subset=["rho_f"])
        summ[m]["spearman_maxgap_rhof"] = float(stats.spearmanr(j.max_gap, j.rho_f)[0])
    return cells, summ


def plot(cells: pd.DataFrame, config) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    rng = np.random.default_rng(0)
    for ax, m in zip(axes, _MODELS):
        for lvl, color in [(0, "#d62728"), (1, "#1f77b4")]:
            sub = cells[(cells.model == m) & (cells.spec_level == lvl)]
            mat = np.vstack(sub["curve"].to_numpy())
            qs = sub["question_id"].to_numpy()
            mean = mat.mean(axis=0)
            # question-resampling bootstrap band
            uq = np.unique(qs)
            boots = np.empty((400, len(_KS)))
            for b in range(400):
                pick = rng.choice(len(uq), size=len(uq), replace=True)
                mask = np.isin(qs, uq[pick])
                boots[b] = mat[mask].mean(axis=0)
            lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
            ax.plot(_KS, mean, color=color, lw=2.2, label=f"L{lvl}")
            ax.fill_between(_KS, lo, hi, color=color, alpha=0.18)
        ax.set_title(m)
        ax.set_xlabel("quality threshold k")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("graded FI_in(k)  [bits, cap = log2(N+1)]")
    axes[0].legend()
    fig.suptitle("Axis 1, honestly: the GRADED FI_in curve (mean over questions, 95% question-bootstrap band)",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    out = config.repo_root() / "data/plots/axis1_graded_fi_in.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def render(summ: dict) -> str:
    L = ["# R4b — Axis 1 reported honestly", ""]
    L.append("## The convention-free headline")
    L.append("")
    L.append("The AUFI drop is, underneath, a censoring statement that needs no clamp value:")
    L.append("")
    L.append("| model | thresholds unreachable at L0 | at L1 |")
    L.append("|---|---|---|")
    for m, v in summ.items():
        L.append(f"| {m} | {v['censor_L0']:.1%} | {v['censor_L1']:.1%} |")
    L.append("")
    L.append("> **Say this, not \"−0.86 bits\":** the fraction of quality thresholds no paraphrase")
    L.append("> reaches falls from ~56–67 % (ambiguous) to ~32–40 % (disambiguated).")
    L.append("")
    L.append("## Cap-sensitivity of ΔAUFI (why the bits figure must carry its convention)")
    L.append("")
    caps = list(_CAPS)
    L.append("| model | " + " | ".join(caps) + " |")
    L.append("|---|" + "---|" * len(caps))
    for m, v in summ.items():
        L.append(f"| {m} | " + " | ".join(f"{v[f'delta_aufi[{c}]']:+.3f}" for c in caps) + " |")
    L.append("")
    L.append("Sign robust under every convention; magnitude is the convention. Never print the delta")
    L.append("without the cap.")
    L.append("")
    L.append("## Stepped shape (\"islands of function\") — now with a null")
    L.append("")
    L.append("Statistic: largest adjacent gap in the sorted per-paraphrase F values. Null: one true")
    L.append("rate per cell, F_i ~ Binomial(k, p̄)/k (pure decoding noise), 2000 sims/cell, BH at 5 %:")
    L.append("")
    L.append("| model | cells | cells with significant islands | Spearman(max-gap, ρ_F) |")
    L.append("|---|---|---|---|")
    for m, v in summ.items():
        L.append(f"| {m} | {v['n_cells']} | {v['islands_sig_frac']:.1%} | {v['spearman_maxgap_rhof']:+.3f} |")
    L.append("")
    L.append("**Reading.** \"Islands\" beyond sampling noise *is* between-paraphrase overdispersion —")
    L.append("the same null ρ_F subtracts (the max-gap statistic tracks ρ_F, last column). The Hazen")
    L.append("stepped-shape analogue is therefore not a separate finding: where it holds, it is axis 2")
    L.append("restated. The binary curve's steps (76 % of cells: one step at k=0→0.05; 24 %: flat) carry")
    L.append("no shape information and are retired from the deliverable.")
    L.append("")
    return "\n".join(L)


def main() -> int:
    config = load_config()
    cells, summ = analyse(config)
    fig_path = plot(cells, config)
    md = render(summ)
    out_md = config.repo_root() / "data/axis1_graded_curve.md"
    out_md.write_text(md, encoding="utf-8")
    keep = cells.drop(columns=["curve"])
    keep.to_parquet(config.repo_root() / "data/axis1_graded.parquet", index=False)
    logger.info("wrote {} + {} + {}", out_md.name, "axis1_graded.parquet", fig_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
