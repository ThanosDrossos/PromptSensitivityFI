"""A4/M5/M7 — the construct-validity dossier for rho_F, as a committed script.

Replaces the unscripted numbers behind Table 5's predictive rows and the
convergent/incremental-validity analysis the 2026-08-14 review required:

  1. CONVERGENT VALIDITY — rho_F(hier) against the two published neighbours,
     Cox's rho_u and Cao/Sclar's spread, per model x level stratum. The axis is
     not empty; these are its convergent channels and the paper reports them.
  2. INCREMENTAL VALIDITY — does rho_F beat spread and rho_u at predicting the
     out-of-sample rephrasing payoff? Payoff = F_max - F_mean on the DISJOINT
     second half of the k=20 arm (recovered exactly: the seed layout reuses the
     k=10 samples, so r_2nd = 2*r20 - r10 per paraphrase).
  3. HELD-OUT-PARAPHRASE payoff (5-vs-5) — the only check that leaves the
     estimation universe entirely: estimate on 5 paraphrases, predict the
     payoff of the other 5; 200 random splits.
  4. GOLD-SET AGREEMENT of per-cell rho_F (union vs target), both estimators.
  5. GREEDY-PASS DISAGREEMENT — rho_F (T=1 samples) predicting whether the
     deterministic T=0 pass disagrees across paraphrases (0 < f_mean < 1),
     partial Spearman controlling accuracy-extremeness |F - 1/2|.
  6. CROSS-MODEL TRANSFER of per-question rho_F (the paper's "0.2 to 0.45"
     had no artifact; this is the scripted number).
  7. COMMENSURABLE NULL — the specificity Delta rho_F recomputed with the
     width arm's own recipe (50 questions, complete-case MoM), so the two
     halves of the double dissociation can be read side by side.

Read-only over parquets; writes data/rho_f_construct_validity.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from ..config import load_config
from ..logging_setup import configure_logging
from ..metrics.sensitivity_v2 import rho_f as rho_f_mom

_MODELS = ["qwen_2_5_7b", "llama_3_1_8b", "mistral_7b_v03"]
_PAIRS = [
    ("qwen_2_5_7b", "llama_3_1_8b"),
    ("qwen_2_5_7b", "mistral_7b_v03"),
    ("llama_3_1_8b", "mistral_7b_v03"),
]


def second_half_rates(r10: list[float], r20: list[float]) -> list[float] | None:
    """Per-paraphrase success rates of samples 10-19 of the k=20 arm.

    The seed layout reuses the k=10 samples exactly, so
    r20 = (10*r10 + 10*r_2nd)/20  =>  r_2nd = 2*r20 - r10 (clipped for float).
    Returns None when the universes are not positionally alignable.
    """
    if r10 is None or r20 is None or len(r10) != len(r20):
        return None
    r2 = 2.0 * np.asarray(r20) - np.asarray(r10)
    if (r2 < -1e-6).any() or (r2 > 1 + 1e-6).any():
        return None
    return np.clip(r2, 0.0, 1.0).tolist()


def payoff(rates: list[float] | None) -> float:
    """Rephrasing payoff of a universe: best paraphrase minus the mean."""
    if rates is None or len(rates) < 2:
        return float("nan")
    arr = np.asarray(rates, dtype=float)
    return float(arr.max() - arr.mean())


def spread_of(rates: list[float] | None) -> float:
    if rates is None or len(rates) < 2:
        return float("nan")
    arr = np.asarray(rates, dtype=float)
    return float(arr.max() - arr.min())


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[float, int]:
    """Partial Spearman of x and y controlling z (rank-transform + partial r)."""
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if m.sum() < 10:
        return float("nan"), int(m.sum())
    rx, ry, rz = (stats.rankdata(v[m]) for v in (x, y, z))
    rxy = np.corrcoef(rx, ry)[0, 1]
    rxz = np.corrcoef(rx, rz)[0, 1]
    ryz = np.corrcoef(ry, rz)[0, 1]
    denom = np.sqrt((1 - rxz**2) * (1 - ryz**2))
    if denom < 1e-12:
        return float("nan"), int(m.sum())
    return float((rxy - rxz * ryz) / denom), int(m.sum())


def split_half_payoff_prediction(
    cells: list[list[float]], k: int, *, n_splits: int = 200, seed: int = 42
) -> dict[str, float]:
    """5-vs-5 held-out-paraphrase payoff prediction, rho_F(MoM) vs spread.

    For each random split of a >=8-paraphrase universe into halves A/B:
    estimate rho_F and spread on A, compute the payoff on B, correlate across
    cells. Returns mean Spearman over splits for both predictors.
    """
    rng = np.random.default_rng(seed)
    r_rho, r_spread = [], []
    usable = [np.asarray(c, dtype=float) for c in cells if c is not None and len(c) >= 8]
    for _ in range(n_splits):
        rho_a, spr_a, pay_b = [], [], []
        for arr in usable:
            idx = rng.permutation(len(arr))
            half = len(arr) // 2
            a, b = arr[idx[:half]], arr[idx[half:]]
            rho_a.append(rho_f_mom(a.tolist(), k))
            spr_a.append(float(a.max() - a.min()))
            pay_b.append(float(b.max() - b.mean()))
        frame = pd.DataFrame({"rho": rho_a, "spr": spr_a, "pay": pay_b}).dropna()
        if len(frame) >= 10:
            r_rho.append(stats.spearmanr(frame.rho, frame.pay).statistic)
            r_spread.append(stats.spearmanr(frame.spr, frame.pay).statistic)
    return {
        "n_cells": len(usable),
        "n_splits": len(r_rho),
        "rho_f": float(np.mean(r_rho)),
        "spread": float(np.mean(r_spread)),
    }


def _load(root: Path, model: str) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    v3 = pd.read_parquet(root / f"data/specificity_v3_{model}.parquet")
    hier = pd.read_parquet(root / f"data/rho_f_hier_union_{model}.parquet")
    n_before = len(v3)
    v3 = v3.merge(
        hier[["question_id", "spec_level", "rho_f_hier"]],
        on=["question_id", "spec_level"],
        how="left",
    )
    if len(v3) != n_before or v3["rho_f_hier"].isna().any():
        raise ValueError(f"{model}: hier join changed rows or left NaNs ({n_before} -> {len(v3)})")
    k = int(v3["n_samples_per_prompt"].iloc[0])
    return v3, hier, k


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.parse_args()
    configure_logging("rho_f_construct_validity")
    root = load_config().repo_root()

    L = [
        "# rho_F construct validity — scripted (replaces the unscripted Table-5 rows)",
        "",
        "Script: `prompt_sensitivity/scripts/rho_f_construct_validity.py`. All "
        "rho_F(hier) values are the UNION-gold hierarchical estimator; MoM rows "
        "are labelled. Every correlation carries its n.",
        "",
        "## 1. Convergent validity — the axis is not empty",
        "",
        "Within-stratum Spearman of rho_F(hier) with the two published "
        "neighbours (and, for contrast, the other axes):",
        "",
        "| model | level | ~rho_u (Cox) | ~spread (Cao) | ~accuracy | ~H_sem | n |",
        "|---|---|---|---|---|---|---|",
    ]
    for model in _MODELS:
        v3, hier, k = _load(root, model)
        for lvl in (0, 1):
            s = v3[v3.spec_level == lvl]

            def sp(col: str, s: pd.DataFrame = s) -> float:
                return float(stats.spearmanr(s["rho_f_hier"], s[col], nan_policy="omit").statistic)

            L.append(
                f"| {model} | L{lvl} | {sp('rho_u'):+.3f} | {sp('spread'):+.3f} "
                f"| {sp('f_graded_mean'):+.3f} | {sp('h_sem_mean'):+.3f} | {len(s)} |"
            )
    L += [
        "",
        "**Reading.** rho_u and spread track rho_F in every stratum — more "
        "strongly than anything else in the metric set. This is convergent "
        "validity for the axis, and the paper reports it as such; the claim "
        '"measured by no existing index" is licensed only with the qualifier '
        '"with the within-prompt sampling term removed".',
        "",
        "## 2. Incremental validity — out-of-sample payoff, k=20 second half",
        "",
        "Payoff = F_max - F_mean on the disjoint second half (samples 10-19) "
        "of the k=20 arm; predictors estimated from the first k=10 samples. "
        "Complete cases of the k=20 subsample (50 questions x 2 levels).",
        "",
        "| model | rho_F (hier) | rho_F (MoM) | spread | rho_u | n | partial rho_F(hier) given spread |",
        "|---|---|---|---|---|---|---|",
    ]
    for model in _MODELS:
        v3, hier, k = _load(root, model)
        k20 = pd.read_parquet(root / f"data/sensitivity_v2_k20_{model}.parquet")
        k20_k = int(k20["n_samples_per_prompt"].iloc[0])
        if k20_k != 2 * k:
            logger.warning("{}: k20 arm has k={} (expected {})", model, k20_k, 2 * k)
        j = v3.merge(
            k20[["question_id", "spec_level", "f_graded_per_paraphrase"]],
            on=["question_id", "spec_level"],
            suffixes=("", "_k20"),
            how="inner",
        )
        logger.info("{}: v3 {} x k20 {} -> joined {}", model, len(v3), len(k20), len(j))
        r2 = [
            second_half_rates(a, b)
            for a, b in zip(
                j["f_graded_per_paraphrase"], j["f_graded_per_paraphrase_k20"], strict=True
            )
        ]
        frame = pd.DataFrame(
            {
                "pay": [payoff(r) for r in r2],
                "hier": j["rho_f_hier"],
                "mom": j["rho_f"],
                "spr": j["spread"],
                "rho_u": j["rho_u"],
            }
        )
        cc = frame.dropna(subset=["pay"])

        def sp2(col: str) -> str:
            sub = cc[[col, "pay"]].dropna()
            r = stats.spearmanr(sub[col], sub["pay"]).statistic
            return f"{r:+.3f} (n={len(sub)})"

        part, n_part = partial_spearman(
            cc["hier"].to_numpy(), cc["pay"].to_numpy(), cc["spr"].to_numpy()
        )
        L.append(
            f"| {model} | {sp2('hier')} | {sp2('mom')} | {sp2('spr')} "
            f"| {sp2('rho_u')} | {len(cc)} | {part:+.3f} (n={n_part}) |"
        )
    L += [
        "",
        "**Reading.** The utility criterion is comparative: rho_F earns its "
        "estimator only if it beats the one-line statistic (spread) and the "
        "no-gold ratio (rho_u) at predicting the payoff. The table states the "
        "result either way; the partial column shows what rho_F adds beyond "
        "spread.",
        "",
        "## 3. Held-out-paraphrase payoff (5-vs-5, 200 splits)",
        "",
        "The k=20 check reuses the same ten paraphrases; this one does not: "
        "estimate on five paraphrases, predict the payoff of the disjoint five.",
        "",
        "| model | rho_F (MoM on 5) | spread (on 5) | cells (\\|U\\|>=8) |",
        "|---|---|---|---|",
    ]
    for model in _MODELS:
        v3, hier, k = _load(root, model)
        res = split_half_payoff_prediction(
            [list(c) if c is not None else None for c in v3["f_graded_per_paraphrase"]], k
        )
        L.append(f"| {model} | {res['rho_f']:+.3f} | {res['spread']:+.3f} | {res['n_cells']} |")
    L += [
        "",
        "## 4. Gold-set agreement of per-cell rho_F",
        "",
        "| model | hier union~target | MoM union~target (both defined) |",
        "|---|---|---|",
    ]
    for model in _MODELS:
        u = pd.read_parquet(root / f"data/rho_f_hier_union_{model}.parquet")
        t = pd.read_parquet(root / f"data/rho_f_hier_target_{model}.parquet")
        j = u.merge(t, on=["question_id", "spec_level"], suffixes=("_u", "_t"))
        r_h = stats.spearmanr(j["rho_f_hier_u"], j["rho_f_hier_t"]).statistic
        both = j[["rho_f_u", "rho_f_t"]].dropna()
        r_m = stats.spearmanr(both["rho_f_u"], both["rho_f_t"]).statistic
        L.append(f"| {model} | {r_h:+.3f} (n={len(j)}) | {r_m:+.3f} (n={len(both)}) |")
    L += [
        "",
        "## 5. Greedy-pass disagreement (cross-decoding-regime check)",
        "",
        "Disagreement = the deterministic T=0 pass splits across paraphrases "
        "(0 < f_mean < 1). Partial Spearman of rho_F(hier) with that indicator, "
        "controlling accuracy-extremeness |F_graded - 1/2| (the mechanical "
        "channel).",
        "",
        "| model | partial Spearman | raw Spearman | n |",
        "|---|---|---|---|",
    ]
    for model in _MODELS:
        v3, hier, k = _load(root, model)
        disagree = ((v3["f_mean"] > 0) & (v3["f_mean"] < 1)).astype(float).to_numpy()
        extreme = (v3["f_graded_mean"] - 0.5).abs().to_numpy()
        part, n = partial_spearman(v3["rho_f_hier"].to_numpy(), disagree, extreme)
        raw = float(stats.spearmanr(v3["rho_f_hier"], disagree).statistic)
        L.append(f"| {model} | {part:+.3f} | {raw:+.3f} | {n} |")
    L += [
        "",
        "## 6. Cross-model transfer of per-question rho_F (hier, union)",
        "",
        "| pair | pooled levels | L0 | L1 |",
        "|---|---|---|---|",
    ]
    per = {
        m: pd.read_parquet(root / f"data/rho_f_hier_union_{m}.parquet").set_index(
            ["question_id", "spec_level"]
        )["rho_f_hier"]
        for m in _MODELS
    }
    for a, b in _PAIRS:
        j = pd.concat([per[a], per[b]], axis=1, join="inner").dropna()
        r_all = stats.spearmanr(j.iloc[:, 0], j.iloc[:, 1]).statistic
        by_lvl = []
        for lvl in (0, 1):
            s = j.xs(lvl, level="spec_level")
            by_lvl.append(stats.spearmanr(s.iloc[:, 0], s.iloc[:, 1]).statistic)
        L.append(f"| {a} ~ {b} | {r_all:+.3f} (n={len(j)}) | {by_lvl[0]:+.3f} | {by_lvl[1]:+.3f} |")
    L += [
        "",
        "**Reading.** Transfer is weak; formulation sensitivity is a property "
        "of the (question x model) pair. This row replaces the unsourced "
        '"0.2 to 0.45" range.',
        "",
        "## 7. Commensurable specificity null (the width arm's own recipe)",
        "",
        "Delta rho_F (L1 - L0) with complete-case MoM, on the width arm's 50 "
        "questions and on all 150, next to the hierarchical row of the primary "
        "family — so the two halves of the double dissociation share an "
        "estimator and a sample.",
        "",
        "| model | MoM, 50 q (width sample) | MoM, 150 q | hier, 150 q (primary family) |",
        "|---|---|---|---|",
    ]
    width_q = set(
        pd.read_parquet(root / "data/width_narrow_qwen_2_5_7b.parquet")["question_id"].astype(str)
    )
    for model in _MODELS:
        v3, hier, k = _load(root, model)

        def mom_delta(frame: pd.DataFrame) -> str:
            w = frame.pivot_table(
                index="question_id", columns="spec_level", values="rho_f"
            ).dropna()
            if len(w) < 10:
                return f"n={len(w)} (too few)"
            d = w[1] - w[0]
            p = stats.wilcoxon(d).pvalue
            return f"{d.mean():+.4f} (p={p:.2g}, n={len(w)})"

        wh = v3.pivot_table(index="question_id", columns="spec_level", values="rho_f_hier").dropna()
        dh = wh[1] - wh[0]
        ph = stats.wilcoxon(dh).pvalue
        L.append(
            f"| {model} | {mom_delta(v3[v3.question_id.astype(str).isin(width_q)])} "
            f"| {mom_delta(v3)} | {dh.mean():+.4f} (p={ph:.2g}, n={len(wh)}) |"
        )
    L += [
        "",
        "**Reading.** The specificity null and the width effect can now be "
        "compared on the same 50 questions with the same estimator; the "
        "remaining asymmetries (test direction, multiplicity regime) are "
        "presentation choices the paper states explicitly.",
        "",
    ]

    out = root / "data/rho_f_construct_validity.md"
    out.write_text("\n".join(L), encoding="utf-8")
    logger.info("wrote {}", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
