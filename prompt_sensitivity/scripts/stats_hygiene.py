"""R8 — statistical hygiene: the single source of truth for the paper's inference.

Replaces the old policy ("we report unadjusted p-values and state the number of
tests") with what the review demanded (2026-08-06 §3.4/§3.4b, SR-01..03):

  1. A DECLARED primary endpoint family, tested once, with Holm and BH columns:
     per model — Δ accuracy under UNION gold (the primary effect), Δ accuracy
     under TARGET gold (protocol comparison), Δ H_sem, Δ rho_F (hierarchical).
     FI_out_fixed is NOT in the family: its paired test is algebraically the
     H_sem test (affine relabeling; identical Wilcoxon p) and reporting both
     double-counts.
  2. Effect sizes with QUESTION-CLUSTERED bootstrap 95% CIs, never p alone.
  3. The cross-model dependence made explicit: the three models are correlated
     measurements on ONE question sample, not replications. We report the
     pairwise correlations of per-question deltas and a pooled per-question
     test (deltas averaged over models -> one Wilcoxon per endpoint), which is
     the honest "one experiment" version of "significant in all three models".
  4. Reliability reported as the disjoint-paraphrase split-half (mean of 200
     random 5/5 splits, Spearman-Brown) — the k10-vs-k20 comparison is retired
     (the k=20 sample CONTAINS the k=10 sample; a statistic correlated with its
     superset is not a reliability).

Output: data/stats_hygiene.md — the Results section quotes numbers from here.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from ..analysis.rho_f_hierarchical import fit_hierarchical_rho_f
from ..config import load_config
from ..scripts.independence_analysis import split_half_reliability

_MODELS = ["qwen_2_5_7b", "llama_3_1_8b", "mistral_7b_v03"]


def paired_deltas(df: pd.DataFrame, col: str) -> pd.Series:
    p = df.pivot_table(index="question_id", columns="spec_level", values=col)
    if 0 not in p.columns or 1 not in p.columns:
        return pd.Series(dtype=float)
    p = p.dropna()
    return (p[1] - p[0]).rename(col)


def cluster_boot_ci(deltas: pd.Series, *, n_boot: int = 5000, seed: int = 42):
    """Question-clustered bootstrap CI of the mean paired delta.

    The unit of analysis is the QUESTION (deltas are already per-question), so
    resampling questions with replacement is the clustered bootstrap here.
    """
    rng = np.random.default_rng(seed)
    v = deltas.to_numpy(dtype=float)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(v.mean()), float(lo), float(hi)


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def bh(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        prev = min(prev, pvals[i] * m / (rank + 1))
        adj[i] = prev
    return adj.tolist()


def main() -> int:
    config = load_config()
    root = config.repo_root()

    # ---- assemble the endpoint deltas per model -----------------------------
    endpoints = []           # rows: model, endpoint, n, effect, ci_lo, ci_hi, p
    per_q_deltas: dict[tuple[str, str], pd.Series] = {}
    for m in _MODELS:
        v3 = pd.read_parquet(root / f"data/specificity_v3_{m}.parquet")
        ug = pd.read_parquet(root / f"data/union_gold_{m}.parquet")
        cells = [list(x) if x is not None else None for x in v3["f_graded_per_paraphrase"]]
        k = int(v3["n_samples_per_prompt"].iloc[0])
        fit = fit_hierarchical_rho_f(cells, k)
        v3 = v3.assign(rho_f_hier=fit.rho_mean)

        frames = {
            "accuracy (union gold)  [PRIMARY]": paired_deltas(ug, "f_graded_union_mean"),
            "accuracy (target gold)": paired_deltas(v3, "f_graded_mean"),
            "H_sem": paired_deltas(v3, "h_sem_mean"),
            "rho_F (hierarchical)": paired_deltas(v3, "rho_f_hier"),
        }
        for name, d in frames.items():
            if d.empty:
                continue
            eff, lo, hi = cluster_boot_ci(d)
            p = stats.wilcoxon(d).pvalue if (d != 0).any() else 1.0
            endpoints.append({"model": m, "endpoint": name, "n": len(d),
                              "effect": eff, "ci_lo": lo, "ci_hi": hi, "p": float(p)})
            per_q_deltas[(m, name)] = d

    dfp = pd.DataFrame(endpoints)
    dfp["p_holm"] = holm(dfp["p"].tolist())
    dfp["p_bh"] = bh(dfp["p"].tolist())

    # ---- cross-model dependence ---------------------------------------------
    dep_rows = []
    pooled_rows = []
    for name in ["accuracy (union gold)  [PRIMARY]", "H_sem"]:
        ds = {m: per_q_deltas[(m, name)] for m in _MODELS if (m, name) in per_q_deltas}
        j = pd.concat(ds.values(), axis=1, keys=ds.keys()).dropna()
        for a, b in [(0, 1), (0, 2), (1, 2)]:
            ma, mb = list(ds)[a], list(ds)[b]
            r = stats.spearmanr(j[ma], j[mb])[0]
            dep_rows.append({"endpoint": name, "pair": f"{ma} ~ {mb}", "spearman": float(r)})
        pooled = j.mean(axis=1)
        eff, lo, hi = cluster_boot_ci(pooled)
        pooled_rows.append({"endpoint": name, "n": len(pooled), "effect": eff,
                            "ci_lo": lo, "ci_hi": hi,
                            "p": float(stats.wilcoxon(pooled).pvalue)})

    # ---- reliability ---------------------------------------------------------
    rel_rows = []
    for m in _MODELS:
        v3 = pd.read_parquet(root / f"data/specificity_v3_{m}.parquet")
        cells = [list(x) if x is not None else None for x in v3["f_graded_per_paraphrase"]]
        rel = split_half_reliability(cells, int(v3["n_samples_per_prompt"].iloc[0]),
                                     n_splits=200)
        rel_rows.append({"model": m, "split_half_SB": rel})

    # ---- render ---------------------------------------------------------------
    L = ["# R8 — statistical hygiene (source of truth for Results)", ""]
    L.append("**Declared primary family** (12 paired Wilcoxon tests; family-wise Holm + BH). "
             "The primary effect of the manipulation is Δ accuracy under UNION gold; "
             "target-gold Δ is the protocol comparison (its excess over union = the grading "
             "lottery); FI_out_fixed is excluded — its test IS the H_sem test (affine "
             "relabeling, identical p).")
    L.append("")
    L.append("| model | endpoint | n | effect (L1−L0) | 95 % CI (question-clustered) | p | Holm | BH |")
    L.append("|---|---|---|---|---|---|---|---|")
    for _, r in dfp.iterrows():
        L.append(f"| {r.model} | {r.endpoint} | {r.n} | {r.effect:+.4f} | "
                 f"[{r.ci_lo:+.4f}, {r.ci_hi:+.4f}] | {r.p:.2g} | {r.p_holm:.2g} | {r.p_bh:.2g} |")
    L.append("")
    L.append("## The three models are correlated measurements, not replications")
    L.append("")
    L.append("Per-question deltas correlate across models:")
    L.append("")
    L.append("| endpoint | pair | Spearman |")
    L.append("|---|---|---|")
    for r in dep_rows:
        L.append(f"| {r['endpoint']} | {r['pair']} | {r['spearman']:+.3f} |")
    L.append("")
    L.append("The honest single-experiment test (per-question deltas averaged over the three "
             "models, one Wilcoxon per endpoint):")
    L.append("")
    L.append("| endpoint | n questions | pooled effect | 95 % CI | p |")
    L.append("|---|---|---|---|---|")
    for r in pooled_rows:
        L.append(f"| {r['endpoint']} | {r['n']} | {r['effect']:+.4f} | "
                 f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | {r['p']:.2g} |")
    L.append("")
    L.append("## Reliability (replaces the retired k10-vs-k20 comparison)")
    L.append("")
    L.append("Disjoint-paraphrase split-half, mean of 200 random 5/5 splits, Spearman-Brown:")
    L.append("")
    L.append("| model | split-half reliability |")
    L.append("|---|---|")
    for r in rel_rows:
        L.append(f"| {r['model']} | {r['split_half_SB']:.3f} |")
    L.append("")
    L.append("Consequences: per-question rho_F point claims are not supportable; observed "
             "cross-metric and cross-model correlations are attenuated by these reliabilities; "
             "the k=20 arm is reported only as a sampling-extension check, never as reliability.")
    L.append("")

    out = root / "data/stats_hygiene.md"
    out.write_text("\n".join(L), encoding="utf-8")
    try:
        print("\n".join(L))
    except UnicodeEncodeError:
        print(f"(console cannot render; see {out})")
    logger.info("wrote {}", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
