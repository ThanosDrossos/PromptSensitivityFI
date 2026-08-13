"""R3 — the independence re-analysis. Replaces "rho_F is orthogonal to ability and
dispersion (.08 / .03)" with something a reviewer can check.

WHAT WAS WRONG (review 2026-08-06 §2.3).
  1. The published .08 / .03 are averages ACROSS models that hide llama's
     rho_F~accuracy = +0.224 (95% CI [+0.088, +0.353], excludes zero).
  2. They are complete-case, i.e. computed after deleting the 34-55% of cells
     where rho_F is undefined -- exactly the accuracy extremes. Range restriction.
  3. ".03 with dispersion" is the MINIMUM over the dispersion family. Against
     variation ratio rho_F is +.41/+.28/+.36; against 1-TVD -.40/-.17/-.36.
  4. No uncertainty is reported, and "orthogonal" is asserted from a point
     estimate. Absence of evidence is not evidence of absence.
  5. The deck's numbers are pooled over levels although main_body.tex says
     correlations are computed WITHIN a specificity level.

WHAT THIS DOES.
  * Every correlation WITHIN (model x level), as the paper's own Methods promises,
    plus a within-model pooled row, never a cross-model average.
  * Three rho_F estimators side by side: complete-case MoM, MoM with 0 imputed on
    zero-variance cells, and the hierarchical posterior (R2) -- the last via
    MULTIPLE IMPUTATION over posterior draws, pooled with Rubin's rules, so
    per-cell uncertainty is propagated rather than hidden by shrinkage.
  * The DISPERSION FACTOR (PC1 of the six dispersion metrics) as the axis-3
    target, not a single hand-picked representative.
  * TOST equivalence tests against a PRE-STATED bound, and the smallest bound the
    data actually support.
  * Disattenuation using the honest split-half reliability (disjoint paraphrase
    halves, averaged over many splits -- not the k10-vs-k20 comparison, whose
    k=20 set contains the k=10 responses).

DECISION RULE (agreed 2026-08-07): orthogonality is claimed ONLY if it holds in
ALL THREE models. If it fails in any model, the claim is dropped and replaced by
the bounded statement the data support.

    uv run python -m prompt_sensitivity.scripts.independence_analysis
    uv run python -m prompt_sensitivity.scripts.independence_analysis --scoring union
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from ..analysis.rho_f_hierarchical import fit_hierarchical_rho_f
from ..config import load_config
from ..metrics.sensitivity_v2 import rho_f as rho_f_mom

_MODELS = ["qwen_2_5_7b", "llama_3_1_8b", "mistral_7b_v03"]
_DISPERSION = ["h_sem_mean", "s_tau_mean", "variation_ratio",
               "fi_out_var", "a_q", "consistency_mean"]
# Pre-stated equivalence bound. Fixed BEFORE looking at the corrected numbers so
# it cannot be tuned to the answer; |rho| < 0.2 is the conventional "small effect"
# threshold and is the loosest bound anyone would accept as "practically
# independent" for a claim that two axes cannot substitute for each other.
_EQUIV_BOUND = 0.20


# ----------------------------------------------------------------- helpers


def _fisher_ci(r: float, n: int, conf: float = 0.95) -> tuple[float, float]:
    if n < 4 or not np.isfinite(r):
        return (np.nan, np.nan)
    r = float(np.clip(r, -0.999999, 0.999999))
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    crit = stats.norm.ppf(0.5 + conf / 2.0)
    return float(np.tanh(z - crit * se)), float(np.tanh(z + crit * se))


def tost_equivalence(r: float, n: int, bound: float = _EQUIV_BOUND) -> dict:
    """Equivalence via the 90% CI rule (operationally identical to TOST at 5%).

    Equivalent iff the whole 90% CI lies inside (-bound, +bound).
    `smallest_supported` is the tightest bound the data would support.
    """
    lo, hi = _fisher_ci(r, n, conf=0.90)
    if not np.isfinite(lo):
        return {"equivalent": False, "ci90_lo": np.nan, "ci90_hi": np.nan,
                "smallest_supported": np.nan}
    return {
        "equivalent": bool(lo > -bound and hi < bound),
        "ci90_lo": lo, "ci90_hi": hi,
        "smallest_supported": float(max(abs(lo), abs(hi))),
    }


def dispersion_factor(df: pd.DataFrame) -> np.ndarray:
    """PC1 of the ranked dispersion family, oriented to point WITH H_sem.

    Using the factor instead of one representative is the point: reporting
    rho_F vs H_sem alone reports the minimum of the family.
    """
    cols = [c for c in _DISPERSION if c in df.columns and df[c].notna().sum() > 10]
    R = df[cols].rank()
    R = (R - R.mean()) / R.std(ddof=0).replace(0, np.nan)
    R = R.dropna(axis=1, how="any").fillna(0.0)
    X = R.to_numpy(dtype=float)
    X = X - X.mean(axis=0, keepdims=True)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    pc1 = U[:, 0] * S[0]
    if "h_sem_mean" in df.columns:
        rr = stats.spearmanr(pc1, df["h_sem_mean"], nan_policy="omit")[0]
        if np.isfinite(rr) and rr < 0:
            pc1 = -pc1
    return pc1


def split_half_reliability(cells: list, k: int, *, n_splits: int = 200,
                           seed: int = 42) -> float:
    """Spearman-Brown-corrected disjoint-PARAPHRASE split-half reliability.

    NOT the k10-vs-k20 comparison: main_body.tex states the larger-k replication
    "reuse[s] the responses of the smaller one exactly", so that correlation is
    between a statistic and a superset of itself. A single 5/5 split is unstable
    (sd ~0.05 across splits), hence the average.
    """
    usable = [list(c) for c in cells if c is not None and len(c) >= 4]
    if len(usable) < 10:
        return float("nan")
    rng = np.random.default_rng(seed)
    rs = []
    for _ in range(n_splits):
        a, b = [], []
        for s in usable:
            idx = rng.permutation(len(s))
            half = len(s) // 2
            ra = rho_f_mom([s[i] for i in idx[:half]], k)
            rb = rho_f_mom([s[i] for i in idx[half:]], k)
            if np.isnan(ra) or np.isnan(rb):
                continue
            a.append(ra)
            b.append(rb)
        if len(a) > 10:
            r = stats.spearmanr(a, b)[0]
            if np.isfinite(r):
                rs.append(r)
    if not rs:
        return float("nan")
    r = float(np.mean(rs))
    return float(2 * r / (1 + r)) if r > -1 else float("nan")


def pooled_correlation_mi(draws: np.ndarray, y: np.ndarray) -> dict:
    """Spearman correlation pooled over posterior draws by Rubin's rules.

    draws: (D, C) posterior draws of rho_F. y: (C,) the other variable.
    Pooling happens on the Fisher-z scale; the point estimate is back-transformed.
    """
    ok = np.isfinite(y)
    y = y[ok]
    draws = draws[:, ok]
    n = int(ok.sum())
    if n < 10:
        return {"r": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "n": n}
    zs = []
    for d in range(draws.shape[0]):
        r = stats.spearmanr(draws[d], y)[0]
        if np.isfinite(r):
            zs.append(np.arctanh(np.clip(r, -0.999999, 0.999999)))
    if not zs:
        return {"r": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "n": n}
    zs = np.asarray(zs)
    z_bar = zs.mean()
    w = 1.0 / (n - 3)                       # within-imputation variance
    b = zs.var(ddof=1) if len(zs) > 1 else 0.0   # between-imputation variance
    t = w + (1.0 + 1.0 / len(zs)) * b
    se = np.sqrt(t)
    return {
        "r": float(np.tanh(z_bar)),
        "ci_lo": float(np.tanh(z_bar - 1.96 * se)),
        "ci_hi": float(np.tanh(z_bar + 1.96 * se)),
        "n": n,
    }


# ----------------------------------------------------------------- the analysis


def analyse(args) -> pd.DataFrame:
    config = load_config()
    rows: list[dict] = []
    reliab: dict[str, float] = {}

    for model in args.models:
        src = (config.repo_root() /
               (f"data/specificity_v3_{model}.parquet" if args.scoring == "target"
                else f"data/union_gold_{model}.parquet"))
        if not src.exists():
            logger.error("missing {}", src)
            continue
        df = pd.read_parquet(src)
        pp_col = ("f_graded_per_paraphrase" if args.scoring == "target"
                  else "f_graded_union_per_paraphrase")
        acc_col = ("f_graded_mean" if args.scoring == "target"
                   else "f_graded_union_mean")
        if args.scoring == "union":
            # The union parquet carries only the re-SCORED quantities. The
            # dispersion family (H_sem, S_tau, ...) is computed from the pooled
            # response clusters and never touches a gold answer, so it is
            # identical under either scoring — join it from the v3 parquet.
            v3 = pd.read_parquet(config.repo_root() / f"data/specificity_v3_{model}.parquet")
            disp_cols = [c for c in _DISPERSION + ["rho_u", "ess_in"] if c in v3.columns]
            df = df.merge(
                v3[["question_id", "spec_level"] + disp_cols],
                on=["question_id", "spec_level"], how="left", validate="1:1",
            )
            # normalise the MoM column name (rescore suffixes it by gold set)
            if "rho_f_union" in df.columns:
                df["rho_f"] = df["rho_f_union"]
        k = int(df["n_samples_per_prompt"].iloc[0]) if "n_samples_per_prompt" in df else 10
        cells = [list(x) if x is not None else None for x in df[pp_col]]

        # three estimators of the same estimand
        if "rho_f" not in df.columns:
            df["rho_f"] = [rho_f_mom(c, k) if c else np.nan for c in cells]
        df["rho_f_imputed0"] = df["rho_f"].fillna(0.0)
        fit = fit_hierarchical_rho_f(cells, k)
        df["rho_f_hier"] = fit.rho_mean
        draws = fit.sample(args.n_draws, seed=config.random_seed)

        df["dispersion_pc1"] = dispersion_factor(df)
        reliab[model] = split_half_reliability(cells, k, n_splits=args.n_splits)

        targets = {"accuracy": acc_col, "H_sem": "h_sem_mean",
                   "dispersion_factor": "dispersion_pc1"}
        levels = [("both", df.index)] + [
            (f"L{lv}", df.index[df.spec_level == lv]) for lv in sorted(df.spec_level.unique())
        ]

        for lvl_name, idx in levels:
            sub = df.loc[idx]
            sub_draws = draws[:, [df.index.get_loc(i) for i in idx]]
            for tname, tcol in targets.items():
                if tcol not in sub.columns:
                    continue
                y = sub[tcol].to_numpy(dtype=float)
                # (a) complete case
                m = sub["rho_f"].notna().to_numpy() & np.isfinite(y)
                r_cc = stats.spearmanr(sub["rho_f"].to_numpy()[m], y[m])[0] if m.sum() > 10 else np.nan
                eq_cc = tost_equivalence(r_cc, int(m.sum()), args.bound)
                # (b) imputed 0
                m2 = np.isfinite(y)
                r_im = stats.spearmanr(sub["rho_f_imputed0"].to_numpy()[m2], y[m2])[0] if m2.sum() > 10 else np.nan
                eq_im = tost_equivalence(r_im, int(m2.sum()), args.bound)
                # (c) hierarchical, multiple imputation
                mi = pooled_correlation_mi(sub_draws, y)
                rows.append({
                    "model": model, "scoring": args.scoring, "level": lvl_name,
                    "target": tname,
                    "n_complete_case": int(m.sum()), "n_all": int(m2.sum()),
                    "coverage": float(sub["rho_f"].notna().mean()),
                    "r_complete_case": r_cc,
                    "cc_ci90_lo": eq_cc["ci90_lo"], "cc_ci90_hi": eq_cc["ci90_hi"],
                    "cc_equivalent": eq_cc["equivalent"],
                    "cc_smallest_bound": eq_cc["smallest_supported"],
                    "r_imputed0": r_im,
                    "im_equivalent": eq_im["equivalent"],
                    "im_smallest_bound": eq_im["smallest_supported"],
                    "r_hier_mi": mi["r"], "hier_ci_lo": mi["ci_lo"], "hier_ci_hi": mi["ci_hi"],
                })

    out = pd.DataFrame(rows)
    out.attrs["reliability"] = reliab
    return out


def report(res: pd.DataFrame, bound: float) -> str:
    """Markdown verdict, including the pass/fail decision rule."""
    rel = res.attrs.get("reliability", {})
    lines = ["# R3 — independence re-analysis", ""]
    lines.append(f"Pre-stated equivalence bound: **|rho| < {bound}** "
                 "(claimed only if it holds in ALL models).")
    lines.append("")
    lines.append("## Split-half reliability of rho_F (disjoint paraphrase halves, Spearman-Brown)")
    lines.append("")
    lines.append("| model | reliability |")
    lines.append("|---|---|")
    for m, v in rel.items():
        lines.append(f"| {m} | {v:.3f} |")
    lines.append("")
    lines.append("Disattenuated correlations below use r / sqrt(rel_rhoF * 0.95).")
    lines.append("")

    for target in ["accuracy", "H_sem", "dispersion_factor"]:
        sub = res[(res.target == target) & (res.level != "both")]
        if sub.empty:
            continue
        lines.append(f"## rho_F vs {target}")
        lines.append("")
        lines.append("| model | level | n (cc) | coverage | complete-case r [90% CI] | imputed-0 r | "
                     "hierarchical r [95% CI] | equivalent? |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for _, r in sub.iterrows():
            rel_m = rel.get(r["model"], np.nan)
            dis = (r["r_complete_case"] / np.sqrt(rel_m * 0.95)
                   if np.isfinite(rel_m) and rel_m > 0 else np.nan)
            lines.append(
                f"| {r['model']} | {r['level']} | {r['n_complete_case']} | {r['coverage']:.0%} | "
                f"{r['r_complete_case']:+.3f} [{r['cc_ci90_lo']:+.3f}, {r['cc_ci90_hi']:+.3f}] "
                f"(disatt {dis:+.3f}) | {r['r_imputed0']:+.3f} | "
                f"{r['r_hier_mi']:+.3f} [{r['hier_ci_lo']:+.3f}, {r['hier_ci_hi']:+.3f}] | "
                f"{'YES' if r['cc_equivalent'] else 'NO'} |"
            )
        holds_cc = bool(sub["cc_equivalent"].all())
        worst = sub["cc_smallest_bound"].max()
        # Direction consistency: a small effect repeated in the same direction
        # across every model x level is evidence a wide single CI cannot show.
        signs = np.sign(sub["r_complete_case"].to_numpy())
        n_pos, n_tot = int((signs > 0).sum()), int(np.isfinite(signs).sum())
        sign_p = float(stats.binomtest(max(n_pos, n_tot - n_pos), n_tot, 0.5).pvalue) if n_tot else np.nan
        hier_max = float(np.nanmax(np.abs(sub["r_hier_mi"].to_numpy())))
        lines.append("")
        lines.append(
            f"**Verdict — equivalence at |rho| < {bound}: "
            f"{'HOLDS' if holds_cc else 'NOT ESTABLISHED'} "
            f"(complete case).** Smallest bound the data support in every model/level: "
            f"**|rho| < {worst:.2f}**. "
            f"Hierarchical (primary) estimates never exceed |{hier_max:.3f}|. "
            f"Direction consistency: {n_pos}/{n_tot} positive, sign test p = {sign_p:.3f}."
        )
        lines.append("")
    lines.append("## How to read the three estimators")
    lines.append("")
    lines.append(
        "**The hierarchical column is primary.** It is the only one defined on all cells and "
        "the only one that propagates per-cell uncertainty (multiple imputation over posterior "
        "draws, pooled by Rubin's rules). It answers: *given what we actually know about rho_F, "
        "how much does it co-vary with the other axis?*"
    )
    lines.append("")
    lines.append(
        "**Complete-case** answers a different and also legitimate question: *among the cells "
        "where rho_F is measurable at all, how much does it co-vary?* It is range-restricted on "
        "accuracy (the excluded cells are the extremes), so it is not a substitute for the "
        "hierarchical estimate, but a consistent sign across model x level is informative."
    )
    lines.append("")
    lines.append(
        "**Imputed-0 is reported only as a sensitivity check and must NOT be used as a headline. "
        "It is an artifact of the imputation rule.** Degenerate cells are heavily skewed toward "
        "all-WRONG rather than all-RIGHT (2.2:1 qwen, 6.8:1 llama, 4.8:1 mistral), and "
        "corr(is-degenerate, accuracy) = -0.21 / -0.34 / -0.38. Substituting a constant 0 "
        "therefore drops a large spike of zeros at low accuracy and manufactures a positive "
        "correlation that is a property of the rule, not of the constructs. (An earlier draft of "
        "the review used these numbers as the counter-estimate; that was wrong.)"
    )
    lines.append("")
    return "\n".join(lines)


def _parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", nargs="+", default=_MODELS)
    ap.add_argument("--scoring", choices=["target", "union"], default="target")
    ap.add_argument("--bound", type=float, default=_EQUIV_BOUND)
    ap.add_argument("--n-draws", type=int, default=200)
    ap.add_argument("--n-splits", type=int, default=200)
    ap.add_argument("--out", default="data/independence_{scoring}.parquet")
    ap.add_argument("--report", default="data/independence_{scoring}.md")
    return ap.parse_args(argv)


def main() -> int:
    args = _parse_args()
    config = load_config()
    res = analyse(args)
    if res.empty:
        logger.error("no results")
        return 1
    out = config.repo_root() / args.out.format(scoring=args.scoring)
    res.to_parquet(out, index=False)
    md = report(res, args.bound)
    rep = config.repo_root() / args.report.format(scoring=args.scoring)
    rep.write_text(md, encoding="utf-8")
    print(md)
    logger.info("wrote {} and {}", out.name, rep.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
