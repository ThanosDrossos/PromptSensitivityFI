"""R6 — generator-width dial analysis: the positive control for rho_F.

PREREGISTERED PREDICTIONS (fixed 2026-08-07, BEFORE any arm data existed;
see RHO_F_CONSTRUCT_VALIDITY_2026-08-07.md and the runbook):

  P0  MANIPULATION CHECK (gates the rest): realized width of the ACCEPTED
      universes is ordered narrow < medium < wide (mean pairwise token edit
      distance; embedding dispersion where available). If the identical
      NLI/gold/dedup gates censored the ordering away, the arm is
      INCONCLUSIVE — report that, do not interpret P1-P4.
  P1  sigma2_between (absolute phrasing variance) increases with width —
      one-sided narrow < wide, per model.
  P2  rho_F (hierarchical) increases with width — one-sided narrow < wide.
  P3  accuracy responds LESS than sigma2_B/rho_F (report the delta with CI;
      no sign prediction).
  P4  H_sem (per-prompt dispersion) responds LESS than sigma2_B/rho_F.
      P3+P4 mirror the specificity dial (which moves accuracy, not rho_F):
      TOGETHER THE TWO DIALS FORM THE DOUBLE DISSOCIATION.
  P5  SWAP (paraphraser-swap ablation, first in this literature per
      LITERATURE_INTEGRATION_2026-08-07 §4.4): per-cell rho_F under the
      OLMo-generated medium-width universes correlates positively with the
      Phi-4 medium arm, and the model ordering (qwen > mistral > llama)
      is preserved.

Inputs (produced by the cluster runbook):
  data/paraphrases_width_{narrow,wide,swap}.parquet  (+ _reject_stats sidecars)
  data/paraphrases_ambigqa.parquet                    (medium == v3 universes)
  data/width_{narrow,wide,swap}_{model}.parquet       (eval cells)
  data/specificity_v3_{model}.parquet                 (medium eval == v3 cells)

Output: data/width_dial_analysis.md + data/width_dial_cells.parquet.
CONTRACT: read-only; `metrics/` untouched.
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from ..analysis.rho_f_hierarchical import fit_hierarchical_rho_f, sigma2_between
from ..config import load_config
from ..paraphrases.deduplicate import levenshtein_tokens

_MODELS = ["qwen_2_5_7b", "llama_3_1_8b", "mistral_7b_v03"]
_ARMS = ["narrow", "medium", "wide"]          # the dial; swap handled separately


# --------------------------------------------------------------------------- #
# P0 — manipulation check                                                      #
# --------------------------------------------------------------------------- #


def universe_width(texts: list[str]) -> dict:
    """Realized-width measures of one accepted universe (no model needed)."""
    if len(texts) < 2:
        return {"pairwise_token_dist": np.nan, "len_cv": np.nan, "n": len(texts)}
    dists = [levenshtein_tokens(a, b) for a, b in combinations(texts, 2)]
    lens = np.array([len(t) for t in texts], dtype=float)
    return {
        "pairwise_token_dist": float(np.mean(dists)),
        "len_cv": float(lens.std() / max(lens.mean(), 1e-9)),
        "n": len(texts),
    }


def subsample_rates(rates: list[float], n: int, *, seed: int) -> list[float]:
    """Seeded paraphrase subsample for the N-matched robustness check."""
    if len(rates) <= n:
        return list(rates)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(rates), size=n, replace=False)
    return [rates[i] for i in sorted(idx)]


def load_universes(config, cache_rel: str) -> pd.DataFrame:
    path = config.repo_root() / cache_rel
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df = df[df["outcome"].isin(["accepted", "singleton_fallback"])]
    rows = []
    for (qid, lvl), g in df.groupby(["question_id", "spec_level"]):
        w = universe_width(g.sort_values("paraphrase_idx")["text"].tolist())
        w.update({"question_id": str(qid), "spec_level": int(lvl)})
        rows.append(w)
    return pd.DataFrame(rows)


def manipulation_check(config, arm_caches: dict[str, str],
                       questions: set[str] | None) -> tuple[pd.DataFrame, bool, str]:
    """Per-arm realized width on the common question set; returns (table, ordered?, note)."""
    per_arm = {}
    for arm, cache in arm_caches.items():
        u = load_universes(config, cache)
        if u.empty:
            continue
        if questions:
            u = u[u.question_id.isin(questions)]
        u["arm"] = arm
        per_arm[arm] = u
    if not per_arm:
        return pd.DataFrame(), False, "no universe caches found"
    tab = pd.concat(per_arm.values(), ignore_index=True)

    have = [a for a in _ARMS if a in per_arm]
    ordered = False
    note = f"arms present: {have}"
    if set(have) >= set(_ARMS):
        # paired per (question, level): is width ordered narrow < medium < wide?
        piv = tab.pivot_table(index=["question_id", "spec_level"],
                              columns="arm", values="pairwise_token_dist").dropna()
        frac_mono = float(((piv["narrow"] < piv["medium"]) &
                           (piv["medium"] < piv["wide"])).mean())
        w_nw = stats.wilcoxon(piv["narrow"], piv["wide"],
                              alternative="less").pvalue if len(piv) > 10 else np.nan
        ordered = bool(
            piv["narrow"].mean() < piv["medium"].mean() < piv["wide"].mean()
            and (np.isnan(w_nw) or w_nw < 0.05)
        )
        note = (f"cells with full ordering narrow<medium<wide: {frac_mono:.0%}; "
                f"means {piv['narrow'].mean():.1f} < {piv['medium'].mean():.1f} "
                f"< {piv['wide'].mean():.1f} (tokens); Wilcoxon narrow<wide "
                f"p = {w_nw:.2g}; ORDERED = {ordered}")
    return tab, ordered, note


def censoring_table(config, arm_caches: dict[str, str]) -> pd.DataFrame:
    rows = []
    for arm, cache in arm_caches.items():
        p = config.repo_root() / cache
        side = p.with_name(p.stem + "_reject_stats.parquet")
        if not side.exists():
            rows.append({"arm": arm, "note": "no sidecar (pre-R6 cache)"})
            continue
        d = pd.read_parquet(side)
        tot_raw = (d.n_accepted + d.n_rejected_nli + d.n_rejected_constraint
                   + d.n_rejected_dedup)
        rows.append({
            "arm": arm, "universes": len(d),
            "nli_reject_rate": float((d.n_rejected_nli / tot_raw).mean()),
            "constraint_reject_rate": float((d.n_rejected_constraint / tot_raw).mean()),
            "dedup_reject_rate": float((d.n_rejected_dedup / tot_raw).mean()),
            "dropped_frac": float(d.dropped.mean()),
            "fallback_nli_frac": float((d.nli_threshold_used < 0.9 - 1e-9).mean()),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# P1-P5 — outcomes                                                             #
# --------------------------------------------------------------------------- #


def load_arm_cells(config, model: str, arm: str, questions: set[str] | None,
                   *, n_cap: dict[tuple[str, int], int] | None = None,
                   seed: int = 42) -> pd.DataFrame:
    """Eval cells for one (model, arm).

    `n_cap`: the N-matched robustness mode — per-(question, level) cap on the
    number of paraphrases, applied by seeded subsampling BEFORE computing
    sigma2_B / rho_F. Needed because the narrow arm's universes are undersized
    (the >=6-char dedup gate collides with minimal edits — 72 % of narrow
    universes filled below target on the 2026-08-07 prep), so a raw arm
    comparison confounds width with |U|. The primary analysis uses full
    universes; the N-matched pass equalises |U| per cell across arms.
    """
    rel = (f"data/specificity_v3_{model}.parquet" if arm == "medium"
           else f"data/width_{arm}_{model}.parquet")
    path = config.repo_root() / rel
    if not path.exists():
        return pd.DataFrame()
    d = pd.read_parquet(path)
    if questions:
        d = d[d.question_id.astype(str).isin(questions)]
    d = d.copy()
    d["arm"] = arm
    k = int(d["n_samples_per_prompt"].iloc[0]) if len(d) else 10
    cells = []
    for _, r in d.iterrows():
        c = list(r["f_graded_per_paraphrase"]) if r["f_graded_per_paraphrase"] is not None else None
        if c is not None and n_cap is not None:
            cap = n_cap.get((str(r["question_id"]), int(r["spec_level"])))
            if cap:
                c = subsample_rates(c, cap, seed=seed)
        cells.append(c)
    d["n_universe"] = [len(c) if c else 0 for c in cells]
    d["sigma2_B"] = [sigma2_between(c, k) for c in cells]
    from ..metrics.sensitivity_v2 import rho_f as _mom
    d["rho_f_mom"] = [(_mom(c, k) if c and len(c) >= 2 else np.nan) for c in cells]
    if len(d) >= 20:
        fit = fit_hierarchical_rho_f(cells, k)
        d["rho_f_hier"] = fit.rho_mean
    else:
        d["rho_f_hier"] = np.nan
    return d


def n_cap_from_arms(config, model: str, questions: set[str] | None) -> dict:
    """Per-(question, level) minimum universe size across the dial arms."""
    caps: dict[tuple[str, int], int] = {}
    for arm in _ARMS:
        d = load_arm_cells(config, model, arm, questions)
        if d.empty:
            continue
        for _, r in d.iterrows():
            key = (str(r["question_id"]), int(r["spec_level"]))
            n = int(r["n_universe"])
            if n >= 3:
                caps[key] = min(caps.get(key, 10**9), n)
    return caps


def paired_arm_test(cells: pd.DataFrame, value: str) -> dict:
    """Per-model paired dial tests on one outcome column."""
    piv = cells.pivot_table(index=["question_id", "spec_level"],
                            columns="arm", values=value)
    have = [a for a in _ARMS if a in piv.columns]
    out: dict = {"n_paired": 0}
    if set(have) >= set(_ARMS):
        p = piv[_ARMS].dropna()
        out["n_paired"] = len(p)
        if len(p) > 10:
            out["means"] = [float(p[a].mean()) for a in _ARMS]
            out["wilcoxon_nw_less_p"] = float(
                stats.wilcoxon(p["narrow"], p["wide"], alternative="less").pvalue)
            out["friedman_p"] = float(stats.friedmanchisquare(
                p["narrow"], p["medium"], p["wide"]).pvalue)
            out["frac_monotone"] = float(
                ((p["narrow"] <= p["medium"]) & (p["medium"] <= p["wide"])).mean())
    return out


def swap_check(config, model: str, questions: set[str] | None) -> dict:
    med = load_arm_cells(config, model, "medium", questions)
    swp = load_arm_cells(config, model, "swap", questions)
    if med.empty or swp.empty:
        return {}
    j = med.merge(swp, on=["question_id", "spec_level"], suffixes=("_med", "_swap"))
    if len(j) < 10:
        return {}
    r_rho, p_rho = stats.spearmanr(j.rho_f_hier_med, j.rho_f_hier_swap)
    r_s2, _ = stats.spearmanr(j.sigma2_B_med, j.sigma2_B_swap, nan_policy="omit")
    return {
        "n": len(j),
        "rho_f_percell_spearman": float(r_rho), "p": float(p_rho),
        "sigma2B_percell_spearman": float(r_s2),
        "rho_f_mean_med": float(j.rho_f_hier_med.mean()),
        "rho_f_mean_swap": float(j.rho_f_hier_swap.mean()),
    }


# --------------------------------------------------------------------------- #
# report                                                                       #
# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", nargs="+", default=_MODELS)
    args = ap.parse_args()
    config = load_config()

    from ..paraphrases.prompts import WIDTH_ARMS
    arm_caches = {a: WIDTH_ARMS[a]["cache"] for a in WIDTH_ARMS}

    # the dial's question set = whatever the narrow/wide caches contain
    qset: set[str] = set()
    for arm in ("narrow", "wide"):
        u = load_universes(config, arm_caches[arm])
        if not u.empty:
            qset |= set(u.question_id)
    questions = qset or None

    L = ["# R6 — generator-width dial (positive control for ρ_F)", ""]
    L.append("Preregistered predictions P0–P5 are in the module docstring and the runbook; they were")
    L.append("fixed before any arm data existed.")
    L.append("")

    tab, ordered, note = manipulation_check(config, arm_caches, questions)
    L.append("## P0 — manipulation check (gates everything below)")
    L.append("")
    L.append(f"**{note}**")
    if not tab.empty:
        agg = tab.groupby("arm")[["pairwise_token_dist", "len_cv", "n"]].mean().round(2)
        L.append("")
        L.append("| arm | pairwise token dist | length CV | mean |U| |")
        L.append("|---|---|---|---|")
        for a, r in agg.iterrows():
            L.append(f"| {a} | {r['pairwise_token_dist']:.2f} | {r['len_cv']:.2f} | {r['n']:.1f} |")
    cens = censoring_table(config, arm_caches)
    if not cens.empty:
        L.append("")
        L.append("Gate censoring per arm (identical gates; differences are the gate reacting to G):")
        L.append("")
        cols = [c for c in cens.columns]
        L.append("| " + " | ".join(cols) + " |")
        L.append("|" + "---|" * len(cols))
        for _, r in cens.round(3).iterrows():
            L.append("| " + " | ".join("" if pd.isna(r[c]) else str(r[c]) for c in cols) + " |")
    L.append("")
    if not ordered:
        L.append("> ⚠ **P0 not (yet) satisfied — the outcome tests below are NOT interpretable as the")
        L.append("> width dial.** Either arms are missing, or the gates censored the ordering away.")
        L.append("")

    all_cells = []
    for model in args.models:
        L.append(f"## {model}")
        L.append("")
        frames = []
        for a in _ARMS:
            c = load_arm_cells(config, model, a, questions)
            if not c.empty:
                frames.append(c)
        cells = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if cells.empty or cells.arm.nunique() < 2:
            L.append("_arms missing — run the cluster jobs first_")
            L.append("")
            continue
        all_cells.append(cells.assign(model=model))
        sizes = cells.groupby("arm")["n_universe"].agg(["mean", "median", "min"]).round(2)
        L.append(f"universe sizes |U| per arm (unequal-N caveat): "
                 + "; ".join(f"{a}: mean {r['mean']:.1f}, median {r['median']:.0f}"
                             for a, r in sizes.iterrows()))
        L.append("")
        # ⚠ hier caveat (verified 2026-08-08): the per-arm empirical-Bayes fit is
        # weakly identified on the NARROW arm (median |U| = 7 with a majority of
        # fully-degenerate cells at near-deterministic decoding) and can invert
        # the ordering (qwen narrow printed .68 while MoM/full-N/N-matched all
        # show narrow LOWEST). Interpret P2 via the MoM row + N-matched rows.
        for pred, col, direction in [("P1 σ²_B", "sigma2_B", "increases"),
                                     ("P2 ρ_F (hier., ⚠ see caveat)", "rho_f_hier", "increases"),
                                     ("P2b ρ_F (MoM, covered cells)", "rho_f_mom", "increases"),
                                     ("P3 accuracy", "f_graded_mean", "little change"),
                                     ("P4 H_sem", "h_sem_mean", "little change")]:
            t = paired_arm_test(cells, col)
            if t.get("n_paired", 0) > 10:
                m = " → ".join(f"{x:.4f}" for x in t["means"])
                L.append(f"- **{pred}** ({direction}): {m} | narrow<wide one-sided "
                         f"p = {t['wilcoxon_nw_less_p']:.3g} | Friedman p = {t['friedman_p']:.3g} "
                         f"| monotone cells {t['frac_monotone']:.0%} (n = {t['n_paired']})")
            else:
                L.append(f"- **{pred}**: insufficient paired arms (n = {t.get('n_paired', 0)})")
        # N-matched robustness: equalise |U| per cell across arms by seeded
        # subsampling, so the dial effect cannot be an unequal-N artifact
        # (narrow's universes are systematically smaller — dedup collision).
        caps = n_cap_from_arms(config, model, questions)
        if caps:
            frames_m = []
            for a in _ARMS:
                c = load_arm_cells(config, model, a, questions, n_cap=caps)
                if not c.empty:
                    frames_m.append(c)
            cells_m = pd.concat(frames_m, ignore_index=True) if frames_m else pd.DataFrame()
            if not cells_m.empty and cells_m.arm.nunique() >= 3:
                for pred, col in [("P1 σ²_B", "sigma2_B"), ("P2 ρ_F (hier.)", "rho_f_hier")]:
                    t = paired_arm_test(cells_m, col)
                    if t.get("n_paired", 0) > 10:
                        m = " → ".join(f"{x:.4f}" for x in t["means"])
                        L.append(f"- **{pred} [N-matched]**: {m} | narrow<wide one-sided "
                                 f"p = {t['wilcoxon_nw_less_p']:.3g} (n = {t['n_paired']})")
        sw = swap_check(config, model, questions)
        if sw:
            L.append(f"- **P5 swap (OLMo medium)**: per-cell ρ_F Spearman = "
                     f"{sw['rho_f_percell_spearman']:+.3f} (p = {sw['p']:.2g}, n = {sw['n']}); "
                     f"means {sw['rho_f_mean_med']:.3f} (Phi-4) vs {sw['rho_f_mean_swap']:.3f} (OLMo); "
                     f"σ²_B per-cell Spearman = {sw['sigma2B_percell_spearman']:+.3f}")
        else:
            L.append("- **P5 swap**: swap arm not present yet")
        L.append("")

    out_md = config.repo_root() / "data/width_dial_analysis.md"
    out_md.write_text("\n".join(L), encoding="utf-8")
    if all_cells:
        keep = pd.concat(all_cells, ignore_index=True)
        drop = [c for c in ("f_graded_per_paraphrase", "fi_in_curve_ks", "fi_in_curve_vals",
                            "fi_in_ci_lower", "fi_in_ci_upper") if c in keep.columns]
        keep.drop(columns=drop).to_parquet(
            config.repo_root() / "data/width_dial_cells.parquet", index=False)
    try:
        print("\n".join(L))
    except UnicodeEncodeError:
        # Windows cp1252 console cannot render the Greek metric names; the
        # report file is already written at this point.
        print(f"(console cannot render the report; see {out_md})")
    logger.info("wrote {}", out_md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
