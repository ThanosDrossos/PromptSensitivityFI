"""Metric selection in two stages: published metrics first, rho_F projected, then added.

WHY (advisor feedback, 2026-09-15). The component analysis that justifies the
three factors must not contain the metric it is meant to justify. The earlier
audit (factor_audit.py) decomposed all fourteen metrics, our own rho_F and the
functional-information variants included. This script:

  Stage 1  decomposes the PUBLISHED metrics only (ten: the eight of the earlier
           audit plus the best- and worst-formulation accuracies of Sclar/
           Mizrahi/Cao, computed here from the graded per-formulation rates),
           runs Horn's parallel analysis, and projects the six constructed
           metrics into that space as supplementary variables -- a held-out
           metric's loading is its correlation with the component score,
           computed without letting it shape the component (Abdi & Williams
           2010, "supplementary elements").
  Stage 2  adds rho_F as an active variable and repeats the retention test.

The questions it answers, from committed artifacts only:
  1. How many components do the published metrics support, and which factor
     is missing an adequate published metric?
  2. In the three-component solution the framework posits, which published
     metric tracks each component best?
  3. Does rho_F track the formulation-dependence component more cleanly than
     any published metric, and does adding it complete the three-component
     structure under the same criterion?

Inputs : data/specificity_v3_*.parquet (the metric columns, plus the graded
         per-formulation rates for F_max / F_min); figures/v3_metric_corr.npy
         as a provenance guard (the 14 shared metrics must reproduce it).
Outputs: data/metric_selection.md (paper source) and data/metric_selection.json
         (table and figure source: the 16-metric matrix, both stages).
         Read-only over the parquets.

    uv run python -m prompt_sensitivity.scripts.metric_selection
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from ..config import load_config
from .factor_audit import horn
from .paper_analyses_b import _varimax

_MODELS = ["qwen_2_5_7b", "llama_3_1_8b", "mistral_7b_v03"]

# (column, label, a priori family, source). Labels of the 14 earlier metrics match
# figures/v3_metric_corr_labels.json; f_max / f_min are derived in `load_strata`.
PUBLISHED: list[tuple[str, str, str, str]] = [
    ("h_sem_mean", "H_sem", "dispersion", "Kuhn et al. 2023; Farquhar et al. 2024"),
    ("s_tau_mean", "S_tau (Errica)", "dispersion", "Errica et al. 2025"),
    ("tvd_sens", "TVD-sens  [M4]", "dispersion", "Errica et al. 2025"),
    ("a_q", "|A_q| observed", "dispersion", "Kuhn et al. 2023 (number of semantic sets)"),
    ("variation_ratio", "variation ratio", "dispersion", "Lu et al. 2024"),
    ("f_graded_mean", "accuracy", "success", "graded accuracy, mean over formulations"),
    ("f_max", "F_max (best formulation)", "success", "Sclar et al. 2024; Mizrahi et al. 2024"),
    ("f_min", "F_min (worst formulation)", "success", "Cao et al. 2024; Sclar et al. 2024"),
    ("spread", "spread (Cao)", "dependence", "Sclar et al. 2024; Cao et al. 2024"),
    ("rho_u", "rho_u (Cox)", "dependence", "Cox et al. 2025"),
]
# (column, label, a priori family) -- held out of the stage-1 decomposition
CONSTRUCTED: list[tuple[str, str, str]] = [
    ("rho_f", "rho_F  [M1]", "dependence"),
    ("aufi_in_graded", "AUFI (graded)", "success"),
    ("fi_premium", "FI premium  [M2]", "success"),
    ("fi_out_fixed", "FI_out_fixed", "dispersion"),
    ("fi_out_var", "Var[FI_out]  [M4]", "dispersion"),
    ("ess_in", "ESS_in", "dependence"),
]
FAMILY_ORDER = ["dispersion", "success", "dependence"]
FAMILY_NAME = {
    "dispersion": "output dispersion",
    "success": "mean task success",
    "dependence": "formulation dependence",
}
PRETTY = {
    "H_sem": "H_sem",
    "S_tau (Errica)": "S_τ",
    "TVD-sens  [M4]": "TVD consistency",
    "|A_q| observed": "\\|A_q\\|",
    "variation ratio": "variation ratio",
    "accuracy": "accuracy",
    "F_max (best formulation)": "F_max (best formulation)",
    "F_min (worst formulation)": "F_min (worst formulation)",
    "spread (Cao)": "spread",
    "rho_u (Cox)": "ρ_u",
    "rho_F  [M1]": "ρ_F",
    "AUFI (graded)": "AUFI",
    "FI premium  [M2]": "ΔFI premium",
    "FI_out_fixed": "FI_out^fixed",
    "Var[FI_out]  [M4]": "Var[FI_out]",
    "ESS_in": "ESS_in",
}
RHO_F = "rho_F  [M1]"
_N_PER_STRATUM = 150  # every published metric is defined on all 150 cells of every stratum
_RHO_F_N = (56, 84, 136, 150)  # smallest, median and legacy complete-case n for rho_F, and full


@dataclass
class Decomposition:
    """Eigendecomposition of a correlation matrix with a varimax-rotated k-component solution."""

    names: list[str]
    evals: np.ndarray
    horn95: np.ndarray
    n_retained: int
    evecs: np.ndarray  # unit eigenvectors as columns, descending eigenvalue
    k: int
    rotation: np.ndarray  # k x k orthogonal varimax rotation, sign flips included
    loadings: np.ndarray  # p x k rotated loadings = corr(variable, rotated component score)


def decompose(C: np.ndarray, names: list[str], n_obs: int, k: int, seed: int = 42) -> Decomposition:
    """PCA of a correlation matrix, Horn's parallel analysis, varimax on the first k."""
    C = (C + C.T) / 2
    C = np.nan_to_num(C, nan=0.0)
    np.fill_diagonal(C, 1.0)
    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(evals)[::-1]
    evals, evecs = np.clip(evals[order], 0, None), evecs[:, order]
    h95 = horn(C, n_obs, seed=seed)
    n_ret = int(np.sum(evals > h95))
    unrotated = evecs[:, :k] * np.sqrt(evals[:k])
    rotated = _varimax(unrotated)
    # Recover the orthogonal rotation R with unrotated @ R = rotated (the pseudo-inverse
    # of unrotated is diag(1/sqrt(lambda)) V^T), so held-out variables rotate alike.
    rotation = np.diag(1 / np.sqrt(evals[:k])) @ evecs[:, :k].T @ rotated
    signs = np.where(rotated[np.abs(rotated).argmax(axis=0), range(k)] < 0, -1.0, 1.0)
    return Decomposition(
        names=list(names),
        evals=evals,
        horn95=h95,
        n_retained=n_ret,
        evecs=evecs,
        k=k,
        rotation=rotation * signs,
        loadings=rotated * signs,
    )


def project(dec: Decomposition, r_sup: np.ndarray) -> np.ndarray:
    """Correlation of a held-out variable with the rotated component scores.

    `r_sup` holds the variable's correlations with the active variables, in
    `dec.names` order. With standardized scores t_j = (z v_j) / sqrt(lambda_j),
    corr(z_s, t_j) = (r_sup . v_j) / sqrt(lambda_j); an orthogonal rotation of
    the scores rotates these correlations by the same matrix. For an active
    variable the formula returns its own rotated loading exactly.
    """
    a = (np.asarray(r_sup, float) @ dec.evecs[:, : dec.k]) / np.sqrt(dec.evals[: dec.k])
    return a @ dec.rotation


def _family_means(dec: Decomposition, family_of: dict[str, str], j: int) -> dict[str, float]:
    return {
        fam: float(
            np.mean(
                [abs(dec.loadings[i, j]) for i, n in enumerate(dec.names) if family_of[n] == fam]
            )
        )
        for fam in FAMILY_ORDER
        if any(family_of[n] == fam for n in dec.names)
    }


def component_families(dec: Decomposition, family_of: dict[str, str]) -> list[str]:
    """Assign each component to the family with the largest mean |loading| on it."""
    return [max(m, key=m.get) for m in (_family_means(dec, family_of, j) for j in range(dec.k))]


def component_titles(dec: Decomposition, family_of: dict[str, str], tie: float = 0.10) -> list[str]:
    """Display name per component; a near-tie between two families is reported as mixed."""
    out = []
    for j in range(dec.k):
        m = _family_means(dec, family_of, j)
        top = sorted(m, key=m.get, reverse=True)
        if len(top) > 1 and m[top[0]] - m[top[1]] < tie:
            out.append(f"mixed: {FAMILY_NAME[top[0]]} + {FAMILY_NAME[top[1]]}")
        else:
            out.append(FAMILY_NAME[top[0]])
    return out


# ------------------------------------------------------------------ data


def derive_extremes(df: pd.DataFrame) -> pd.DataFrame:
    """Best- and worst-formulation graded accuracy per cell, from the persisted rates."""
    rates = df["f_graded_per_paraphrase"].apply(lambda r: np.asarray(r, dtype=float))
    return df.assign(
        f_max=rates.apply(lambda r: float(r.max()) if r.size else np.nan),
        f_min=rates.apply(lambda r: float(r.min()) if r.size else np.nan),
    )


def load_strata(root: Path) -> list[tuple[str, pd.DataFrame]]:
    """The six model x level strata with tvd_sens, f_max and f_min added."""
    out = []
    for m in _MODELS:
        d = pd.read_parquet(root / f"data/specificity_v3_{m}.parquet")
        if "tvd_sens" not in d.columns and "consistency_mean" in d.columns:
            d = d.assign(tvd_sens=1.0 - d["consistency_mean"])
        d = derive_extremes(d)
        for lvl in sorted(d["spec_level"].unique()):
            out.append((f"{m} L{lvl}", d[d.spec_level == lvl]))
    return out


def mean_corr(strata: list[tuple[str, pd.DataFrame]], cols: list[str]) -> np.ndarray:
    """Element-wise mean over strata of the within-stratum pairwise Spearman matrix.

    Same recipe as make_metric_corr.py / factor_audit.py; correlations are never
    pooled across the specificity levels.
    """
    mats = []
    for _, sub in strata:
        M = np.full((len(cols), len(cols)), np.nan)
        for i, a in enumerate(cols):
            for j, b in enumerate(cols):
                x, y = sub[a], sub[b]
                ok = x.notna() & y.notna()
                if int(ok.sum()) > 5 and x[ok].nunique() > 1 and y[ok].nunique() > 1:
                    M[i, j] = stats.spearmanr(x[ok], y[ok])[0]
        mats.append(M)
    return np.nanmean(np.stack(mats), axis=0)


def provenance_guard(root: Path, C: np.ndarray, cols: list[str], tol: float = 1e-9) -> float:
    """The 14 metrics shared with figures/v3_metric_corr.npy must reproduce it."""
    old = np.load(root / "figures/v3_metric_corr.npy")
    meta = json.loads((root / "figures/v3_metric_corr_labels.json").read_text(encoding="utf-8"))
    idx = [cols.index(c) for c in meta["metrics"]]
    diff = float(np.nanmax(np.abs(C[np.ix_(idx, idx)] - old)))
    if diff > tol:
        raise RuntimeError(
            f"matrix recipe drifted from figures/v3_metric_corr.npy (max diff {diff})"
        )
    return diff


def stratum_cross_check(
    strata: list[tuple[str, pd.DataFrame]],
    dec: Decomposition,
    active_cols: list[str],
    probe_cols: list[str],
) -> pd.DataFrame:
    """Empirical check of the projection: score each stratum with the pooled weights.

    Within a stratum the active metrics are rank-transformed and standardized,
    scored with the pooled eigenvectors and rotation, and each probe metric is
    Spearman-correlated with every component score on the cells where the probe
    is defined (rho_F is complete-case). The pooled matrix is a mean over
    strata, so the two need not coincide exactly; agreement is the check.
    """
    weights = (dec.evecs[:, : dec.k] / np.sqrt(dec.evals[: dec.k])) @ dec.rotation
    rows = []
    for name, sub in strata:
        Z = sub[active_cols].rank().to_numpy(dtype=float)
        Z = (Z - Z.mean(axis=0)) / Z.std(axis=0, ddof=1)
        T = Z @ weights
        for c in probe_cols:
            x = sub[c].to_numpy(dtype=float)
            ok = np.isfinite(x)
            row: dict = {"stratum": name, "metric": c, "n": int(ok.sum())}
            for j in range(dec.k):
                row[f"C{j + 1}"] = float(stats.spearmanr(x[ok], T[ok, j])[0])
            rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ rendering


def _fmt(x: float) -> str:
    return f"{x:+.2f}"


def _loading_row(name: str, fam: str, v: np.ndarray, comp_fams: list[str]) -> str:
    a = np.abs(v)
    dom = int(a.argmax())
    margin = float(a[dom] - np.sort(a)[-2]) if len(a) > 1 else float(a[dom])
    match = "yes" if comp_fams[dom] == fam else "NO"
    cells = " | ".join(f"**{_fmt(x)}**" if j == dom else _fmt(x) for j, x in enumerate(v))
    return f"| {name} | {FAMILY_NAME[fam]} | {cells} | C{dom + 1} ({match}) | {margin:.2f} |"


def _loading_table(
    dec: Decomposition,
    family_of: dict[str, str],
    comp_fams: list[str],
    sup: dict[str, np.ndarray],
) -> list[str]:
    titles = component_titles(dec, family_of)
    heads = " | ".join(f"C{j + 1} ({titles[j]})" for j in range(dec.k))
    L = [
        f"| metric | a priori factor | {heads} | dominant | margin |",
        "|---|---|" + "---|" * dec.k + "---|---|",
    ]
    for i, n in enumerate(dec.names):
        L.append(_loading_row(PRETTY[n], family_of[n], dec.loadings[i], comp_fams))
    if sup:
        L.append(
            "| *held out (supplementary)* | | " + " | ".join("" for _ in range(dec.k)) + " | | |"
        )
        for n, v in sup.items():
            L.append(_loading_row(PRETTY[n] + " †", family_of[n], v, comp_fams))
    return L


def _evals(v: np.ndarray, n: int = 4) -> str:
    return ", ".join(f"{x:.2f}" for x in v[:n])


def render(res: dict) -> str:
    s1, s1k2, s2 = res["stage1_k3"], res["stage1_k2"], res["stage2"]
    d1, d2 = s1["dec"], s2["dec"]
    fam = res["family_of"]
    L = [
        "# Metric selection — published metrics first, ρ_F projected, then added (generated)",
        "",
        "The component analysis that justifies the three factors must not contain the metric it",
        "justifies. **Stage 1** decomposes the ten published metrics only (mean within-stratum Spearman",
        "matrix, six model × level strata) and projects the six constructed metrics into that space as",
        "*supplementary* variables: a held-out metric's loading is its correlation with the component",
        "score, computed without letting it shape the component. **Stage 2** adds ρ_F as an active",
        "variable and repeats the retention test. The earlier fourteen-metric decomposition remains in",
        "`data/factor_audit.md` and is repeated in §7 as a sensitivity variant.",
        "",
        "## 1. The metric sets",
        "",
        "| metric | a priori factor | source | role |",
        "|---|---|---|---|",
    ]
    for _, lab, f, src in PUBLISHED:
        L.append(f"| {PRETTY[lab]} | {FAMILY_NAME[f]} | {src} | active (published) |")
    for _, lab, f in CONSTRUCTED:
        L.append(f"| {PRETTY[lab]} | {FAMILY_NAME[f]} | this work | held out in stage 1 |")
    L += [
        "",
        "F_max and F_min are the best- and worst-formulation graded accuracies of a cell, computed",
        "from the persisted per-formulation rates (`f_graded_per_paraphrase`), the same basis as",
        "accuracy and ρ_F; they were added to the audit on 2026-09-16 (FORKING_PATHS fork 14). The",
        "persisted `spread` keeps the pipeline's original scoring rule (Spearman 0.66–0.85 with",
        "F_max − F_min on graded rates); a graded-spread variant is in §7. POSIX (Chatterjee et al.",
        "2024) is measured on a separate 100-cell arm and is analyzed in `data/metric_reductions.md`.",
        "Provenance guard: the 14 metrics shared with `figures/v3_metric_corr.npy` reproduce it to",
        f"{res['guard_diff']:.1e}.",
        "",
        "## 2. Stage 1 — how many components do the published metrics support?",
        "",
        f"Eigenvalues: {[round(float(x), 3) for x in d1.evals]}",
        "",
        f"Horn 95th percentile (random data, n = {res['n_obs']} per stratum, 500 draws): "
        f"{[round(float(x), 3) for x in d1.horn95]}",
        "",
        f"**Horn retains {d1.n_retained}.** The third eigenvalue ({d1.evals[2]:.3f}) falls below its parallel",
        f"threshold ({d1.horn95[2]:.3f}); at n = 56 to 150 and ten seeds the count is always "
        f"{sorted(res['stage1_counts'])}. The first two components explain {s1k2['share']:.1%} of the",
        f"variance, the first three {s1['share']:.1%}. The two published formulation-dependence metrics",
        f"correlate {res['r_spread_rho_u']:+.2f} with each other, too little to carry a component of their own.",
        "",
        "### The two retained components (varimax)",
        "",
    ]
    L += _loading_table(s1k2["dec"], fam, s1k2["families"], s1k2["sup"])
    L += [
        "",
        "## 3. Stage 1 — the three-component solution the framework posits",
        "",
        "Three causes of a wrong answer call for three components; extracting three from the",
        "published metrics gives each a priori family its own component. Loadings are correlations",
        "with the varimax-rotated component scores; † marks held-out (supplementary) metrics; the",
        "margin is the dominant |loading| minus the next largest, a measure of how specific a metric",
        "is to one component.",
        "",
    ]
    L += _loading_table(d1, fam, s1["families"], s1["sup"])
    L += ["", "## 4. Representative per component (stage 1)", ""]
    for j in range(3):
        L.append(
            f"**C{j + 1} = {component_titles(d1, fam)[j]}.** " + res["representative_notes"][j]
        )
        L.append("")
    L += [
        "## 5. Stage 2 — ρ_F added as an active variable",
        "",
        f"Eigenvalues: {[round(float(x), 3) for x in d2.evals]}",
        "",
        f"Horn 95th percentile (n = {res['n_obs']}): {[round(float(x), 3) for x in d2.horn95]}",
        "",
        f"**Horn retains {d2.n_retained}.** The third eigenvalue ({d2.evals[2]:.3f}) exceeds its threshold at",
        f"every n from 56 to 150 and every seed (counts {sorted(res['stage2_counts'])}; thresholds "
        + ", ".join(f"{t:.2f} at n = {n}" for n, t in res["stage2_thresholds"])
        + f"). The first three explain {s2['share']:.1%}. Loadings of the published metrics move by at most",
        f"{res['max_loading_shift']:.2f} relative to stage 1.",
        "",
    ]
    L += _loading_table(d2, fam, s2["families"], {})
    L += [
        "",
        "## 6. Empirical cross-check of the projection (per stratum, stage-1 components)",
        "",
        "Each stratum is scored with the pooled weights (rank-standardized active metrics); the table",
        "gives the Spearman correlation of each probe metric with the component scores, on the cells",
        "where the probe is defined (ρ_F is complete-case; n in parentheses).",
        "",
        "| stratum | "
        + " | ".join(
            f"{PRETTY[res['label_of'][c]]} ~ C{j + 1}" for c in res["probe_cols"] for j in range(3)
        )
        + " |",
        "|---|" + "---|" * (3 * len(res["probe_cols"])),
    ]
    xc = res["cross_check"]
    for s, g in xc.groupby("stratum", sort=False):
        cells = []
        for c in res["probe_cols"]:
            r = g[g.metric == c].iloc[0]
            for j in range(3):
                cell = f"{r[f'C{j + 1}']:+.2f}"
                if j == 0 and c == "rho_f":
                    cell += f" (n={int(r['n'])})"
                cells.append(cell)
        L.append(f"| {s} | " + " | ".join(cells) + " |")
    means = xc.groupby("metric")[["C1", "C2", "C3"]].mean()
    L.append(
        "| **mean** | "
        + " | ".join(
            f"**{means.loc[c, f'C{j + 1}']:+.2f}**" for c in res["probe_cols"] for j in range(3)
        )
        + " |"
    )
    L += [
        "",
        "## 7. Sensitivity of the count and of the key loadings",
        "",
        "| variant | p | eigenvalues 1–4 | Horn retains | ρ_F on C_dep | spread on C_dep / C_succ "
        "| ρ_u on C_dep | H_sem on C_disp | accuracy on C_succ |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for v in res["variants"]:
        L.append(
            f"| {v['name']} | {v['p']} | {v['evals4']} | {v['n_retained']} | {v['rho_f']} | "
            f"{v['spread']} | {v['rho_u']} | {v['h_sem']} | {v['accuracy']} |"
        )
    L += [
        "",
        "Loadings marked † are supplementary projections; the others are active loadings. Without",
        "F_max and F_min the published set has a single success metric, and a component needs several",
        "indicators to be retained; with them, success is retained and formulation dependence is the",
        "one factor the published metrics do not carry. With all fourteen earlier metrics Horn retains",
        "three because AUFI (an exact alias of accuracy) and the FI premium add success indicators and",
        "ρ_F a dependence indicator; that count is superseded by the two-stage result above.",
        "",
        "## 8. Conventions",
        "",
        "- Matrix: mean over the six model × level strata of the within-stratum pairwise Spearman",
        "  correlation (recipe of `make_metric_corr.py`); correlations never pool across levels.",
        f"- Horn: 500 random normal data sets of n × p, 95th percentile per eigenvalue, seed 42; n = {res['n_obs']}",
        "  (every published metric is defined on all 150 cells of every stratum). For stage 2, ρ_F is",
        "  complete-case (56 to 117 cells per stratum), so the count is also reported at n = 56 and 84.",
        "- Rotation: varimax (Kaiser 1958) on the first k components; each component's sign is fixed so",
        "  that its largest |loading| is positive. Supplementary loadings use the same rotation.",
        "",
    ]
    return "\n".join(L)


# ------------------------------------------------------------------ driver


def _key_loadings(
    name: str,
    C: np.ndarray,
    cols: list[str],
    labels: list[str],
    active: list[str],
    family_of: dict[str, str],
    n_obs: int,
) -> dict:
    """One sensitivity variant: decomposition on `active`, key loadings on the family components."""
    names = [labels[cols.index(c)] for c in active]
    dec = decompose(
        C[np.ix_([cols.index(c) for c in active], [cols.index(c) for c in active])],
        names,
        n_obs,
        k=3,
    )
    fams = component_families(dec, family_of)

    def loading(label: str, fam: str) -> str:
        if fam not in fams or label not in labels:
            return "n/a"
        j = fams.index(fam)
        if label in names:
            return _fmt(dec.loadings[names.index(label), j])
        r = C[labels.index(label), [cols.index(c) for c in active]]
        return _fmt(project(dec, r)[j]) + " †"

    return {
        "name": name,
        "p": len(active),
        "evals4": _evals(dec.evals),
        "n_retained": dec.n_retained,
        "families": fams,
        "rho_f": loading(RHO_F, "dependence"),
        "spread": loading("spread (Cao)", "dependence")
        + " / "
        + loading("spread (Cao)", "success"),
        "rho_u": loading("rho_u (Cox)", "dependence"),
        "h_sem": loading("H_sem", "dispersion"),
        "accuracy": loading("accuracy", "success"),
    }


def run(root: Path, n_obs: int = _N_PER_STRATUM) -> dict:
    strata = load_strata(root)
    family_of = {lab: f for _, lab, f, _ in PUBLISHED} | {lab: f for _, lab, f in CONSTRUCTED}
    label_of = {c: lab for c, lab, _, _ in PUBLISHED} | {c: lab for c, lab, _ in CONSTRUCTED}
    cols = [c for c, _, _, _ in PUBLISHED] + [c for c, _, _ in CONSTRUCTED]
    labels = [label_of[c] for c in cols]
    C = mean_corr(strata, cols)
    guard_diff = provenance_guard(root, C, cols)

    pub_cols = [c for c, _, _, _ in PUBLISHED]
    pub_labels = [label_of[c] for c in pub_cols]
    pub_idx = [cols.index(c) for c in pub_cols]
    Cp = C[np.ix_(pub_idx, pub_idx)]

    stage1 = {}
    for k in (2, 3):
        dec = decompose(Cp, pub_labels, n_obs=n_obs, k=k)
        sup = {lab: project(dec, C[labels.index(lab), pub_idx]) for _, lab, _ in CONSTRUCTED}
        stage1[k] = {
            "dec": dec,
            "share": float(dec.evals[:k].sum() / dec.evals.sum()),
            "families": component_families(dec, family_of),
            "sup": sup,
        }
    d1 = stage1[3]["dec"]
    stage1_counts = {
        int(np.sum(d1.evals > horn(Cp, n, seed=s))) for n in _RHO_F_N for s in range(1, 11)
    }

    # stage 2: rho_F active
    s2_cols = pub_cols + ["rho_f"]
    s2_idx = [cols.index(c) for c in s2_cols]
    C2 = C[np.ix_(s2_idx, s2_idx)]
    d2 = decompose(C2, [label_of[c] for c in s2_cols], n_obs=n_obs, k=3)
    stage2 = {
        "dec": d2,
        "share": float(d2.evals[:3].sum() / d2.evals.sum()),
        "families": component_families(d2, family_of),
    }
    stage2_counts = {
        int(np.sum(d2.evals > horn(C2, n, seed=s))) for n in _RHO_F_N for s in range(1, 11)
    }
    stage2_thresholds = [(n, float(horn(C2, n, seed=42)[2])) for n in _RHO_F_N]
    # loading shift of the published metrics between the two three-component solutions,
    # after matching components by family
    order = [stage2["families"].index(f) for f in stage1[3]["families"]]
    shift = np.abs(d2.loadings[: len(pub_cols)][:, order] - d1.loadings)
    max_loading_shift = float(shift.max())

    # representative notes, one per stage-1 component, written from the numbers
    fams3, sup3 = stage1[3]["families"], stage1[3]["sup"]
    notes = []
    for j in range(3):
        f = fams3[j]
        members = sorted(
            [(n, d1.loadings[i, j]) for i, n in enumerate(pub_labels) if family_of[n] == f],
            key=lambda t: -abs(t[1]),
        )
        best, best_v = members[0]
        cross = max(
            [(n, d1.loadings[i, j]) for i, n in enumerate(pub_labels) if family_of[n] != f],
            key=lambda t: abs(t[1]),
        )
        held = [(n, sup3[n][j]) for n in sup3 if family_of[n] == f]
        text = f"Best published metric: {PRETTY[best]} ({_fmt(best_v)})"
        text += (
            "; other members: " + ", ".join(f"{PRETTY[n]} {_fmt(v)}" for n, v in members[1:])
            if len(members) > 1
            else "; it is the only published metric of this factor"
        )
        text += f". Largest loading from another family: {PRETTY[cross[0]]} ({_fmt(cross[1])})."
        if held:
            text += " Held-out metrics of this factor: " + ", ".join(
                f"{PRETTY[n]} {_fmt(v)} (elsewhere "
                + ", ".join(_fmt(sup3[n][m]) for m in range(3) if m != j)
                + ")"
                for n, v in held
            )
            text += "."
        notes.append(text)

    probe_cols = ["rho_f", "spread", "rho_u"]
    xc = stratum_cross_check(strata, d1, pub_cols, probe_cols)

    # graded-spread variant: replace the pipeline's spread by F_max - F_min on graded rates
    strata_g = [(n, s.assign(spread=s["f_max"] - s["f_min"])) for n, s in strata]
    Cg = mean_corr(strata_g, cols)
    pub8 = [c for c in pub_cols if c not in ("f_max", "f_min")]
    variants = [
        _key_loadings(
            "stage 1: ten published metrics", C, cols, labels, pub_cols, family_of, n_obs
        ),
        _key_loadings("stage 2: ten published + ρ_F", C, cols, labels, s2_cols, family_of, n_obs),
        _key_loadings(
            "eight published (without F_max, F_min)", C, cols, labels, pub8, family_of, n_obs
        ),
        _key_loadings("eight published + ρ_F", C, cols, labels, pub8 + ["rho_f"], family_of, n_obs),
        _key_loadings(
            "stage 2 with spread on graded rates", Cg, cols, labels, s2_cols, family_of, n_obs
        ),
        _key_loadings(
            "stage 2 + AUFI, FI premium (accuracy aliases)",
            C,
            cols,
            labels,
            s2_cols + ["aufi_in_graded", "fi_premium"],
            family_of,
            n_obs,
        ),
        _key_loadings(
            "all fourteen earlier metrics (factor_audit.md, n = 136)",
            C,
            cols,
            labels,
            [c for c in cols if c not in ("f_max", "f_min")],
            family_of,
            136,
        ),
    ]
    return {
        "matrix": C,
        "cols": cols,
        "labels": labels,
        "guard_diff": guard_diff,
        "family_of": family_of,
        "label_of": label_of,
        "n_obs": n_obs,
        "stage1_k2": stage1[2],
        "stage1_k3": stage1[3],
        "stage1_counts": stage1_counts,
        "stage2": stage2,
        "stage2_counts": stage2_counts,
        "stage2_thresholds": stage2_thresholds,
        "max_loading_shift": max_loading_shift,
        "r_spread_rho_u": float(C[cols.index("spread"), cols.index("rho_u")]),
        "representative_notes": notes,
        "probe_cols": probe_cols,
        "cross_check": xc,
        "variants": variants,
    }


def to_json(res: dict) -> dict:
    def sol(d: dict) -> dict:
        dec = d["dec"]
        return {
            "eigenvalues": [round(float(x), 4) for x in dec.evals],
            "horn95": [round(float(x), 4) for x in dec.horn95],
            "n_retained": dec.n_retained,
            "explained_share": round(d["share"], 4),
            "component_families": d["families"],
            "loadings": {
                n: [round(float(x), 3) for x in dec.loadings[i]] for i, n in enumerate(dec.names)
            },
            "supplementary": {
                n: [round(float(x), 3) for x in v] for n, v in d.get("sup", {}).items()
            },
        }

    xc = res["cross_check"]
    return {
        "published": [
            {"column": c, "label": lab, "family": f, "source": s} for c, lab, f, s in PUBLISHED
        ],
        "constructed": [{"column": c, "label": lab, "family": f} for c, lab, f in CONSTRUCTED],
        "matrix": {
            "labels": res["labels"],
            "columns": res["cols"],
            "values": np.round(res["matrix"], 4).tolist(),
        },
        "n_obs": res["n_obs"],
        "stage1_two_components": sol(res["stage1_k2"]),
        "stage1_three_components": sol(res["stage1_k3"]),
        "stage1_counts_over_n_and_seeds": sorted(res["stage1_counts"]),
        "stage2_three_components": sol(res["stage2"]),
        "stage2_counts_over_n_and_seeds": sorted(res["stage2_counts"]),
        "stage2_third_threshold_by_n": [
            {"n": n, "horn95_3": round(t, 4)} for n, t in res["stage2_thresholds"]
        ],
        "max_loading_shift_stage1_to_stage2": round(res["max_loading_shift"], 4),
        "cross_check_mean": {
            c: [round(float(v), 3) for v in xc[xc.metric == c][["C1", "C2", "C3"]].mean()]
            for c in res["probe_cols"]
        },
        "cross_check_rows": xc.round(3).to_dict(orient="records"),
        "variants": res["variants"],
    }


def main() -> int:
    root = load_config().repo_root()
    res = run(root)
    md = render(res)
    (root / "data/metric_selection.md").write_text(md, encoding="utf-8")
    (root / "data/metric_selection.json").write_text(
        json.dumps(to_json(res), indent=1, ensure_ascii=False), encoding="utf-8"
    )
    print(md)
    logger.info("wrote data/metric_selection.md + .json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
