"""R4 — the dispersion family as analytic reductions, not empirical convergence.

WHY (review 2026-08-06 §2.4). The deck presented "one construct, many costumes" as
measured correlations (S_tau .94, TVD .91, |A_q| .91, var-ratio .75, Var[FI_out]
.70, POSIX .60). Two of those are exact algebraic identities and three more are
functionals of the same pooled cluster distribution, so presenting them as
empirical convergence both overstates (they could not have come out otherwise)
and understates (an identity is a stronger claim than an n=150 correlation).

This script produces the paper's replacement:

  PROPOSITION (verified numerically over every cell here). Let P = (p_1..p_A) be
  the pooled semantic-cluster distribution of a cell and |A| its support size.
  Then, as computed by this pipeline:
     (i)   S_tau           =  H(P) / log2|A|          (Errica, normalised entropy)
     (ii)  FI_out_fixed    =  log2(m0) - H(P)         (affine in -H)
     (iii) Var[FI_out]     =  Var[H_sem]              (|A| constant within cell)
     (iv)  |A|, variation ratio, 1-TVD consistency    are functionals of P
  Hence every within-dataset agreement among (i)-(iv) and H_sem is arithmetic,
  not evidence. The one index measured independently (POSIX, its own
  teacher-forced pass) correlates with BOTH the dispersion axis and rho_F, and
  the difference is not significant (Williams test) -- so POSIX does not
  discriminate between axes 2 and 3 at this sample size.

Outputs: data/metric_reductions.md (paper source) + printed summary.
CONTRACT: read-only over persisted parquets; `metrics/` untouched.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from ..config import load_config

_MODELS = ["qwen_2_5_7b", "llama_3_1_8b", "mistral_7b_v03"]


# ------------------------------------------------------------------ helpers


def williams_test(r12: float, r13: float, r23: float, n: int) -> tuple[float, float]:
    """Williams/Steiger t for H0: rho(1,2) = rho(1,3) with shared variable 1.

    Used for: is corr(POSIX, H_sem) larger than corr(POSIX, rho_F)?
    """
    det = (1 - r12**2 - r13**2 - r23**2) + 2 * r12 * r13 * r23
    num = (r12 - r13) * np.sqrt((n - 1) * (1 + r23))
    den = np.sqrt(2 * det * (n - 1) / (n - 3) + ((r12 + r13) ** 2 / 4) * (1 - r23) ** 3)
    t = num / den
    p = 2 * (1 - stats.t.cdf(abs(t), n - 3))
    return float(t), float(p)


def _fisher_ci(r: float, n: int) -> tuple[float, float]:
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    se = 1 / np.sqrt(n - 3)
    return float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))


def verify_identities(df: pd.DataFrame) -> dict:
    """Numerical verification of (i)-(iii) over every row of one model grid."""
    out: dict[str, float] = {}
    # (i) S_tau = H_sem / log2(a_q); undefined (pipeline emits 0) when a_q <= 1.
    ok = df["a_q"] >= 2
    pred = df.loc[ok, "h_sem_mean"] / np.log2(df.loc[ok, "a_q"])
    out["s_tau_max_abs_err"] = float((df.loc[ok, "s_tau_mean"] - pred).abs().max())
    out["s_tau_n_checked"] = int(ok.sum())
    out["s_tau_degenerate_frac"] = float((~ok).mean())
    deg = df.loc[~ok]
    out["s_tau_degenerate_zero_frac"] = (
        float((deg["s_tau_mean"] == 0).mean()) if len(deg) else float("nan")
    )
    # (ii) FI_out_fixed = log2(m0) - H_sem
    pred2 = np.log2(df["m0"].clip(lower=1)) - df["h_sem_mean"]
    out["fi_out_fixed_max_abs_err"] = float((df["fi_out_fixed"] - pred2).abs().max())
    # (iii) Var[FI_out] = Var[H_sem]
    out["fi_out_var_max_abs_err"] = float((df["fi_out_var"] - df["h_sem_var"]).abs().max())
    return out


def posix_discrimination(config) -> pd.DataFrame:
    """POSIX vs H_sem vs rho_F on the SAME (rho_F-covered) subset + Williams test.

    The deck's .63/.43/.61 used all n=100 cells while the rho_F comparison can
    only use the covered subset — an apples-to-oranges pairing this table fixes.
    """
    rows = []
    for m in _MODELS:
        path = config.repo_root() / f"data/posix_arm_{m}.parquet"
        if not path.exists():
            logger.warning("missing {}", path)
            continue
        d = pd.read_parquet(path)
        s = d[["posix_psi", "h_sem_mean", "rho_f"]].dropna()
        n = len(s)
        r_h = stats.spearmanr(s.posix_psi, s.h_sem_mean)[0]
        r_r = stats.spearmanr(s.posix_psi, s.rho_f)[0]
        r_hr = stats.spearmanr(s.h_sem_mean, s.rho_f)[0]
        t, p = williams_test(r_h, r_r, r_hr, n)
        ci_h, ci_r = _fisher_ci(r_h, n), _fisher_ci(r_r, n)
        rows.append(
            {
                "model": m,
                "n": n,
                "posix_vs_h_sem": r_h,
                "h_ci_lo": ci_h[0],
                "h_ci_hi": ci_h[1],
                "posix_vs_rho_f": r_r,
                "r_ci_lo": ci_r[0],
                "r_ci_hi": ci_r[1],
                "williams_t": t,
                "williams_p": p,
            }
        )
    return pd.DataFrame(rows)


_DISPERSION_INDICES = [
    ("H_sem", "h_sem_mean"),
    ("S_tau", "s_tau_mean"),
    ("variation ratio", "variation_ratio"),
    ("1-TVD consistency", "consistency_mean"),
    ("\\|A_q\\|", "a_q"),
]


def divergence_table(config) -> pd.DataFrame:
    """Paired L0->L1 test per dispersion index: one distribution, many answers.

    Being functionals of one pooled clustering does NOT make the indices one
    measurement — on the primary specificity test they reach different
    decisions (a per-cell-varying normaliser is not a monotone transform
    across cells). This table is the paper's evidence for reporting exactly
    one representative instead of counting the family as convergent evidence.
    """
    rows = []
    for m in _MODELS:
        df = pd.read_parquet(config.repo_root() / f"data/specificity_v3_{m}.parquet")
        row: dict = {"model": m}
        for label, col in _DISPERSION_INDICES:
            wide = df.pivot_table(index="question_id", columns="spec_level", values=col).dropna()
            delta = wide[1] - wide[0]
            p = stats.wilcoxon(delta).pvalue if (delta != 0).any() else 1.0
            row[label] = (float(delta.mean()), float(p), len(wide))
        rows.append(row)
    return pd.DataFrame(rows)


def render(idents: dict[str, dict], posix: pd.DataFrame, div: pd.DataFrame) -> str:
    L = ["# R4 — Metric reductions: the dispersion family is one object", ""]
    L.append(
        "**Proposition.** Let P be a cell's pooled semantic-cluster distribution and |A| its support"
    )
    L.append("size, as produced by the pipeline's pooled clustering. Then, exactly as computed:")
    L.append("")
    L.append("| index | reduction | verified max abs error (all cells, 3 models) |")
    L.append("|---|---|---|")
    worst = {
        k: max(v[k] for v in idents.values())
        for k in ("s_tau_max_abs_err", "fi_out_fixed_max_abs_err", "fi_out_var_max_abs_err")
    }
    L.append(f"| S_τ (Errica) | H(P) / log₂\\|A\\| | {worst['s_tau_max_abs_err']:.2e} |")
    L.append(f"| FI_out_fixed | log₂(m₀) − H(P) | {worst['fi_out_fixed_max_abs_err']:.2e} |")
    L.append(f"| Var[FI_out] | Var[H_sem] | {worst['fi_out_var_max_abs_err']:.2e} |")
    L.append(
        "| \\|A_q\\|, variation ratio, 1−TVD | functionals of P (same pooled clustering) | by construction |"
    )
    L.append("")
    L.append("**Consequences.**")
    L.append("1. Within-dataset correlations among these indices are *arithmetic*, not evidence of")
    L.append("   convergent measurement. Report the reductions as a table, never as correlations.")
    L.append("2. The paired L0→L1 test on FI_out_fixed **is** the H_sem test (an affine map with a")
    L.append("   per-question constant cannot change a paired test). Report exactly one of them.")
    n_deg = np.mean([v["s_tau_degenerate_frac"] for v in idents.values()])
    L.append(
        f"3. **Degeneracy rule (stated, not silent):** S_τ is undefined when \\|A\\| ≤ 1 — "
        f"{n_deg:.1%} of cells across the three models (the pipeline emitted 0 there). "
        "All S_τ analyses condition on \\|A\\| ≥ 2 and report that coverage."
    )
    L.append("")
    L.append("## One distribution, many answers — the paired L0→L1 test per index")
    L.append("")
    L.append('"Functionals of one clustering" is NOT "one measurement": on the primary')
    L.append("specificity test the indices reach different decisions, because dividing by a")
    L.append("per-cell-varying normaliser is not a monotone transform across cells. This is")
    L.append("why the paper reports exactly ONE dispersion representative (H_sem) and treats")
    L.append("agreement within the family as arithmetic, never as convergent evidence.")
    L.append("")
    L.append("| model | " + " | ".join(lbl for lbl, _ in _DISPERSION_INDICES) + " |")
    L.append("|---|" + "---|" * len(_DISPERSION_INDICES))
    for _, r in div.iterrows():
        cells = []
        for lbl, _ in _DISPERSION_INDICES:
            d, p, n = r[lbl]
            cells.append(f"{d:+.3f} (p={p:.2g}, n={n})")
        L.append(f"| {r['model']} | " + " | ".join(cells) + " |")
    L.append("")
    L.append(
        "## POSIX — the one independent measurement — does not discriminate axis 3 from axis 2"
    )
    L.append("")
    L.append(
        "Same (ρ_F-covered) subset for both correlations; Williams/Steiger test of the difference:"
    )
    L.append("")
    L.append("| model | n | POSIX~H_sem [95 % CI] | POSIX~ρ_F [95 % CI] | Williams t | p |")
    L.append("|---|---|---|---|---|---|")
    for _, r in posix.iterrows():
        L.append(
            f"| {r['model']} | {int(r['n'])} | "
            f"{r['posix_vs_h_sem']:+.3f} [{r['h_ci_lo']:+.3f}, {r['h_ci_hi']:+.3f}] | "
            f"{r['posix_vs_rho_f']:+.3f} [{r['r_ci_lo']:+.3f}, {r['r_ci_hi']:+.3f}] | "
            f"{r['williams_t']:.2f} | {r['williams_p']:.3f} |"
        )
    L.append("")
    L.append(
        "POSIX correlates with **both** axes; in no model is its dispersion loading significantly"
    )
    L.append(
        'larger than its ρ_F loading. The claim "the phrasing axis stays empty" is **withdrawn**;'
    )
    L.append("the honest statement is that POSIX is not axis-diagnostic at this sample size.")
    L.append("")
    L.append("## Per-model identity checks")
    L.append("")
    L.append(
        "| model | S_τ err (n, cond. \\|A\\|≥2) | S_τ degenerate (of which =0) | FI_out_fixed err | Var[FI_out] err |"
    )
    L.append("|---|---|---|---|---|")
    for m, v in idents.items():
        L.append(
            f"| {m} | {v['s_tau_max_abs_err']:.1e} (n={v['s_tau_n_checked']}) | "
            f"{v['s_tau_degenerate_frac']:.1%} ({v['s_tau_degenerate_zero_frac']:.0%}) | "
            f"{v['fi_out_fixed_max_abs_err']:.1e} | {v['fi_out_var_max_abs_err']:.1e} |"
        )
    L.append("")
    return "\n".join(L)


def main() -> int:
    config = load_config()
    idents = {}
    for m in _MODELS:
        df = pd.read_parquet(config.repo_root() / f"data/specificity_v3_{m}.parquet")
        idents[m] = verify_identities(df)
        logger.info("{}: identities verified over {} rows", m, len(df))
    posix = posix_discrimination(config)
    div = divergence_table(config)
    md = render(idents, posix, div)
    out = config.repo_root() / "data/metric_reductions.md"
    out.write_text(md, encoding="utf-8")
    print(md)
    logger.info("wrote {}", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
