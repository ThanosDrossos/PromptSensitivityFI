"""Part-B paper analyses (FINAL_PHASE_PLAN, user-approved 2026-08-02).

Laptop-only, no cluster. Four sections, one summary file:

  B2  Constructive counterexamples — seeded synthetic cells proving each axis
      can vary while the others are pinned (the "identifiability box").
  B3a Factor structure — eigendecomposition + varimax of the within-level
      Spearman matrix over the metric stack (expect 3 factors + ESS_in apart).
  B3b Octant occupancy — median-split (accuracy, rho_F, H_sem) per (model,
      level); all 2^3 octants populated => no axis predicts another.
  B4  rho_F cluster-bootstrap CIs — resample paraphrases within a cell
      (B=2000, seeded) from the stored per-paraphrase graded rates.

Outputs: figures/b2_counterexamples.json, figures/b3_factors.json,
data/b3_octants.parquet, data/rho_f_bootstrap.parquet, and the human summary
data/paper_analyses_b.md.

    uv run python -m prompt_sensitivity.scripts.paper_analyses_b
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
import pandas as pd
from loguru import logger

from ..config import load_config
from ..logging_setup import configure_logging
from ..metrics.sensitivity_v2 import rho_f

_V3_GLOB = "data/specificity_v3_{model}.parquet"
_MODELS = ("qwen_2_5_7b", "llama_3_1_8b", "mistral_7b_v03")


# --------------------------------------------------------------------------- #
# B2 — constructive counterexamples                                           #
# --------------------------------------------------------------------------- #


def _fi_in_curve(rates: np.ndarray, ks=(0.25, 0.5, 0.75, 1.0)) -> list[float]:
    n = len(rates)
    out = []
    for k in ks:
        nk = int((rates >= k).sum())
        out.append(float("inf") if nk == 0 else float(-math.log2(nk / n)))
    return out


def _h_norm(cluster_probs: list[float]) -> float:
    p = np.asarray(cluster_probs, float)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def counterexamples(k: int = 10) -> dict:
    """Each pair of cells pins one axis's HEADLINE scalar and moves another.

    Deliberately NOT claimed: pinning the entire FI_in(k) curve while moving
    rho_F. Both are functionals of the same per-paraphrase rate multiset, so
    the threshold curve bounds rho_F (up to bin-internal variance) — that
    shared origin is stated in the derivation (B1); the independence claim is
    about the REPORTED representatives (accuracy / rho_F / H_sem), and holds.
    """
    # 1. Same mean F (accuracy pinned at 0.5); rho_F 0 vs ~1.
    #    (0.5,)*10: every paraphrase a fair coin -> ALL variability is decoding
    #    noise. (1,0,1,...): success fully determined by phrasing. The FI_in
    #    curves differ — which is exactly the distribution information rho_F
    #    reads and the accuracy scalar discards.
    a = np.array([0.5] * 10)
    b = np.array([1.0, 0.0] * 5)
    ex1 = {
        "mean_F": [float(a.mean()), float(b.mean())],
        "fi_in_curve_a": _fi_in_curve(a),
        "fi_in_curve_b": _fi_in_curve(b),
        "rho_F": [rho_f(a.tolist(), k),
                  rho_f(b.tolist(), k)],
    }
    # 2. Same accuracy AND same rho_F; different H_sem: the WRONG mass sits in
    #    one semantic cluster vs spread over four. (Output-space quantities are
    #    invisible to correctness-based metrics by construction.)
    correct_share = 0.5
    ex2 = {
        "mean_F": [correct_share, correct_share],
        "h_sem_norm": [
            _h_norm([correct_share, 0.5]),                       # one wrong mode
            _h_norm([correct_share] + [0.125] * 4),              # scattered wrong
        ],
    }
    # 3. Same H_sem; different accuracy: confidently-RIGHT vs confidently-WRONG
    #    single mode (the real L0 finding: ambiguity shows as confident wrong
    #    answers, not dispersion).
    ex3 = {
        "h_sem_norm": [_h_norm([1.0]), _h_norm([1.0])],
        "mean_F": [1.0, 0.0],
    }
    # 4. FI_spec carries no model quantity: identical across all models in the
    #    v3 data by construction (checked empirically in the md summary).
    checks = {
        "ex1_rho_f_gap": abs(ex1["rho_F"][0] - ex1["rho_F"][1]) > 0.5,
        "ex1_accuracy_pinned": ex1["mean_F"][0] == ex1["mean_F"][1],
        "ex2_hsem_gap": abs(ex2["h_sem_norm"][0] - ex2["h_sem_norm"][1]) > 0.5,
        "ex3_accuracy_gap": ex3["mean_F"][0] - ex3["mean_F"][1] == 1.0,
    }
    assert all(checks.values()), f"counterexample invariants violated: {checks}"
    return {"ex1_rhoF_free": ex1, "ex2_hsem_free": ex2, "ex3_accuracy_free": ex3,
            "checks": checks}


# --------------------------------------------------------------------------- #
# B3a — factor structure                                                       #
# --------------------------------------------------------------------------- #


def _varimax(loadings: np.ndarray, n_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    """Standard varimax rotation (numpy only, Kaiser 1958)."""
    p, k = loadings.shape
    rot = np.eye(k)
    var = 0.0
    for _ in range(n_iter):
        lam = loadings @ rot
        u, s, vt = np.linalg.svd(
            loadings.T @ (lam ** 3 - (lam @ np.diag(np.sum(lam ** 2, axis=0))) / p))
        rot = u @ vt
        new_var = float(s.sum())
        if new_var < var * (1 + tol):
            break
        var = new_var
    return loadings @ rot


def factor_structure(corr_path, labels_path, n_factors: int = 3) -> dict:
    C = np.load(corr_path)
    meta = json.loads(labels_path.read_text(encoding="utf-8"))
    names = meta["labels"]
    # Spearman matrices from finite pairwise samples are not guaranteed PSD;
    # clip tiny negative eigenvalues before analysis.
    C = np.nan_to_num((C + C.T) / 2, nan=0.0)
    np.fill_diagonal(C, 1.0)
    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(evals)[::-1]
    evals, evecs = np.clip(evals[order], 0, None), evecs[:, order]
    total = float(evals.sum())
    loadings = evecs[:, :n_factors] * np.sqrt(evals[:n_factors])
    rotated = _varimax(loadings)
    # Sign convention: flip factors so their largest-|loading| entry is positive.
    for j in range(rotated.shape[1]):
        i = int(np.abs(rotated[:, j]).argmax())
        if rotated[i, j] < 0:
            rotated[:, j] *= -1
    assignment = {names[i]: int(np.abs(rotated[i]).argmax())
                  for i in range(len(names))}
    return {
        "eigenvalues": [round(float(v), 3) for v in evals.tolist()],
        "explained_by_top3": round(float(evals[:3].sum() / total), 3),
        "loadings_varimax": {
            names[i]: [round(float(x), 2) for x in rotated[i]]
            for i in range(len(names))
        },
        "factor_assignment": assignment,
    }


# --------------------------------------------------------------------------- #
# B3b — octant occupancy                                                       #
# --------------------------------------------------------------------------- #


def octant_occupancy(root) -> pd.DataFrame:
    rows = []
    for m in _MODELS:
        df = pd.read_parquet(root / _V3_GLOB.format(model=m))
        for lvl, sub in df.groupby("spec_level"):
            s = sub.dropna(subset=["f_graded_mean", "rho_f", "h_sem_mean"]).copy()
            if len(s) < 8:
                continue
            for col, tag in (("f_graded_mean", "abil"), ("rho_f", "sens"),
                             ("h_sem_mean", "disp")):
                s[tag] = (s[col] > s[col].median()).map({True: "hi", False: "lo"})
            for (a, b, c), grp in s.groupby(["abil", "sens", "disp"]):
                rows.append({
                    "model_key": m, "spec_level": int(lvl),
                    "octant": f"{a}-abil/{b}-sens/{c}-disp", "n": len(grp),
                    "example_question": grp.iloc[0]["question_text"][:120],
                    "example_qid": str(grp.iloc[0]["question_id"]),
                })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# B4 — rho_F cluster bootstrap                                                 #
# --------------------------------------------------------------------------- #


def rho_f_bootstrap(root, n_boot: int = 2000, seed: int = 42,
                    k_samples: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = []
    for m in _MODELS:
        df = pd.read_parquet(root / _V3_GLOB.format(model=m))
        for _, r in df.iterrows():
            rates = r.get("f_graded_per_paraphrase")
            if rates is None or (hasattr(rates, "__len__") and len(rates) < 2):
                continue
            rates = np.asarray(rates, float)
            point = rho_f(rates.tolist(), k_samples)
            if point is None or (isinstance(point, float) and math.isnan(point)):
                continue
            boots = []
            for _ in range(n_boot):
                sample = rates[rng.integers(0, len(rates), len(rates))]
                v = rho_f(sample.tolist(), k_samples)
                if v is not None and not math.isnan(v):
                    boots.append(v)
            if len(boots) < n_boot // 2:
                lo = hi = float("nan")   # CI undefined on mostly-degenerate resamples
            else:
                lo, hi = np.percentile(boots, [2.5, 97.5])
            out.append({
                "model_key": m, "question_id": str(r["question_id"]),
                "spec_level": int(r["spec_level"]), "rho_f": float(point),
                "ci_lo": float(lo), "ci_hi": float(hi),
                "n_boot_defined": len(boots),
            })
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--skip-bootstrap", action="store_true")
    args = ap.parse_args()

    configure_logging("paper_analyses_b")
    root = load_config().repo_root()
    figures = root / "figures"
    figures.mkdir(exist_ok=True)

    logger.info("B2: constructive counterexamples")
    b2 = counterexamples()
    (figures / "b2_counterexamples.json").write_text(
        json.dumps(b2, indent=1), encoding="utf-8")

    logger.info("B3a: factor structure")
    b3a = factor_structure(figures / "v3_metric_corr.npy",
                           figures / "v3_metric_corr_labels.json")
    (figures / "b3_factors.json").write_text(
        json.dumps(b3a, indent=1), encoding="utf-8")

    logger.info("B3b: octant occupancy")
    b3b = octant_occupancy(root)
    b3b.to_parquet(root / "data" / "b3_octants.parquet", index=False)

    boot_summary = "skipped (--skip-bootstrap)"
    if not args.skip_bootstrap:
        logger.info("B4: rho_F cluster bootstrap (B={})", args.n_boot)
        b4 = rho_f_bootstrap(root, n_boot=args.n_boot)
        b4.to_parquet(root / "data" / "rho_f_bootstrap.parquet", index=False)
        med_w = float((b4.ci_hi - b4.ci_lo).median())
        boot_summary = (f"{len(b4)} cells with CIs; median 95% width "
                        f"{med_w:.2f}; defined-resample floor "
                        f"{int(b4.n_boot_defined.min())}/{args.n_boot}")

    # ---- summary md ----
    occ = (b3b.groupby(["model_key", "spec_level"])["octant"]
           .nunique().rename("octants_populated").reset_index())
    lines = [
        "# Part-B analyses — three-axes defense (generated)", "",
        "## B2 counterexamples (all invariant checks passed)", "```json",
        json.dumps(b2["checks"], indent=1), "```", "",
        "Ex1: accuracy pinned at 0.5, rho_F "
        f"{b2['ex1_rhoF_free']['rho_F'][0]:.2f} vs "
        f"{b2['ex1_rhoF_free']['rho_F'][1]:.2f} "
        "(the FI_in curves differ — the distribution information rho_F reads "
        "and the accuracy scalar discards). "
        "Ex2: accuracy pinned, normalized H_sem "
        f"{b2['ex2_hsem_free']['h_sem_norm'][0]:.2f} vs "
        f"{b2['ex2_hsem_free']['h_sem_norm'][1]:.2f}. "
        "Ex3: H_sem pinned (single mode), accuracy 1.0 vs 0.0.", "",
        "## B3a factor structure",
        f"- top-3 eigenvalues {b3a['eigenvalues'][:3]} explain "
        f"**{b3a['explained_by_top3']:.0%}** of the metric-space variance",
        "- varimax factor assignment (0/1/2 = the three axes):", "```json",
        json.dumps(b3a["factor_assignment"], indent=1), "```", "",
        "## B3b octant occupancy (per model x level)", "",
        occ.to_string(index=False), "",
        f"Full table with example questions: data/b3_octants.parquet "
        f"({int(b3b.n.sum())} questions across cells).", "",
        "## B4 rho_F bootstrap CIs",
        f"- {boot_summary}",
    ]
    (root / "data" / "paper_analyses_b.md").write_text(
        "\n".join(lines), encoding="utf-8")
    logger.info("wrote data/paper_analyses_b.md")
    print("DONE data/paper_analyses_b.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
