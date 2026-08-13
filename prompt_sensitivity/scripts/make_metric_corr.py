"""R9 — the (previously missing) generator for figures/v3_metric_corr.npy.

The independence figure and the factor analysis both consume
figures/v3_metric_corr.npy + v3_metric_corr_labels.json, but no script in the
repository produced them (review 2026-08-06 §3.10 / F10). This script is now
the canonical recipe:

  RECIPE. For each of the 6 (model x specificity-level) strata of the v3 grid,
  compute the pairwise-complete Spearman correlation matrix over the 14 labeled
  metrics (tvd_sens := 1 - consistency_mean, the Errica consistency complement;
  every other column is read as persisted). The published matrix is the
  ELEMENT-WISE MEAN over the 6 strata (NaN-aware): correlations are computed
  within model and within level, never pooled across the manipulated variable
  (main_body.tex §Aggregation).

  PROVENANCE NOTE. The archived matrix (created ~2026-07-27 by an untracked
  one-off) is reproduced by this recipe to max |diff| <= 0.009 on common cells
  — rounding-level for Spearman at n = 80-150, but not bit-exact; the archived
  file also left `spread` all-NaN where this recipe derives it. On first run
  the archived file is preserved as *_archived_20260727.npy and the canonical,
  script-generated matrix replaces it. Downstream consumers (fig_independence,
  paper_analyses_b.factor_structure) drop all-NaN variables and are unaffected
  beyond the third decimal.

    uv run python -m prompt_sensitivity.scripts.make_metric_corr
"""

from __future__ import annotations

import json
import shutil
import sys

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from ..config import load_config

_MODELS = ["qwen_2_5_7b", "llama_3_1_8b", "mistral_7b_v03"]

# column -> display label, in the figure's canonical order
_METRICS: list[tuple[str, str]] = [
    ("f_graded_mean", "accuracy"),
    ("aufi_in_graded", "AUFI (graded)"),
    ("rho_f", "rho_F  [M1]"),
    ("fi_premium", "FI premium  [M2]"),
    ("spread", "spread (Cao)"),
    ("h_sem_mean", "H_sem"),
    ("fi_out_fixed", "FI_out_fixed"),
    ("fi_out_var", "Var[FI_out]  [M4]"),
    ("tvd_sens", "TVD-sens  [M4]"),
    ("s_tau_mean", "S_tau (Errica)"),
    ("variation_ratio", "variation ratio"),
    ("a_q", "|A_q| observed"),
    ("rho_u", "rho_u (Cox)"),
    ("ess_in", "ESS_in"),
]


def build_matrix(config) -> tuple[np.ndarray, list[str], list[str]]:
    cols = [c for c, _ in _METRICS]
    mats = []
    for m in _MODELS:
        d = pd.read_parquet(config.repo_root() / f"data/specificity_v3_{m}.parquet")
        if "tvd_sens" not in d.columns and "consistency_mean" in d.columns:
            d = d.assign(tvd_sens=1.0 - d["consistency_mean"])
        for lvl in sorted(d["spec_level"].unique()):
            sub = d[d.spec_level == lvl]
            M = np.full((len(cols), len(cols)), np.nan)
            for i, a in enumerate(cols):
                for j, b in enumerate(cols):
                    if a not in sub.columns or b not in sub.columns:
                        continue
                    x, y = sub[a], sub[b]
                    ok = x.notna() & y.notna()
                    if int(ok.sum()) > 5:
                        M[i, j] = stats.spearmanr(x[ok], y[ok])[0]
            mats.append(M)
    mean = np.nanmean(np.stack(mats), axis=0)
    return mean, cols, [lab for _, lab in _METRICS]


def main() -> int:
    config = load_config()
    out_npy = config.repo_root() / "figures/v3_metric_corr.npy"
    out_json = config.repo_root() / "figures/v3_metric_corr_labels.json"

    mean, cols, labels = build_matrix(config)

    if out_npy.exists():
        old = np.load(out_npy)
        both = np.isfinite(old) & np.isfinite(mean)
        max_diff = float(np.abs(mean - old)[both].max()) if both.any() else float("nan")
        logger.info("existing matrix: max |diff| on common cells = {:.4f} "
                    "(finite: old {}, new {})", max_diff,
                    int(np.isfinite(old).sum()), int(np.isfinite(mean).sum()))
        archive = out_npy.with_name("v3_metric_corr_archived_20260727.npy")
        if not archive.exists():
            shutil.copy2(out_npy, archive)
            logger.info("archived untracked original -> {}", archive.name)
        if not np.isnan(max_diff) and max_diff > 0.02:
            logger.error("reproduction diverges (> .02) — NOT overwriting; investigate")
            return 1

    np.save(out_npy, mean)
    out_json.write_text(json.dumps({"metrics": cols, "labels": labels}, indent=1),
                        encoding="utf-8")
    logger.info("wrote {} + labels ({} metrics, mean over 6 model x level strata)",
                out_npy.name, len(cols))
    return 0


if __name__ == "__main__":
    sys.exit(main())
