"""R2 driver — fit the partially-pooled rho_F on the v3 grid and persist it.

Produces, per model, a parquet with one row per cell carrying THREE estimators of
the same estimand so the paper can show the result under each:

    rho_f                 method-of-moments ICC, complete case (the v3 column;
                          NaN on zero-variance cells -> 45-66% coverage)
    rho_f_imputed0        the same, with 0 substituted on zero-variance cells
                          (the defensible alternative missingness rule)
    rho_f_hier            posterior mean of the hierarchical beta-binomial
                          (defined everywhere -> 100% coverage), plus its SD

and `sigma2_between`, the ABSOLUTE noise-corrected variance component, which is
comparable across models in a way the share is not.

Also writes posterior DRAWS (n_draws x n_cells) per model so R3 can propagate
per-cell uncertainty by multiple imputation instead of correlating shrunken
point estimates.

    uv run python -m prompt_sensitivity.scripts.fit_rho_f_hierarchical
    uv run python -m prompt_sensitivity.scripts.fit_rho_f_hierarchical \\
        --scoring union      # after the R1 arm lands

CONTRACT: `metrics/` untouched; this only reads persisted columns.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from ..analysis.rho_f_hierarchical import fit_hierarchical_rho_f, sigma2_between
from ..config import load_config

_MODELS = ["qwen_2_5_7b", "llama_3_1_8b", "mistral_7b_v03"]


def _source(config, model: str, scoring: str) -> tuple[Path, str, str, str]:
    """(parquet, per-paraphrase col, accuracy col, MoM rho_F col) for the scoring.

    The union parquet (rescore_union_gold.py) suffixes its MoM columns by gold
    set — `rho_f_union` / `rho_f_target` — while v3 has plain `rho_f`.
    """
    root = config.repo_root()
    if scoring == "target":
        return (root / f"data/specificity_v3_{model}.parquet",
                "f_graded_per_paraphrase", "f_graded_mean", "rho_f")
    if scoring == "union":
        # written by scripts/rescore_union_gold.py (R1)
        return (root / f"data/union_gold_{model}.parquet",
                "f_graded_union_per_paraphrase", "f_graded_union_mean", "rho_f_union")
    raise ValueError(f"unknown scoring: {scoring}")


def run(args) -> int:
    config = load_config()
    for model in args.models:
        path, col, acc_col, mom_col = _source(config, model, args.scoring)
        if not path.exists():
            logger.error("missing {} — run the R1 arm first for --scoring union", path)
            continue
        df = pd.read_parquet(path)
        if col not in df.columns:
            logger.error("{} has no column {}", path, col)
            continue
        cells = [list(x) if x is not None else None for x in df[col]]
        k = int(df["n_samples_per_prompt"].iloc[0])

        fit = fit_hierarchical_rho_f(cells, k)
        df["rho_f_hier"] = fit.rho_mean
        df["rho_f_hier_sd"] = fit.rho_sd
        df["sigma2_between"] = [sigma2_between(c, k) for c in cells]
        if mom_col in df.columns:
            df["rho_f"] = df[mom_col]      # normalised name for downstream code
            df["rho_f_imputed0"] = df["rho_f"].fillna(0.0)
            cov = df["rho_f"].notna().mean()
        else:
            cov = float("nan")

        keep = [c for c in (
            "question_id", "spec_level", "model_key", "m0", "m_valid",
            "target_collision", acc_col, "h_sem_mean", "rho_u", "s_tau_mean",
            "variation_ratio", "consistency_mean", "fi_out_var", "a_q",
            "posix_psi", "ess_in", "fi_premium", "aufi_in_graded",
            "rho_f", "rho_f_imputed0", "rho_f_hier", "rho_f_hier_sd",
            "sigma2_between",
        ) if c in df.columns]
        out = config.repo_root() / f"data/rho_f_hier_{args.scoring}_{model}.parquet"
        df[keep].to_parquet(out, index=False)

        draws = fit.sample(args.n_draws, seed=config.random_seed)
        np.save(config.repo_root() / f"data/rho_f_hier_draws_{args.scoring}_{model}.npy", draws)

        logger.info(
            "{} [{}]: n={} k={} | MoM coverage {:.1%} mean {:.3f} | hier mean {:.3f} "
            "(sd across cells {:.3f}, mean post sd {:.3f}) | prior rho~Beta({:.2f},{:.2f}) "
            "mu~Beta({:.2f},{:.2f}) -> {}",
            model, args.scoring, len(df), k, cov,
            float(np.nanmean(df["rho_f"])) if "rho_f" in df else float("nan"),
            float(df["rho_f_hier"].mean()), float(df["rho_f_hier"].std()),
            float(df["rho_f_hier_sd"].mean()),
            fit.prior_a, fit.prior_b, fit.mu_a, fit.mu_b, out.name,
        )
    return 0


def _parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", nargs="+", default=_MODELS)
    ap.add_argument("--scoring", choices=["target", "union"], default="target")
    ap.add_argument("--n-draws", type=int, default=200)
    return ap.parse_args(argv)


def main() -> int:
    return run(_parse_args())


if __name__ == "__main__":
    sys.exit(main())
