"""Backfill rho_f + fi_premium (METRIC_PROPOSALS M1+M2) into existing metrics
parquets — both derive from the persisted `f_graded_per_paraphrase`, so historic
runs (v3) get the new metric set WITHOUT any cluster re-run.

Idempotent (recomputes and overwrites the two columns), atomic replace, and
uses the SAME compute path as the run driver (sensitivity_v2.compute_row_metrics).

    python -m prompt_sensitivity.scripts.backfill_sensitivity_v2 \
        data/specificity_v3_qwen_2_5_7b.parquet [more.parquet ...]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from ..config import load_config
from ..logging_setup import configure_logging
from ..metrics.sensitivity_v2 import compute_row_metrics


def backfill_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Add/overwrite rho_f + fi_premium columns. Pure; returns a copy."""
    out = df.copy()
    vals = []
    for _, r in out.iterrows():
        f = r.get("f_graded_per_paraphrase")
        f = list(f) if f is not None and not (np.isscalar(f) and pd.isna(f)) else None
        k = int(r.get("n_samples_per_prompt") or 10)
        vals.append(compute_row_metrics(f, max(k, 2)))
    out["rho_f"] = [v["rho_f"] for v in vals]
    out["fi_premium"] = [v["fi_premium"] for v in vals]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("parquets", nargs="+")
    args = ap.parse_args()
    configure_logging("backfill_sensitivity_v2")
    root = load_config().repo_root()
    for p in args.parquets:
        path = Path(p) if Path(p).is_absolute() else root / p
        df = backfill_frame(pd.read_parquet(path))
        tmp = path.with_suffix(path.suffix + ".tmp")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
        ok = df.rho_f.notna().mean() * 100
        logger.info("{}: rho_f defined {:.0f}% (mean {:.3f}), fi_premium nonzero {:.0f}% (mean {:.3f})",
                    path.name, ok, df.rho_f.mean(),
                    (df.fi_premium.fillna(0) > 0).mean() * 100, df.fi_premium.mean())
    print("BACKFILL DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
