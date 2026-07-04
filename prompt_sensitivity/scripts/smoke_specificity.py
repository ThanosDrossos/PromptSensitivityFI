"""Specificity-pivot verification gate (pivot spec §11).

Runs ~5 AmbigQA questions x 2 levels x 1 local model at small budgets
(--max-paraphrases 6, --k-samples 3), then asserts on the aggregated parquet:

  HARD (exit != 0 on failure):
    - required columns present (fi_spec, spec_level, m_valid, f_mean, aufi_in,
      fi_out_mean, h_sem_mean),
    - mean(fi_spec | level 1) > mean(fi_spec | level 0),
    - mean(f_mean  | level 1) >= mean(f_mean | level 0),
    - the parquet round-trips.
  SOFT (warning only at this N):
    - mean(aufi_in | level 1) <= mean(aufi_in | level 0)   (FI_in should fall)
    - h_sem / fi_out direction checks.

Needs the local models + DeBERTa -> cluster-run (sbatch cluster/smoke_specificity.sbatch).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

from ..config import load_config
from ..logging_setup import configure_logging

_OUT = "data/smoke_specificity.parquet"
_REQUIRED = ["fi_spec", "spec_level", "m_valid", "f_mean", "aufi_in",
             "fi_out_mean", "h_sem_mean"]


def main() -> int:
    configure_logging("smoke_specificity")
    config = load_config()
    out_path = config.repo_root() / _OUT

    cmd = [
        sys.executable, "-m", "prompt_sensitivity.scripts.run_specificity",
        "--n-questions", "5", "--models", "qwen_2_5_7b",
        "--max-paraphrases", "6", "--k-samples", "3",
        "--out", _OUT,
    ]
    logger.info("gate: running {}", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=config.repo_root())
    if proc.returncode != 0:
        logger.error("run_specificity exited {}", proc.returncode)
        return proc.returncode

    return check_parquet(out_path)


def check_parquet(out_path: Path) -> int:
    """The gate's assert block — separated so it can be run on an existing parquet."""
    df = pd.read_parquet(out_path)  # hard assert: loads back cleanly
    logger.info("gate: {} cells loaded from {}", len(df), out_path)

    failures: list[str] = []
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        failures.append(f"missing columns: {missing}")
    else:
        by = df.groupby("spec_level")
        fi_spec = by["fi_spec"].mean()
        f_mean = by["f_mean"].mean()
        if not (fi_spec.get(1, 0.0) > fi_spec.get(0, 0.0)):
            failures.append(f"fi_spec not increasing: {dict(fi_spec)}")
        if not (f_mean.get(1, 0.0) >= f_mean.get(0, 0.0)):
            failures.append(f"f_mean decreased with specificity: {dict(f_mean)}")

        # SOFT direction checks (small N -> warning only).
        for col, direction in [("aufi_in", "<="), ("h_sem_mean", "<="), ("fi_out_mean", ">=")]:
            if col not in df.columns:
                continue
            m = by[col].mean()
            v0, v1 = m.get(0), m.get(1)
            ok = (v1 <= v0) if direction == "<=" else (v1 >= v0)
            if v0 is not None and v1 is not None and not ok:
                logger.warning("soft check: expected {}(L1) {} {}(L0), got L0={:.3f} L1={:.3f}",
                               col, direction, col, v0, v1)

    print()
    print("=" * 80)
    cols = [c for c in _REQUIRED + ["a_q"] if c in df.columns]
    print(df.groupby("spec_level")[cols].mean().to_string(float_format=lambda x: f"{x:.3f}"))
    print("=" * 80)
    if failures:
        for f in failures:
            logger.error("GATE FAIL: {}", f)
        return 1
    logger.info("GATE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
