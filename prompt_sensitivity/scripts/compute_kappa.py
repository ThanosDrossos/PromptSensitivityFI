"""Sprint 2 gate — Cohen's κ on the annotation sheet Thanos+Diener filled in.

Reads `data/annotation_sample_v1.csv` (or the path passed via --in), pulls
the two annotator columns (`thanos`, `diener`), and computes Cohen's κ. If
κ < 0.8, the gate fails and the sprint brief calls for tightening the NLI
threshold and regenerating.

We compute κ by hand (not via scikit-learn) because the inter-annotator
calculation is so small — keeps the dep surface narrow and the math
auditable.
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd
from loguru import logger

from ..config import load_config
from ..logging_setup import configure_logging


def cohens_kappa(a: list[int], b: list[int]) -> dict[str, float]:
    """Cohen's κ for two raters with binary labels {0, 1}."""
    if len(a) != len(b) or not a:
        raise ValueError(f"length mismatch or empty: |a|={len(a)} |b|={len(b)}")
    n = len(a)
    # Observed agreement.
    p_obs = sum(1 for x, y in zip(a, b) if x == y) / n
    # Expected agreement under independence: sum_k P(a=k) * P(b=k).
    pa1 = sum(a) / n
    pb1 = sum(b) / n
    pa0 = 1 - pa1
    pb0 = 1 - pb1
    p_exp = pa0 * pb0 + pa1 * pb1
    if abs(1 - p_exp) < 1e-12:
        kappa = 1.0
    else:
        kappa = (p_obs - p_exp) / (1 - p_exp)
    return {"kappa": kappa, "p_observed": p_obs, "p_expected": p_exp, "n_items": n}


def _parse_label(v) -> int | None:
    """Accept 1/0, '1'/'0', 'yes'/'no', case-insensitive. Blank -> None."""
    if v is None or (isinstance(v, float) and pd.isna(v)) or v == "":
        return None
    s = str(v).strip().lower()
    if s in {"1", "yes", "y", "true", "t"}:
        return 1
    if s in {"0", "no", "n", "false", "f"}:
        return 0
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="inp", type=str, default="data/annotation_sample_v1.csv")
    args = parser.parse_args()

    configure_logging("compute_kappa")
    config = load_config()
    repo_root = config.repo_root()

    path = repo_root / args.inp
    if not path.exists():
        logger.error("missing {} — run `export-annotation` and fill in the columns", path)
        return 1
    df = pd.read_csv(path)
    if "thanos" not in df.columns or "diener" not in df.columns:
        logger.error("CSV missing required columns thanos/diener")
        return 1

    a_raw = [_parse_label(v) for v in df["thanos"]]
    b_raw = [_parse_label(v) for v in df["diener"]]
    pairs = [(x, y) for x, y in zip(a_raw, b_raw) if x is not None and y is not None]
    if not pairs:
        logger.error("no annotated rows; have both annotators filled in the columns?")
        return 1
    a = [p[0] for p in pairs]
    b = [p[1] for p in pairs]

    result = cohens_kappa(a, b)
    n_blank = len(df) - len(pairs)
    result["rows_skipped_blank"] = float(n_blank)

    print(json.dumps(result, indent=2))
    if result["kappa"] < 0.8:
        logger.warning(
            "κ = {:.3f} below the 0.8 gate; the brief calls for tightening "
            "the NLI threshold (Sprint 2 §3.1) before continuing.",
            result["kappa"],
        )
        return 2
    logger.info("κ = {:.3f} — gate passes", result["kappa"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
