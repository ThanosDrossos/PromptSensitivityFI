"""Frozen-head evaluation on the vagueness HOLDOUT (laptop, post-pull).

Applies the trained FeedbackModel bundles — frozen, no retraining — to the
holdout dumps (dump_vagueness_holdout.py): AmbigQA's annotator-labeled
ambiguous vs non-ambiguous questions. Questions that appear in the v3
training parquets are EXCLUDED, so every scored prompt is (a) a question the
head never saw and (b) labeled by human judgment, not by the L0/L1 rewriting
mechanism the head was trained on. Reports AUROC + a prompt-length baseline.

    uv run python -m prompt_sensitivity.scripts.eval_vagueness_holdout
"""

from __future__ import annotations

import argparse
import glob

import numpy as np
import pandas as pd
from loguru import logger

from ..config import load_config
from ..feedback.heads import FeedbackModel, build_features
from ..logging_setup import configure_logging

_MODELS = ("qwen_2_5_7b", "llama_3_1_8b", "mistral_7b_v03")


def _auroc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=str, default="data/vagueness_holdout_results.parquet")
    args = ap.parse_args()

    configure_logging("eval_vagueness_holdout")
    root = load_config().repo_root()

    # Questions the heads were trained on (any v3 metrics parquet) -> excluded.
    seen: set[str] = set()
    for f in glob.glob(str(root / "data/specificity_v3_*.parquet")):
        seen |= set(pd.read_parquet(f)["question_id"].astype(str))
    logger.info("excluding {} training questions", len(seen))

    rows = []
    for m in _MODELS:
        hs_path = root / f"data/vagueness_holdout_{m}.parquet"
        fm_path = root / f"data/feedback_model_{m}.joblib"
        if not hs_path.exists() or not fm_path.exists():
            logger.warning("[{}] holdout dump or bundle missing — skipped", m)
            continue
        hs = pd.read_parquet(hs_path)
        labels = (hs.groupby(hs["question_id"].astype(str))["ambiguous"]
                  .first().to_dict())
        fm = FeedbackModel.load(fm_path)
        X, meta = build_features(hs, layer_fracs=fm.layer_fracs)
        keep = ~meta["question_id"].isin(seen)
        X, meta = X[keep.values], meta[keep].reset_index(drop=True)
        y = meta["question_id"].map(labels).astype(int).values
        if len(np.unique(y)) < 2:
            logger.warning("[{}] holdout has a single class after exclusion", m)
            continue
        score = fm.gauges(X.astype(np.float32))["vagueness"].values
        base = meta["paraphrase"].str.len().values.astype(float)
        au, au_len = _auroc(y, score), _auroc(y, base)
        logger.info("[{}] n={} ({} ambiguous) AUROC {:.3f} | length-baseline {:.3f}",
                    m, len(y), int(y.sum()), au, au_len)
        rows.append({"model_key": m, "n": len(y), "n_ambiguous": int(y.sum()),
                     "auroc": au, "auroc_length_baseline": au_len})
    if not rows:
        logger.error("nothing evaluated — run the holdout dump + pull first")
        return 1
    out = pd.DataFrame(rows)
    out.to_parquet(root / args.out, index=False)
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
