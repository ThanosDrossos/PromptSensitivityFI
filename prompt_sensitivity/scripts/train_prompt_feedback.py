"""Train + verify the prompt-feedback deliverable (4 heads x 3 models).

Per model: build shared TBG features (50/75/100% depth concat), train the
vagueness / reliability / dispersion / fragility heads with question-grouped
5-fold OOF verification, isotonic calibration, permutation + length controls,
then save the shipped bundle (data/feedback_model_<model>.joblib) and one
verification table (data/feedback_verification.parquet).

    python -m prompt_sensitivity.scripts.train_prompt_feedback \
        [--models qwen_2_5_7b,...] [--exclude-questions ID,...]
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd
from loguru import logger

from ..config import load_config
from ..feedback.heads import FeedbackModel, build_features, head_labels, train_head
from ..logging_setup import configure_logging

_HEADS = [
    ("vagueness", True, False),
    ("reliability", False, False),
    ("dispersion", True, False),
    ("fragility", True, True),      # experimental
]
_DEFAULT_EXCLUDE = "7308933918215095839"   # v3 singleton universe


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default="qwen_2_5_7b,llama_3_1_8b,mistral_7b_v03")
    ap.add_argument("--exclude-questions", default=_DEFAULT_EXCLUDE)
    ap.add_argument("--n-splits", type=int, default=5)
    args = ap.parse_args()

    configure_logging("train_prompt_feedback")
    root = load_config().repo_root()
    excl = {q.strip() for q in args.exclude_questions.split(",") if q.strip()}
    rows = []
    for m in [x.strip() for x in args.models.split(",") if x.strip()]:
        hs = pd.read_parquet(root / f"data/hidden_states_{m}.parquet")
        hs["question_id"] = hs.question_id.astype(str)
        hs = hs[~hs.question_id.isin(excl)]
        metrics = pd.read_parquet(root / f"data/specificity_v3_{m}.parquet")
        if "rho_f" not in metrics.columns:
            logger.error("{}: rho_f missing — run backfill_sensitivity_v2 first", m)
            return 1
        X, meta = build_features(hs)
        labels = head_labels(meta, metrics)
        lengths = meta.paraphrase.str.len().to_numpy(dtype=float)
        logger.info("[{}] features {} x {}", m, *X.shape)

        heads = {}
        for name, binarize, experimental in _HEADS:
            h = train_head(name, X, labels[name].to_numpy(dtype=float),
                           meta.question_id, binarize=binarize,
                           n_splits=args.n_splits, prompt_lengths=lengths,
                           experimental=experimental)
            heads[name] = h
            v = h.verification
            logger.info("[{}] {:11s} {}", m, name,
                        " ".join(f"{k}={v[k]:.3f}" if isinstance(v[k], float) else f"{k}={v[k]}"
                                 for k in sorted(v)))
            rows.append({"model_key": m, "head": name, "experimental": experimental, **v})

        fm = FeedbackModel(model_key=m, layer_fracs=(0.5, 0.75, 1.0), heads=heads,
                           meta={"trained_on": "specificity_v3", "n_prompts": int(X.shape[0])})
        out = root / f"data/feedback_model_{m}.joblib"
        fm.save(out)
        logger.info("[{}] bundle -> {}", m, out.name)

    ver = pd.DataFrame(rows)
    ver.to_parquet(root / "data/feedback_verification.parquet", index=False)
    print()
    print(ver.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nwrote data/feedback_verification.parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
