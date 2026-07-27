"""Prompt-feedback demo: prompt in -> gauges + advice out.

Two modes:

  --replay N     (laptop, default) Sample N prompts from the feature dump,
                 run the trained gauges, print the composed feedback NEXT TO
                 the ground-truth labels — the end-to-end verification demo.
                 No GPU, no model weights needed.

  --prompt TEXT  (cluster / GPU) Embed an arbitrary prompt with the target
                 model (TBG states via chat_hidden_states, closed-book message
                 format) and print its gauges. CAVEAT, printed at runtime: the
                 heads were trained on uniform-evidence prompts; a bare prompt
                 is off-distribution, so treat live gauges as a mechanism demo,
                 not a calibrated judgment.

    python -m prompt_sensitivity.scripts.prompt_feedback_demo --replay 6
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from loguru import logger

from ..config import load_config
from ..feedback.heads import FeedbackModel, build_features, compose_feedback, head_labels
from ..logging_setup import configure_logging


def _replay(fm: FeedbackModel, root, model_key: str, n: int, seed: int) -> None:
    hs = pd.read_parquet(root / f"data/hidden_states_{model_key}.parquet")
    metrics = pd.read_parquet(root / f"data/specificity_v3_{model_key}.parquet")
    X, meta = build_features(hs, fm.layer_fracs)
    labels = head_labels(meta, metrics)
    rng = np.random.default_rng(seed)
    # stratify: half L0, half L1, only rows with a reliability label
    ok = labels.reliability.notna().to_numpy()
    picks = []
    for lvl in (0, 1):
        pool = np.flatnonzero(ok & (meta.spec_level == lvl).to_numpy())
        picks += list(rng.choice(pool, size=max(1, n // 2), replace=False))
    G = fm.gauges(X[picks])
    for row_i, (i, g) in enumerate(zip(picks, G.itertuples(index=False))):
        g = g._asdict()
        lab = labels.iloc[i]
        print("=" * 96)
        print(f"PROMPT (true level L{int(lab.spec_level)}): {meta.paraphrase.iloc[i]}")
        print(f"  gauges: vague={g['vagueness']:.2f}  reliab={g['reliability']:.2f}  "
              f"disperse={g['dispersion']:.2f}  fragile={g.get('fragility', float('nan')):.2f}")
        print(f"  truth : vague={lab.vagueness:.0f}        f={lab.reliability:.2f}   "
              f"H_sem={lab.dispersion:.2f}"
              + (f"  rho_F={lab.fragility:.2f}" if np.isfinite(lab.fragility) else "  rho_F=n/a"))
        for msg in compose_feedback(g):
            print(f"  -> {msg}")


def _live(fm: FeedbackModel, config, model_key: str, prompt: str) -> None:
    from ..models.registry import get_client
    print("CAVEAT: heads were trained on uniform-evidence prompts; a bare prompt is "
          "off-distribution — mechanism demo, not a calibrated judgment.\n")
    client = get_client(model_key, config)
    messages = [[{"role": "system",
                  "content": "You are a precise question answering assistant. "
                             "Answer with a short factual answer."},
                 {"role": "user", "content": prompt}]]
    arr, _ = client.chat_hidden_states(messages, layer_fracs=fm.layer_fracs)
    X = arr.reshape(1, -1).astype(np.float32)
    g = fm.gauges(X).iloc[0].to_dict()
    print(f"PROMPT: {prompt}")
    print("  gauges: " + "  ".join(f"{k}={v:.2f}" for k, v in g.items()))
    for msg in compose_feedback(g):
        print(f"  -> {msg}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="qwen_2_5_7b")
    ap.add_argument("--replay", type=int, default=None, metavar="N")
    ap.add_argument("--prompt", type=str, default=None)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    if (args.replay is None) == (args.prompt is None):
        print("choose exactly one of --replay N | --prompt TEXT", file=sys.stderr)
        return 2

    configure_logging("prompt_feedback_demo")
    config = load_config()
    root = config.repo_root()
    bundle = root / f"data/feedback_model_{args.model}.joblib"
    if not bundle.exists():
        logger.error("bundle {} missing — run train_prompt_feedback first", bundle.name)
        return 1
    fm = FeedbackModel.load(bundle)
    if args.replay is not None:
        _replay(fm, root, args.model, args.replay, args.seed)
    else:
        _live(fm, config, args.model, args.prompt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
