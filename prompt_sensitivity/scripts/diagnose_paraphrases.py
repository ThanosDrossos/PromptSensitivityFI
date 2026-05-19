"""Diagnose a paraphrase parquet — `make diagnose-paraphrases [PATH]`.

Reads `data/paraphrases_v1.parquet` (or whatever `--in` points at), reports:

  - per-question outcome (accepted, rejected-by-reason)
  - distribution of NLI fwd/bwd scores split by accepted vs rejected
  - 5 example rejections per reason per question (truncated)
  - threshold-counterfactual: "if we lowered NLI to X, we'd accept Y more"

Used to decide whether the brief's NLI=0.9 is realistic for this generator
on these questions, before sending data to Diener for the κ gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from ..config import load_config
from ..logging_setup import configure_logging


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="inp", type=str, default="data/paraphrases_smoke.parquet")
    parser.add_argument("--examples", type=int, default=3)
    args = parser.parse_args()

    configure_logging("diagnose_paraphrases")
    config = load_config()
    repo_root = config.repo_root()

    path = repo_root / args.inp
    if not path.exists():
        logger.error("missing {}", path)
        return 1
    df = pd.read_parquet(path)
    logger.info("loaded {} rows across {} questions", len(df), df["question_id"].nunique())

    target = config.paraphrases.n_per_question
    strict = config.paraphrases.nli.bidirectional_threshold
    fallback = config.paraphrases.nli.fallback_threshold

    print()
    print("=" * 90)
    print(f"OVERALL — target {target} accepted/question; thresholds strict={strict} fallback={fallback}")
    print("=" * 90)

    summary_rows = []
    for qid, sub in df.groupby("question_id"):
        n_acc = (sub["outcome"] == "accepted").sum()
        n_total = len(sub)
        reasons = sub.loc[sub["outcome"] == "rejected", "reason"].value_counts().to_dict()
        dropped = bool(sub["dropped"].any())
        thresh_used = sub["nli_threshold_used"].iloc[0]
        attempts = int(sub["regeneration_attempts"].max())
        summary_rows.append({
            "question_id": qid,
            "accepted": int(n_acc),
            "rejected": int(n_total - n_acc),
            "dropped": dropped,
            "attempts": attempts,
            "threshold_used": float(thresh_used),
            **{f"r_{k}": int(v) for k, v in reasons.items()},
        })
    summary_df = pd.DataFrame(summary_rows).fillna(0)
    print(summary_df.to_string(index=False))

    print()
    print("=" * 90)
    print("NLI SCORE DISTRIBUTIONS (across all questions)")
    print("=" * 90)
    # Only rows that ran through NLI have non-null nli_fwd.
    has_nli = df.dropna(subset=["nli_fwd", "nli_bwd"])
    for outcome in ("accepted", "rejected"):
        sub = has_nli[has_nli["outcome"] == outcome]
        if sub.empty:
            print(f"  {outcome:<8}  no rows with NLI scores")
            continue
        for col in ("nli_fwd", "nli_bwd"):
            v = sub[col].astype(float).to_numpy()
            qs = np.percentile(v, [10, 25, 50, 75, 90])
            print(f"  {outcome:<8} {col}: n={len(v):>4}  min={v.min():.3f}  p10={qs[0]:.3f}  p25={qs[1]:.3f}  p50={qs[2]:.3f}  p75={qs[3]:.3f}  p90={qs[4]:.3f}  max={v.max():.3f}")

    print()
    print("=" * 90)
    print("THRESHOLD COUNTERFACTUAL — for each candidate τ, how many extra would pass NLI?")
    print("=" * 90)
    rej = has_nli[has_nli["outcome"] == "rejected"]
    rej_min = rej[["nli_fwd", "nli_bwd"]].astype(float).min(axis=1)
    for tau in (0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5):
        n_pass = int((rej_min >= tau).sum())
        print(f"  τ={tau:.2f}  : {n_pass:>5} additional candidates would pass NLI (out of {len(rej)} currently rejected)")

    print()
    print("=" * 90)
    print(f"SAMPLE REJECTIONS (--examples={args.examples} per reason per question)")
    print("=" * 90)
    for qid, sub in df.groupby("question_id"):
        print(f"\n  qid={qid}")
        rej_sub = sub[sub["outcome"] == "rejected"]
        for reason, grp in rej_sub.groupby("reason"):
            print(f"    reason={reason}  (n={len(grp)})")
            for _, r in grp.head(args.examples).iterrows():
                fwd = r["nli_fwd"]
                bwd = r["nli_bwd"]
                jac = r["jaccard"]
                fwd_s = f"{fwd:.3f}" if pd.notna(fwd) else "  -  "
                bwd_s = f"{bwd:.3f}" if pd.notna(bwd) else "  -  "
                jac_s = f"{jac:.3f}" if pd.notna(jac) else "  -  "
                role = r["role"]
                text = (r["text"] or "")[:90]
                print(f"      role={role:<14} fwd={fwd_s} bwd={bwd_s} jac={jac_s}  {text!r}")

    out = {
        "n_questions": int(df["question_id"].nunique()),
        "n_accepted": int((df["outcome"] == "accepted").sum()),
        "n_rejected": int((df["outcome"] == "rejected").sum()),
        "n_dropped_questions": int(df.loc[df["dropped"], "question_id"].nunique()),
        "target": target,
    }
    print()
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
