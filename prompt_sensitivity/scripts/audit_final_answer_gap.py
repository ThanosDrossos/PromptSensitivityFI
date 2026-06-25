"""P0-2a: triage the chain-F vs final-answer-F gap.

Smoke: mean chain-F 0.754 but mean final-answer-F 0.161 — the graded chain
scorer does not predict final-answer accuracy as parameterised (Smoke_Run §4.4,
Glossary M04). This classifies WHY, per response, into:

  parse_failure  — parse_answer_line returned "" (extraction missed the answer)
  nli_too_strict — entail in [0.4, entail_threshold) with no contradiction
                   (the answer is right but the asymmetric NLI under-fires)
  model_wrong    — entail < 0.4 OR contradict >= contradict_threshold

It reloads the per-paraphrase F-responses from the SQLite cache by the `purpose`
field (e2e_smoke writes purpose `e2e_f::{qid}::{ladder_type}::L{level}::{model}`).

MUST run where the matching cache lives (i.e. the cluster, whose `local`-provider
responses differ in hash from the gateway-era local cache):
  python -m prompt_sensitivity.scripts.audit_final_answer_gap --in data/cluster_e2e_musique.parquet
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

from ..config import load_config
from ..data.load_musique import load_musique_validation
from ..logging_setup import configure_logging
from ..models.schemas import LLMResponse
from ..prompts import parse_answer_line
from ..scoring.nli_with_gold import score_nli_with_gold


def _gold_by_qid(config) -> dict[str, str]:
    m = config.sampling.musique
    qs = load_musique_validation(
        hf_dataset=m.hf_dataset or None, hf_config=m.hf_config, split=m.split,
        local_path=m.local_path, repo_root=config.repo_root(),
    )
    return {q.id: q.answer for q in qs}


def _responses_for_cell(conn: sqlite3.Connection, purpose: str) -> list[str]:
    rows = conn.execute(
        "SELECT response_json FROM llm_cache WHERE purpose = ?", (purpose,)
    ).fetchall()
    texts: list[str] = []
    for (rj,) in rows:
        try:
            texts.append(LLMResponse.model_validate_json(rj).text)
        except Exception:  # noqa: BLE001
            continue
    return texts


def _classify(parsed: str, ent: float | None, contr: float | None,
              entail_thr: float, contra_thr: float) -> str:
    if not parsed:
        return "parse_failure"
    assert ent is not None and contr is not None
    if ent < 0.4 or contr >= contra_thr:
        return "model_wrong"
    if 0.4 <= ent < entail_thr and contr < contra_thr:
        return "nli_too_strict"
    return "model_wrong"  # ent >= entail_thr would have passed strict already


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", default="data/cluster_e2e_musique.parquet")
    ap.add_argument("--cache", default=None, help="sqlite cache path (default config.cache.path)")
    ap.add_argument("--out", default="audit_final_answer_gap.json")
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    configure_logging("audit_final_answer_gap")
    config = load_config()
    root = config.repo_root()

    df = pd.read_parquet(root / args.inp)
    if "scoring_mode" in df.columns:
        df = df[df["scoring_mode"] == "chain_completion"]

    sel = df[(df["f_mean"] >= 0.8) & (df["final_answer_f_mean"] == 0)]
    if len(sel) < args.n:  # pad with the next-lowest final-answer cells
        extra = df[~df.index.isin(sel.index)].sort_values("final_answer_f_mean").head(args.n - len(sel))
        sel = pd.concat([sel, extra])
    sel = sel.head(args.n)
    logger.info("selected {} cells for audit", len(sel))

    gold = _gold_by_qid(config)
    cache_path = Path(args.cache) if args.cache else config.cache_path()
    if not cache_path.exists():
        logger.error("cache not found at {} — run this where the matching cache lives", cache_path)
        return 1
    conn = sqlite3.connect(str(cache_path))

    entail_thr = config.scoring.entail_threshold
    contra_thr = config.scoring.contradict_threshold

    counts = {"parse_failure": 0, "nli_too_strict": 0, "model_wrong": 0}
    detail: list[dict] = []
    for _, c in sel.iterrows():
        purpose = f"e2e_f::{c['question_id']}::{c['ladder_type_raw']}::L{int(c['level'])}::{c['model_key']}"
        g = gold.get(c["question_id"], "")
        for text in _responses_for_cell(conn, purpose):
            parsed = parse_answer_line(text)
            ent = contr = None
            if parsed:
                r = score_nli_with_gold(g, parsed, config=config)
                ent, contr = float(r.entail_prob), float(r.contradict_prob)
            cat = _classify(parsed, ent, contr, entail_thr, contra_thr)
            counts[cat] += 1
            detail.append({
                "purpose": purpose, "gold": g, "parsed": parsed,
                "entail": ent, "contradict": contr, "category": cat,
                "raw_text": text[:300],
            })

    n = sum(counts.values())
    pct = {k: (100.0 * v / n if n else 0.0) for k, v in counts.items()}
    summary = {
        "n_cells": int(len(sel)), "n_responses": int(n),
        "counts": counts, "percentages": {k: round(v, 1) for k, v in pct.items()},
        "dominant": (max(counts, key=counts.get) if n else None),
        "cache": str(cache_path), "in": str(root / args.inp),
    }
    print(json.dumps(summary, indent=2))
    (root / args.out).write_text(
        json.dumps({"summary": summary, "rows": detail}, indent=2), encoding="utf-8"
    )
    logger.info("wrote {}", root / args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
