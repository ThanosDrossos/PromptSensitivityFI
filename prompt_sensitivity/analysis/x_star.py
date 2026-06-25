"""x*(q;M) embedding-distance analysis. Section_7 §7.9 C3 / §7.7.

The supervisor's geometric intuition (§7.7): the highest-functioning paraphrase
x* sits at a special point in the model's embedding geometry, and a paraphrase's
function F(x) should DECAY with its embedding distance from x*. We test it: for
each (question, model, ladder_family, level) cell,

  1. x* = the paraphrase with the maximum F (ties: shorter text, then lexical),
  2. dist(x, x*) = || e_M(x) - e_M(x*) ||_2 on the L2-normalized own-encoder
     embedding (P1-1),
  3. rho = Spearman( F(x), dist(x, x*) ) over the N-1 non-x* paraphrases.

C3 target: across-question mean rho <= -0.4 (farther => less functional).

The pure core (`pick_x_star`, `cell_x_star_rho`, `aggregate`) is unit-tested.
`main()` reconstructs per-paraphrase F (re-scored from the cached responses) and
embeddings (re-encoded via LocalHFClient.embed_hidden) — so it must run where the
cache + model weights live (the cluster). Writes data/x_star_analysis.parquet and
data/plots/10_x_star_distance.png.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from loguru import logger


# --------------------------------------------------------------------------- #
# Pure core (unit-tested)                                                     #
# --------------------------------------------------------------------------- #


def pick_x_star(f_scores: Sequence[float], texts: Sequence[str]) -> int:
    """Index of the max-F paraphrase. Deterministic tie-break (§7.7): highest F,
    then shortest text, then lexicographically smallest text."""
    return min(range(len(f_scores)), key=lambda i: (-f_scores[i], len(texts[i]), texts[i]))


def cell_x_star_rho(
    f_scores: Sequence[float],
    embeddings: Sequence[np.ndarray],
    texts: Sequence[str],
) -> float | None:
    """Spearman(F(x), dist(x, x*)) over the non-x* paraphrases. None if degenerate
    (fewer than 2 comparison points, or no variance in F or distance)."""
    n = len(f_scores)
    if n < 3:
        return None
    star = pick_x_star(f_scores, texts)
    e_star = np.asarray(embeddings[star], dtype=float)
    others = [i for i in range(n) if i != star]
    dists = [float(np.linalg.norm(np.asarray(embeddings[i], dtype=float) - e_star)) for i in others]
    fs = [float(f_scores[i]) for i in others]
    if len(set(fs)) < 2 or len(set(dists)) < 2:
        return None
    from scipy.stats import spearmanr

    rho = spearmanr(fs, dists)[0]
    return float(rho) if rho == rho else None  # filter NaN


def aggregate(per_cell: list[dict]) -> dict:
    """Mean rho per (model_key, ladder_family, level) and overall."""
    import pandas as pd

    df = pd.DataFrame(per_cell)
    if df.empty or "rho" not in df:
        return {"overall_mean_rho": float("nan"), "by_group": {}}
    valid = df.dropna(subset=["rho"])
    by_group = (
        valid.groupby(["model_key", "ladder_family", "level"])["rho"].mean().to_dict()
        if not valid.empty else {}
    )
    return {
        "overall_mean_rho": float(valid["rho"].mean()) if not valid.empty else float("nan"),
        "n_cells": int(len(valid)),
        "by_group": {str(k): float(v) for k, v in by_group.items()},
    }


# --------------------------------------------------------------------------- #
# Cluster reconstruction + driver                                             #
# --------------------------------------------------------------------------- #


def _reconstruct_cell(config, q, row, model_key, paraphrases):
    """Per-paraphrase (F, prompt-embedding) for one cell, reusing the e2e helpers.

    Generation is served from the SQLite cache (the run already populated it), so
    this re-scores + re-embeds without new generation. Returns (f_scores, embs,
    texts) or None if the model can't embed (no own-encoder)."""
    from ..models.registry import get_client
    from ..scoring import chain_completion_score_batch, f_score_batch
    from ..scripts.e2e_smoke import _assemble_messages, _sample_response
    from ..ladders import render_reasoning_scaffold

    entry = config.models[model_key]
    client = get_client(model_key, config)
    use_cot = q.has_decomposition()
    gen_max = config.generation.cot_max_tokens if use_cot else config.generation.answer_max_tokens

    msgs = [_assemble_messages(q, p, row, use_cot=use_cot) for p in paraphrases]
    responses = [
        _sample_response(
            client, entry, m, temperature=0.0, seed=42,
            purpose=f"e2e_f::{q.id}::{row.ladder_type}::L{row.level}::{model_key}",
            max_tokens=gen_max,
        )
        for m in msgs
    ]
    if use_cot:
        scaffold = (
            render_reasoning_scaffold(q, row.hops_provided or 0)
            if row.ladder_family == "reasoning" else None
        )
        f_scores = chain_completion_score_batch(
            q.question_decomposition, responses, config=config, scaffold_text=scaffold
        )
    else:
        f_scores = [float(x) for x in f_score_batch(q.answer, responses, config=config)]

    if not (entry.has_hidden and hasattr(client, "embed_hidden")):
        return None
    prompt_texts = [m[1].content for m in msgs]
    embs = client.embed_hidden(prompt_texts)   # L2-normalized (P1-1)
    return f_scores, [embs[i] for i in range(len(prompt_texts))], paraphrases


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", default="data/full_run.parquet")
    ap.add_argument("--out", default="data/x_star_analysis.parquet")
    ap.add_argument("--plot", default="data/plots/10_x_star_distance.png")
    args = ap.parse_args()

    import pandas as pd

    from ..config import load_config
    from ..logging_setup import configure_logging
    from ..ladders import build_reasoning_ladder
    from ..scripts.e2e_smoke import _context_rows, _index_questions

    configure_logging("x_star")
    config = load_config()
    root = config.repo_root()
    df = pd.read_parquet(root / args.inp)
    logger.info("loaded {} cells from {}", len(df), args.inp)

    q_idx = _index_questions(config)
    para = pd.read_parquet(root / "data" / "paraphrases_musique.parquet")
    para = para[para["outcome"] == "accepted"]
    texts_by_q = {
        str(qid): sub.sort_values("paraphrase_idx")["text"].tolist()
        for qid, sub in para.groupby("question_id")
    }

    per_cell: list[dict] = []
    cells = df[["question_id", "model_key", "ladder_family", "ladder_type_raw", "level"]].drop_duplicates()
    for _, c in cells.iterrows():
        q = q_idx.get(c["question_id"])
        paraphrases = texts_by_q.get(str(c["question_id"]))
        if q is None or not paraphrases:
            continue
        if c["ladder_family"] == "reasoning":
            rows = [r for r in build_reasoning_ladder(q) if r.level == int(c["level"])]
        else:
            rows = _context_rows(q, c["ladder_type_raw"], [int(c["level"])])
        if not rows:
            continue
        try:
            rec = _reconstruct_cell(config, q, rows[0], c["model_key"], paraphrases)
        except Exception as exc:  # noqa: BLE001
            logger.warning("x* recon failed for {} {}: {}", c["question_id"], c["model_key"], exc)
            continue
        if rec is None:
            continue
        f_scores, embs, txts = rec
        rho = cell_x_star_rho(f_scores, embs, txts)
        per_cell.append({
            "question_id": c["question_id"], "model_key": c["model_key"],
            "ladder_family": c["ladder_family"], "level": int(c["level"]), "rho": rho,
        })

    out_df = pd.DataFrame(per_cell)
    (root / args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(root / args.out, index=False)
    agg = aggregate(per_cell)
    logger.info("x* overall mean rho = {} (C3 target <= -0.4)", agg["overall_mean_rho"])
    _plot(out_df, root / args.plot)
    import json
    print(json.dumps(agg, indent=2))
    return 0


def _plot(out_df, out_png: Path) -> None:
    if out_df.empty or out_df["rho"].dropna().empty:
        logger.info("x* plot skipped: no rho values")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = out_df.dropna(subset=["rho"]).groupby("model_key")["rho"]
    models = list(g.groups)
    means = [g.get_group(m).mean() for m in models]
    sems = [g.get_group(m).std(ddof=1) / max(1, len(g.get_group(m))) ** 0.5 for m in models]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(models)), means, yerr=sems, capsize=4, color="#55A868")
    ax.axhline(-0.4, ls="--", color="red", label="C3 target -0.4")
    ax.set_xticks(range(len(models))); ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("mean Spearman rho(F, dist to x*)")
    ax.set_title("x*(q;M) embedding-distance analysis (§7.9 C3)")
    ax.legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)
    logger.info("wrote {}", out_png)


if __name__ == "__main__":
    sys.exit(main())
