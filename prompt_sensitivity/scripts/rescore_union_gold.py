"""R1 — union-gold control arm. Re-scores the EXISTING v3 generations against the
union of every interpretation's answers, at BOTH specificity levels.

WHY (review 2026-08-06 §2.1). The v3 headline (L0->L1 graded accuracy +0.22/+0.24/
+0.25) is arithmetically indistinguishable from removing a 1-in-m0 grading lottery:
at L0 the model is shown a question that does not determine which of m0 equally
valid answers is wanted, and is graded against one of them chosen by
sha256(question_id + seed). The null acc_L0 ~= acc_L1/m0 predicts the observed L0
accuracy to -0.0003 (95% CI [-0.041, +0.044]); and on the 45 cells the pipeline
already flags as `target_collision` (the pinned answer is shared, so the lottery
CANNOT be lost) the gain is -0.044 (CI [-0.135, +0.035], p = 0.43) versus +0.268
elsewhere.

AmbigQA's own protocol (Min et al. 2020 §3.2) scores an ambiguous question against
ALL interpretations via a set F1. Scoring BOTH levels against the union keeps the
gold set IDENTICAL across levels, so the fixed-gold guardrail is satisfied exactly
-- contrary to the objection in main_body.tex that crediting any interpretation
"would change the gold set between levels". It does not.

WHAT THIS BUYS. The decomposition

    Delta_target  =  Delta_union  +  Delta_targeting

separates the part of the effect that is the model reading better (Delta_union)
from the part that is the grader being told which reading it wanted
(Delta_targeting). Delta_union is the non-tautological quantity.

HOW. Strictly a RE-SCORING job:
  * every generation is recovered from the SQLite LLM cache by rebuilding the
    byte-identical LLMRequest that `run_specificity._run_spec_cell` issued
    (same messages, temperature, seed, purpose, max_tokens -> same SHA256 key);
  * NO eval model is ever loaded and NO generation is ever issued -- a cache miss
    is recorded and the cell is skipped, never silently regenerated;
  * only DeBERTa (the NLI scorer) runs, exactly as in the original scoring pass.

Run `--probe` FIRST: it counts cache coverage without loading DeBERTa, so you can
tell in seconds whether the cache on this machine can support the arm. The v3
generations live on bwUniCluster; the laptop cache is gateway-era only.

    # 1. is the cache here?
    uv run python -m prompt_sensitivity.scripts.rescore_union_gold --probe \
        --models qwen_2_5_7b llama_3_1_8b mistral_7b_v03

    # 2. the actual arm (needs a GPU for DeBERTa to be quick)
    uv run python -m prompt_sensitivity.scripts.rescore_union_gold \
        --models qwen_2_5_7b llama_3_1_8b mistral_7b_v03 \
        --out data/union_gold_{model}.parquet

CONTRACT: `metrics/` is not modified. This script only feeds it new inputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from ..config import load_config
from ..metrics.fi_in import aufi_in_from_scores
from ..metrics.sensitivity_v2 import compute_row_metrics
from ..models.registry import _get_cache
from ..models.schemas import LLMRequest
from ..scoring.nli_with_gold import f_score_batch_multi_gold
from ..specificity.build_levels import SpecRow
from .e2e_smoke import _assemble_messages
from .run_specificity import (
    _SpecQuestionView,
    _evidence_paragraphs,
    _graded_f_scores,
    _ladder_row_for,
    load_spec_rows,
)

_PARAPHRASE_PARQUET = "data/paraphrases_ambigqa.parquet"


# --------------------------------------------------------------------------- #
# paraphrase universes (cache-only; never generates)                          #
# --------------------------------------------------------------------------- #


def load_paraphrase_universes(config, path: str | None = None) -> dict[tuple[str, int], list[str]]:
    """{(question_id, spec_level) -> [paraphrase texts ordered by paraphrase_idx]}.

    Reads the persisted universe cache only. `singleton_fallback` rows count as
    accepted (that is how the run driver treated them), so the enumeration here
    matches the enumeration that produced the cached generations.
    """
    parquet = config.repo_root() / (path or _PARAPHRASE_PARQUET)
    if not parquet.exists():
        raise RuntimeError(f"paraphrase universe cache missing: {parquet}")
    df = pd.read_parquet(parquet)
    df = df[df["outcome"].isin(["accepted", "singleton_fallback"])]
    out: dict[tuple[str, int], list[str]] = {}
    for (qid, lvl), g in df.groupby(["question_id", "spec_level"], sort=False):
        g = g.sort_values("paraphrase_idx")
        out[(str(qid), int(lvl))] = [str(t) for t in g["text"]]
    logger.info("paraphrase universes: {} cells from {}", len(out), parquet)
    return out


# --------------------------------------------------------------------------- #
# cache-only response recovery                                                #
# --------------------------------------------------------------------------- #


def _request(model_entry, messages, *, temperature: float, seed: int,
             purpose: str, max_tokens: int) -> LLMRequest:
    """Byte-identical to the request `run_specificity._sample_response` built.

    Any drift here silently turns into a cache miss (never a wrong answer),
    which is why the miss counter is reported rather than swallowed.
    """
    return LLMRequest(
        provider=model_entry.provider,
        model_id=model_entry.model_id,
        messages=messages,
        temperature=temperature,
        top_p=1.0,
        max_tokens=max_tokens,
        seed=seed,
        purpose=purpose,
    )


def recover_cell_responses(
    config, cache, row: SpecRow, model_key: str, paraphrases: list[str],
    *, k_samples: int, evidence_max_chars: int, context_mode: str,
) -> tuple[list[str] | None, dict[int, list[str]] | None, int, int]:
    """(T=0 responses, {paraphrase_idx -> k temperature samples}, n_hit, n_miss).

    Returns (None, None, hit, miss) when ANY request of the cell is missing: a
    partially recovered cell would silently change the estimand.
    """
    model_entry = config.models[model_key]
    ev = _evidence_paragraphs(row, evidence_max_chars) if context_mode == "uniform_evidence" else []
    view = _SpecQuestionView(row, paragraphs=ev)
    lrow = _ladder_row_for(row, n_paragraphs=len(ev))
    gen_max = config.generation.answer_max_tokens
    msgs_per_paraphrase = [
        _assemble_messages(view, p, lrow, use_cot=False) for p in paraphrases
    ]

    n_hit = n_miss = 0
    f_responses: list[str] = []
    for msgs in msgs_per_paraphrase:
        req = _request(
            model_entry, msgs, temperature=0.0, seed=42,
            purpose=f"spec_f::{row.question_id}::L{row.spec_level}::{model_key}",
            max_tokens=gen_max,
        )
        hit = cache.get(req)
        if hit is None:
            n_miss += 1
        else:
            n_hit += 1
            f_responses.append(hit.text.strip())

    samples: dict[int, list[str]] = {}
    for i, msgs in enumerate(msgs_per_paraphrase):
        got: list[str] = []
        for kk in range(k_samples):
            req = _request(
                model_entry, msgs,
                temperature=config.h_sem.sampling_temperature,
                seed=10000 + i * 100 + kk,
                purpose=(f"spec_hsem::{row.question_id}::L{row.spec_level}"
                         f"::{model_key}::s{kk}"),
                max_tokens=gen_max,
            )
            hit = cache.get(req)
            if hit is None:
                n_miss += 1
            else:
                n_hit += 1
                got.append(hit.text.strip())
        samples[i] = got

    complete = (
        n_miss == 0
        and len(f_responses) == len(paraphrases)
        and all(len(v) == k_samples for v in samples.values())
    )
    if not complete:
        return None, None, n_hit, n_miss
    return f_responses, samples, n_hit, n_miss


# --------------------------------------------------------------------------- #
# the arm                                                                     #
# --------------------------------------------------------------------------- #


def score_cell(config, row: SpecRow, f_responses, samples, k_samples: int) -> dict:
    """Score ONE recovered cell under both gold sets.

    target gold = row.target_answers  (the v3 estimand; reproduced as a check)
    union gold  = row.all_answers     (every interpretation's answers; IDENTICAL
                                       at both levels, so the gold set does not
                                       drift across the manipulation)
    """
    target_gold = list(row.target_answers)
    union_gold = list(row.all_answers) or target_gold

    f_target = [float(x) for x in f_score_batch_multi_gold(target_gold, f_responses, config=config)]
    f_union = [float(x) for x in f_score_batch_multi_gold(union_gold, f_responses, config=config)]

    g_target = _graded_f_scores(target_gold, samples, config)
    g_union = _graded_f_scores(union_gold, samples, config)

    rec: dict = {
        "question_id": row.question_id,
        "spec_level": row.spec_level,
        "m0": row.m0,
        "m_valid": row.m_valid,
        "target_idx": row.target_idx,
        "target_collision": bool(row.target_collision),
        "n_paraphrases": len(f_responses),
        "n_samples_per_prompt": k_samples,
        "n_target_golds": len(target_gold),
        "n_union_golds": len(union_gold),
        # T=0 binary
        "f_target_mean": float(np.mean(f_target)) if f_target else None,
        "f_union_mean": float(np.mean(f_union)) if f_union else None,
        # graded
        "f_graded_target_mean": float(np.mean(g_target)) if g_target else None,
        "f_graded_union_mean": float(np.mean(g_union)) if g_union else None,
        "f_graded_target_per_paraphrase": list(g_target),
        "f_graded_union_per_paraphrase": list(g_union),
        # axis-1 summaries under both golds
        "aufi_in_graded_target": aufi_in_from_scores(g_target) if g_target else None,
        "aufi_in_graded_union": aufi_in_from_scores(g_union) if g_union else None,
    }
    # axis-2 under both golds: does formulation sensitivity survive when the
    # grading lottery is removed? (the review's open question)
    for tag, scores in (("target", g_target), ("union", g_union)):
        for k, v in compute_row_metrics(scores, k_samples).items():
            rec[f"{k}_{tag}"] = v
    return rec


def _checkpoint(recs: list[dict], out_path: Path) -> None:
    """Atomically persist progress so a walltime kill loses at most one window.

    Mirrors `e2e_smoke._checkpoint`: write a sibling .tmp and os.replace, so an
    interrupted write can never leave a truncated parquet behind.
    """
    import os

    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    pd.DataFrame(recs).to_parquet(tmp, index=False)
    os.replace(tmp, out_path)


def _load_done(out_path: Path) -> tuple[list[dict], set[tuple[str, int]]]:
    """Existing rows + the (question_id, spec_level) keys already scored."""
    if not out_path.exists():
        return [], set()
    df = pd.read_parquet(out_path)
    recs = df.to_dict("records")
    done = {(str(r["question_id"]), int(r["spec_level"])) for r in recs}
    return recs, done


def run(args) -> int:
    config = load_config()
    rows, questions, context_mode, evidence_max_chars = load_spec_rows(
        config, n_questions=args.n_questions, context_mode=args.context_mode,
    )
    universes = load_paraphrase_universes(config, args.paraphrase_cache)
    cache = _get_cache(config)
    logger.info("LLM cache: {} rows at {}", cache.size(), cache.path)

    k_samples = args.k_samples or config.h_sem.n_samples_per_prompt

    for model_key in args.models:
        if model_key not in config.models:
            logger.error("unknown model key {}", model_key)
            return 2
        out = Path(str(args.out).format(model=model_key))
        out = out if out.is_absolute() else config.repo_root() / out
        out.parent.mkdir(parents=True, exist_ok=True)

        # Resume: a singleton chain re-enters this script every 30-min window,
        # so already-scored cells must be skipped and a surplus window must be a
        # clean no-op rather than a re-scoring pass.
        recs, done = ([], set()) if args.probe else _load_done(out)
        if done:
            logger.info("{}: resuming — {} cells already scored in {}",
                        model_key, len(done), out.name)

        tot_hit = tot_miss = 0
        n_complete = n_resumed = n_skipped = n_nouniverse = 0
        for row in rows:
            key = (row.question_id, row.spec_level)
            if key in done:
                n_resumed += 1
                continue
            paraphrases = universes.get(key)
            if not paraphrases:
                n_nouniverse += 1
                continue
            if args.max_paraphrases:
                paraphrases = paraphrases[: args.max_paraphrases]
            f_resp, samples, hit, miss = recover_cell_responses(
                config, cache, row, model_key, paraphrases,
                k_samples=k_samples, evidence_max_chars=evidence_max_chars,
                context_mode=context_mode,
            )
            tot_hit += hit
            tot_miss += miss
            if f_resp is None:
                n_skipped += 1
                continue
            n_complete += 1
            if args.probe:
                continue
            rec = score_cell(config, row, f_resp, samples, k_samples)
            rec["model_key"] = model_key
            recs.append(rec)
            if len(recs) % args.checkpoint_every == 0:
                _checkpoint(recs, out)
                logger.info("  checkpoint: {} cells in {} ({})",
                            len(recs), out.name, model_key)

        total = tot_hit + tot_miss
        logger.info(
            "{}: cache {}/{} hits ({:.1%}) | new {} / resumed {} / skipped {} / no-universe {}",
            model_key, tot_hit, total, (tot_hit / total) if total else 0.0,
            n_complete, n_resumed, n_skipped, n_nouniverse,
        )
        if args.probe:
            continue
        if not recs:
            logger.error(
                "{}: no cell fully recovered from the cache — the v3 generations are "
                "not on this machine. Run this on the cluster (or sync data/cache/"
                "llm_cache.sqlite down first).", model_key)
            continue
        _checkpoint(recs, out)
        logger.info("{}: wrote {} cells -> {}", model_key, len(recs), out)
        _print_arm_summary(pd.DataFrame(recs), model_key)
    return 0


def _print_arm_summary(df: pd.DataFrame, model_key: str) -> None:
    """The decomposition the arm exists to produce."""
    p = df.pivot_table(index="question_id", columns="spec_level",
                       values=["f_graded_target_mean", "f_graded_union_mean"])
    try:
        t0 = p[("f_graded_target_mean", 0)]
        t1 = p[("f_graded_target_mean", 1)]
        u0 = p[("f_graded_union_mean", 0)]
        u1 = p[("f_graded_union_mean", 1)]
    except KeyError:
        logger.warning("{}: both levels not present; skipping summary", model_key)
        return
    m = pd.concat([t0, t1, u0, u1], axis=1).dropna()
    m.columns = ["t0", "t1", "u0", "u1"]
    d_target = (m.t1 - m.t0).mean()
    d_union = (m.u1 - m.u0).mean()
    print(f"\n=== R1 union-gold arm — {model_key} (n={len(m)} questions) ===")
    print(f"  target gold : L0={m.t0.mean():.3f}  L1={m.t1.mean():.3f}  Delta={d_target:+.3f}")
    print(f"  union  gold : L0={m.u0.mean():.3f}  L1={m.u1.mean():.3f}  Delta={d_union:+.3f}")
    print(f"  decomposition: Delta_target {d_target:+.3f} = "
          f"Delta_union {d_union:+.3f} + Delta_targeting {d_target - d_union:+.3f}")
    print("  reading: Delta_union is the non-tautological effect (the model reading better);")
    print("           Delta_targeting is the grader being told which reading it wanted.")


def _parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", default="data/union_gold_{model}.parquet",
                    help="output path; {model} is substituted")
    # 150 = the v3 grid. The config default (50) is the smoke size; using it here
    # would silently rescore a third of the run. Verified 2026-08-07: at 150 the
    # (question_id, spec_level) enumeration matches data/specificity_v3_*.parquet
    # exactly — 150/150 questions, 0 either-way difference, universe sizes
    # identical (299 cells of 10 + 1 singleton).
    ap.add_argument("--n-questions", type=int, default=150)
    ap.add_argument("--k-samples", type=int, default=None)
    ap.add_argument("--max-paraphrases", type=int, default=None)
    ap.add_argument("--context-mode", default=None)
    ap.add_argument("--paraphrase-cache", default=None)
    ap.add_argument("--checkpoint-every", type=int, default=10,
                    help="persist progress every N scored cells (30-min windows)")
    ap.add_argument("--probe", action="store_true",
                    help="count cache coverage only; never loads DeBERTa")
    return ap.parse_args(argv)


def main() -> int:
    return run(_parse_args())


if __name__ == "__main__":
    sys.exit(main())
