"""AmbigQA specificity driver (pivot spec §8) — thin, reuses the e2e cell helpers.

The manipulated axis is QUESTION SPECIFICITY: each AmbigQA question yields two
CLOSED-BOOK cells (spec_level 0 = ambiguous Q, 1 = disambiguated Q_i; empty
paragraph list -> `_assemble_messages` builds the no-context prompt). The
scoring gold a_i is FIXED across both levels (build_levels guardrail); scoring
is binary NLI-with-gold, OR'd over the interpretation's answer variants
(`f_score_batch_multi_gold`). FI_spec = log2(m0/m_valid) is attached from
`metrics.fi_spec` — dataset-side, not orchestrator-side.

Per-cell checkpoint to --out (resume key: question_id, spec_level, model_key).
Paraphrase universes are built per (question, level) over that level's question
text and persisted to data/paraphrases_ambigqa.parquet.

Cluster-only in practice (provider:local models); run via cluster sbatch.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
from loguru import logger

from ..config import load_config
from ..data.ambigqa_schemas import AmbigQuestion
from ..data.load_ambigqa import load_ambigqa
from ..ladders.schemas import LadderRow
from ..logging_setup import configure_logging
from ..metrics import build_metric_tuple
from ..metrics.fi_spec import fi_spec_bits
from ..metrics.h_sem import cluster_responses_pooled
from ..models.embedding import encode_texts
from ..models.registry import get_client
from ..scoring.nli_with_gold import f_score_batch_multi_gold
from ..data.schemas import HotpotParagraph
from ..specificity.build_levels import SpecRow, build_spec_levels, target_in_evidence
from .e2e_smoke import _assemble_messages, _checkpoint, _clustering_inputs, _sample_response
from .local_check import _free_vram


_AMBIGQA_PARAPHRASE_PARQUET = "data/paraphrases_ambigqa.parquet"


# --------------------------------------------------------------------------- #
# Adapters (spec §8.5): closed-book cell via the EXISTING machinery            #
# --------------------------------------------------------------------------- #


class _SpecQuestionView:
    """Duck-typed stand-in for MultiHopQuestion, restricted to what the reused
    cell helpers touch. MultiHopQuestion itself is NOT changed (scope guard §12).

    v2: `paragraphs` carries the question's evidence snippets wrapped as
    HotpotParagraph, so the EXISTING context-block prompt path renders them;
    closed-book mode passes an empty list (v1 behaviour).
    """

    def __init__(self, row: SpecRow, paragraphs: list | None = None) -> None:
        self.id = row.question_id
        self.dataset = "ambigqa"
        self.question = row.question_text
        self.answer = row.target_answers[0]   # primary gold (multi-gold OR in scoring)
        self.paragraphs: list = paragraphs or []
        self.question_decomposition: list = []
        self.n_hops = None

    def has_decomposition(self) -> bool:      # -> binary NLI path, no CoT
        return False


def _evidence_paragraphs(row: SpecRow, max_chars: int) -> list[HotpotParagraph]:
    """Wrap the row's evidence snippets as context paragraphs.

    Whole snippets in dataset order until the char cap (median bundle ~4.8k,
    p75 ~6.3k). The SAME list is used at both levels and for every paraphrase —
    the v2 uniformity guardrail.
    """
    out: list[HotpotParagraph] = []
    used = 0
    for ev in row.evidence:
        n = len(ev.title) + len(ev.snippet)
        if out and used + n > max_chars:
            break
        out.append(HotpotParagraph(title=ev.title, sentences=[ev.snippet]))
        used += n
    return out


def _ladder_row_for(row: SpecRow, n_paragraphs: int = 0) -> LadderRow:
    """Context-family LadderRow over the (possibly empty) evidence block."""
    return LadderRow(
        question_id=row.question_id,
        ladder_type="random",
        ladder_family="context",
        level_idx=row.spec_level,
        level=row.spec_level,
        paragraph_indices=list(range(n_paragraphs)),
        paragraph_titles=[],
        gold_count=0,
    )


# --------------------------------------------------------------------------- #
# Paraphrase universes per (question, spec level)                              #
# --------------------------------------------------------------------------- #


def _generate_spec_paraphrases(
    config, rows: list[SpecRow], max_paraphrases: int
) -> dict[tuple[str, int], list[str]]:
    """Build (and persist) the paraphrase universe per (question_id, spec_level).

    Each LEVEL has its own universe over its own question text — the NLI
    equivalence filter keeps specificity constant WITHIN a level. Cached in
    data/paraphrases_ambigqa.parquet keyed by (question_id, spec_level); a
    question that yields nothing falls back to its own text as a singleton.
    """
    from ..paraphrases.pipeline import build_paraphrase_set

    parquet_path = config.repo_root() / _AMBIGQA_PARAPHRASE_PARQUET

    persisted: dict[tuple[str, int], list[str]] = {}
    if parquet_path.exists():
        try:
            prev = pd.read_parquet(parquet_path)
            # singleton_fallback rows count as cached too — a universe that
            # legitimately yielded nothing must not be re-attempted every resume
            # (each retry costs a full Phi-4 pipeline pass).
            prev = prev[prev["outcome"].isin(["accepted", "singleton_fallback"])]
            for (qid, lvl), sub in prev.groupby(["question_id", "spec_level"]):
                persisted[(str(qid), int(lvl))] = (
                    sub.sort_values("paraphrase_idx")["text"].tolist()
                )
            logger.info("loaded persisted AmbigQA paraphrases for {} cells", len(persisted))
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not read {} ({}); regenerating", parquet_path, exc)

    out: dict[tuple[str, int], list[str]] = {}
    n_todo = sum(1 for r in rows if (r.question_id, r.spec_level) not in persisted)
    n_gen = 0
    for row in rows:
        key = (row.question_id, row.spec_level)
        if key in persisted and persisted[key]:
            out[key] = persisted[key][:max_paraphrases]
            continue
        n_gen += 1
        logger.info("paraphrase universe {}/{}: qid={} L{}",
                    n_gen, n_todo, row.question_id, row.spec_level)
        try:
            pset = build_paraphrase_set(
                f"{row.question_id}::L{row.spec_level}",
                row.question_text,
                config=config,
                gold_answer=row.target_answers[0],
            )
            texts = [ap.text for ap in pset.accepted][:max_paraphrases]
            outcome = "accepted"
        except Exception as exc:  # noqa: BLE001
            logger.warning("paraphrase gen failed for {} L{}: {}",
                           row.question_id, row.spec_level, exc)
            texts = []
            outcome = "accepted"
        if not texts:
            logger.warning("qid={} L{} no paraphrases; singleton fallback",
                           row.question_id, row.spec_level)
            texts = [row.question_text]
            outcome = "singleton_fallback"
        out[key] = texts
        # Persist THIS universe immediately (atomic replace) so a walltime kill
        # mid-prep loses at most the universe in flight — required for the
        # 30-min gpu_a100_short singleton-chain to make monotonic progress.
        _append_paraphrase_rows(parquet_path, [
            {"question_id": row.question_id, "spec_level": row.spec_level,
             "outcome": outcome, "paraphrase_idx": idx, "text": text}
            for idx, text in enumerate(texts)
        ])
    if n_gen:
        logger.info("generated + persisted {} new universes -> {}", n_gen, parquet_path)
    return out


def _append_paraphrase_rows(parquet_path, rows_new: list[dict]) -> None:
    """Append rows to the paraphrase cache with an atomic replace."""
    import os

    df_new = pd.DataFrame(rows_new)
    if parquet_path.exists():
        try:
            df_new = pd.concat([pd.read_parquet(parquet_path), df_new], ignore_index=True)
        except Exception:  # noqa: BLE001
            pass
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = parquet_path.with_suffix(parquet_path.suffix + ".tmp")
    df_new.to_parquet(tmp, index=False)
    os.replace(tmp, parquet_path)


# --------------------------------------------------------------------------- #
# One closed-book specificity cell                                             #
# --------------------------------------------------------------------------- #


def _run_spec_cell(
    config,
    row: SpecRow,
    model_key: str,
    paraphrases: list[str],
    *,
    k_samples: int,
    fast: bool,
    inspect: bool = False,
    context_mode: str = "uniform_evidence",
    evidence_max_chars: int = 6000,
) -> tuple[dict, dict | None]:
    """One (SpecRow, model) cell -> (flat result row, inspection record | None).
    Mirrors the e2e binary (non-CoT) path but scores with the multi-gold OR and
    attaches FI_spec. v2: uniform evidence block unless context_mode=closed_book."""
    model_entry = config.models[model_key]
    client = get_client(model_key, config)
    ev_paragraphs = (
        _evidence_paragraphs(row, evidence_max_chars)
        if context_mode == "uniform_evidence" else []
    )
    view = _SpecQuestionView(row, paragraphs=ev_paragraphs)
    lrow = _ladder_row_for(row, n_paragraphs=len(ev_paragraphs))
    gen_max = config.generation.answer_max_tokens
    logger.info("cell qid={} spec_level={} model={} N={} m0={} m_valid={} evidence={}snip",
                row.question_id, row.spec_level, model_key,
                len(paraphrases), row.m0, row.m_valid, len(ev_paragraphs))

    prompt_messages = [
        _assemble_messages(view, p, lrow, use_cot=False) for p in paraphrases
    ]
    prompt_user_texts = [m[1].content for m in prompt_messages]

    # --- F(x) at T=0, multi-gold OR over the target's answer variants --------
    f_responses = [
        _sample_response(
            client, model_entry, msgs, temperature=0.0, seed=42,
            purpose=f"spec_f::{row.question_id}::L{row.spec_level}::{model_key}",
            max_tokens=gen_max,
        )
        for msgs in prompt_messages
    ]
    f_scores = [float(x) for x in f_score_batch_multi_gold(
        row.target_answers, f_responses, config=config
    )]
    f_scores_perm = [float(x) for x in f_score_batch_multi_gold(
        row.target_answers, f_responses, config=config, permissive=True
    )]
    logger.info("  F: mean={:.3f} (permissive {:.3f})",
                float(np.mean(f_scores)) if f_scores else 0.0,
                float(np.mean(f_scores_perm)) if f_scores_perm else 0.0)

    n = len(paraphrases)
    if fast:
        cluster_assignments: dict[int, list[int]] = {}
        prompt_embeddings = np.zeros((n, 1), dtype=np.float32)
        response_embeddings: dict[int, np.ndarray] = {}
    else:
        responses_per_paraphrase: dict[int, list[str]] = {}
        for i, msgs in enumerate(prompt_messages):
            responses_per_paraphrase[i] = [
                _sample_response(
                    client, model_entry, msgs,
                    temperature=config.h_sem.sampling_temperature,
                    seed=10000 + i * 100 + kk,
                    purpose=(f"spec_hsem::{row.question_id}::L{row.spec_level}"
                             f"::{model_key}::s{kk}"),
                    max_tokens=gen_max,
                )
                for kk in range(k_samples)
            ]
        clustering_inputs = _clustering_inputs(
            responses_per_paraphrase, config.h_sem.cluster_on
        )
        cluster_assignments = cluster_responses_pooled(clustering_inputs, config=config)
        prompt_embeddings = encode_texts(prompt_user_texts, config=config)
        response_embeddings = {
            i: encode_texts(samples, config=config)
            for i, samples in responses_per_paraphrase.items()
        }

    tup = build_metric_tuple(
        question_id=row.question_id,
        ladder_type="random",
        level=row.spec_level,
        model_key=model_key,
        scores=f_scores,
        cluster_assignments=cluster_assignments,
        prompt_embeddings=prompt_embeddings,
        response_embeddings=response_embeddings,
        posix_log_p=None,
        posix_lengths=None,
        encoder_label="external_mpnet",
        config=config,
        fi_spec=fi_spec_bits(row.m0, row.m_valid),
        spec_level=row.spec_level,
        m_valid=row.m_valid,
        m0=row.m0,
        target_idx=row.target_idx,
    )
    row_dict = tup.model_dump()
    # Post-dump extras, e2e pattern (metrics/ untouched).
    row_dict["dataset"] = "ambigqa"
    row_dict["question_text"] = row.question_text
    row_dict["f_mean_permissive"] = float(np.mean(f_scores_perm)) if f_scores_perm else None
    row_dict["context_mode"] = context_mode
    row_dict["n_evidence_snippets"] = len(ev_paragraphs)

    # Inspection capture (per-run audit deliverable): everything needed to check
    # this cell start->finish by hand. H_sem is summarised (distinct clusters +
    # one representative answer per cluster) rather than dumped in full.
    inspect_rec: dict | None = None
    if inspect:
        hsem_summary = None
        if not fast and cluster_assignments:
            reps: dict[int, str] = {}
            for i, assigns in cluster_assignments.items():
                for s_idx, cl in enumerate(assigns):
                    reps.setdefault(int(cl), responses_per_paraphrase[i][s_idx][:80])
            hsem_summary = {"k": k_samples, "n_clusters": len(reps), "representatives": reps}
        inspect_rec = {
            "question_id": row.question_id, "spec_level": row.spec_level,
            "question_text": row.question_text, "target_answers": list(row.target_answers),
            "m0": row.m0, "m_valid": row.m_valid, "target_idx": row.target_idx,
            "model_key": model_key,
            "context_mode": context_mode,
            "evidence": [f"{p.title}: {p.sentences[0][:120]}" for p in ev_paragraphs[:4]],
            "n_evidence": len(ev_paragraphs),
            "paraphrases": [
                {"idx": i, "paraphrase": p, "answer_t0": f_responses[i][:200],
                 "f": f_scores[i]}
                for i, p in enumerate(paraphrases)
            ],
            "hsem": hsem_summary,
            "metrics": {k: row_dict.get(k) for k in
                        ("f_mean", "aufi_in", "fi_out_mean", "h_sem_mean", "a_q", "fi_spec")},
        }
    return row_dict, inspect_rec


def _cell_key(qid, spec_level, model_key) -> tuple:
    return (str(qid), int(spec_level), str(model_key))


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-questions", type=int, default=None,
                        help="default: config.sampling.ambigqa.n_questions (50)")
    parser.add_argument("--models", type=str, default="qwen_2_5_7b")
    parser.add_argument("--k-samples", type=int, default=None,
                        help="H_sem samples/prompt; default config.h_sem.n_samples_per_prompt")
    parser.add_argument("--max-paraphrases", type=int, default=10)
    parser.add_argument("--fast", action="store_true",
                        help="skip H_sem sampling/clustering/embeddings "
                             "(FI_in + accuracy + FI_spec only)")
    parser.add_argument("--inspect-n", type=int, default=0,
                        help="write a start->finish audit bundle for the first N "
                             "questions (both levels) to data/inspect_<out-stem>.md")
    parser.add_argument("--context-mode", choices=["closed_book", "uniform_evidence"],
                        default=None,
                        help="override config.specificity.context_mode "
                             "(closed_book reproduces the v1 design)")
    parser.add_argument("--out", type=str, default="data/specificity_metrics.parquet")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    configure_logging("run_specificity")
    config = load_config()
    repo_root = config.repo_root()

    acfg = config.sampling.ambigqa
    if acfg is None:
        logger.error("config.sampling.ambigqa missing — add the block from the pivot spec §9")
        return 1
    spec_cfg = config.specificity
    target_seed = spec_cfg.target_seed if spec_cfg is not None else config.random_seed
    n_questions = args.n_questions if args.n_questions is not None else acfg.n_questions
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in models:
        if m not in config.models:
            logger.error("unknown model_key {!r}; available: {}", m, list(config.models))
            return 1
    k_samples = args.k_samples if args.k_samples is not None else config.h_sem.n_samples_per_prompt
    if not args.fast and k_samples < 5:
        logger.warning("H_sem k={} (<5): near-zero semantic-entropy resolution", k_samples)

    # --- context mode (v2: uniform evidence) --------------------------------
    context_mode = args.context_mode or (
        spec_cfg.context_mode if spec_cfg is not None else "uniform_evidence"
    )
    evidence_max_chars = spec_cfg.evidence_max_chars if spec_cfg is not None else 6000
    require_cov = (
        context_mode == "uniform_evidence"
        and (spec_cfg.require_target_in_evidence if spec_cfg is not None else True)
    )

    # --- questions -> spec rows ---------------------------------------------
    ambiguous = [
        q for q in load_ambigqa(
            hf_dataset=acfg.hf_dataset, hf_config=acfg.hf_config, split=acfg.split,
            min_interpretations=acfg.min_interpretations,
            include_single_answer_anchor=acfg.include_single_answer_anchor,
        )
        if q.is_ambiguous()
    ]
    if require_cov:
        # v2 evidence-coverage filter: dataset-side + model-free (no selection on
        # model knowledge). ~52% of the ambiguous split passes.
        n_before = len(ambiguous)
        ambiguous = [q for q in ambiguous if target_in_evidence(q, seed=target_seed)]
        logger.info("evidence-coverage filter: {} -> {} questions (target answer in bundle)",
                    n_before, len(ambiguous))
    questions: list[AmbigQuestion] = ambiguous[:n_questions]
    if not questions:
        logger.error("no AmbigQA questions left after filtering; bailing")
        return 1
    rows: list[SpecRow] = []
    for q in questions:
        rows.extend(build_spec_levels(
            q, seed=target_seed,
            include_evidence=context_mode == "uniform_evidence",
        ))
    logger.info("context_mode={} (evidence cap {} chars)", context_mode, evidence_max_chars)

    n_cells = len(rows) * len(models)
    per_cell_calls = args.max_paraphrases * (1 + (0 if args.fast else k_samples))
    logger.info("plan: {} questions -> {} spec rows -> {} cells, ~{} LLM calls (k={}, fast={})",
                len(questions), len(rows), n_cells, n_cells * per_cell_calls,
                k_samples, args.fast)
    if args.dry_run:
        print(json.dumps({
            "dry_run": True, "n_questions": len(questions), "n_cells": n_cells,
            "estimated_llm_calls": n_cells * per_cell_calls,
            "models": models, "levels": sorted({r.spec_level for r in rows}),
            "m0_values": sorted({r.m0 for r in rows}),
            "context_mode": context_mode,
            "evidence_snippets_median": float(np.median([len(r.evidence) for r in rows])),
        }, indent=2))
        return 0

    # --- paraphrase universes (per question x level) -------------------------
    paraphrases = _generate_spec_paraphrases(config, rows, args.max_paraphrases)

    # Free the generator's VRAM before the eval model loads (job 5762430
    # post-mortem): the Phi-4 generator/judge (28 GiB bf16) stayed resident after
    # prep, and Phi-4 + Qwen (~15 GiB) exceed the 40 GB A100 on gpu_a100_short ->
    # CUDA OOM in every cell. The MuSiQue flow dodged this structurally (prep ran
    # as a separate job); this driver runs both phases in one process, so it must
    # drop the weights explicitly. DeBERTa (~1.6 GB, needed again for scoring +
    # clustering) lives in a different cache and survives. No-op when the
    # universes came from the parquet cache and no generator was ever loaded.
    _free_vram()

    # --- cells with per-cell checkpoint + resume ------------------------------
    out_path = repo_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows_out: list[dict] = []
    done: set[tuple] = set()
    if out_path.exists():
        try:
            prev = pd.read_parquet(out_path)
            rows_out = prev.to_dict("records")
            done = {
                _cell_key(r.get("question_id"), r.get("spec_level"), r.get("model_key"))
                for r in rows_out
            }
            logger.info("resume: {} cells already in {}", len(done), out_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not read {} for resume ({}); starting fresh", out_path, exc)
            rows_out, done = [], set()

    inspect_qids = {q.id for q in questions[: args.inspect_n]} if args.inspect_n > 0 else set()
    inspect_recs: list[dict] = []
    n_done = n_failed = n_skipped = 0
    for row in rows:
        for model_key in models:
            key = _cell_key(row.question_id, row.spec_level, model_key)
            if key in done:
                n_skipped += 1
                continue
            t0 = time.perf_counter()
            try:
                row_dict, inspect_rec = _run_spec_cell(
                    config, row, model_key,
                    paraphrases[(row.question_id, row.spec_level)],
                    k_samples=k_samples, fast=args.fast,
                    inspect=row.question_id in inspect_qids,
                    context_mode=context_mode,
                    evidence_max_chars=evidence_max_chars,
                )
            except Exception:  # noqa: BLE001 — isolate cell failures
                n_failed += 1
                logger.exception("cell FAILED qid={} L{} model={} — continuing",
                                 row.question_id, row.spec_level, model_key)
                continue
            rows_out.append(row_dict)
            if inspect_rec is not None:
                inspect_recs.append(inspect_rec)
            done.add(key)
            n_done += 1
            logger.info("cell {} done in {:.1f}s — qid={} L{} model={}",
                        n_done, time.perf_counter() - t0,
                        row.question_id, row.spec_level, model_key)
            _checkpoint(rows_out, out_path)

    logger.info("done: {} new, {} skipped (resume), {} failed. total: {}",
                n_done, n_skipped, n_failed, len(rows_out))
    if not rows_out:
        logger.error("no cells produced")
        return 1
    if inspect_recs:
        insp = out_path.parent / f"inspect_{out_path.stem}.md"
        insp.write_text(_render_spec_inspection_md(inspect_recs), encoding="utf-8")
        logger.info("inspection bundle: {} cell(s) -> {}", len(inspect_recs), insp)
    _print_summary(pd.DataFrame(rows_out))
    return 0 if n_failed == 0 else 2


def _render_spec_inspection_md(records: list[dict]) -> str:
    """Human-readable start->finish audit for the inspected questions: ambiguous
    vs disambiguated text, the fixed gold, every paraphrase with the model's T=0
    answer and its F, an H_sem cluster summary, and the cell metrics."""
    def t(s, n: int = 110) -> str:
        s = " ".join(str(s).split()).replace("|", "\\|")
        return s if len(s) <= n else s[:n] + "…"

    by_q: dict[str, list[dict]] = {}
    for r in records:
        by_q.setdefault(r["question_id"], []).append(r)

    out = ["# Specificity run — inspection bundle", "",
           f"{len(by_q)} question(s), {len(records)} cell(s). Gold is FIXED across "
           "levels (the guardrail); level 0 = ambiguous, level 1 = disambiguated.", ""]
    for qi, (qid, cells) in enumerate(by_q.items(), 1):
        cells = sorted(cells, key=lambda c: c["spec_level"])
        c0 = cells[0]
        out += [f"## {qi}. `{qid}` — m0={c0['m0']}, target_idx={c0['target_idx']}, "
                f"model `{c0['model_key']}`",
                f"**Fixed gold (a_i variants):** {', '.join(c0['target_answers'])}", ""]
        if c0.get("n_evidence"):
            out.append(f"**Uniform evidence** ({c0['n_evidence']} snippets, identical for "
                       "both levels & all paraphrases); first entries:")
            out += [f"- {t(e, 130)}" for e in c0.get("evidence", [])]
            out.append("")
        for c in cells:
            m = c["metrics"]
            out += [f"### Level {c['spec_level']} — “{t(c['question_text'])}”",
                    f"FI_spec={m['fi_spec']:.3f} · f_mean={_fmtf(m['f_mean'])} · "
                    f"AUFI_in={_fmtf(m['aufi_in'])} · FI_out={_fmtf(m['fi_out_mean'])} · "
                    f"H_sem={_fmtf(m['h_sem_mean'])} · |A_q|={m['a_q']}", "",
                    "| # | paraphrase | model answer (T=0) | F |",
                    "|--:|------------|--------------------|--:|"]
            for p in c["paraphrases"]:
                out.append(f"| {p['idx']} | {t(p['paraphrase'], 80)} | "
                           f"{t(p['answer_t0'], 80)} | {p['f']:.0f} |")
            out.append("")
            if c["hsem"]:
                h = c["hsem"]
                reps = " · ".join(f"c{cl}: “{t(txt, 40)}”"
                                  for cl, txt in sorted(h["representatives"].items())[:8])
                out += [f"H_sem (k={h['k']}): {h['n_clusters']} clusters — {reps}", ""]
    return "\n".join(out)


def _fmtf(v) -> str:
    return "—" if v is None else f"{v:.3f}"


def _print_summary(df: pd.DataFrame) -> None:
    print()
    print("=" * 90)
    print("SPECIFICITY PIVOT — per-level means")
    print("=" * 90)
    cols = [c for c in ["f_mean", "f_mean_permissive", "aufi_in", "fi_out_mean",
                        "h_sem_mean", "a_q", "fi_spec"] if c in df.columns]
    print(df.groupby("spec_level")[cols].mean().to_string(float_format=lambda x: f"{x:.3f}"))
    print()


if __name__ == "__main__":
    sys.exit(main())
