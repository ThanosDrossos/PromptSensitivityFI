"""TBG states for the vagueness-head HOLDOUT: AmbigQA's non-ambiguous rows.

The vagueness head was trained to separate L0 (ambiguous) from L1
(disambiguated) prompts of the SAME questions, OOF by question. This dump
enables the stronger, held-out test the paper needs: AmbigQA's annotators also
marked ~800 NQ questions as NOT ambiguous (single interpretation — rows the
main pipeline drops via min_interpretations=2). Those carry a human
"unambiguous" label produced by a different mechanism than disambiguation
rewriting, for questions the head has never seen.

This script dumps ONE prompt per question (the question text itself — no
paraphrases, no scoring, no gold) in the exact training prompt format
(same system message, same uniform-evidence block, same chat assembly as
run_specificity/dump_hidden_states), for:

  * every non-ambiguous anchor row with a non-empty evidence bundle
    (label 0 = specific), and
  * every AMBIGUOUS question with non-empty evidence (label 1 = vague;
    the overlap with the 149 v3 training questions is excluded at EVAL
    time, laptop-side, so the dump stays complete and reusable).

Output: data/vagueness_holdout_{model}.parquet — same schema as the main
hidden-state dumps plus an `ambiguous` label column. Evaluate laptop-side by
applying the FROZEN feedback bundles (no retraining).

    python -m prompt_sensitivity.scripts.dump_vagueness_holdout \
        --models qwen_2_5_7b,llama_3_1_8b,mistral_7b_v03
"""

from __future__ import annotations

import argparse
import time

import pandas as pd
from loguru import logger

from ..config import load_config
from ..data.load_ambigqa import load_ambigqa
from ..logging_setup import configure_logging
from ..models.registry import get_client
from ..specificity.build_levels import SpecRow, choose_target_idx
from .dump_hidden_states import _DEFAULT_LAYER_FRACS, _append_rows, encode_vec
from .run_specificity import (
    _SpecQuestionView,
    _evidence_paragraphs,
    _ladder_row_for,
)


def _holdout_rows(config) -> list[tuple[SpecRow, int]]:
    """[(single-prompt SpecRow, ambiguous_label)] for the holdout pool."""
    acfg = config.sampling.ambigqa
    seed = (config.specificity.target_seed
            if config.specificity is not None else config.random_seed)
    questions = load_ambigqa(
        hf_dataset=acfg.hf_dataset, hf_config=acfg.hf_config, split=acfg.split,
        min_interpretations=1,               # keep the non-ambiguous anchors
        include_single_answer_anchor=True,
    )
    out: list[tuple[SpecRow, int]] = []
    n_no_ev = 0
    for q in questions:
        if not q.evidence:
            n_no_ev += 1                     # format match needs an evidence block
            continue
        m0 = q.m0()
        idx = choose_target_idx(q.id, m0, seed=seed)
        target = q.interpretations[idx]
        row = SpecRow(
            question_id=q.id, spec_level=0, question_text=q.question,
            target_answers=list(target.answers), m_valid=m0, m0=m0,
            target_idx=idx, evidence=list(q.evidence),
        )
        out.append((row, int(m0 >= 2)))
    n_amb = sum(lbl for _, lbl in out)
    logger.info("holdout pool: {} prompts ({} ambiguous / {} specific; "
                "{} dropped for empty evidence)",
                len(out), n_amb, len(out) - n_amb, n_no_ev)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", type=str, default="qwen_2_5_7b")
    ap.add_argument("--layer-fracs", type=str,
                    default=",".join(map(str, _DEFAULT_LAYER_FRACS)))
    ap.add_argument("--out-template", type=str,
                    default="data/vagueness_holdout_{model}.parquet")
    args = ap.parse_args()

    configure_logging("dump_vagueness_holdout")
    config = load_config()
    root = config.repo_root()
    fracs = tuple(float(x) for x in args.layer_fracs.split(",") if x.strip())
    evidence_max_chars = (config.specificity.evidence_max_chars
                         if config.specificity is not None else 6000)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in models:
        entry = config.models.get(m)
        if entry is None or entry.provider != "local" or not entry.has_hidden:
            logger.error("model {!r} missing / not provider:local / no hidden states", m)
            return 1

    pool = _holdout_rows(config)
    from .run_specificity import _assemble_messages  # late import (heavy module)

    for model_key in models:
        out_path = root / args.out_template.format(model=model_key)
        done: set[str] = set()
        if out_path.exists():
            done = set(pd.read_parquet(out_path)["question_id"].astype(str))
        client = get_client(model_key, config)
        n_new = 0
        t0 = time.perf_counter()
        for row, label in pool:
            if str(row.question_id) in done:
                continue
            ev = _evidence_paragraphs(row, evidence_max_chars)
            view = _SpecQuestionView(row, paragraphs=ev)
            lrow = _ladder_row_for(row, n_paragraphs=len(ev))
            messages = [[
                {"role": msg.role, "content": msg.content}
                for msg in _assemble_messages(view, row.question_text, lrow,
                                              use_cot=False)
            ]]
            arr, layer_idxs = client.chat_hidden_states(messages, layer_fracs=fracs)
            dim = arr.shape[-1]
            _append_rows(out_path, [
                {"question_id": str(row.question_id), "spec_level": 0,
                 "model_key": model_key, "paraphrase_idx": 0,
                 "paraphrase": row.question_text,
                 "context_mode": "uniform_evidence", "position": "tbg",
                 "layer_idx": int(layer_idxs[pos]), "layer_frac": float(fracs[pos]),
                 "dim": int(dim), "dtype": "float16",
                 "vec": encode_vec(arr[0, pos]),
                 "ambiguous": int(label), "m0": int(row.m0)}
                for pos in range(arr.shape[1])
            ])
            n_new += 1
            if n_new % 50 == 0:
                logger.info("[{}] {} prompts dumped ({:.1f}s/prompt)",
                            model_key, n_new, (time.perf_counter() - t0) / n_new)
        logger.info("[{}] holdout dump done: {} new -> {}", model_key, n_new, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
