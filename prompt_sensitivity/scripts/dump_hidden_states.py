"""TBG hidden-state dump for FI probes (FI_PROBES_PLAN.md §2-3).

For every (question, spec level, model, paraphrase) cell member, run ONE
forward pass of the exact prompt the eval runs use (same `_assemble_messages`,
same uniform-evidence block, via `run_specificity.load_spec_rows`) and persist
the last-prompt-token hidden state at ~4 fractional-depth layers. These are the
probe features; labels (aufi_in / f_graded / h_sem) come from the metrics
parquet and join on (question_id, spec_level, model_key, paraphrase_idx).

Design properties:
  * STANDALONE — separate from the eval chains; hidden states cannot share the
    generate() forward anyway, so a dedicated resumable job is cleaner than a
    flag inside the battle-tested run driver, and it can backfill the finished
    v2 dataset (first 50 questions) before the v3 run exists.
  * CACHE-ONLY — reads data/paraphrases_ambigqa.parquet and NEVER generates; a
    missing universe is skipped with a warning, so this job can run alongside
    an eval chain without ever racing Phi-4 for VRAM or the cache file.
  * RESUMABLE — per-cell append with atomic replace (singleton-chain safe);
    a cell already in the parquet is skipped.

Storage: one row per (paraphrase, layer); the vector is raw float16 bytes
(`vec` + `dim` + `dtype`) — pyarrow-safe and half the size of float32 lists.

    python -m prompt_sensitivity.scripts.dump_hidden_states \
        --n-questions 50 --models qwen_2_5_7b
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from ..config import load_config
from ..logging_setup import configure_logging
from ..models.registry import get_client
from .e2e_smoke import _assemble_messages
from .local_check import _free_vram
from .run_specificity import (
    _AMBIGQA_PARAPHRASE_PARQUET,
    _evidence_paragraphs,
    _ladder_row_for,
    _SpecQuestionView,
    load_spec_rows,
)

_DEFAULT_LAYER_FRACS = (0.25, 0.5, 0.75, 1.0)


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested)                                                   #
# --------------------------------------------------------------------------- #


def encode_vec(vec: np.ndarray) -> bytes:
    """float16 raw bytes for the parquet `vec` column."""
    return np.ascontiguousarray(vec, dtype=np.float16).tobytes()


def decode_vec(blob: bytes, dim: int) -> np.ndarray:
    """Inverse of encode_vec -> (dim,) float16."""
    out = np.frombuffer(blob, dtype=np.float16)
    if out.shape[0] != dim:
        raise ValueError(f"vector blob has {out.shape[0]} dims, expected {dim}")
    return out


def universe_texts(para: pd.DataFrame, qid: str, level: int, max_n: int) -> list[str] | None:
    """Paraphrase texts of one universe from the cache, in paraphrase_idx order.
    None when the universe is absent (dump SKIPS — it never generates)."""
    sub = para[
        (para["question_id"].astype(str) == str(qid))
        & (para["spec_level"] == level)
        & (para["outcome"].isin(["accepted", "singleton_fallback"]))
    ].sort_values("paraphrase_idx")
    if sub.empty:
        return None
    return sub["text"].tolist()[:max_n]


def done_cells(existing: pd.DataFrame | None) -> set[tuple[str, int]]:
    """(question_id, spec_level) cells already fully in the dump (resume)."""
    if existing is None or existing.empty:
        return set()
    return {
        (str(q), int(lvl))
        for q, lvl in existing.groupby(["question_id", "spec_level"]).groups
    }


def _append_rows(path: Path, rows_new: list[dict]) -> None:
    """Append with atomic replace (same pattern as the paraphrase cache)."""
    new = pd.DataFrame(rows_new)
    if path.exists():
        new = pd.concat([pd.read_parquet(path), new], ignore_index=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    new.to_parquet(tmp, index=False)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-questions", type=int, default=None)
    ap.add_argument("--models", type=str, default="qwen_2_5_7b")
    ap.add_argument("--max-paraphrases", type=int, default=10)
    ap.add_argument("--layer-fracs", type=str, default=",".join(map(str, _DEFAULT_LAYER_FRACS)),
                    help="fractional transformer depths for the TBG states")
    ap.add_argument("--context-mode", choices=["closed_book", "uniform_evidence"], default=None)
    ap.add_argument("--out-template", type=str, default="data/hidden_states_{model}.parquet")
    args = ap.parse_args()

    configure_logging("dump_hidden_states")
    config = load_config()
    root = config.repo_root()
    fracs = tuple(float(x) for x in args.layer_fracs.split(",") if x.strip())

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in models:
        entry = config.models.get(m)
        if entry is None or entry.provider != "local" or not entry.has_hidden:
            logger.error("model {!r} missing / not provider:local / no hidden states", m)
            return 1

    rows, _questions, context_mode, evidence_max_chars = load_spec_rows(
        config, n_questions=args.n_questions, context_mode=args.context_mode
    )

    ppath = root / _AMBIGQA_PARAPHRASE_PARQUET
    if not ppath.exists():
        logger.error("paraphrase cache {} not found — run prep first", ppath)
        return 1
    para = pd.read_parquet(ppath)

    for model_key in models:
        out_path = root / args.out_template.format(model=model_key)
        existing = pd.read_parquet(out_path) if out_path.exists() else None
        skip = done_cells(existing)
        client = get_client(model_key, config)

        n_new = n_skipped = n_missing = 0
        t0 = time.perf_counter()
        for row in rows:
            key = (str(row.question_id), int(row.spec_level))
            if key in skip:
                n_skipped += 1
                continue
            texts = universe_texts(para, row.question_id, row.spec_level, args.max_paraphrases)
            if texts is None:
                logger.warning("universe missing for qid={} L{} — skipped (dump never generates)",
                               row.question_id, row.spec_level)
                n_missing += 1
                continue
            ev = (
                _evidence_paragraphs(row, evidence_max_chars)
                if context_mode == "uniform_evidence" else []
            )
            view = _SpecQuestionView(row, paragraphs=ev)
            lrow = _ladder_row_for(row, n_paragraphs=len(ev))
            messages_batch = [
                [{"role": m.role, "content": m.content}
                 for m in _assemble_messages(view, p, lrow, use_cot=False)]
                for p in texts
            ]
            arr, layer_idxs = client.chat_hidden_states(messages_batch, layer_fracs=fracs)
            dim = arr.shape[-1]
            _append_rows(out_path, [
                {"question_id": str(row.question_id), "spec_level": int(row.spec_level),
                 "model_key": model_key, "paraphrase_idx": p_idx, "paraphrase": texts[p_idx],
                 "context_mode": context_mode, "position": "tbg",
                 "layer_idx": int(layer_idxs[l_pos]), "layer_frac": float(fracs[l_pos]),
                 "dim": int(dim), "dtype": "float16",
                 "vec": encode_vec(arr[p_idx, l_pos])}
                for p_idx in range(arr.shape[0])
                for l_pos in range(arr.shape[1])
            ])
            n_new += 1
            if n_new % 20 == 0:
                logger.info("[{}] {} cells dumped ({:.1f}s/cell)",
                            model_key, n_new, (time.perf_counter() - t0) / n_new)
        logger.info("[{}] done: {} new, {} resumed, {} missing -> {}",
                    model_key, n_new, n_skipped, n_missing, out_path)
        _free_vram()   # next model needs the VRAM

    print("DUMP DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
