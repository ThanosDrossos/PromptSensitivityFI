"""MuSiQue-Answerable loader. Dataset_Evaluation_v6_Dual_Ladder.md §5.

MuSiQue (Trivedi et al., TACL 2022, arXiv:2108.00573; repo
`StonyBrookNLP/musique`) is the primary dataset for GRADED chain-completion
scoring. Each record carries a gold `question_decomposition` (one entry per
hop) alongside the paragraph pool, so F can be the fraction of reasoning hops
the model recovers instead of a binary final-answer match (v6 §2).

We load the **MuSiQue-Ans** (answerable) variant, not the Full variant with
unanswerable twins.

Record schema (canonical jsonl from `download_data.sh`, and most HF mirrors):

    {
      "id": "2hop__128801_128819",
      "question": "...",
      "answer": "...",
      "answer_aliases": [...],            # optional, ignored
      "paragraphs": [
        {"idx": 0, "title": "...", "paragraph_text": "...", "is_supporting": false},
        ...
      ],
      "question_decomposition": [
        {"id": ..., "question": "...", "answer": "...", "paragraph_support_idx": 3},
        ...
      ]
    }

Mapping to `MultiHopQuestion`:
  paragraphs[].paragraph_text  -> HotpotParagraph(sentences=[text], is_gold=is_supporting)
  question_decomposition[]     -> DecompositionHop(hop_idx, sub_question, sub_answer,
                                                   supporting_paragraph_idx)
  n_hops                        = len(question_decomposition)
  question_type                 = f"{n_hops}hop"   (clamped to 2hop/3hop/4hop)

NOTE: MuSiQue paragraph counts vary (~20). Nothing here hardcodes 10.

Loading strategy: this loader supports two sources, tried in order:
  1. A local jsonl file (the canonical release), if `local_path` is given or a
     default lives under `data/raw/musique/`.
  2. A Hugging Face dataset id (a community mirror), via `datasets.load_dataset`.
The HF mirrors vary in field names; `parse_musique_record` normalises the
two shapes we've seen (`paragraph_text` vs `text`, decomposition `answer`
field present).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .schemas import DecompositionHop, HotpotParagraph, MultiHopQuestion


def _hop_count_label(n_hops: int) -> str:
    """MuSiQue is 2-4 hops; clamp to the QuestionType literal labels."""
    n = max(2, min(4, n_hops))
    return f"{n}hop"


def _paragraph_text(p: dict[str, Any]) -> str:
    """Tolerate both `paragraph_text` (canonical) and `text` (some mirrors)."""
    return p.get("paragraph_text") or p.get("text") or ""


def parse_musique_record(record: dict[str, Any]) -> MultiHopQuestion:
    """Parse one MuSiQue jsonl/HF record dict into MultiHopQuestion."""
    raw_paragraphs = record["paragraphs"]
    # Preserve the dataset's paragraph order; `idx` (if present) is the
    # canonical position and we sort by it defensively.
    if raw_paragraphs and isinstance(raw_paragraphs[0], dict) and "idx" in raw_paragraphs[0]:
        raw_paragraphs = sorted(raw_paragraphs, key=lambda p: p.get("idx", 0))

    paragraphs = [
        HotpotParagraph(
            title=str(p.get("title", "")),
            sentences=[_paragraph_text(p)],
            is_gold=bool(p.get("is_supporting", False)),
        )
        for p in raw_paragraphs
    ]

    raw_decomp = record.get("question_decomposition") or []
    decomposition: list[DecompositionHop] = []
    for i, hop in enumerate(raw_decomp):
        decomposition.append(
            DecompositionHop(
                hop_idx=i,
                sub_question=str(hop.get("question", "")),
                sub_answer=str(hop.get("answer", "")),
                supporting_paragraph_idx=hop.get("paragraph_support_idx"),
            )
        )

    n_hops = len(decomposition)

    return MultiHopQuestion(
        id=str(record["id"]),
        dataset="musique",
        question=record["question"],
        answer=record["answer"],
        question_type=_hop_count_label(n_hops),  # type: ignore[arg-type]
        level=None,
        paragraphs=paragraphs,
        supporting_facts=[],          # MuSiQue has no HotpotQA-style supporting_facts
        question_decomposition=decomposition,
        n_hops=n_hops,
    )


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _default_local_paths(repo_root: Path, split: str) -> list[Path]:
    """Canonical MuSiQue-Ans filenames under data/raw/musique/."""
    base = repo_root / "data" / "raw" / "musique"
    # `download_data.sh` produces musique_ans_v1.0_{train,dev,test}.jsonl
    split_alias = {"validation": "dev", "valid": "dev", "dev": "dev",
                   "train": "train", "test": "test"}.get(split, split)
    return [
        base / f"musique_ans_v1.0_{split_alias}.jsonl",
        base / f"musique_ans_v1.0_{split}.jsonl",
        base / f"{split_alias}.jsonl",
    ]


def load_musique_validation(
    *,
    hf_dataset: str | None = None,
    hf_config: str | None = None,
    split: str = "validation",
    local_path: str | None = None,
    repo_root: Path | None = None,
    cache_dir: str | None = None,
) -> list[MultiHopQuestion]:
    """Load MuSiQue-Ans for the given split.

    Resolution order:
      1. explicit `local_path` jsonl, if provided and existing;
      2. a default jsonl under `data/raw/musique/`, if present;
      3. the Hugging Face dataset `hf_dataset` (community mirror).

    At least one source must resolve, else FileNotFoundError / the datasets
    error propagates.
    """
    # 1 + 2: local jsonl.
    candidate_paths: list[Path] = []
    if local_path:
        candidate_paths.append(Path(local_path))
    if repo_root is not None:
        candidate_paths.extend(_default_local_paths(repo_root, split))
    for p in candidate_paths:
        if p.exists():
            return [parse_musique_record(r) for r in _iter_jsonl(p)]

    # 3: Hugging Face mirror.
    if hf_dataset:
        from datasets import load_dataset  # noqa: WPS433 — heavy import, lazy

        # MuSiQue mirrors commonly expose split "validation" or "dev".
        hf_split = split
        ds = load_dataset(hf_dataset, hf_config, split=hf_split, cache_dir=cache_dir)
        return [parse_musique_record(dict(r)) for r in ds]  # type: ignore[arg-type]

    raise FileNotFoundError(
        "MuSiQue not found. Provide a local jsonl via `local_path=` / place it "
        "under data/raw/musique/, or pass `hf_dataset=` for a HF mirror. "
        f"Tried local paths: {[str(p) for p in candidate_paths]}"
    )
