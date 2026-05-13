"""HotpotQA distractor-config validation loader.

Per Research_Design_v3 §2 (and the HotpotQA dataset card on HuggingFace), each
record has fields:

    id                str
    question          str
    answer            str
    type              str  ("bridge" | "comparison")
    level             str  ("easy" | "medium" | "hard")
    context           {title: list[str], sentences: list[list[str]]}
    supporting_facts  {title: list[str], sent_id: list[int]}

We parse this into our schema-unified `MultiHopQuestion`.
"""

from __future__ import annotations

from typing import Any, Iterable

from .schemas import (
    HotpotParagraph,
    HotpotSupportingFact,
    MultiHopQuestion,
)


def parse_hotpotqa_record(record: dict[str, Any]) -> MultiHopQuestion:
    """Parse one HotpotQA dict (HF datasets row) into MultiHopQuestion.

    The HuggingFace `hotpotqa/hotpot_qa` row shape uses paired-list columns
    (e.g. context.title is a list, context.sentences is a list-of-lists indexed
    in the same order). We zip them to a list of paragraph dicts before
    Pydantic validation.
    """

    ctx = record["context"]
    titles = ctx["title"]
    sentences_lists = ctx["sentences"]
    if len(titles) != len(sentences_lists):
        raise ValueError(
            f"context arrays mismatched: titles={len(titles)} sentences={len(sentences_lists)}"
        )

    sf = record["supporting_facts"]
    sf_pairs = list(zip(sf["title"], sf["sent_id"], strict=True))
    gold_titles = {t for t, _ in sf_pairs}

    paragraphs = [
        HotpotParagraph(title=t, sentences=list(s), is_gold=(t in gold_titles))
        for t, s in zip(titles, sentences_lists, strict=True)
    ]

    supporting = [HotpotSupportingFact(title=t, sent_id=int(i)) for t, i in sf_pairs]

    return MultiHopQuestion(
        id=str(record["id"]),
        dataset="hotpotqa",
        question=record["question"],
        answer=record["answer"],
        question_type=record["type"],
        level=record.get("level"),
        paragraphs=paragraphs,
        supporting_facts=supporting,
    )


def load_hotpotqa_validation(
    *,
    hf_dataset: str = "hotpotqa/hotpot_qa",
    hf_config: str = "distractor",
    split: str = "validation",
    cache_dir: str | None = None,
) -> list[MultiHopQuestion]:
    """Load and parse the full HotpotQA distractor validation split.

    HotpotQA was migrated to parquet on the HF Hub in 2024; the legacy script
    loader (which needed `trust_remote_code=True`) is gone and the kwarg now
    raises a deprecation warning. We drop it.
    """
    from datasets import load_dataset  # noqa: WPS433 — heavy import, lazy

    ds = load_dataset(
        hf_dataset,
        hf_config,
        split=split,
        cache_dir=cache_dir,
    )
    return [parse_hotpotqa_record(r) for r in _iterable(ds)]


def _iterable(ds) -> Iterable[dict[str, Any]]:  # type: ignore[no-untyped-def]
    """`datasets.Dataset` is iterable but not typed as such by some stubs."""
    return ds  # type: ignore[return-value]
