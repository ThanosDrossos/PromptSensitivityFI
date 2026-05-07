"""2WikiMultihopQA loader (framolfese repackage).

Per Research_Design_v3 §2.1, framolfese/2WikiMultihopQA on HuggingFace stores
the data in the *exact* HotpotQA field layout — same context/supporting_facts
shape, same sentence-list-per-paragraph structure. The only structural
difference is that 2WikiMultihopQA has four `type` values
(comparison, inference, compositional, bridge_comparison) instead of the
HotpotQA two, and there is no `level` field.

This means the parser is a near-clone of `parse_hotpotqa_record`, but kept
separate so dataset-specific quirks have a clear home.
"""

from __future__ import annotations

from typing import Any, Iterable

from .schemas import (
    HotpotParagraph,
    HotpotSupportingFact,
    MultiHopQuestion,
)


def parse_twiki_record(record: dict[str, Any]) -> MultiHopQuestion:
    """Parse one framolfese/2WikiMultihopQA row into MultiHopQuestion."""
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
        dataset="2wikimultihopqa",
        question=record["question"],
        answer=record["answer"],
        question_type=record["type"],
        level=None,
        paragraphs=paragraphs,
        supporting_facts=supporting,
    )


def load_twiki_validation(
    *,
    hf_dataset: str = "framolfese/2WikiMultihopQA",
    hf_config: str | None = None,
    split: str = "validation",
    cache_dir: str | None = None,
) -> list[MultiHopQuestion]:
    """Load and parse the full 2WikiMultihopQA validation split."""
    from datasets import load_dataset  # noqa: WPS433

    ds = load_dataset(hf_dataset, hf_config, split=split, cache_dir=cache_dir)
    return [parse_twiki_record(r) for r in _iterable(ds)]


def _iterable(ds) -> Iterable[dict[str, Any]]:  # type: ignore[no-untyped-def]
    return ds  # type: ignore[return-value]
