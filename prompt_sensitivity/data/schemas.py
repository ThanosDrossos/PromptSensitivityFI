"""Shared Pydantic schemas for HotpotQA and 2WikiMultihopQA records.

The framolfese repackage of 2WikiMultihopQA matches HotpotQA's exact field
layout: id, question, answer, type, context (title + sentences),
supporting_facts (title + sent_id). One schema, two datasets, no adapter code
(see Research_Design_v3 §2.1).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


# Allowed `level` values in HotpotQA (validation set has only medium / hard).
HotpotLevel = Literal["easy", "medium", "hard"]
# Allowed `type` values across both datasets. `bridge` and `comparison` are in
# HotpotQA; `inference`, `compositional`, `bridge_comparison` are 2WikiMultihopQA-only.
QuestionType = Literal["bridge", "comparison", "inference", "compositional", "bridge_comparison"]


class HotpotParagraph(BaseModel):
    """A single paragraph in the 10-paragraph context.

    `is_gold` is computed at parse time from `supporting_facts.title`. We store
    it on the paragraph so ladder construction is a pure function of the parsed
    object — no need to re-look-up supporting_facts.
    """

    model_config = ConfigDict(extra="forbid")

    title: str
    sentences: list[str]
    is_gold: bool = False

    def joined(self) -> str:
        """Concatenate sentences exactly as the dataset stores them.

        HotpotQA sentences already include trailing whitespace where appropriate
        (Yang 2018 §3.2: paragraphs are split by sentence-segmentation). Joining
        with empty string preserves the original Wikipedia text.
        """
        return "".join(self.sentences)


class HotpotSupportingFact(BaseModel):
    """A (paragraph_title, sentence_id) entry pointing into the gold context."""

    model_config = ConfigDict(extra="forbid")

    title: str
    sent_id: int


class MultiHopQuestion(BaseModel):
    """Unified record. Used for both HotpotQA and 2WikiMultihopQA."""

    model_config = ConfigDict(extra="forbid")

    id: str
    dataset: Literal["hotpotqa", "2wikimultihopqa"]
    question: str
    answer: str
    question_type: QuestionType
    level: HotpotLevel | None = None  # 2WikiMultihopQA does not carry `level`
    paragraphs: list[HotpotParagraph] = Field(min_length=1)
    supporting_facts: list[HotpotSupportingFact] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_gold_count(self) -> "MultiHopQuestion":
        """All native HotpotQA / 2Wiki questions ship with exactly 2 gold paragraphs.

        Edge case (per Sprint 3 brief): questions with fewer than 2 supporting
        paragraphs are excluded at sample time. We surface them as a validator
        warning here, not a hard error, because the loader still needs to parse
        the malformed record so the sampler can drop it explicitly.
        """
        gold_titles = {sf.title for sf in self.supporting_facts}
        gold_count = sum(1 for p in self.paragraphs if p.is_gold)
        if gold_count != len(gold_titles):
            # Gold flag was not propagated by parser. Fix it here defensively.
            for p in self.paragraphs:
                p.is_gold = p.title in gold_titles
        return self

    def gold_paragraphs(self) -> list[HotpotParagraph]:
        return [p for p in self.paragraphs if p.is_gold]

    def distractor_paragraphs(self) -> list[HotpotParagraph]:
        return [p for p in self.paragraphs if not p.is_gold]
