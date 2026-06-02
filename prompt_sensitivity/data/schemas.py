"""Shared Pydantic schemas for HotpotQA, 2WikiMultihopQA, and MuSiQue records.

The framolfese repackage of 2WikiMultihopQA matches HotpotQA's exact field
layout: id, question, answer, type, context (title + sentences),
supporting_facts (title + sent_id). One schema, two datasets, no adapter code
(see Research_Design_v3 §2.1).

MuSiQue (Dataset_Evaluation_v6_Dual_Ladder.md) adds a `question_decomposition`:
a chain of reasoning hops, each with a sub-question, gold sub-answer, and the
index of the paragraph that supports it. This is the signal that enables
GRADED chain-completion scoring (v6 §2): F becomes the fraction of gold hops
the model recovers, instead of a binary final-answer match. MuSiQue also has
a variable, larger paragraph pool (~20) and 2-4 supporting paragraphs, so the
"exactly 10 paragraphs / exactly 2 gold" assumptions are relaxed below.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


# Allowed `level` values in HotpotQA (validation set has only medium / hard).
HotpotLevel = Literal["easy", "medium", "hard"]
# Allowed `type` values. `bridge` / `comparison` are HotpotQA; `inference`,
# `compositional`, `bridge_comparison` are 2WikiMultihopQA; `2hop`/`3hop`/
# `4hop` are MuSiQue hop-count labels (MuSiQue encodes hop count in the id).
QuestionType = Literal[
    "bridge",
    "comparison",
    "inference",
    "compositional",
    "bridge_comparison",
    "2hop",
    "3hop",
    "4hop",
]


class HotpotParagraph(BaseModel):
    """A single context paragraph.

    `is_gold` marks a supporting paragraph. For HotpotQA / 2Wiki it is derived
    from `supporting_facts.title` at parse time; for MuSiQue it is set directly
    from the `is_supporting` flag. We store it on the paragraph so ladder
    construction is a pure function of the parsed object.
    """

    model_config = ConfigDict(extra="forbid")

    title: str
    sentences: list[str]
    is_gold: bool = False

    def joined(self) -> str:
        """Concatenate sentences exactly as the dataset stores them.

        HotpotQA sentences already include trailing whitespace where appropriate
        (Yang 2018 §3.2: paragraphs are split by sentence-segmentation). Joining
        with empty string preserves the original Wikipedia text. MuSiQue stores
        each paragraph as a single string, wrapped in a one-element list, so the
        join is an identity there.
        """
        return "".join(self.sentences)


class HotpotSupportingFact(BaseModel):
    """A (paragraph_title, sentence_id) entry pointing into the gold context."""

    model_config = ConfigDict(extra="forbid")

    title: str
    sent_id: int


class DecompositionHop(BaseModel):
    """One reasoning hop of a MuSiQue `question_decomposition`.

    `sub_question` may contain placeholders like "#1", "#2" that refer to the
    gold answer of an earlier hop. `sub_answer` is the gold intermediate
    answer. `supporting_paragraph_idx` indexes into `MultiHopQuestion.paragraphs`
    (the paragraph that supports this hop), or None if unmapped.
    """

    model_config = ConfigDict(extra="forbid")

    hop_idx: int                                   # 0-based position in the chain
    sub_question: str                              # may contain "#1", "#2" placeholders
    sub_answer: str                                # the gold intermediate answer
    supporting_paragraph_idx: int | None = None    # index into paragraphs


class MultiHopQuestion(BaseModel):
    """Unified record for HotpotQA, 2WikiMultihopQA, and MuSiQue."""

    model_config = ConfigDict(extra="forbid")

    id: str
    dataset: Literal["hotpotqa", "2wikimultihopqa", "musique"]
    question: str
    answer: str
    question_type: QuestionType
    level: HotpotLevel | None = None  # 2WikiMultihopQA / MuSiQue do not carry `level`
    paragraphs: list[HotpotParagraph] = Field(min_length=1)
    # HotpotQA / 2Wiki ship supporting_facts; MuSiQue does not (it uses
    # per-paragraph is_supporting + question_decomposition instead), so this
    # defaults to empty rather than min_length=1.
    supporting_facts: list[HotpotSupportingFact] = Field(default_factory=list)
    # MuSiQue-only: the gold reasoning chain. Non-empty => chain-completion
    # scoring is available for this question (v6 §2).
    question_decomposition: list[DecompositionHop] = Field(default_factory=list)
    # MuSiQue-only: number of reasoning hops. None for HotpotQA / 2Wiki.
    n_hops: int | None = None

    @model_validator(mode="after")
    def _validate_gold_count(self) -> "MultiHopQuestion":
        """Sync the per-paragraph is_gold flags with supporting_facts.

        HotpotQA / 2Wiki: gold is defined by supporting_facts.title — propagate
        it to the paragraphs (idempotent; native records have 2+ gold).

        MuSiQue: supporting_facts is empty and is_gold is set directly by the
        loader from the `is_supporting` flag, so we MUST NOT wipe it. Only
        propagate when supporting_facts is non-empty.
        """
        if not self.supporting_facts:
            return self  # MuSiQue path — trust loader-set is_gold flags.
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

    def has_decomposition(self) -> bool:
        """True iff chain-completion scoring is available (MuSiQue)."""
        return len(self.question_decomposition) > 0
