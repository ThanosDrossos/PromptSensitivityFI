"""Schemas for AmbigQA (Min et al. 2020) — the specificity pivot's dataset.

An AmbigQA record is an (often) ambiguous open-domain question plus its
disambiguated interpretations, each with its own answer set. The specificity
manipulation (specificity/build_levels.py) maps one record to two closed-book
cells: level 0 = the ambiguous question Q, level 1 = one disambiguated Q_i,
with the scoring gold a_i held FIXED across both levels.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AmbigInterpretation(BaseModel):
    """One disambiguated interpretation Q_i with its accepted answer variants a_i."""

    model_config = ConfigDict(extra="forbid")

    disambiguated_question: str                 # the disambiguated rewrite Q_i
    answers: list[str] = Field(min_length=1)    # accepted answer variants for a_i


class AmbigQuestion(BaseModel):
    """One AmbigQA record: the original (ambiguous) question + interpretations."""

    model_config = ConfigDict(extra="forbid")

    id: str
    dataset: Literal["ambigqa"] = "ambigqa"
    question: str                               # the original (ambiguous) question
    interpretations: list[AmbigInterpretation] = Field(min_length=1)

    def m0(self) -> int:
        """Number of valid interpretations — the ambiguity of the raw question."""
        return len(self.interpretations)

    def is_ambiguous(self) -> bool:
        return self.m0() > 1
