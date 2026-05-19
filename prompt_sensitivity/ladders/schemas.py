"""Schemas for ladder rows.

Per Research_Design_v3 §4.2, each (question, ladder_type, level) row is the
paragraph subset to splice into the LLM prompt at that level. We store the
subset as paragraph INDICES into the question's 10-paragraph context — the
denormalised paragraph titles travel along for convenient analytics without
needing a separate JOIN.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


LadderType = Literal["random", "gold_first", "distractor_first"]


class LadderRow(BaseModel):
    """One row of `data/ladders.parquet`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    question_id: str
    ladder_type: LadderType
    level_idx: int = Field(ge=0, le=5)        # 0..5, position in the {0,2,4,6,8,10} sequence
    level: int = Field(ge=0)                  # number of paragraphs at this level
    paragraph_indices: list[int] = Field(default_factory=list)
    paragraph_titles: list[str] = Field(default_factory=list)
    gold_count: int = Field(ge=0)             # how many gold paragraphs in this prefix
    permutation: list[int] | None = None      # for random: the full per-question shuffle


class LevelSlice(BaseModel):
    """Lightweight helper — just (indices, titles) for a single level.

    Used internally by ladder builders before they assemble the full row set.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    paragraph_indices: list[int]
    paragraph_titles: list[str]
    gold_count: int
