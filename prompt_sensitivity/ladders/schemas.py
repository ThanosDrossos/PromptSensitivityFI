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


# Context ladders order paragraphs; the reasoning ladder feeds decomposition
# hops. `ladder_family` separates the two manipulations (v6 §5).
LadderType = Literal["random", "gold_first", "distractor_first", "reasoning"]
LadderFamily = Literal["context", "reasoning"]


class LadderRow(BaseModel):
    """One row of `data/ladders.parquet`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    question_id: str
    ladder_type: LadderType
    # Position in the level sequence. Was hardcoded le=5 for the 6-rung
    # {0,2,4,6,8,10} context ladder; relaxed because the reasoning ladder has
    # n_hops rungs (variable) and MuSiQue context sweeps may use more levels.
    level_idx: int = Field(ge=0)
    level: int = Field(ge=0)                  # #paragraphs (context) or #hops fed (reasoning)
    paragraph_indices: list[int] = Field(default_factory=list)
    paragraph_titles: list[str] = Field(default_factory=list)
    gold_count: int = Field(ge=0)             # how many gold paragraphs in this prefix
    permutation: list[int] | None = None      # for random: the full per-question shuffle
    # v6: which manipulation this row belongs to. Defaults to "context" so the
    # existing three context ladders are unchanged.
    ladder_family: LadderFamily = "context"
    # Reasoning ladder only: number of decomposition hops handed to the model
    # as scaffold at this rung (0..n_hops-1; final hop always withheld).
    hops_provided: int | None = None


class LevelSlice(BaseModel):
    """Lightweight helper — just (indices, titles) for a single level.

    Used internally by ladder builders before they assemble the full row set.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    paragraph_indices: list[int]
    paragraph_titles: list[str]
    gold_count: int
