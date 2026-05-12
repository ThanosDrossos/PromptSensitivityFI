"""Pydantic v2 schemas for the paraphrase pipeline."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


RoleName = Literal["neutral", "journalist", "casual_user", "domain_expert"]
RejectionReason = Literal[
    "nli_low",
    "nli_one_direction",
    "constraint_mismatch",
    "edit_distance_close",
    "exact_duplicate",
    "empty",
]


class RawParaphrase(BaseModel):
    """A candidate from the generator, pre-filter."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    question_id: str
    role: RoleName
    sample_idx: int
    text: str
    generator_model_key: str
    generator_seed: int
    request_hash: str          # ties this row back to the cache entry


class AcceptedParaphrase(BaseModel):
    """A candidate that passed all four filters."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    question_id: str
    paraphrase_idx: int        # 0..29 — position in the final ParaphraseSet
    role: RoleName
    text: str
    # Filter diagnostics for the audit log.
    nli_entail_fwd: float
    nli_entail_bwd: float
    constraint_jaccard: float
    generator_model_key: str
    generator_seed: int
    request_hash: str


class RejectedParaphrase(BaseModel):
    """A candidate that failed at least one filter — kept for audit only."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    question_id: str
    role: RoleName
    sample_idx: int
    text: str
    reason: RejectionReason
    nli_entail_fwd: float | None = None
    nli_entail_bwd: float | None = None
    constraint_jaccard: float | None = None


class ParaphraseSet(BaseModel):
    """Per-question result: the 30 accepted paraphrases plus diagnostics."""

    model_config = ConfigDict(extra="forbid")

    question_id: str
    accepted: list[AcceptedParaphrase] = Field(default_factory=list)
    rejected: list[RejectedParaphrase] = Field(default_factory=list)
    regeneration_attempts: int = 0
    dropped: bool = False           # true if we could not reach n_per_question
    nli_threshold_used: float = 0.9 # 0.9 normally, 0.85 if we hit the fallback

    def n_accepted(self) -> int:
        return len(self.accepted)

    def is_complete(self, target: int) -> bool:
        return len(self.accepted) >= target and not self.dropped
