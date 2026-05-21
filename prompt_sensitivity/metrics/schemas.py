"""Shared schemas for the 10-metric stack.

The "10-tuple" of Research_Design_v3 §3 is actually 11 named scalars (the doc
calls it a 10-tuple but enumerates 11 fields). We use a Pydantic dataclass
with explicit field names rather than a tuple — survives schema evolution and
makes parquet rows self-describing.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


LadderType = Literal["random", "gold_first", "distractor_first"]


class ResponseSample(BaseModel):
    """One sampled (paraphrase, model) response — input to most metrics.

    `embedding` is the sentence-encoder vector (mpnet) of the *response*,
    populated only by callers that need it (rho_u). For h_sem / errica /
    fi_out the cluster id alone suffices.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    paraphrase_idx: int
    sample_idx: int                    # 0..k-1, k = config.h_sem.n_samples_per_prompt
    text: str
    cluster_id: int | None = None
    embedding: list[float] | None = None


class MetricTuple(BaseModel):
    """The 11-scalar metric vector per (question, ladder_type, level, model).

    Order matches Research_Design_v3 §3 "Reporting protocol" line for line.

    Any field that cannot be computed (e.g. POSIX for GPT-4o because there's
    no echo path through the gateway) is left as `None` and the writeup must
    flag the limitation explicitly rather than imputing a value.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    question_id: str
    ladder_type: LadderType
    level: int
    model_key: str

    # Raw accuracy — mean F(x) across paraphrases. Not in §3 reporting tuple
    # but materially useful for plots and supervisor presentation: the FI_in
    # metric is meaningful only relative to a baseline F-rate, so we record
    # the rate explicitly rather than reading it back from AUFI_in.
    f_mean: float | None = None         # in [0, 1]; mean of F(x) over paraphrases

    # Tier A — primary novel contribution
    aufi_in: float | None = None        # area under FI_in(k) curve, k in [0,1]
    fi_out_mean: float | None = None    # mean FI_out across paraphrases

    # Tier B — Errica two-number deliverable
    s_tau_mean: float | None = None     # mean S_tau across paraphrases
    consistency_mean: float | None = None  # mean (1 - TVD) across paraphrase pairs

    # Tier C — diagnostic complementary metrics
    spread: float | None = None         # max(F) - min(F) across paraphrases
    variation_ratio: float | None = None
    posix_psi: float | None = None      # null for GPT-4o (no echo on chat)
    ess_in: float | None = None         # trace(Cov(input embeddings))
    rho_u: float | None = None          # Cox 2025 U_e / U_t
    h_sem_mean: float | None = None
    h_sem_var: float | None = None

    # Diagnostics for the audit trail.
    n_paraphrases: int = 0
    n_samples_per_prompt: int = 0
    encoder_label: str = "external_mpnet"     # "external_mpnet" | "own_hidden" | "openai_gateway"
