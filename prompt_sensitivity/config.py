"""Config loader. Single source of truth for hyperparameters.

Per Research_Design_v3 §7.4, all hyperparameters live in `config.yaml` at repo
root and are loaded into a frozen Pydantic model so call sites cannot mutate
them mid-run.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SamplingDatasetConfig(_Frozen):
    n_questions: int
    stratify_by: str
    split: str
    hf_dataset: str
    hf_config: str | None
    # MuSiQue may be loaded from a local jsonl (the canonical release) instead
    # of a HF mirror. Optional + default None keeps HotpotQA / 2Wiki unchanged.
    local_path: str | None = None


class AmbigQASamplingConfig(_Frozen):
    """AmbigQA sampling (specificity pivot, 2026-07). Own block rather than a
    SamplingDatasetConfig because the knobs differ (interpretation filters, no
    stratify/local_path)."""

    hf_dataset: str = "ambig_qa"
    hf_config: str = "light"
    split: str = "validation"          # AmbigQA test split is not public
    n_questions: int = 50
    min_interpretations: int = 2       # keep only genuinely ambiguous records
    include_single_answer_anchor: bool = False


class SamplingConfig(_Frozen):
    hotpotqa: SamplingDatasetConfig
    twiki: SamplingDatasetConfig
    # v6: MuSiQue is the primary graded-scoring dataset. Optional so existing
    # configs without a `musique` block still validate.
    musique: SamplingDatasetConfig | None = None
    # Specificity pivot: AmbigQA. Optional for the same reason.
    ambigqa: AmbigQASamplingConfig | None = None


class LadderConfig(_Frozen):
    levels: list[int]
    k_gold: int
    n_total_paragraphs: int
    variants: list[str]
    # v6: optional family selector. The context ladder uses `levels` over the
    # paragraph pool; the reasoning ladder feeds decomposition hops 0..k-1.
    families: list[str] = Field(default_factory=lambda: ["context"])


class NLIConfig(_Frozen):
    model: str
    bidirectional_threshold: float
    fallback_threshold: float


class ConstraintFilterConfig(_Frozen):
    judge_model: str
    jaccard_threshold: float
    judge_max_tokens: int = 1024


class DedupConfig(_Frozen):
    min_edit_distance: int
    metric: str = "char"          # "char" | "token"


class ParaphraseConfig(_Frozen):
    n_per_question: int
    generator_model: str
    generator_temperature: float
    templates: list[str]
    raw_candidates_per_question: int
    samples_per_template: int
    max_regeneration_attempts: int
    nli: NLIConfig
    constraint_filter: ConstraintFilterConfig
    deduplication: DedupConfig


class ModelEntry(_Frozen):
    """Capability flags per model. See `Research_Design_v3` §5 + gateway capability matrix.

    `chat_logprobs`: chat-completions returns top_logprobs<=20 for this model.
    `echo_completions`: /v1/completions supports echo=true (POSIX prerequisite).
    `has_hidden`: model's own last-layer hidden state is reachable (cluster-only).
    """

    provider: str
    model_id: str
    chat_logprobs: bool
    echo_completions: bool
    has_hidden: bool


class ScoringConfig(_Frozen):
    method: str
    nli_model: str
    entail_threshold: float
    contradict_threshold: float
    exact_match_appendix_only: bool
    # P0-2b: a looser entailment bar for the SECONDARY final-answer score only.
    # The strict `entail_threshold` is unchanged; this gives a parallel permissive
    # column so a too-strict NLI can't masquerade as model failure.
    entail_threshold_permissive: float = 0.5
    # v6 graded chain-completion. Default None => reuse the binary thresholds
    # above (the chain scorer reads entail_threshold / contradict_threshold).
    chain_entail_threshold: float | None = None
    chain_contradict_threshold: float | None = None


class GenerationConfig(_Frozen):
    """Token budgets for model generation. Separate from paraphrase generation."""

    # Baseline "answer briefly" responses are short.
    answer_max_tokens: int = 64
    # CoT responses must fit a full reasoning chain — 64 is far too small.
    cot_max_tokens: int = 512


class HSemConfig(_Frozen):
    n_samples_per_prompt: int
    sampling_temperature: float
    cluster_nli_model: str
    cluster_threshold: float
    # Output-space clustering collapse fix (2026-06-28). cluster_criterion: how
    # two samples merge — "label" (argmax NLI == entailment both ways; strict,
    # default) vs "prob" (entail prob >= cluster_threshold both ways; legacy,
    # lenient). cluster_on: WHAT to cluster — "answer" (parse_answer_line of each
    # sample; default, avoids style/verbosity over-merging) vs "response" (the
    # full generation). Defaults so older configs without these keys still load.
    cluster_criterion: Literal["prob", "label"] = "label"
    cluster_on: Literal["answer", "response"] = "answer"


class SpecificityConfig(_Frozen):
    """AmbigQA specificity manipulation (pivot spec §9; v2 amendment 2026-07-06).

    v2: `uniform_evidence` — every cell gets the question's own AmbigNQ
    `used_queries` snippet bundle as context, IDENTICAL across both specificity
    levels and all paraphrases, so specificity stays the only manipulated
    variable while answerability comes from READING, not parametric recall
    (kills the v1 closed-book knowledge floor). `closed_book` reproduces v1.
    """

    levels: list[int] = Field(default_factory=lambda: [0, 1])
    target_seed: int = 42              # deterministic target-interpretation choice
    context_mode: Literal["closed_book", "uniform_evidence"] = "uniform_evidence"
    evidence_max_chars: int = 6000     # cap the bundle (whole snippets, dataset order)
    require_target_in_evidence: bool = True   # v2 filter (dataset-side, model-free)


class BootstrapConfig(_Frozen):
    n_iterations: int
    confidence: float


class EmbeddingConfig(_Frozen):
    external_encoder: str
    gateway_encoder: str


class APIConfig(_Frozen):
    """LiteLLM-gateway access config. Single endpoint, single key."""

    base_url_env: str
    api_key_env: str
    default_base_url: str
    max_retries: int
    initial_backoff_s: float
    max_backoff_s: float
    rate_limit_qps: dict[str, float]


class CacheConfig(_Frozen):
    backend: str
    path: str


class SpendConfig(_Frozen):
    pilot_usd_cap: float


class Config(_Frozen):
    """Root config. Populated from `config.yaml`."""

    config_version: int
    random_seed: int
    sampling: SamplingConfig
    ladders: LadderConfig
    paraphrases: ParaphraseConfig
    models: dict[str, ModelEntry] = Field(default_factory=dict)
    scoring: ScoringConfig
    # v6 token budgets. Optional with all-default fields so configs without a
    # `generation` block still validate.
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    # Specificity pivot: optional so pre-pivot configs still validate.
    specificity: SpecificityConfig | None = None
    h_sem: HSemConfig
    bootstrap: BootstrapConfig
    embedding: EmbeddingConfig
    api: APIConfig
    cache: CacheConfig
    spend: SpendConfig

    def repo_root(self) -> Path:
        return REPO_ROOT

    def cache_path(self) -> Path:
        return (REPO_ROOT / self.cache.path).resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@lru_cache(maxsize=1)
def load_config(path: Path | str | None = None) -> Config:
    """Read `config.yaml` once per process. Override with PROMPT_SENSITIVITY_CONFIG env var."""

    if path is None:
        env = os.environ.get("PROMPT_SENSITIVITY_CONFIG")
        path = Path(env) if env else DEFAULT_CONFIG_PATH
    path = Path(path)
    raw = _load_yaml(path)
    return Config.model_validate(raw)
