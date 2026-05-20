"""Config loader. Single source of truth for hyperparameters.

Per Research_Design_v3 §7.4, all hyperparameters live in `config.yaml` at repo
root and are loaded into a frozen Pydantic model so call sites cannot mutate
them mid-run.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

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


class SamplingConfig(_Frozen):
    hotpotqa: SamplingDatasetConfig
    twiki: SamplingDatasetConfig


class LadderConfig(_Frozen):
    levels: list[int]
    k_gold: int
    n_total_paragraphs: int
    variants: list[str]


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


class HSemConfig(_Frozen):
    n_samples_per_prompt: int
    sampling_temperature: float
    cluster_nli_model: str
    cluster_threshold: float


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
