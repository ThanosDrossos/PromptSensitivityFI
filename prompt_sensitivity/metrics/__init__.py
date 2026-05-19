"""The 10 metrics. Research_Design_v3 §3.

Pre-cluster capability matrix (from registry.py comment):
  - chat-completions logprobs <=20:  Llama/Teuken/Qwen/GPT-4o (gateway)
  - /v1/completions echo (POSIX):    Llama/Teuken/Qwen only; GPT-4o gives null
  - own-encoder hidden states:       none through gateway; mpnet is used

Module layout:

    metrics/
    ├── schemas.py          MetricTuple Pydantic v2 model
    ├── fi_in.py            FI_in(q, k, l) + curve + AUFI_in + bootstrap CI
    ├── h_sem.py            Farquhar-clustering semantic entropy
    ├── fi_out.py           log2|A_q| - H_sem(Y|X=x)
    ├── errica.py           S_tau + 1-TVD (Errica two-number axis)
    ├── posix.py            cross-assignment log-prob divergence
    ├── ess_in.py           input-embedding dispersion (mpnet)
    ├── rho_u.py            Cox 2025 within/across-variance decomposition
    ├── spread.py           max(F) - min(F)
    ├── variation_ratio.py  1 - mode_count / N
    └── orchestrator.py     glue: returns the MetricTuple per (q, ladder, level, model)
"""

from .schemas import MetricTuple, ResponseSample
from .fi_in import fi_in, fi_in_curve, aufi_in, fi_in_bootstrap
from .h_sem import h_sem, cluster_responses, cluster_responses_pooled
from .fi_out import fi_out
from .errica import s_tau_freeform, s_tau_multiple_choice, tvd_consistency
from .posix import posix
from .spread import spread
from .variation_ratio import variation_ratio
from .ess_in import ess_in
from .rho_u import rho_u
from .orchestrator import build_metric_tuple

__all__ = [
    "MetricTuple",
    "ResponseSample",
    "fi_in",
    "fi_in_curve",
    "aufi_in",
    "fi_in_bootstrap",
    "h_sem",
    "cluster_responses",
    "cluster_responses_pooled",
    "fi_out",
    "s_tau_freeform",
    "s_tau_multiple_choice",
    "tvd_consistency",
    "posix",
    "spread",
    "variation_ratio",
    "ess_in",
    "rho_u",
    "build_metric_tuple",
]
