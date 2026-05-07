# PromptSensitivityFI

Implementation of **Functional Information for Prompts** (`FI_in`), the novel
metric proposed in the KIT seminar project *Prompt Sensitivity Metric*
(Thanos Drossos, supervisor Moritz Diener).

`FI_in(q, k, l) = -log₂(N_k(q, l) / |U_q,l|)` adapts Hazen-Griffin-Carothers-Szostak
(2007) functional information from biopolymer space to LLM prompt space, as
formalised in `Section_7_Functional_Information_for_Prompts.md`. Companion
metrics: Errica `(S_τ, 1-TVD)`, Cox 2025 `ρ_u`, POSIX `ψ`, Farquhar `H_sem`,
performance spread, variation ratio, ESS_in.

## Layout

```
prompt_sensitivity/
├── data/             # dataset loaders + Pydantic schemas (HotpotQA, 2WikiMultihopQA)
├── ladders/          # three-ladder construction + bit-cost
├── paraphrases/      # generate / NLI-filter / constraint-filter / dedup
├── models/           # API wrappers (Together, OpenAI) + cache
├── metrics/          # the 10 metric implementations
├── analysis/         # correlations, IRT, plots
├── prompts/templates/# prompt templates per model
├── scripts/          # CLI entry points (`api_check`, `download_datasets`, ...)
├── pipeline.py       # orchestrator
└── config.py         # config loader
```

Authoritative design docs (one level up):
- `Research_Design_v3_Context_Ladder.md`
- `Section_7_Functional_Information_for_Prompts.md`
- `Research_Synthesis_Prompt_Sensitivity_Metric.md`

## Setup

```bash
uv sync --all-extras
cp .env.example .env  # fill in OPENAI_API_KEY and TOGETHER_API_KEY
make test             # unit tests, no API calls
make sprint1-verify   # full Sprint-1 verification (needs .env)
```
