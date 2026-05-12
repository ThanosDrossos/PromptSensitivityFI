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
├── ladders/          # three-ladder construction + bit-cost                    (Sprint 3)
├── paraphrases/      # generate / NLI-filter / constraint-filter / dedup       (Sprint 2)
├── models/           # LiteLLM gateway client (OpenAI SDK + custom base_url) + cache
├── metrics/          # the 10 metric implementations                           (Sprint 4)
├── analysis/         # correlations, IRT, plots                                (Sprint 5)
├── prompts/templates/# prompt templates per model
├── scripts/          # CLI entry points (`api_check`, `list_models`, ...)
├── pipeline.py       # orchestrator                                            (Sprint 5)
└── config.py         # config loader
```

Authoritative design docs (one level up):
- `Research_Design_v3_Context_Ladder.md`
- `Section_7_Functional_Information_for_Prompts.md`
- `Research_Synthesis_Prompt_Sensitivity_Metric.md`

## LiteLLM gateway: model & capability matrix

All four models route through the supervisor-provided LiteLLM gateway
(default `https://ai-gateway.dsi-experimente.de/v1`). The OpenAI Python SDK
is reused — only the `base_url` and `api_key` differ.

| Model                       | chat_logprobs (top-K) | echo_completions | own hidden state |
|-----------------------------|-----------------------|------------------|------------------|
| Llama-3.1-8B-Instruct       | ✅ ≤ 20               | ✅ (POSIX OK)    | ❌ cluster-only  |
| Mistral-7B-Instruct-v0.3    | ✅ ≤ 20               | ✅ (POSIX OK)    | ❌ cluster-only  |
| Qwen2.5-7B-Instruct         | ✅ ≤ 20               | ✅ (POSIX OK)    | ❌ cluster-only  |
| GPT-4o                      | ✅ ≤ 20               | ❌ (no echo)     | ❌ cluster-only  |

**Sprint-4+ implications.**
- **POSIX `ψ`** uses `/v1/completions` with `echo=true, logprobs=1, max_tokens=0`
  on Llama/Mistral/Qwen. GPT-4o gets `null` for POSIX (record limitation, do not
  hide).
- **Errica `S_τ` on free-form output** is computed via Monte-Carlo over semantic
  clusters (effectively H_sem-derived). The full-vocab token-entropy variant
  is unavailable through any chat-completions API.
- **ESS_in** uses the external `all-mpnet-base-v2` encoder for all four models
  pre-cluster. The "own-encoder" variant is Sprint-6 (KIT cluster + vLLM).

## Setup

```bash
# 1) install dependencies
uv sync --all-extras

# 2) set the LiteLLM API key
cp .env.example .env
#    then edit .env: paste LITELLM_API_KEY = "sk-..."

# 3) tests that don't touch the gateway
make test          # or .\tasks.ps1 test on PowerShell

# 4) Sprint-1 full verification (needs .env and network)
make sprint1-verify
```

`sprint1-verify` runs `list-models -> data-download -> sample -> api-check` in
order.

### Running `make` on Windows

You have three options, in order of recommendation:

1. **Use the PowerShell helper** (no install needed):
   ```powershell
   .\tasks.ps1 install
   .\tasks.ps1 test
   .\tasks.ps1 sprint1-verify
   ```
   `tasks.ps1` mirrors every Makefile target one-to-one.

2. **Install GNU make via `winget`** (one-time, then `make ...` works in any
   shell):
   ```powershell
   winget install ezwinports.make
   # close + reopen your terminal so PATH refreshes
   make test
   ```
   Alternative if `winget` is missing: `choco install make` (needs Chocolatey).

3. **Use Git Bash** and treat the project like Linux. Git for Windows ships
   bash and most coreutils; install `make` via the `make-for-windows` package
   or use option 2 above.

Inside any of the above, every `make <target>` ultimately just runs
`uv run python -m prompt_sensitivity.scripts.<target>`. Nothing is magic;
you can always copy the command from `Makefile`/`tasks.ps1` and run it
directly.

## Verifying Sprint 1

After `.env` is filled in:

```bash
# Confirms gateway is reachable + prints every alias the proxy exposes.
make list-models

# Pulls HotpotQA distractor (~1 GB) and framolfese/2WikiMultihopQA validation.
make data-download

# Writes data/sample_v1.json with 100+50 stratified question IDs.
make sample

# Round-trip + logprob probe for all 4 models. Writes logs/api_check_summary.json.
make api-check
```

If `list-models` shows a model alias that's missing from `config.yaml`,
edit `config.yaml.models.<key>.model_id` to match one of the printed IDs
and re-run. (LiteLLM admins can register custom aliases; the canonical
LiteLLM names baked into the default config may not be the ones registered.)
