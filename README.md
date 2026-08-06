# PromptSensitivityFI

Measuring **prompt sensitivity in bits**. KIT seminar project (Thanos Drossos,
supervisor Moritz Diener).

Every metric here is the same ruler — **bits = −log₂(surviving fraction)** —
adapted from Szostak (2003) / Hazen et al. (2007) functional information and
licensed for non-biological systems by Wong et al. (2023). Applying it to
prompts is the project's contribution.

**Status: data collection is complete.** 3 models × 149 questions × 2
specificity levels, plus stability/POSIX/holdout arms. The paper is the
remaining work.

## The frame: three orthogonal axes + one dial

| | metric | question it answers |
|---|---|---|
| **Ability** | FI_in(q,k) = −log₂(N_k/\|U_q\|) curve; AUFI = ∫ | how well does the model do, and how much rephrasing luck does that take? |
| **Formulation sensitivity** | **ρ_F** — one-way ICC over rephrasings | of the variation in success, how much is caused by *which phrasing*, vs decoding noise? |
| **Output dispersion** | **H_sem**; FI_out = log₂\|A_q\| − H_sem | how scattered are the answers? |
| **The dial** (manipulated) | **FI_spec** = log₂(m₀/m_valid) | how much ambiguity does the question text itself remove? *(model-free)* |

The three axes are empirically independent (ρ_F ⊥ accuracy .08, ⊥ H_sem .03;
a factor analysis recovers exactly these three blocks). AUFI turned out to be
accuracy in a log wrapper (ρ = −1.00) and is reported only in the appendix.

## The experiment

AmbigQA supplies real ambiguous questions **and** their human-written
disambiguated versions, so nothing is constructed by us:

- **L0** = the ambiguous question · **L1** = the target interpretation's
  disambiguated question (+1.58 bits of FI_spec on average)
- **Guardrails**: the gold answer is *fixed* across levels, and the evidence
  block is *identical* across levels and across all rephrasings — the question
  text is the only thing that changes.
- Per cell: **10 NLI-verified rephrasings × 10 samples**, graded F(x) scored
  semantically (never exact match).

**Headline result** — paying ~1.6 bits of question specificity: accuracy
roughly doubles (ΔF̄ +0.22…+0.25), the FI_in curve drops ~0.8 bits, answers
converge — in all three models, Wilcoxon p ≤ 5e-9.

## Models

All in-process HF transformers (`provider: local`) on bwUniCluster 3.0:

- **Eval**: `llama_3_1_8b`, `mistral_7b_v03`, `qwen_2_5_7b`
- **Generator + judge**: `phi_4_14b` — never an eval model
- `gpt_4o` (LiteLLM gateway) is legacy and unused

## Layout

```
prompt_sensitivity/
├── data/            # AmbigQA loader + Pydantic schemas
├── specificity/     # level construction (L0/L1), the fixed-gold guardrail
├── paraphrases/     # generate → NLI-filter → gold-constraint-filter → dedup
├── models/          # LocalHFClient (transformers) + SQLite request cache
├── metrics/         # FI_in, FI_out/H_sem, FI_spec, rho_F, ESS_in, POSIX …  (frozen)
├── feedback/        # the prompt-checker heads (linear probes on hidden states)
├── analysis/        # correlations, factor structure, x*-geometry
└── scripts/         # CLI entry points
cluster/             # sbatch files + sync/run dispatchers
app/                 # Streamlit demo
docs/archive/        # ⛔ superseded documents — do not use
```

## Running it

Windows: `make` is not installed — use `.\tasks.ps1 <target>` or
`uv run python -m prompt_sensitivity.scripts.<name>`. GPU work runs on the
cluster (`bash cluster/run.sh push` → submit on the login node →
`bash cluster/run.sh pull`); see `cluster/README.md`.

```bash
uv sync --extra app --extra dev     # BOTH extras, or ruff/pytest disappear
uv run pytest -q
```

## Documentation

| read this | for |
|---|---|
| `PIPELINE_WALKTHROUGH.md` | the whole pipeline, for someone new, with a worked example |
| `EXPLAINER_Three_Dimensions.md` | the metric frame + every number verified |
| `METRIC_PROPOSALS.md` | which metrics were adopted/rejected and why |
| `REBUILD_PLAN_AmbigQA_Specificity.md` | the dataset/driver design (German) |
| `FI_PROBES_PLAN.md` | the probe / prompt-checker capstone |
| `data/final_run_results.md` | the results of record |
| `CODEBASE_WALKTHROUGH.md` | code mechanics — ⚠️ its framing predates the axes pivot |

Design docs one level up: `Section_7_Functional_Information_for_Prompts.md` is
the formula authority; `Research_Design_v2_Specificity_FI.md` the research
cycle. Anything labelled v3–v6 there is a superseded era and carries a banner.
