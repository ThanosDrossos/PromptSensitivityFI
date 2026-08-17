# PromptSensitivityFI

Measuring **prompt sensitivity in bits**. KIT seminar project (Thanos Drossos,
supervisor Moritz Diener).

Every metric here is the same ruler — **bits = −log₂(surviving fraction)** —
adapted from Szostak (2003) / Hazen et al. (2007) functional information and
licensed for non-biological systems by Wong et al. (2023). Applying it to
prompts is the project's contribution.

**Status: data collection is complete.** 3 models × 150 questions × 2
specificity levels, plus width/swap/POSIX/k20/holdout arms. The paper
(`../SensitivityFunctionalInformationPaper`) is the remaining work.

## The frame: three axes + one dial

| | metric | question it answers |
|---|---|---|
| **Competence** | graded accuracy; FI_in(q,k) curve | how well does the model do, and how much rephrasing luck does that take? |
| **Formulation sensitivity** | **ρ_F** — noise-corrected ICC over rephrasings; σ²_B | of the variation in success, how much is caused by *which phrasing*, vs decoding noise? |
| **Output dispersion** | **H_sem** (sole representative) | how scattered are the answers? |
| **The dial** (manipulated) | **FI_spec** = log₂(m₀/m_valid) | how much ambiguity does the question text itself remove? *(model-free)* |

The axes are **distinct but not orthogonal**: no cross-axis association
exceeds |ρ| = 0.14 under the primary estimator, but the sample supports
equivalence bounds only of |ρ| < 0.34, and ρ_F has a small consistent
positive association with dispersion (complete-case 6/6 strata). The
dispersion "family" (S_τ, TVD-consistency, |A_q|, variation ratio,
Var[FI_out]) is **one pooled-clustering object** — agreement within it is
arithmetic, not convergent evidence. AUFI is accuracy in a log wrapper
(ρ = −1.00) and lives in the appendix.

## The experiment

AmbigQA supplies real ambiguous questions **and** their human-written
disambiguated versions:

- **L0** = the ambiguous question · **L1** = the target interpretation's
  disambiguated question (target pinned by a seeded hash)
- **Guardrails**: gold fixed across levels; evidence identical across levels
  and rephrasings; scored under **two gold sets** (pinned target + union of
  all readings) from identical cached responses.
- Per cell: **10 NLI-verified rephrasings × 10 samples**, graded F(x) scored
  semantically (never exact match).

**Headline (union gold, the primary endpoint)** — disambiguation buys
**+0.064/+0.125/+0.119** accuracy (Qwen/Llama/Mistral; BH-significant 3/3,
Holm-robust 2/3; pooled p = 3.3e-5). The naive target-gold numbers
(+0.22…+0.25) are 47–72 % **grading lottery** and are reported only as the
protocol comparison. H_sem falls (Holm-robust in Llama); ρ_F does not move
(and the estimator's resolution for that null is quantified in
`data/rho_f_recovery_sim.md`). The width dial moves ρ_F in the predicted
direction (significant in Qwen; n's and seed ranges in
`data/width_dial_analysis.md`); the paraphraser-swap arm preserves per-cell
structure and the model ordering **Qwen > Mistral > Llama**, the most robust
result in the project. The probes: underspecification transfers zero-shot at
AUROC **.667/.655/.670** (n = 1,852, training questions excluded) vs frozen
text baselines .543–.571; the ρ_F head is a clean, properly-nulled negative.

## Models

All in-process HF transformers (`provider: local`) on bwUniCluster 3.0:

- **Eval**: `llama_3_1_8b`, `mistral_7b_v03`, `qwen_2_5_7b`
- **Generator + judge**: `phi_4_14b` — never an eval model; the R6 swap arm
  uses `olmo_2_13b` for both roles
- `gpt_4o` (LiteLLM gateway) is legacy and unused

## Layout

```
prompt_sensitivity/
├── data/            # AmbigQA loader + Pydantic schemas
├── specificity/     # level construction (L0/L1), the fixed-gold guardrail
├── paraphrases/     # generate → NLI-filter → gold-constraint-filter → dedup
├── models/          # LocalHFClient (transformers) + SQLite request cache
├── metrics/         # FI_in, H_sem, FI_spec, rho_F, POSIX …  (frozen)
├── feedback/        # the prompt-checker heads (linear probes on hidden states)
├── analysis/        # hierarchical rho_F estimator, x*-geometry
└── scripts/         # CLI entry points incl. the R-series analyses
cluster/             # sbatch files + runbooks (no further runs planned)
app/                 # Streamlit demo
papers/              # reference-PDF library (gitignored)
docs/                # reviews, results notes, literature, design authorities
```

## Running it

```bash
uv sync --extra app --extra dev     # BOTH extras, or ruff/pytest disappear
uv run pytest -q                    # 382 tests, CPU-only
```

Every table and figure re-derives from the committed `data/*.parquet` +
`data/*.md` artifacts; `data/run_manifest.json` pins seeds, thresholds, the
question sample, and file hashes. The large probe inputs
(`hidden_states_*`, `vagueness_holdout_*`, ~450 MB) are not in git.
Paper figures: `uv run python -m prompt_sensitivity.scripts.make_paper_figures
--out ../SensitivityFunctionalInformationPaper/1_Figures` (reads artifacts;
no hardcoded numbers).

## Documentation

| read this | for |
|---|---|
| `data/stats_hygiene.md` | **source of truth for Results numbers** (+ `.json` for figures) |
| `FORKING_PATHS.md` | the 13 analysis forks and which results depend on them |
| `docs/reviews/REVIEW_2026-08-16_Consolidated_Action_Plan.md` | the working review + action plan |
| `docs/PROJECT_STATE_2026-08-16.md` | current status snapshot |
| `PIPELINE_WALKTHROUGH.md` | the whole pipeline, for someone new |
| `EXPLAINER_Three_Dimensions.md` | the metric frame (R5-corrected) |
| `docs/design/` | formula + research-design authorities |
| `CODEBASE_WALKTHROUGH.md` | code mechanics — ⚠️ framing predates the axes pivot |
