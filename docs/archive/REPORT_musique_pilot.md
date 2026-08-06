> # ⛔ OUTDATED — ARCHIVED, DO NOT USE AS A SOURCE OF TRUTH
>
> Archived 2026-08-05. Belongs to a superseded era of the project (MuSiQue dual-ladder pilot, 63 cells / 3 questions).
>
> **Current source of truth:** `data/final_run_results.md`
>
> Kept only so past decisions stay traceable. See `docs/archive/README.md`.

---

# Pilot results — preliminary

**Cells run:** 63  
**Questions:** 3 (4hop1__93963_17066..., 4hop3__862_846_326..., 4hop3__862_846_613...)  
**Ladders:** random  
**Levels:** [0, 1, 2, 3, 4, 8]  
**Models:** llama_3_1_8b, mistral_7b_v03, qwen_2_5_7b  

> Sample size is intentionally small (this is a smoke-test pilot). 
> All numbers should be treated as illustrative; the Sprint-5 full 
> pilot (50 questions × 4 models) will produce statistically 
> reportable values.

## 1. Accuracy responds to context (the headline)

![F-mean by level](01_f_mean_by_level.png)

Each panel is one question; lines are the three ladders. Aggregated 
panel shows the mean across questions. The expected pattern: 
F-mean climbs with context level, with gold_first as upper envelope 
and distractor_first as lower envelope. Per-question trajectories at 
the top level (L=8, random ladder):

| qid | F-mean at L=top |
|---|---|
| 4hop1__93963_170667_443779_52195 | 0.88 |
| 4hop3__862_846_326964_7713 | 0.88 |
| 4hop3__862_846_613770_7713 | 0.75 |

## 2. The novel FI_in metric

![AUFI_in by level](02_aufi_in_by_level.png)

AUFI_in (Area under FI_in(k) curve) is the design doc's primary 
scalar (Section_7 §7.3). **Units: BITS.** Bounded in [0, log₂(N+1)] — 
for N=30 paraphrases the cap is ~4.95 bits; for N=10 it is ~3.46 bits. 
Lower = more paraphrases pass the threshold (less prompt-sensitivity). 
Should decrease as context grows for questions the model can answer.

**Quick units cheat-sheet (because supervisors always ask):**

| Metric | Range | Units |
|---|---|---|
| F-mean (plot 1) | [0, 1] | accuracy (fraction of paraphrases correct) |
| AUFI_in (plot 2) | [0, log₂(N+1)] | **bits** |
| FI_in(k) (curve, not plotted) | [0, log₂(N)] ∪ {+∞} | **bits** |
| H_sem (plot 3) | [0, log₂(|A_q|)] | **bits** |
| S_τ (Errica) | [0, 1] | normalised entropy (dimensionless) |
| spread | [0, 1] | max(F) − min(F) on binary F |
| variation_ratio | [0, 1] | 1 − mode_count/N |
| ρ_u (Cox 2025) | [0, 1] | variance ratio (dimensionless) |
| POSIX ψ | [0, ∞) | natural log per token (rare units) |

## 3. Farquhar 2024 semantic entropy (baseline)

![H_sem by level](03_h_sem_by_level.png)

Farquhar's H_sem on cluster proportions. Expected to drop with 
context (model converges to one answer when context is informative).

## 4. V3 — three-ladder bound consistency

![Three-ladder F-mean per level](04_three_ladder_envelope.png)

**Bound check (gold_first ≥ random ≥ distractor_first on F-mean):**

- Insufficient data (need all 3 ladders × overlapping levels).

Failures (random > gold_first) would be interesting — they would 
indicate that distractor paragraphs prime parametric retrieval more 
effectively than direct gold facts for that question.

## 4b. Quality vs context bits (does more b_theo info mean better accuracy?)

![Quality vs context bits](08_quality_vs_context_bits.png)

**Plot reading.** Blue (left axis) = F-mean accuracy. Red dashed 
(right axis) = b_theo — bits of *theoretical surprise* about 
whether at least one gold paragraph is in a random subset of size l 
(Sprint 3 §4.4 closed-form from `prompt_sensitivity.ladders.bit_cost`). 
b_theo drops monotonically from +∞ at L=0 (no chance of gold) to 
0 bits at L≥9 (gold guaranteed by pigeonhole).

**The story for the supervisor.** As context level grows, b_theo falls 
— each added paragraph delivers more bits of certainty that gold 
is in the prompt. The hypothesis the metric is *designed* to test: 
more bits delivered → better accuracy. Verify visually that the blue 
curve climbs as the red curve falls. Where they DON'T move together 
(e.g. q1 stays flat at F=0 even as b_theo drops to 0) is itself a 
finding: that question is bottlenecked by something other than 
gold-paragraph availability (model knowledge gap, prompt template, 
answer-extraction failure, etc.).

## 5. Model comparison (at random ladder)

![Model comparison: F-mean](06_model_comparison_f_mean.png)

![Model comparison: AUFI_in](07_model_comparison_aufi_in.png)

Direct visual comparison of the two evaluated models on the random ladder (the realistic-user condition). Top panel shows per-question F-mean trajectories; aggregate panel shows mean across questions. If one model's curve sits consistently above the other, that model handles paraphrase + context perturbations more robustly. If their curves cross, there's a context-amount regime where the smaller model wins — itself an interesting finding.

## 6. V1 — metric inter-correlation (small-N illustrative)

![Metric correlations](05_metric_correlations.png)

Spearman ρ between the metric scalars. Design doc target: ρ ∈ 
[0.4, 0.8] between AUFI_in and existing metrics (FI_in is a new 
axis, not a re-derivation). Sample size here is too small to 
conclude; the full Sprint-5 pilot is needed for the real V1 check.

## Caveats

- Only one model in this pilot (`kit.gpt-4.1` via gateway). Adding 
  the three open-weight models is straightforward once budget allows.
- POSIX is `None` for kit.gpt-4.1 (no echo path on OpenAI chat models). 
  Will populate when Llama/Teuken/Qwen are added.
- ESS_in is small for context-heavy cells — this is a known mpnet 
  encoder limitation (paraphrases project to nearby points). The 
  own-encoder variant in Sprint 6 will not have this property.
