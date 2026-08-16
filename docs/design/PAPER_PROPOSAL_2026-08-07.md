# Paper proposal — what we claim now (post R1–R3)

Basis: `RESULTS_R1_R2_R3_2026-08-07.md`, `data/independence_{target,union}.md`, review + corrections.
Framing per Thanos's decisions (2026-08-07): measurement-model paper; metric analysis is a major part;
contributions = {axes/ρ_F, measurement model, prompt-checker}; the lottery result lives in **Methods**.

---

## 1. Key takeaways after R1–R3

1. **The specificity effect is real, at half the advertised size.** Under union gold (the dataset's own
   protocol), disambiguation improves accuracy by **+0.064 / +0.125 / +0.119** (BH-significant in all three
   models). The old +0.22–0.25 headline was **47–72 % grading lottery**, now quantified and removed.
2. **ρ_F survives every stress test applied to it as a *construct*** — gold-robust (per-cell agreement
   .53–.69 across scoring rules; same model ordering; same dial response), trait-like (does **not** respond
   to the specificity dial, now a properly powered null on n = 150 per model, replicated under both golds),
   and its model ordering (qwen .49 > mistral .25 > llama .13) holds for the share **and** the absolute
   variance component σ²_B, killing the decoding-noise objection.
3. **But "orthogonal" is gone.** The data support equivalence only to |ρ| < 0.33–0.47; and ρ_F has a small,
   *consistent* positive association with the dispersion family (6/6 model×level positive, sign test
   p = .031, replicated under both golds). Also honest now: reliability 0.35–0.58, coverage was outcome-
   dependent (fixed by the hierarchical estimator), cross-model transfer weak → a (question × model) trait.
4. **The dispersion "family" is one object, provably.** S_τ ≡ H_sem/log₂|A|; Var[FI_out] ≡ Var[H_sem];
   FI_out_fixed ≡ log₂m₀ − H_sem; |A_q|, variation ratio, 1−TVD are functionals of the same pooled
   clustering. Only POSIX is an independent measurement — and it fails to discriminate axis 3 from axis 2.
5. **Axis 1 is accuracy.** AUFI ≡ accuracy (ρ = −.9997); the persisted binary FI_in curve encodes one number;
   Δ-bits scale with an arbitrary cap. And Hazen 2007 already did FI-of-letter-sequences, so the FI dressing
   cannot carry novelty.

## 2. Do the three axes still hold?

**As a structure: yes — and the evidence for it is now better than before.**
- Horn parallel analysis retains **exactly 3 factors** (the one strong result of the old independence
  section).
- Under the primary (hierarchical, uncertainty-propagating) estimator, no cross-axis association exceeds
  **|0.14|** in any model × level, under either gold.
- **New, and stronger than any correlation: the axes dissociate under the manipulation.** The same dial that
  moves ability (+6–13 pts, 3/3 models) leaves ρ_F exactly flat (Δ ≤ 0.013, 3/3 models, both golds) and
  moves H_sem weakly (Holm-robust in 1/3 models). Differential responsiveness to an intervention is textbook
  construct-separation evidence, and it does not suffer from the power problem the correlations have.
- **UPDATE 2026-08-08 — the double dissociation is now complete on BOTH sides** (R6 width dial,
  `RESULTS_R6_width_dial_2026-08-08.md`): widening the paraphrase distribution raises σ²_B in 3/3 models
  (p .021–.060) and ρ_F (MoM) monotonically in 3/3 (qwen p = .0035 seed-stable; mistral .051; llama .079 —
  the effect tracks the model's own sensitivity level), while leaving accuracy AND H_sem cleanly flat
  (Friedman .47–.68). Plus the first paraphraser-swap ablation in this literature (P5): per-cell ρ_F
  agreement at the reliability ceiling and the model ordering preserved under OLMo-2-13B —
  **levels are G-relative, structure is G-robust.**

**As originally claimed: no.** Three renamings:

| axis | was | becomes |
|---|---|---|
| 1 | "FI_in, the bits-view of ability" | **Competence** — graded accuracy F̄, reported plainly. FI_in^G(k) kept only as a *presentation* (graded curve per R4b; AUFI and the binary curve to appendix; no headline in "bits") |
| 2 | "ρ_F, orthogonal to everything — THE novel metric" | **Formulation sensitivity** — ρ_F with the hierarchical estimator as primary (100 % coverage, posterior SD), σ²_B alongside, reliability and bounds reported. Claim: *distinct and non-redundant*, not orthogonal. Novelty = the **conjunction**: per-question × noise-corrected × task-success-referenced (Cox: no gold, no correction; BrittleBench: model-level, deterministic) |
| 3 | "H_sem + a family of convergent indices" | **Output dispersion** — H_sem as the *sole* representative, with a Proposition showing the others are reductions of it. FI_out_fixed = unit conversion only, never a second result |
| dial | "FI_spec, bits of specificity, dose–response" | **The manipulated variable** — two-point (ambiguous/disambiguated); drop per-bit dose language (unvalidatable in a 2-point design); effect stated as Δ_union |

## 3. Proposed contributions (abstract-ready)

> **C1 — A measurement model that tidies the prompt-sensitivity metric zoo.** Three questions — how well
> (competence), how much does phrasing decide it (formulation sensitivity), how scattered are the answers
> (dispersion) — each with one representative. We prove, not estimate, that widely used indices reduce to
> the dispersion representative (S_τ, TVD-consistency, |A|, variation ratio, Var[FI_out] are functions of
> one cluster distribution; two are exact relabelings of H_sem), and we show empirically that POSIX does not
> discriminate between dispersion and formulation sensitivity. Independence of the three axes is reported
> honestly: bounded association (all |ρ| ≤ 0.14 under the primary estimator; equivalence establishable only
> to 0.33–0.47) plus **experimental dissociation** — the specificity manipulation moves competence in all
> models while leaving formulation sensitivity flat.
>
> **C2 — ρ_F, a per-question, sampling-noise-corrected, task-success-referenced sensitivity share, with an
> estimator that works.** The ICC-style share published indices lack; a hierarchical beta-binomial estimator
> giving 100 % coverage (the MoM version is undefined on exactly the outcome extremes) with per-cell
> uncertainty; honest psychometrics (split-half reliability .38–.57, canonical per `data/stats_hygiene.md`; cross-model transfer .2–.45 ⇒ a
> question × model trait); robustness across grading conventions; and a stable model ranking confirmed by
> the absolute variance component.
>
> **C3 — A prompt-checker artifact.** Linear probes on last-prompt-token states detect underspecified
> questions **zero-shot** on annotator-labelled data (AUROC .670–.678, hardened eval) where matched-protocol text baselines
> get .54–.59; with in-domain labels, bag-of-words catches up — so the value is label-efficiency/transfer,
> not raw detection. (Full R7 baseline suite pending.)

**In Methods, prominently (not a contribution bullet, per decision):** the union-gold protocol. AmbigQA's own
evaluation scores against all interpretations; scoring one pinned reading manufactures a pseudo-effect worth
47–72 % of the naive gain (decomposition table as protocol validation). This is also the paper's shield: we
found and removed our own confound.

**One-sentence pitch:** *Prompt sensitivity is three measurements, not one number: most published indices are
provably a single dispersion quantity; the missing quantity — the phrasing-attributable share of task success —
is measurable per question, stable across grading conventions, and untouched by the very manipulation that
doubles accuracy.*

## 4. The results table the paper leads with

| | competence (F̄, union gold) | ρ_F (hier.) | H_sem |
|---|---|---|---|
| dial L0→L1 | **+.064/+.125/+.119** (BH-sig 3/3) | −.013/+.013/+.013 (n.s. 3/3) | falls 3/3, Holm-robust 1/3 |
| model ordering | — | qwen .49 > mistral .25 > llama .13 (both golds, share + σ²_B) | — |
| reading | specificity raises competence | trait of (question × model), not moved by specificity | weak level response; the axis where the literature lives |

## 5. Claim language — say / don't say

| don't say | say |
|---|---|
| orthogonal / independent axes | distinct, non-redundant; \|ρ\| ≤ 0.14 (primary), equivalence bound 0.33–0.47; dissociation under manipulation |
| "accuracy roughly doubles" | +6–13 pts against the full valid-answer set; the naive doubling was 47–72 % grading artifact (Methods) |
| "first application of FI to prompts" | generator-relative instantiation FI^G of Hazen's letter-sequence construction with an LLM receiver |
| "one construct, many costumes (r = .60–.94)" | Proposition: five indices are functionals of one cluster distribution (two exactly); POSIX shown separately |
| "stability .81–.95" | split-half reliability .38–.57 (superset comparison retired) |
| "significant in all three models" (unqualified) | BH-sig 3/3, Holm-robust 2/3 (qwen's Δ_union: BH .045, Holm .15); plus the pooled single-experiment test p = 3.3e-05 |
| "the three axes explain 75.5 %" | top-3 explain **74.6 %** on the canonical script-generated 14-variable matrix (Horn still retains exactly 3) |
| "phrasing axis stays empty" (POSIX) | POSIX correlates with both axes; difference n.s. (Williams p = .42/.63/.054) |

## 5b. Construct validity of ρ_F (added 2026-08-07, after Thanos's challenge)

Full dossier: `RHO_F_CONSTRUCT_VALIDITY_2026-08-07.md`. The flat dial is the *discriminant* half of
validity (FI_spec is an ability dial; ρ_F's own dials act on the generator G, the decoding, and the
model). The positive-control half now has three data-backed pieces:
- **cross-regime convergence**: ρ_F (T=1) predicts deterministic T=0 phrasing disagreement at partial
  r = +.40/+.51/+.54 (controlling accuracy-extremeness);
- **out-of-sample payoff prediction (promote into Results)**: ρ_F from the first k=10 samples predicts
  the rephrasing payoff F_max − F̄ on the disjoint k=20 second half at +.61/+.39/+.70, surviving
  partialling out accuracy — "rephrase or give up" is decidable before trying;
- the generator-width dial remains the named missing experiment (was R6; ESS_in range-restriction makes
  it untestable in-data).

## 6. Open risks to manage in the draft

1. **qwen's Δ_union is marginal** (+0.064, p = .031) — phrase as "6–13 pts, smallest in the strongest model"
   (which is itself sensible: qwen starts at 0.53 union accuracy at L0).
2. **H_sem's level effect is Holm-robust in one model only** — never write "all three models" for axis 3.
3. ~~**Literature verification pending**~~ **RESOLVED 2026-08-07 afternoon** — verdicts in
   `LITERATURE_REVIEW_2026-08-07_VERIFIED.md`, applied changes in `LITERATURE_INTEGRATION_2026-08-07.md`:
   - **Żatuchin: real, does NOT scoop ρ_F's estimand** (sentiment outcome, corpus-level, brand object) —
     but it publishes the G-theory apparatus first, and Urbano (SIGIR 2013) shows the MoM estimator is
     textbook IR-evaluation practice. **C2 must drop the word "estimator" from its novelty clause** —
     the surviving conjunction is *gold-referenced task success × per question* with a standard estimator.
   - **Pecher: real, scoops the prose thesis** ("sensitivity ∝ underspecification") for text
     classification at T=0, instruction-side — differentiated in the related-work rewrite.
   - **Kossen SEP: real (ICML 2024 *workshop*), same token position as ours** — token position is NOT a
     differentiator; their App. A.12 already tried continuous SE targets. **Zhang sparse-neurons: real,
     EMNLP 2025 main, on AmbigQA, with transfer — the ambiguity probe is scooped**; C3 survives only as
     the paraphrase-distribution heads + zero-shot transfer evaluation (probes section repositioned).
   - **New since the proposal:** the ρ_F ~ accuracy complete-case association is fully explained by a
     mechanical prevalence-coupling null (`data/mechanical_null.md`, 24/24 inside the band) — the
     bounded-association story in C1 gets *stronger*, and the imputed-0 numbers are retired for good.
4. ~~**R4/R4b/R5 must land before the draft**~~ **ALL LANDED** — R4/R4b/R5 (2026-08-07), R6/R7
   (2026-08-08), R8/R9 (2026-08-08). **The Results section is WRITTEN** (main_body.tex §Results,
   2026-08-08): 4.1 three distinct axes, 4.2 specificity dial (+ endpoint table), 4.3 width dial +
   swap (+ table), 4.4 measurement properties, 4.5 probes. Methods gained the width-dial
   subsubsection and the hardened-probe-evaluation paragraph; §Aggregation carries the R8 inference
   policy. Numbers sourced from `data/stats_hygiene.md` (primary), `data/metric_reductions.md`,
   `data/axis1_graded_curve.md`, `data/width_dial_analysis.md`, `data/probe_eval_hardened.md`.
   Still open: Discussion (findings/contributions/limitations are stubs + TODO notes), Conclusion,
   abstract/intro contribution alignment, pending-bib verifications (Gulliford, Jiang, Cegin, …).
