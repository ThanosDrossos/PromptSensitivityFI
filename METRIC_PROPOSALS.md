# Metric proposals v2 — prompt sensitivity beyond AUFI

**Status:** analysis + proposal, 2026-07-24. Grounded in the v3 full run
(149 q × 2 levels × 3 models, graded F) and the governing docs
(Section_7 §7.3–7.9, Research_Design v2/v3). Every number below recomputed from
`data/specificity_v3_<model>.parquet` + `data/hidden_states_<model>.parquet`;
per-cell candidate values in `data/metric_candidates_v3.parquet`,
report card in `figures/v3_metric_report_card.png`.

## 1. The problem, stated exactly

Under binary F, AUFI_in = 0.975·(−log₂ accuracy) — a deterministic transform
(Spearman −1.000 exactly; verified to 2e-16). The graded track softens this to
ρ ≈ −0.999: still, **AUFI is accuracy in a logarithmic wrapper**. The FI_in(k)
*curve* remains the right presentation of the construct (§7.3.2), but the
project needs scalar sensitivity metrics that do NOT re-measure accuracy.

Report card criteria (all measured on v3):
- **redundancy**: |Spearman| with graded accuracy, computed WITHIN specificity
  level (pooling levels would smuggle the manipulation into the correlation);
- **response**: paired L0→L1 delta + Wilcoxon (does it react to the
  specificity manipulation?);
- **reliability**: cross-model per-cell Spearman (does it measure a stable
  property of questions?);
- **coverage**: fraction of cells where it is defined/informative.

| candidate | redundancy | response (sig/3) | reliability | coverage | verdict |
|---|---|---|---|---|---|
| AUFI_graded (baseline) | **0.999** | ✓ 3/3 | 0.79 | 100% | accuracy in disguise |
| reformulation gain log₂(F_max/F̄) | **0.981** | ✓ 3/3 | 0.67 | 66% | rejected — same disguise |
| **ρ_F (functional ICC)** | **0.157** | flat 0/3 | 0.33 | 56% | **adopt — the sensitivity trait** |
| **ΔFI reliability premium** | 0.507 | ✓ 2/3 (+) | 0.42 | 100% | **adopt — the curve-shape scalar** |
| Var[FI_out] (§7.4.2) | 0.220 | ✓ 2/3 (−) | 0.37 | 86% | adopt — output-side 2nd-order |
| TVD sensitivity (1−consistency) | 0.259 | ✓ 2/3 (−) | 0.45 | 87% | adopt — gold-free triangulation |
| x*-geometry ρ (§7.7) | 0.138 | (population-level) | 0.17 | ~50% | **adopt as FINDING, not per-q metric** |

## 2. Proposed metric set

### M1 — ρ_F: functional ICC ("formulation share of functional variance")

One-way random-effects ICC(1) on the N×k correctness outcomes, grouped by
paraphrase (exact for binomial cells from the stored per-paraphrase rates):

    SS_between = k·Σᵢ(F̄ᵢ − F̄)²,  SS_within = Σᵢ k·F̄ᵢ(1−F̄ᵢ)
    ρ_F = (MSB − MSW) / (MSB + (k−1)·MSW),  clamped to [0,1]

**Reading:** the fraction of the model's success variability attributable to
*phrasing choice* rather than sampling noise. ρ_F = 0: rephrasing is
irrelevant (all variability is decoding noise). ρ_F = 1: success is fully
determined by which paraphrase you picked.

**Evidence:** redundancy 0.16 (vs 0.999) — a genuinely different quantity;
FI-native (variance decomposition of F over U_q); the F-space analogue of Cox's
ρ_u, closing the loop with the Tier-C stack. **Its non-response to specificity
is a feature, correctly read:** disambiguation raises *ability* but leaves the
*share* of phrasing-induced variability unchanged — sensitivity-as-trait is a
different axis from difficulty, which is precisely what a sensitivity metric
should look like. Honest limits: undefined on all-0/all-1 cells (56% coverage —
principled: no variance ⇒ sensitivity unmeasurable there; always report
coverage); moderate reliability 0.33 at k=10, N=10 (CIs needed; grows with k).

### M2 — ΔFI(k₁→k₂): the reliability premium (two-threshold Szostak contrast)

    ΔFI(q) = FI_in(q, k=1.0) − FI_in(q, k=0.5)
            = log₂( N_{F≥0.5} / N_{F=1.0} )   [clamped as usual]

**Reading:** the extra bits of phrasing-rarity demanded by *perfect*
reliability over *usable* reliability. Pure curve-shape: identically 0 under
binary F, so it isolates exactly the information the graded track added. Stays
100% inside the Hazen/Szostak formalism (§7.3.2 explicitly frames FI as
threshold-indexed; this is the canonical two-level contrast).

**Evidence:** redundancy 0.51 (moderate — it measures the tail, which is
legitimately related to ability), responds to specificity in 2/3 models with a
**new finding: the premium RISES with disambiguation** (+0.12/+0.49/+0.17;
llama p=1e-4) — *disambiguation buys usability, not perfection*: half-reliable
phrasings become common while perfectly-reliable ones stay rare.

### M3 — x*-geometry (§7.7 H1): report as a confirmed hypothesis

Per cell with function variation: x* = argmax F (ties → shortest), then
ρ(F(x), ‖e(x) − e(x*)‖) over the remaining paraphrases, TBG states at 75%
depth. **H1 CONFIRMED in all 3 models:** mean per-cell ρ −0.36/−0.31/−0.31,
78–85% of cells negative, Wilcoxon vs 0: p ≤ 1e-13 each; pooled within-cell
ρ −0.25/−0.19/−0.20. Function decays with embedding distance from the best
phrasing — the geometric reading of FI_in (curve = distribution function of
distances to x*) is empirically supported, and it ties the probes (same
hidden states) to the metric story. As a *per-question* scalar it is too noisy
at N=10 (cross-model reliability 0.17) → report at population level per
(model, level); per-question version is future work with larger N.

### M4 — output-side pair (already computed; promote from diagnostics)

- **Var_x[FI_out(x)]** (§7.4.2 verbatim; = h_sem_var): does *how much the
  output concentrates* itself depend on phrasing? Redundancy 0.22, responds
  (−) in 2/3 models. The doc's own "second-order sensitivity signal".
- **TVD sensitivity = 1 − consistency** (Errica-anchored): mean pairwise
  distribution shift across paraphrases. **Gold-free** — usable on live
  traffic with no ground truth. Redundancy 0.26, responds (−) in 2/3.

### Rejected

- **Reformulation gain** log₂(F_max/F̄): redundancy 0.98 — dominated by the
  mean; same disguise as AUFI.
- **Residualized AUFI**: keep ONLY as the P1 probe label (statistical
  orthogonalization, not a conceptual metric).

## 3. The reporting frame this implies

Prompt sensitivity is not one number; the v3 data supports a **two-axis
summary** per (model, level): **ability** (accuracy / the FI_in curve height)
× **sensitivity** (ρ_F as the trait scalar, ΔFI premium as the shape scalar),
triangulated by the output-side pair, with the x*-geometry as the mechanism
story and the FI_in(k) curve as the primary visual. AUFI drops to an appendix
alias of accuracy.

## 4. Pipeline changes (small, additive)

1. Emit `rho_F` and `fi_premium` as first-class columns in the spec driver
   (both computable from `f_graded_per_paraphrase` — backfillable post-hoc for
   v3, no re-run needed).
2. Switch the P1 probe target to ρ_F (or residualized AUFI) — fixes the
   P1≈P2 collinearity with a label that is 0.16-, not 0.999-, correlated with
   accuracy.
3. Writeup: bootstrap CIs for ρ_F (cluster bootstrap over paraphrases);
   Kendall-τ generator-stability check (§7.8 P1) stays on the roadmap.
