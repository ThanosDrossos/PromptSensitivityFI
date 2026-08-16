# R4 — Metric reductions: the dispersion family is one object

**Proposition.** Let P be a cell's pooled semantic-cluster distribution and |A| its support
size, as produced by the pipeline's pooled clustering. Then, exactly as computed:

| index | reduction | verified max abs error (all cells, 3 models) |
|---|---|---|
| S_τ (Errica) | H(P) / log₂\|A\| | 1.67e-16 |
| FI_out_fixed | log₂(m₀) − H(P) | 0.00e+00 |
| Var[FI_out] | Var[H_sem] | 4.44e-16 |
| \|A_q\|, variation ratio, 1−TVD | functionals of P (same pooled clustering) | by construction |

**Consequences.**
1. Within-dataset correlations among these indices are *arithmetic*, not evidence of
   convergent measurement. Report the reductions as a table, never as correlations.
2. The paired L0→L1 test on FI_out_fixed **is** the H_sem test (an affine map with a
   per-question constant cannot change a paired test). Report exactly one of them.
3. **Degeneracy rule (stated, not silent):** S_τ is undefined when \|A\| ≤ 1 — 13.8% of cells across the three models (the pipeline emitted 0 there). All S_τ analyses condition on \|A\| ≥ 2 and report that coverage.

## One distribution, many answers — the paired L0→L1 test per index

"Functionals of one clustering" is NOT "one measurement": on the primary
specificity test the indices reach different decisions, because dividing by a
per-cell-varying normaliser is not a monotone transform across cells. This is
why the paper reports exactly ONE dispersion representative (H_sem) and treats
agreement within the family as arithmetic, never as convergent evidence.

| model | H_sem | S_tau | variation ratio | 1-TVD consistency | \|A_q\| |
|---|---|---|---|---|---|
| qwen_2_5_7b | -0.124 (p=0.011, n=150) | -0.017 (p=0.21, n=150) | -0.064 (p=0.0056, n=150) | +0.084 (p=0.002, n=150) | -1.600 (p=8.9e-05, n=150) |
| llama_3_1_8b | -0.433 (p=3.7e-07, n=150) | -0.053 (p=0.00028, n=150) | -0.048 (p=0.042, n=150) | +0.120 (p=2.8e-06, n=150) | -8.120 (p=3.6e-08, n=150) |
| mistral_7b_v03 | -0.139 (p=0.034, n=150) | -0.029 (p=0.057, n=150) | -0.018 (p=0.48, n=150) | +0.040 (p=0.16, n=150) | -1.760 (p=0.052, n=150) |

## POSIX — the one independent measurement — does not discriminate axis 3 from axis 2

Same (ρ_F-covered) subset for both correlations; Williams/Steiger test of the difference:

| model | n | POSIX~H_sem [95 % CI] | POSIX~ρ_F [95 % CI] | Williams t | p |
|---|---|---|---|---|---|
| qwen_2_5_7b | 42 | +0.409 [+0.119, +0.634] | +0.237 [-0.072, +0.505] | 0.82 | 0.415 |
| llama_3_1_8b | 62 | +0.380 [+0.143, +0.575] | +0.297 [+0.051, +0.509] | 0.48 | 0.630 |
| mistral_7b_v03 | 62 | +0.560 [+0.360, +0.710] | +0.318 [+0.074, +0.526] | 1.97 | 0.054 |

POSIX correlates with **both** axes; in no model is its dispersion loading significantly
larger than its ρ_F loading. The claim "the phrasing axis stays empty" is **withdrawn**;
the honest statement is that POSIX is not axis-diagnostic at this sample size.

## Per-model identity checks

| model | S_τ err (n, cond. \|A\|≥2) | S_τ degenerate (of which =0) | FI_out_fixed err | Var[FI_out] err |
|---|---|---|---|---|
| qwen_2_5_7b | 1.1e-16 (n=221) | 26.3% (100%) | 0.0e+00 | 3.3e-16 |
| llama_3_1_8b | 1.7e-16 (n=280) | 6.7% (100%) | 0.0e+00 | 4.4e-16 |
| mistral_7b_v03 | 1.7e-16 (n=275) | 8.3% (100%) | 0.0e+00 | 2.2e-16 |
