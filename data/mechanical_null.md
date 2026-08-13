# Mechanical-coupling null for rho_F ~ accuracy (Gulliford check)

True rho held CONSTANT across cells; per-cell mean success held at its observed
value; the full paraphrase x sample grid re-simulated from the beta-binomial and
re-estimated with the pipeline's own MoM estimator. Any correlation that appears
is produced by estimator mechanics alone (binary-outcome variance ceiling +
complete-case selection), not by the constructs. Motivated by Gulliford et al.
(2005): ICC on binary outcomes is mechanically coupled to outcome prevalence.

n_sims = 500, seed = 42. Null band = 5th-95th percentile.

## At each model's own hierarchical population rho (the realistic null)

| model | level | true rho | observed cc r | null 90% band | inside? | coverage obs vs null |
|---|---|---|---|---|---|---|
| qwen_2_5_7b | L0 | 0.487 | +0.074 | [-0.146, +0.379] | YES | 0.37 vs 0.33 |
| qwen_2_5_7b | L1 | 0.487 | +0.052 | [-0.304, +0.149] | YES | 0.54 vs 0.45 |
| llama_3_1_8b | L0 | 0.130 | +0.373 | [+0.123, +0.490] | YES | 0.54 vs 0.50 |
| llama_3_1_8b | L1 | 0.130 | +0.173 | [-0.088, +0.244] | YES | 0.79 vs 0.74 |
| mistral_7b_v03 | L0 | 0.252 | +0.186 | [-0.023, +0.409] | YES | 0.47 vs 0.41 |
| mistral_7b_v03 | L1 | 0.252 | -0.086 | [-0.243, +0.141] | YES | 0.68 vs 0.61 |

**6/6 observed complete-case correlations sit inside the
mechanical null band.** Where the observed value is inside the band, the
rho_F ~ accuracy association licenses no claim about the constructs — it is
the size the estimator produces on its own at constant true rho.

Notes. (1) The observed r here is recomputed with the identical estimator on
the identical (modal-N, complete-case) cells the null simulates, so it can
differ slightly from the cc column of `independence_target.md` (different cell
joins); per-level coverage here reproduces the review's pooled 45/66/57 %
exactly. (2) The null makes NO claim about the true rho_F ~ accuracy relation;
it shows the complete-case estimate is uninformative about it at this design
size — which is precisely why the hierarchical estimator (R2) is primary.
(3) The mechanical sign pattern (positive band at the L0 accuracy floor,
centred band near 0.5 at L1) matches the observed sign pattern, as Gulliford's
prevalence-coupling predicts.

## Coverage-selection artifact under the null

| model | level | corr(defined, acc) observed | null 90% band |
|---|---|---|---|
| qwen_2_5_7b | L0 | +0.555 | [+0.471, +0.544] |
| qwen_2_5_7b | L1 | +0.038 | [-0.023, +0.076] |
| llama_3_1_8b | L0 | +0.797 | [+0.773, +0.851] |
| llama_3_1_8b | L1 | +0.356 | [+0.294, +0.459] |
| mistral_7b_v03 | L0 | +0.758 | [+0.661, +0.770] |
| mistral_7b_v03 | L1 | +0.327 | [+0.151, +0.311] |

A non-zero corr(defined, accuracy) under the null shows the coverage-selection
artifact (review §2.3) is reproduced by mechanics alone: cells at the accuracy
extremes are exactly where a binary ICC degenerates. Full grid incl. the
sensitivity sweep over true rho in `data/mechanical_null.parquet`.
