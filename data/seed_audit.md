# Seed-robustness audit (supervisor feedback, 2026-08-11)

Four seed layers, varied wherever no new model calls are needed.

## L1a - Bootstrap seed (question-clustered CIs, 5,000 resamples)

| model | endpoint | effect | CI seed 1 | seed 2 | seed 3 | seed 4 | seed 5 | max CI drift |
|---|---|---|---|---|---|---|---|---|
| qwen_2_5_7b | Δ accuracy (union) | +0.0639 | [+0.001,+0.126] | [+0.001,+0.126] | [+0.001,+0.124] | [-0.001,+0.128] | [+0.001,+0.128] | 0.0021 |
| qwen_2_5_7b | Δ H_sem | -0.1239 | [-0.204,-0.047] | [-0.202,-0.047] | [-0.205,-0.048] | [-0.204,-0.045] | [-0.204,-0.045] | 0.0020 |
| llama_3_1_8b | Δ accuracy (union) | +0.1251 | [+0.066,+0.185] | [+0.066,+0.185] | [+0.066,+0.184] | [+0.066,+0.186] | [+0.066,+0.186] | 0.0016 |
| llama_3_1_8b | Δ H_sem | -0.4333 | [-0.587,-0.284] | [-0.581,-0.282] | [-0.588,-0.284] | [-0.580,-0.282] | [-0.584,-0.285] | 0.0078 |
| mistral_7b_v03 | Δ accuracy (union) | +0.1189 | [+0.056,+0.185] | [+0.056,+0.182] | [+0.055,+0.180] | [+0.053,+0.183] | [+0.055,+0.183] | 0.0048 |
| mistral_7b_v03 | Δ H_sem | -0.1393 | [-0.279,-0.001] | [-0.280,-0.001] | [-0.273,-0.003] | [-0.278,-0.003] | [-0.284,-0.002] | 0.0065 |

The effect estimate is a mean and does not depend on a seed at all; only the CI endpoints move, in the fourth decimal.

## L1b - Split-half draw seed (reliability of rho_F)

| model | 200 splits, seeds 0..4 | spread |
|---|---|---|
| qwen_2_5_7b | 0.380, 0.359, 0.371, 0.370, 0.373 | 0.020 |
| llama_3_1_8b | 0.518, 0.521, 0.516, 0.526, 0.518 | 0.010 |
| mistral_7b_v03 | 0.571, 0.574, 0.572, 0.571, 0.563 | 0.011 |

## L1c - Hierarchical rho_F estimator

| model | mean rho_F, repeat fits 1..3 | identical? |
|---|---|---|
| qwen_2_5_7b | 0.487197, 0.487197, 0.487197 | yes |
| llama_3_1_8b | 0.130492, 0.130492, 0.130492 | yes |
| mistral_7b_v03 | 0.252090, 0.252090, 0.252090 | yes |

The empirical-Bayes fit is a deterministic grid + Nelder-Mead search from a fixed start: no seed enters the point estimates.

## L3 - Target-interpretation seed (the expensive one)

Changing it re-pins which annotator reading defines L1, so BOTH the L1 prompt and the target gold change: a full re-generation. Two observations bound how much it could matter.

**(i) The primary endpoint is scored against the union of all readings.** The gold set is then seed-invariant by construction; only the L1 prompt text still depends on the draw.

**(ii) Which reading got pinned DOES weakly predict the effect** - the draw is uniform over a question's m0 readings, but the per-question effect declines with the drawn index in all three models (one significant, two marginal). So this layer is not innocuous and the size of its influence has to be quantified rather than asserted:

| model | Spearman(Δ_union, pinned index) | p | Spearman(Δ_union, m0) | p |
|---|---|---|---|---|
| qwen_2_5_7b | -0.213 | 0.01 | -0.105 | 0.20 |
| llama_3_1_8b | -0.158 | 0.05 | -0.126 | 0.12 |
| mistral_7b_v03 | -0.148 | 0.07 | -0.049 | 0.55 |

**(iii) The association is real, and it is a moderator worth reporting.** AmbigQA lists readings in a canonical order, and disambiguating to the FIRST-listed reading helps far more than disambiguating to a later one:

| model | Δ_union, pinned index = 0 | pinned index > 0 | gap |
|---|---|---|---|
| qwen_2_5_7b | +0.175 (n = 60) | -0.011 (n = 90) | +0.186 |
| llama_3_1_8b | +0.175 (n = 60) | +0.092 (n = 90) | +0.083 |
| mistral_7b_v03 | +0.184 (n = 60) | +0.075 (n = 90) | +0.109 |

**(iv) But the ESTIMAND is stable across seeds anyway.** A new seed redraws each question's reading uniformly, so what varies is the *share* of questions that land on index 0. That share is a sum of independent Bernoulli(1/m0) draws over 150 questions, so it is tightly concentrated; propagating its spread through the gap above gives the seed-induced spread of the headline:

Expected share at index 0: **0.399** (observed 0.400); SD of that share across seeds: **0.039**.

| model | headline Δ_union | implied SD across target seeds | 95 % CI half-width |
|---|---|---|---|
| qwen_2_5_7b | +0.0639 | ±0.0072 | ±0.0640 |
| llama_3_1_8b | +0.1251 | ±0.0032 | ±0.0596 |
| mistral_7b_v03 | +0.1189 | ±0.0042 | ±0.0629 |

The seed-induced spread is roughly an order of magnitude smaller than the sampling CI, so re-running with a new target seed would move the headline well inside its existing interval. The heterogeneity by reading rank, however, is a substantive result and should be reported rather than averaged away.

## What a seed replication would still add (needs the cluster)

| layer | what changes | cost | status |
|---|---|---|---|
| L1 analysis | resamples, folds, sims | free | **done, stable (above)** |
| L2 sampling | which k = 10 generations | cache-only re-score, no new generation (the k = 20 arm already holds a disjoint second half) | **recommended, cheap** |
| L3 target reading | L1 prompt + target gold | full re-generation of the L1 half of the grid | bounded above; not run |
| L4 generator | who writes the paraphrases | already done as the R6 swap arm (OLMo-2 replaces Phi-4 as generator AND judge) | **done** |
