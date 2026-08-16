# rho_F construct validity — scripted (replaces the unscripted Table-5 rows)

Script: `prompt_sensitivity/scripts/rho_f_construct_validity.py`. All rho_F(hier) values are the UNION-gold hierarchical estimator; MoM rows are labelled. Every correlation carries its n.

## 1. Convergent validity — the axis is not empty

Within-stratum Spearman of rho_F(hier) with the two published neighbours (and, for contrast, the other axes):

| model | level | ~rho_u (Cox) | ~spread (Cao) | ~accuracy | ~H_sem | n |
|---|---|---|---|---|---|---|
| qwen_2_5_7b | L0 | +0.455 | +0.213 | -0.163 | -0.011 | 150 |
| qwen_2_5_7b | L1 | +0.240 | +0.175 | -0.126 | -0.215 | 150 |
| llama_3_1_8b | L0 | +0.432 | +0.244 | +0.050 | -0.141 | 150 |
| llama_3_1_8b | L1 | +0.529 | +0.370 | +0.092 | -0.047 | 150 |
| mistral_7b_v03 | L0 | +0.517 | +0.282 | +0.063 | -0.064 | 150 |
| mistral_7b_v03 | L1 | +0.505 | +0.339 | +0.064 | -0.067 | 150 |

**Reading.** rho_u and spread track rho_F in every stratum — more strongly than anything else in the metric set. This is convergent validity for the axis, and the paper reports it as such; the claim "measured by no existing index" is licensed only with the qualifier "with the within-prompt sampling term removed".

## 2. Incremental validity — out-of-sample payoff, k=20 second half

Payoff = F_max - F_mean on the disjoint second half (samples 10-19) of the k=20 arm; predictors estimated from the first k=10 samples. Complete cases of the k=20 subsample (50 questions x 2 levels).

| model | rho_F (hier) | rho_F (MoM) | spread | rho_u | n | partial rho_F(hier) given spread |
|---|---|---|---|---|---|---|
| qwen_2_5_7b | +0.020 (n=100) | +0.615 (n=42) | +0.800 (n=100) | +0.297 (n=100) | 100 | -0.347 (n=100) |
| llama_3_1_8b | +0.097 (n=100) | +0.397 (n=62) | +0.633 (n=100) | +0.047 (n=100) | 100 | -0.037 (n=100) |
| mistral_7b_v03 | +0.095 (n=100) | +0.702 (n=62) | +0.618 (n=100) | +0.286 (n=100) | 100 | -0.259 (n=100) |

**Reading.** The utility criterion is comparative: rho_F earns its estimator only if it beats the one-line statistic (spread) and the no-gold ratio (rho_u) at predicting the payoff. The table states the result either way; the partial column shows what rho_F adds beyond spread.

## 3. Held-out-paraphrase payoff (5-vs-5, 200 splits)

The k=20 check reuses the same ten paraphrases; this one does not: estimate on five paraphrases, predict the payoff of the disjoint five.

| model | rho_F (MoM on 5) | spread (on 5) | cells (\|U\|>=8) |
|---|---|---|---|
| qwen_2_5_7b | +0.294 | +0.385 | 299 |
| llama_3_1_8b | +0.369 | +0.526 | 299 |
| mistral_7b_v03 | +0.376 | +0.496 | 299 |

## 4. Gold-set agreement of per-cell rho_F

| model | hier union~target | MoM union~target (both defined) |
|---|---|---|
| qwen_2_5_7b | +0.534 (n=300) | +0.817 (n=131) |
| llama_3_1_8b | +0.692 (n=300) | +0.849 (n=197) |
| mistral_7b_v03 | +0.681 (n=300) | +0.756 (n=170) |

## 5. Greedy-pass disagreement (cross-decoding-regime check)

Disagreement = the deterministic T=0 pass splits across paraphrases (0 < f_mean < 1). Partial Spearman of rho_F(hier) with that indicator, controlling accuracy-extremeness |F_graded - 1/2| (the mechanical channel).

| model | partial Spearman | raw Spearman | n |
|---|---|---|---|
| qwen_2_5_7b | +0.436 | +0.188 | 300 |
| llama_3_1_8b | +0.313 | +0.313 | 300 |
| mistral_7b_v03 | +0.372 | +0.314 | 300 |

## 6. Cross-model transfer of per-question rho_F (hier, union)

| pair | pooled levels | L0 | L1 |
|---|---|---|---|
| qwen_2_5_7b ~ llama_3_1_8b | +0.162 (n=300) | +0.187 | +0.145 |
| qwen_2_5_7b ~ mistral_7b_v03 | +0.117 (n=300) | +0.123 | +0.125 |
| llama_3_1_8b ~ mistral_7b_v03 | +0.225 (n=300) | +0.154 | +0.293 |

**Reading.** Transfer is weak; formulation sensitivity is a property of the (question x model) pair. This row replaces the unsourced "0.2 to 0.45" range.

## 7. Commensurable specificity null (the width arm's own recipe)

Delta rho_F (L1 - L0) with complete-case MoM, on the width arm's 50 questions and on all 150, next to the hierarchical row of the primary family — so the two halves of the double dissociation share an estimator and a sample.

| model | MoM, 50 q (width sample) | MoM, 150 q | hier, 150 q (primary family) |
|---|---|---|---|
| qwen_2_5_7b | -0.0775 (p=0.64, n=11) | +0.0186 (p=0.71, n=36) | -0.0131 (p=0.54, n=150) |
| llama_3_1_8b | +0.0417 (p=0.73, n=23) | +0.0392 (p=0.26, n=70) | +0.0126 (p=0.43, n=150) |
| mistral_7b_v03 | -0.0917 (p=0.41, n=22) | +0.0167 (p=0.88, n=58) | +0.0134 (p=0.41, n=150) |

**Reading.** The specificity null and the width effect can now be compared on the same 50 questions with the same estimator; the remaining asymmetries (test direction, multiplicity regime) are presentation choices the paper states explicitly.
