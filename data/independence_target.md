# R3 — independence re-analysis

Pre-stated equivalence bound: **|rho| < 0.2** (claimed only if it holds in ALL models).

## Split-half reliability of rho_F (disjoint paraphrase halves, Spearman-Brown)

| model | reliability |
|---|---|
| qwen_2_5_7b | 0.380 |
| llama_3_1_8b | 0.520 |
| mistral_7b_v03 | 0.571 |

Disattenuated correlations below use r / sqrt(rel_rhoF * 0.95).

## rho_F vs accuracy

| model | level | n (cc) | coverage | complete-case r [90% CI] | imputed-0 r | hierarchical r [95% CI] | equivalent? |
|---|---|---|---|---|---|---|---|
| qwen_2_5_7b | L0 | 56 | 37% | +0.044 [-0.180, +0.264] (disatt +0.074) | +0.467 | -0.019 [-0.233, +0.198] | NO |
| qwen_2_5_7b | L1 | 80 | 53% | +0.023 [-0.163, +0.207] (disatt +0.038) | -0.001 | +0.015 [-0.213, +0.242] | NO |
| llama_3_1_8b | L0 | 81 | 54% | +0.259 [+0.079, +0.423] (disatt +0.369) | +0.603 | +0.009 [-0.197, +0.214] | NO |
| llama_3_1_8b | L1 | 117 | 78% | +0.140 [-0.013, +0.287] (disatt +0.199) | +0.251 | +0.061 [-0.151, +0.268] | NO |
| mistral_7b_v03 | L0 | 70 | 47% | +0.132 [-0.068, +0.322] (disatt +0.179) | +0.610 | +0.017 [-0.192, +0.225] | NO |
| mistral_7b_v03 | L1 | 102 | 68% | -0.115 [-0.273, +0.050] (disatt -0.156) | +0.137 | +0.030 [-0.191, +0.248] | NO |

**Verdict — equivalence at |rho| < 0.2: NOT ESTABLISHED (complete case).** Smallest bound the data support in every model/level: **|rho| < 0.42**. Hierarchical (primary) estimates never exceed |0.061|. Direction consistency: 5/6 positive, sign test p = 0.219.

## rho_F vs H_sem

| model | level | n (cc) | coverage | complete-case r [90% CI] | imputed-0 r | hierarchical r [95% CI] | equivalent? |
|---|---|---|---|---|---|---|---|
| qwen_2_5_7b | L0 | 56 | 37% | -0.067 [-0.285, +0.157] (disatt -0.112) | +0.328 | -0.045 [-0.263, +0.178] | NO |
| qwen_2_5_7b | L1 | 80 | 53% | +0.009 [-0.177, +0.194] (disatt +0.014) | +0.515 | -0.086 [-0.292, +0.127] | YES |
| llama_3_1_8b | L0 | 81 | 54% | -0.045 [-0.227, +0.141] (disatt -0.064) | +0.088 | -0.041 [-0.249, +0.170] | NO |
| llama_3_1_8b | L1 | 117 | 78% | +0.023 [-0.131, +0.175] (disatt +0.032) | +0.142 | -0.040 [-0.242, +0.164] | YES |
| mistral_7b_v03 | L0 | 70 | 47% | +0.107 [-0.093, +0.299] (disatt +0.145) | +0.130 | -0.008 [-0.222, +0.207] | NO |
| mistral_7b_v03 | L1 | 102 | 68% | +0.173 [+0.009, +0.328] (disatt +0.235) | +0.152 | -0.003 [-0.214, +0.209] | NO |

**Verdict — equivalence at |rho| < 0.2: NOT ESTABLISHED (complete case).** Smallest bound the data support in every model/level: **|rho| < 0.33**. Hierarchical (primary) estimates never exceed |0.086|. Direction consistency: 4/6 positive, sign test p = 0.688.

## rho_F vs dispersion_factor

| model | level | n (cc) | coverage | complete-case r [90% CI] | imputed-0 r | hierarchical r [95% CI] | equivalent? |
|---|---|---|---|---|---|---|---|
| qwen_2_5_7b | L0 | 56 | 37% | +0.131 [-0.094, +0.343] (disatt +0.218) | +0.370 | +0.005 [-0.216, +0.225] | NO |
| qwen_2_5_7b | L1 | 80 | 53% | +0.209 [+0.024, +0.379] (disatt +0.347) | +0.579 | -0.025 [-0.237, +0.190] | NO |
| llama_3_1_8b | L0 | 81 | 54% | +0.105 [-0.081, +0.283] (disatt +0.149) | +0.150 | +0.017 [-0.190, +0.223] | NO |
| llama_3_1_8b | L1 | 117 | 78% | +0.162 [+0.010, +0.307] (disatt +0.231) | +0.244 | +0.042 [-0.164, +0.244] | NO |
| mistral_7b_v03 | L0 | 70 | 47% | +0.244 [+0.048, +0.422] (disatt +0.331) | +0.190 | +0.056 [-0.158, +0.265] | NO |
| mistral_7b_v03 | L1 | 102 | 68% | +0.311 [+0.156, +0.452] (disatt +0.423) | +0.251 | +0.083 [-0.130, +0.289] | NO |

**Verdict — equivalence at |rho| < 0.2: NOT ESTABLISHED (complete case).** Smallest bound the data support in every model/level: **|rho| < 0.45**. Hierarchical (primary) estimates never exceed |0.083|. Direction consistency: 6/6 positive, sign test p = 0.031.

## How to read the three estimators

**The hierarchical column is primary.** It is the only one defined on all cells and the only one that propagates per-cell uncertainty (multiple imputation over posterior draws, pooled by Rubin's rules). It answers: *given what we actually know about rho_F, how much does it co-vary with the other axis?*

**Complete-case** answers a different and also legitimate question: *among the cells where rho_F is measurable at all, how much does it co-vary?* It is range-restricted on accuracy (the excluded cells are the extremes), so it is not a substitute for the hierarchical estimate, but a consistent sign across model x level is informative.

**Imputed-0 is reported only as a sensitivity check and must NOT be used as a headline. It is an artifact of the imputation rule.** Degenerate cells are heavily skewed toward all-WRONG rather than all-RIGHT (2.2:1 qwen, 6.8:1 llama, 4.8:1 mistral), and corr(is-degenerate, accuracy) = -0.21 / -0.34 / -0.38. Substituting a constant 0 therefore drops a large spike of zeros at low accuracy and manufactures a positive correlation that is a property of the rule, not of the constructs. (An earlier draft of the review used these numbers as the counter-estimate; that was wrong.)
