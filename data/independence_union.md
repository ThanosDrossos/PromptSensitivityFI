# R3 — independence re-analysis

Pre-stated equivalence bound: **|rho| < 0.2** (claimed only if it holds in ALL models).

## Split-half reliability of rho_F (disjoint paraphrase halves, Spearman-Brown)

| model | reliability |
|---|---|
| qwen_2_5_7b | 0.363 |
| llama_3_1_8b | 0.509 |
| mistral_7b_v03 | 0.495 |

Disattenuated correlations below use r / sqrt(rel_rhoF * 0.95).

## rho_F vs accuracy

| model | level | n (cc) | coverage | complete-case r [90% CI] | imputed-0 r | hierarchical r [95% CI] | equivalent? |
|---|---|---|---|---|---|---|---|
| qwen_2_5_7b | L0 | 91 | 61% | -0.167 [-0.331, +0.007] (disatt -0.284) | -0.080 | -0.023 [-0.249, +0.205] | NO |
| qwen_2_5_7b | L1 | 88 | 59% | -0.198 [-0.362, -0.023] (disatt -0.337) | -0.229 | +0.003 [-0.220, +0.226] | NO |
| llama_3_1_8b | L0 | 122 | 81% | +0.235 [+0.088, +0.371] (disatt +0.337) | +0.312 | +0.136 [-0.077, +0.338] | NO |
| llama_3_1_8b | L1 | 126 | 84% | +0.106 [-0.042, +0.249] (disatt +0.152) | +0.130 | +0.092 [-0.113, +0.290] | NO |
| mistral_7b_v03 | L0 | 116 | 77% | -0.126 [-0.274, +0.028] (disatt -0.184) | +0.072 | -0.001 [-0.222, +0.220] | NO |
| mistral_7b_v03 | L1 | 117 | 78% | -0.255 [-0.392, -0.106] (disatt -0.371) | -0.155 | -0.039 [-0.254, +0.180] | NO |

**Verdict — equivalence at |rho| < 0.2: NOT ESTABLISHED (complete case).** Smallest bound the data support in every model/level: **|rho| < 0.39**. Hierarchical (primary) estimates never exceed |0.136|. Direction consistency: 2/6 positive, sign test p = 0.688.

## rho_F vs H_sem

| model | level | n (cc) | coverage | complete-case r [90% CI] | imputed-0 r | hierarchical r [95% CI] | equivalent? |
|---|---|---|---|---|---|---|---|
| qwen_2_5_7b | L0 | 91 | 61% | -0.001 [-0.175, +0.172] (disatt -0.002) | +0.506 | -0.073 [-0.282, +0.143] | YES |
| qwen_2_5_7b | L1 | 88 | 59% | +0.005 [-0.172, +0.181] (disatt +0.008) | +0.588 | -0.138 [-0.338, +0.074] | YES |
| llama_3_1_8b | L0 | 122 | 81% | -0.080 [-0.227, +0.071] (disatt -0.114) | +0.053 | -0.036 [-0.235, +0.166] | NO |
| llama_3_1_8b | L1 | 126 | 84% | +0.077 [-0.071, +0.222] (disatt +0.111) | +0.196 | +0.019 [-0.178, +0.215] | NO |
| mistral_7b_v03 | L0 | 116 | 77% | +0.188 [+0.035, +0.332] (disatt +0.274) | +0.235 | +0.003 [-0.206, +0.212] | NO |
| mistral_7b_v03 | L1 | 117 | 78% | +0.223 [+0.073, +0.364] (disatt +0.326) | +0.344 | +0.038 [-0.172, +0.245] | NO |

**Verdict — equivalence at |rho| < 0.2: NOT ESTABLISHED (complete case).** Smallest bound the data support in every model/level: **|rho| < 0.36**. Hierarchical (primary) estimates never exceed |0.138|. Direction consistency: 4/6 positive, sign test p = 0.688.

## rho_F vs dispersion_factor

| model | level | n (cc) | coverage | complete-case r [90% CI] | imputed-0 r | hierarchical r [95% CI] | equivalent? |
|---|---|---|---|---|---|---|---|
| qwen_2_5_7b | L0 | 91 | 61% | +0.159 [-0.015, +0.324] (disatt +0.271) | +0.578 | -0.004 [-0.217, +0.210] | NO |
| qwen_2_5_7b | L1 | 88 | 59% | +0.168 [-0.008, +0.335] (disatt +0.287) | +0.645 | -0.067 [-0.275, +0.146] | NO |
| llama_3_1_8b | L0 | 122 | 81% | +0.054 [-0.097, +0.202] (disatt +0.077) | +0.135 | +0.048 [-0.151, +0.244] | NO |
| llama_3_1_8b | L1 | 126 | 84% | +0.208 [+0.063, +0.345] (disatt +0.299) | +0.307 | +0.111 [-0.088, +0.302] | NO |
| mistral_7b_v03 | L0 | 116 | 77% | +0.285 [+0.137, +0.420] (disatt +0.416) | +0.285 | +0.080 [-0.130, +0.282] | NO |
| mistral_7b_v03 | L1 | 117 | 78% | +0.341 [+0.198, +0.469] (disatt +0.497) | +0.441 | +0.126 [-0.084, +0.325] | NO |

**Verdict — equivalence at |rho| < 0.2: NOT ESTABLISHED (complete case).** Smallest bound the data support in every model/level: **|rho| < 0.47**. Hierarchical (primary) estimates never exceed |0.126|. Direction consistency: 6/6 positive, sign test p = 0.031.

## How to read the three estimators

**The hierarchical column is primary.** It is the only one defined on all cells and the only one that propagates per-cell uncertainty (multiple imputation over posterior draws, pooled by Rubin's rules). It answers: *given what we actually know about rho_F, how much does it co-vary with the other axis?*

**Complete-case** answers a different and also legitimate question: *among the cells where rho_F is measurable at all, how much does it co-vary?* It is range-restricted on accuracy (the excluded cells are the extremes), so it is not a substitute for the hierarchical estimate, but a consistent sign across model x level is informative.

**Imputed-0 is reported only as a sensitivity check and must NOT be used as a headline. It is an artifact of the imputation rule.** Degenerate cells are heavily skewed toward all-WRONG rather than all-RIGHT (2.2:1 qwen, 6.8:1 llama, 4.8:1 mistral), and corr(is-degenerate, accuracy) = -0.21 / -0.34 / -0.38. Substituting a constant 0 therefore drops a large spike of zeros at low accuracy and manufactures a positive correlation that is a property of the rule, not of the constructs. (An earlier draft of the review used these numbers as the counter-estimate; that was wrong.)
