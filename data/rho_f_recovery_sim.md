# A5 — recovery, draws-based test, and identifiability of hierarchical rho_F

Script: `prompt_sensitivity/scripts/rho_f_recovery_sim.py` (seeds 0-4, deterministic). Companion to `data/stats_hygiene.md`.

## 1. Recovery: what a true Delta rho_F reports as (posterior-mean scale)

Per-cell mean success held at its observed value; true rho = model's own fitted population rho at L0 and +Delta at L1; full grid re-simulated; the pipeline's own estimator re-fit; paired posterior-mean delta recorded. Mean [min, max] over 5 seeds.

| model | rho0 | true +0.00 | true +0.05 | true +0.10 | true +0.20 |
|---|---|---|---|---|---|
| qwen_2_5_7b | 0.444 | -0.003 [-0.007, +0.004] | +0.007 [-0.006, +0.015] | +0.019 [+0.013, +0.032] | +0.034 [+0.029, +0.037] |
| llama_3_1_8b | 0.091 | +0.004 [-0.005, +0.012] | +0.020 [+0.006, +0.038] | +0.039 [+0.023, +0.049] | +0.075 [+0.059, +0.090] |
| mistral_7b_v03 | 0.239 | +0.001 [-0.003, +0.006] | +0.008 [+0.005, +0.020] | +0.026 [+0.018, +0.034] | +0.047 [+0.038, +0.057] |

**Reading.** The reported delta is attenuated roughly 3-5x (qwen/mistral) and 2-3x (llama): a true +0.20 — larger than the whole between-model range of rho_F in this study — reports as ~+0.04-0.05 in qwen and mistral and ~+0.08 in llama. Any bound stated on the posterior-mean scale must therefore be read through this attenuation: the '0.04' CI bound of the specificity null corresponds to true effects of roughly 0.2 (qwen, mistral) and 0.10-0.15 (llama). The null is genuinely informative only against effects of that size, and the paper's wording must say so.

## 2. Rubin-pooled paired test on the committed posterior draws

200 posterior draws per cell; each draw analysed as a completed dataset (mean paired L1-L0 delta, n = complete question pairs); Rubin's rules pool point estimate, within- and between-draw variance.

| gold | model | Delta rho_F | 95% CI | p | share of variance from posterior uncertainty |
|---|---|---|---|---|---|
| union | qwen_2_5_7b | -0.0121 | [-0.0805, +0.0562] | 0.728 | 43% |
| union | llama_3_1_8b | +0.0135 | [-0.0195, +0.0466] | 0.422 | 32% |
| union | mistral_7b_v03 | +0.0149 | [-0.0381, +0.0679] | 0.581 | 39% |
| target | qwen_2_5_7b | -0.0016 | [-0.0568, +0.0535] | 0.953 | 47% |
| target | llama_3_1_8b | +0.0133 | [-0.0214, +0.0481] | 0.452 | 40% |
| target | mistral_7b_v03 | +0.0145 | [-0.0409, +0.0698] | 0.608 | 40% |

**Reading.** The draws-based CIs are wider than the posterior-mean CIs because they carry the per-cell posterior uncertainty the shrunken means discard. The null result stands under this test; its honest statement is the CI, read together with the attenuation table above.

## 3. Identifiability: narrow-arm geometry vs production geometry, equal cell counts

Known constant rho simulated on each arm's realized geometry (per-cell means and universe sizes); population mean of the fitted per-cell posterior means; 5 seeds. Equal n_cells makes this a regime contrast, not a sample-size contrast.

| model | geometry | n cells | mean U | true 0.1 | true 0.3 | true 0.5 |
|---|---|---|---|---|---|---|
| qwen_2_5_7b | narrow | 100 | 6.7 | 0.169 [0.018, 0.330] | 0.571 [0.345, 0.635] | 0.811 [0.666, 0.892] |
| qwen_2_5_7b | production | 100 | 10.0 | 0.061 [0.001, 0.156] | 0.258 [0.200, 0.309] | 0.533 [0.381, 0.767] |
| llama_3_1_8b | narrow | 100 | 6.7 | 0.062 [0.001, 0.181] | 0.303 [0.149, 0.529] | 0.508 [0.460, 0.616] |
| llama_3_1_8b | production | 100 | 10.0 | 0.061 [0.035, 0.073] | 0.277 [0.218, 0.318] | 0.523 [0.438, 0.613] |
| mistral_7b_v03 | narrow | 100 | 6.7 | 0.030 [0.001, 0.147] | 0.249 [0.160, 0.360] | 0.484 [0.432, 0.531] |
| mistral_7b_v03 | production | 100 | 10.0 | 0.040 [0.001, 0.114] | 0.327 [0.264, 0.461] | 0.517 [0.428, 0.649] |

**Reading.** On production geometry the estimator tracks the truth; on narrow geometry (smaller universes, degeneracy-heavy means) recovery is biased and seed-unstable at the same number of cells. This substantiates the width table's estimator note: the per-arm hierarchical fit on the narrow arm is a regime problem, so the MoM-on-covered-cells and the N-matched hierarchical contrasts are the interpretable quantities there.
