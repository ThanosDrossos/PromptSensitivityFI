# R8 — statistical hygiene (source of truth for Results)

**Declared primary family** (12 paired Wilcoxon tests; family-wise Holm + BH). The primary effect of the manipulation is Δ accuracy under UNION gold; target-gold Δ is the protocol comparison (its excess over union = the grading lottery); FI_out_fixed is excluded — its test IS the H_sem test (affine relabeling, identical p).

| model | endpoint | n | effect (L1−L0) | 95 % CI (question-clustered) | p | Holm | BH |
|---|---|---|---|---|---|---|---|
| qwen_2_5_7b | accuracy (union gold)  [PRIMARY] | 150 | +0.0639 | [-0.0002, +0.1279] | 0.031 | 0.15 | 0.045 |
| qwen_2_5_7b | accuracy (target gold) | 150 | +0.2247 | [+0.1611, +0.2921] | 1.3e-09 | 1.3e-08 | 5.4e-09 |
| qwen_2_5_7b | H_sem | 150 | -0.1239 | [-0.2049, -0.0454] | 0.011 | 0.068 | 0.019 |
| qwen_2_5_7b | rho_F (hierarchical, union gold) | 150 | -0.0131 | [-0.0406, +0.0137] | 0.54 | 1 | 0.54 |
| llama_3_1_8b | accuracy (union gold)  [PRIMARY] | 150 | +0.1251 | [+0.0667, +0.1859] | 3.8e-05 | 0.00031 | 9.2e-05 |
| llama_3_1_8b | accuracy (target gold) | 150 | +0.2379 | [+0.1826, +0.2965] | 1.3e-13 | 1.6e-12 | 1.6e-12 |
| llama_3_1_8b | H_sem | 150 | -0.4333 | [-0.5831, -0.2842] | 3.7e-07 | 3.4e-06 | 1.1e-06 |
| llama_3_1_8b | rho_F (hierarchical, union gold) | 150 | +0.0126 | [-0.0083, +0.0332] | 0.43 | 1 | 0.47 |
| mistral_7b_v03 | accuracy (union gold)  [PRIMARY] | 150 | +0.1189 | [+0.0559, +0.1818] | 0.00026 | 0.0018 | 0.00053 |
| mistral_7b_v03 | accuracy (target gold) | 150 | +0.2471 | [+0.1831, +0.3143] | 1.7e-11 | 1.9e-10 | 1e-10 |
| mistral_7b_v03 | H_sem | 150 | -0.1393 | [-0.2774, +0.0011] | 0.034 | 0.15 | 0.045 |
| mistral_7b_v03 | rho_F (hierarchical, union gold) | 150 | +0.0134 | [-0.0118, +0.0389] | 0.41 | 1 | 0.47 |

## Replication lines (outside the declared family)

The family's rho_F endpoint is union gold since 2026-08-16 (it was target gold while the family was declared union-primary — an inconsistency, review 08-14 §3.5.3). The target-gold rho_F test and the permissive-threshold scoring promised in Methods are reported here, uncorrected, as replication checks:

| model | endpoint | n | effect (L1−L0) | 95 % CI | p |
|---|---|---|---|---|---|
| qwen_2_5_7b | rho_F (hierarchical, target gold) | 150 | -0.0019 | [-0.0172, +0.0137] | 0.98 |
| qwen_2_5_7b | accuracy (target gold, permissive scoring) | 150 | +0.2047 | [+0.1360, +0.2740] | 1.2e-07 |
| llama_3_1_8b | rho_F (hierarchical, target gold) | 150 | +0.0127 | [-0.0031, +0.0286] | 0.22 |
| llama_3_1_8b | accuracy (target gold, permissive scoring) | 150 | +0.2607 | [+0.1880, +0.3353] | 1.6e-09 |
| mistral_7b_v03 | rho_F (hierarchical, target gold) | 150 | +0.0134 | [-0.0111, +0.0382] | 0.48 |
| mistral_7b_v03 | accuracy (target gold, permissive scoring) | 150 | +0.2473 | [+0.1793, +0.3180] | 9.6e-10 |

The evidential strength of the rho_F null is bounded by the estimator's attenuation and by the draws-based test — see `data/rho_f_recovery_sim.md` (a true Delta of +0.20 reports as ~+0.03-0.08; the Rubin-pooled draws CIs span roughly ±0.05-0.08).

## The three models are correlated measurements, not replications

Per-question deltas correlate across models:

| endpoint | pair | Spearman |
|---|---|---|
| accuracy (union gold)  [PRIMARY] | qwen_2_5_7b ~ llama_3_1_8b | +0.635 |
| accuracy (union gold)  [PRIMARY] | qwen_2_5_7b ~ mistral_7b_v03 | +0.546 |
| accuracy (union gold)  [PRIMARY] | llama_3_1_8b ~ mistral_7b_v03 | +0.523 |
| H_sem | qwen_2_5_7b ~ llama_3_1_8b | +0.437 |
| H_sem | qwen_2_5_7b ~ mistral_7b_v03 | +0.370 |
| H_sem | llama_3_1_8b ~ mistral_7b_v03 | +0.427 |

The honest single-experiment test (per-question deltas averaged over the three models, one Wilcoxon per endpoint):

| endpoint | n questions | pooled effect | 95 % CI | p |
|---|---|---|---|---|
| accuracy (union gold)  [PRIMARY] | 150 | +0.1026 | [+0.0487, +0.1568] | 3.3e-05 |
| H_sem | 150 | -0.2322 | [-0.3300, -0.1369] | 6.1e-05 |

## Reliability (replaces the retired k10-vs-k20 comparison)

Disjoint-paraphrase split-half, mean of 200 random 5/5 splits, Spearman-Brown:

| model | split-half reliability |
|---|---|
| qwen_2_5_7b | 0.380 |
| llama_3_1_8b | 0.520 |
| mistral_7b_v03 | 0.571 |

Consequences: per-question rho_F point claims are not supportable; observed cross-metric and cross-model correlations are attenuated by these reliabilities; the k=20 arm is reported only as a sampling-extension check, never as reliability.
