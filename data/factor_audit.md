# Factor-structure audit (supervisor feedback, 2026-08-11)

## Q1 - What went in, and does the structure match the claim?

14 metrics, 6 model x level strata, mean within-stratum Spearman. Top-3 explain **0.746**; Horn retains **3**.

| metric | claimed axis | F1 | F2 | F3 | dominant | match |
|---|---|---|---|---|---|---|
| accuracy | comp | -0.21 | +0.93 | -0.01 | F2 (comp) | yes |
| AUFI (graded) | comp | +0.21 | -0.92 | +0.01 | F2 (comp) | yes |
| rho_F | sens | +0.03 | +0.13 | +0.90 | F3 (sens) | yes |
| FI premium | comp | +0.20 | +0.74 | +0.27 | F2 (comp) | yes |
| spread (Cao) | sens | +0.27 | +0.48 | +0.64 | F3 (sens) | yes |
| H_sem | disp | +0.98 | -0.09 | +0.03 | F1 (disp) | yes |
| FI_out_fixed | disp | -0.72 | -0.07 | +0.06 | F1 (disp) | yes |
| Var[FI_out] | disp | +0.71 | -0.11 | +0.33 | F1 (disp) | yes |
| TVD-sens | disp | +0.91 | -0.10 | +0.33 | F1 (disp) | yes |
| S_tau (Errica) | disp | +0.92 | -0.05 | +0.01 | F1 (disp) | yes |
| variation ratio | disp | +0.77 | -0.05 | +0.37 | F1 (disp) | yes |
| |A_q| observed | disp | +0.90 | -0.13 | +0.07 | F1 (disp) | yes |
| rho_u (Cox) | sens | +0.29 | -0.09 | +0.82 | F3 (sens) | yes |
| ESS_in | sens | -0.01 | -0.09 | +0.42 | F3 (sens) | yes |

Factor identity: F1 = disp, F2 = comp, F3 = sens. **14/14 metrics land on the factor of the axis they were assigned to a priori.**

## Q2 - Is '3 factors' an artifact of stacking aliases?

Seven of the 14 inputs are provable functions of one clustering (dispersion family) and AUFI is accuracy. Re-run with ONE representative per identity class plus every separate measurement (7 variables: accuracy, rho_F, FI premium, spread (Cao), H_sem, rho_u (Cox), ESS_in).

Eigenvalues: [2.642, 1.571, 1.043, 0.854, 0.423, 0.317, 0.149]
Horn 95th pct: [1.47, 1.27, 1.149, 1.045, 0.957, 0.874, 0.79]
**Horn retains 2**; top-3 explain 0.751.

| metric | F1 | F2 | F3 |
|---|---|---|---|
| accuracy | -0.10 | +0.72 | -0.52 |
| rho_F | +0.78 | +0.37 | +0.07 |
| FI premium | +0.11 | +0.05 | +0.88 |
| spread (Cao) | +0.81 | +0.10 | +0.31 |
| H_sem | +0.47 | +0.72 | +0.25 |
| rho_u (Cox) | +0.67 | -0.20 | -0.29 |
| ESS_in | +0.11 | +0.87 | +0.07 |

## Q3 - FI_spec: was it in the PCA, and where does it fall?

**It was not, and it cannot be.** Correlations are computed WITHIN a specificity level (so the manipulation is not smuggled into the matrix). At L0, FI_spec = log2(m0/m0) = 0 bits for every question: zero variance, correlation undefined. Its question-level carrier log2(m0) - the count of annotator-listed readings, identical at both levels and equal to FI_spec at L1 - is defined throughout, so we add that instead.

15-variable re-run: Horn retains **4**, top-3 explain 0.705.

log2(m0) loadings: F1 -0.13 | F2 -0.50 | F3 +0.28 -> dominant F2 (comp), |loading| = 0.50

Direct within-stratum Spearman of log2(m0) with the three representatives:

| stratum | vs accuracy | vs rho_F | vs H_sem |
|---|---|---|---|
| qwen_2_5_7b L0 | -0.138 | -0.179 | +0.246 |
| qwen_2_5_7b L1 | -0.284 | -0.022 | +0.070 |
| llama_3_1_8b L0 | -0.192 | -0.185 | +0.107 |
| llama_3_1_8b L1 | -0.244 | -0.052 | +0.118 |
| mistral_7b_v03 L0 | -0.135 | -0.040 | +0.046 |
| mistral_7b_v03 L1 | -0.161 | -0.052 | +0.169 |
| **mean** | **-0.192** | **-0.088** | **+0.126** |

## Q4 - Is 'Horn retains 3' seed- and n-stable?

| assumed n | seeds 1..10 -> retained | eigenvalue 3 | eigenvalue 4 |
|---|---|---|---|
| 80 | [3] | 1.616 | 0.944 |
| 136 | [3] | 1.616 | 0.944 |
| 150 | [3] | 1.616 | 0.944 |
| 299 | [3] | 1.616 | 0.944 |

(Eigenvalue 3 = 1.616 sits well above every Horn threshold; eigenvalue 4 = 0.944 sits below every one. The retention decision is not close, so it cannot flip with the simulation seed.)
