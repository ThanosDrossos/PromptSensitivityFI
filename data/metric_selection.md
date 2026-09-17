# Metric selection — published metrics first, ρ_F projected, then added (generated)

The component analysis that justifies the three factors must not contain the metric it
justifies. **Stage 1** decomposes the ten published metrics only (mean within-stratum Spearman
matrix, six model × level strata) and projects the six constructed metrics into that space as
*supplementary* variables: a held-out metric's loading is its correlation with the component
score, computed without letting it shape the component. **Stage 2** adds ρ_F as an active
variable and repeats the retention test. The earlier fourteen-metric decomposition remains in
`data/factor_audit.md` and is repeated in §7 as a sensitivity variant.

## 1. The metric sets

| metric | a priori factor | source | role |
|---|---|---|---|
| H_sem | output dispersion | Kuhn et al. 2023; Farquhar et al. 2024 | active (published) |
| S_τ | output dispersion | Errica et al. 2025 | active (published) |
| TVD consistency | output dispersion | Errica et al. 2025 | active (published) |
| \|A_q\| | output dispersion | Kuhn et al. 2023 (number of semantic sets) | active (published) |
| variation ratio | output dispersion | Lu et al. 2024 | active (published) |
| accuracy | mean task success | graded accuracy, mean over formulations | active (published) |
| F_max (best formulation) | mean task success | Sclar et al. 2024; Mizrahi et al. 2024 | active (published) |
| F_min (worst formulation) | mean task success | Cao et al. 2024; Sclar et al. 2024 | active (published) |
| spread | formulation dependence | Sclar et al. 2024; Cao et al. 2024 | active (published) |
| ρ_u | formulation dependence | Cox et al. 2025 | active (published) |
| ρ_F | formulation dependence | this work | held out in stage 1 |
| AUFI | mean task success | this work | held out in stage 1 |
| ΔFI premium | mean task success | this work | held out in stage 1 |
| FI_out^fixed | output dispersion | this work | held out in stage 1 |
| Var[FI_out] | output dispersion | this work | held out in stage 1 |
| ESS_in | formulation dependence | this work | held out in stage 1 |

F_max and F_min are the best- and worst-formulation graded accuracies of a cell, computed
from the persisted per-formulation rates (`f_graded_per_paraphrase`), the same basis as
accuracy and ρ_F; they were added to the audit on 2026-09-16 (FORKING_PATHS fork 14). The
persisted `spread` keeps the pipeline's original scoring rule (Spearman 0.66–0.85 with
F_max − F_min on graded rates); a graded-spread variant is in §7. POSIX (Chatterjee et al.
2024) is measured on a separate 100-cell arm and is analyzed in `data/metric_reductions.md`.
Provenance guard: the 14 metrics shared with `figures/v3_metric_corr.npy` reproduce it to
0.0e+00.

## 2. Stage 1 — how many components do the published metrics support?

Eigenvalues: [5.144, 2.465, 1.089, 0.549, 0.353, 0.183, 0.144, 0.038, 0.025, 0.01]

Horn 95th percentile (random data, n = 150 per stratum, 500 draws): [1.55, 1.382, 1.264, 1.172, 1.075, 1.004, 0.928, 0.855, 0.784, 0.703]

**Horn retains 2.** The third eigenvalue (1.089) falls below its parallel
threshold (1.264); at n = 56 to 150 and ten seeds the count is always [2]. The first two components explain 76.1% of the
variance, the first three 87.0%. The two published formulation-dependence metrics
correlate +0.42 with each other, too little to carry a component of their own.

### The two retained components (varimax)

| metric | a priori factor | C1 (output dispersion) | C2 (mean task success) | dominant | margin |
|---|---|---|---|---|---|
| H_sem | output dispersion | **+0.92** | -0.19 | C1 (yes) | 0.73 |
| S_τ | output dispersion | **+0.86** | -0.15 | C1 (yes) | 0.71 |
| TVD consistency | output dispersion | **+0.97** | -0.12 | C1 (yes) | 0.85 |
| \|A_q\| | output dispersion | **+0.85** | -0.20 | C1 (yes) | 0.64 |
| variation ratio | output dispersion | **+0.88** | -0.03 | C1 (yes) | 0.86 |
| accuracy | mean task success | -0.17 | **+0.96** | C2 (yes) | 0.79 |
| F_max (best formulation) | mean task success | -0.03 | **+0.98** | C2 (yes) | 0.95 |
| F_min (worst formulation) | mean task success | -0.44 | **+0.72** | C2 (yes) | 0.28 |
| spread | formulation dependence | **+0.51** | +0.48 | C1 (NO) | 0.03 |
| ρ_u | formulation dependence | **+0.59** | +0.08 | C1 (NO) | 0.51 |
| *held out (supplementary)* | |  |  | | |
| ρ_F † | formulation dependence | **+0.38** | +0.30 | C1 (NO) | 0.08 |
| AUFI † | mean task success | +0.17 | **-0.96** | C2 (yes) | 0.79 |
| ΔFI premium † | mean task success | +0.28 | **+0.63** | C2 (yes) | 0.35 |
| FI_out^fixed † | output dispersion | **-0.58** | +0.03 | C1 (yes) | 0.55 |
| Var[FI_out] † | output dispersion | **+0.72** | -0.10 | C1 (yes) | 0.62 |
| ESS_in † | formulation dependence | **+0.14** | +0.04 | C1 (NO) | 0.10 |

## 3. Stage 1 — the three-component solution the framework posits

Three causes of a wrong answer call for three components; extracting three from the
published metrics gives each a priori family its own component. Loadings are correlations
with the varimax-rotated component scores; † marks held-out (supplementary) metrics; the
margin is the dominant |loading| minus the next largest, a measure of how specific a metric
is to one component.

| metric | a priori factor | C1 (output dispersion) | C2 (mean task success) | C3 (formulation dependence) | dominant | margin |
|---|---|---|---|---|---|---|
| H_sem | output dispersion | **+0.98** | -0.15 | +0.07 | C1 (yes) | 0.83 |
| S_τ | output dispersion | **+0.93** | -0.11 | +0.05 | C1 (yes) | 0.82 |
| TVD consistency | output dispersion | **+0.90** | -0.15 | +0.34 | C1 (yes) | 0.56 |
| \|A_q\| | output dispersion | **+0.87** | -0.18 | +0.12 | C1 (yes) | 0.69 |
| variation ratio | output dispersion | **+0.78** | -0.08 | +0.41 | C1 (yes) | 0.38 |
| accuracy | mean task success | -0.15 | **+0.97** | +0.03 | C2 (yes) | 0.82 |
| F_max (best formulation) | mean task success | -0.08 | **+0.96** | +0.20 | C2 (yes) | 0.76 |
| F_min (worst formulation) | mean task success | -0.27 | **+0.82** | -0.37 | C2 (yes) | 0.45 |
| spread | formulation dependence | +0.23 | +0.33 | **+0.74** | C3 (yes) | 0.41 |
| ρ_u | formulation dependence | +0.24 | -0.11 | **+0.84** | C3 (yes) | 0.59 |
| *held out (supplementary)* | |  |  |  | | |
| ρ_F † | formulation dependence | -0.01 | +0.10 | **+0.89** | C3 (yes) | 0.80 |
| AUFI † | mean task success | +0.15 | **-0.97** | -0.04 | C2 (yes) | 0.82 |
| ΔFI premium † | mean task success | +0.18 | **+0.58** | +0.32 | C2 (yes) | 0.26 |
| FI_out^fixed † | output dispersion | **-0.63** | +0.00 | -0.03 | C1 (yes) | 0.60 |
| Var[FI_out] † | output dispersion | **+0.64** | -0.14 | +0.31 | C1 (yes) | 0.33 |
| ESS_in † | formulation dependence | +0.04 | -0.02 | **+0.23** | C3 (yes) | 0.19 |

## 4. Representative per component (stage 1)

**C1 = output dispersion.** Best published metric: H_sem (+0.98); other members: S_τ +0.93, TVD consistency +0.90, \|A_q\| +0.87, variation ratio +0.78. Largest loading from another family: F_min (worst formulation) (-0.27). Held-out metrics of this factor: FI_out^fixed -0.63 (elsewhere +0.00, -0.03), Var[FI_out] +0.64 (elsewhere -0.14, +0.31).

**C2 = mean task success.** Best published metric: accuracy (+0.97); other members: F_max (best formulation) +0.96, F_min (worst formulation) +0.82. Largest loading from another family: spread (+0.33). Held-out metrics of this factor: AUFI -0.97 (elsewhere +0.15, -0.04), ΔFI premium +0.58 (elsewhere +0.18, +0.32).

**C3 = formulation dependence.** Best published metric: ρ_u (+0.84); other members: spread +0.74. Largest loading from another family: variation ratio (+0.41). Held-out metrics of this factor: ρ_F +0.89 (elsewhere -0.01, +0.10), ESS_in +0.23 (elsewhere +0.04, -0.02).

## 5. Stage 2 — ρ_F added as an active variable

Eigenvalues: [5.232, 2.744, 1.612, 0.551, 0.36, 0.241, 0.183, 0.082, 0.033, 0.01, 0.0]

Horn 95th percentile (n = 150): [1.59, 1.401, 1.285, 1.194, 1.115, 1.043, 0.964, 0.893, 0.831, 0.753, 0.678]

**Horn retains 3.** The third eigenvalue (1.612) exceeds its threshold at
every n from 56 to 150 and every seed (counts [3]; thresholds 1.47 at n = 56, 1.40 at n = 84, 1.32 at n = 136, 1.29 at n = 150). The first three explain 86.8%. Loadings of the published metrics move by at most
0.08 relative to stage 1.

| metric | a priori factor | C1 (output dispersion) | C2 (mean task success) | C3 (formulation dependence) | dominant | margin |
|---|---|---|---|---|---|---|
| H_sem | output dispersion | **+0.98** | -0.15 | +0.02 | C1 (yes) | 0.83 |
| S_τ | output dispersion | **+0.93** | -0.11 | +0.02 | C1 (yes) | 0.82 |
| TVD consistency | output dispersion | **+0.92** | -0.15 | +0.31 | C1 (yes) | 0.61 |
| \|A_q\| | output dispersion | **+0.89** | -0.18 | +0.04 | C1 (yes) | 0.71 |
| variation ratio | output dispersion | **+0.80** | -0.08 | +0.36 | C1 (yes) | 0.44 |
| accuracy | mean task success | -0.15 | **+0.97** | +0.05 | C2 (yes) | 0.82 |
| F_max (best formulation) | mean task success | -0.09 | **+0.94** | +0.28 | C2 (yes) | 0.66 |
| F_min (worst formulation) | mean task success | -0.28 | **+0.83** | -0.37 | C2 (yes) | 0.46 |
| spread | formulation dependence | +0.27 | +0.31 | **+0.72** | C3 (yes) | 0.41 |
| ρ_u | formulation dependence | +0.30 | -0.12 | **+0.77** | C3 (yes) | 0.46 |
| ρ_F | formulation dependence | +0.02 | +0.06 | **+0.97** | C3 (yes) | 0.91 |

## 6. Empirical cross-check of the projection (per stratum, stage-1 components)

Each stratum is scored with the pooled weights (rank-standardized active metrics); the table
gives the Spearman correlation of each probe metric with the component scores, on the cells
where the probe is defined (ρ_F is complete-case; n in parentheses).

| stratum | ρ_F ~ C1 | ρ_F ~ C2 | ρ_F ~ C3 | spread ~ C1 | spread ~ C2 | spread ~ C3 | ρ_u ~ C1 | ρ_u ~ C2 | ρ_u ~ C3 |
|---|---|---|---|---|---|---|---|---|---|
| qwen_2_5_7b L0 | -0.09 (n=56) | +0.08 | +0.85 | +0.26 | +0.42 | +0.68 | +0.42 | -0.04 | +0.82 |
| qwen_2_5_7b L1 | -0.02 (n=80) | -0.08 | +0.91 | +0.51 | +0.04 | +0.79 | +0.58 | -0.23 | +0.87 |
| llama_3_1_8b L0 | -0.05 (n=81) | +0.21 | +0.54 | +0.21 | +0.50 | +0.54 | -0.06 | -0.12 | +0.83 |
| llama_3_1_8b L1 | +0.01 (n=117) | +0.09 | +0.73 | +0.12 | +0.19 | +0.72 | +0.14 | -0.11 | +0.88 |
| mistral_7b_v03 L0 | +0.07 (n=70) | +0.08 | +0.75 | +0.16 | +0.47 | +0.60 | +0.15 | -0.02 | +0.84 |
| mistral_7b_v03 L1 | +0.11 (n=102) | -0.19 | +0.84 | +0.17 | +0.10 | +0.74 | +0.34 | -0.24 | +0.89 |
| **mean** | **+0.00** | **+0.03** | **+0.77** | **+0.24** | **+0.29** | **+0.68** | **+0.26** | **-0.12** | **+0.86** |

## 7. Sensitivity of the count and of the key loadings

| variant | p | eigenvalues 1–4 | Horn retains | ρ_F on C_dep | spread on C_dep / C_succ | ρ_u on C_dep | H_sem on C_disp | accuracy on C_succ |
|---|---|---|---|---|---|---|---|---|
| stage 1: ten published metrics | 10 | 5.14, 2.47, 1.09, 0.55 | 2 | +0.89 † | +0.74 / +0.33 | +0.84 | +0.98 | +0.97 |
| stage 2: ten published + ρ_F | 11 | 5.23, 2.74, 1.61, 0.55 | 3 | +0.97 | +0.72 / +0.31 | +0.77 | +0.98 | +0.97 |
| eight published (without F_max, F_min) | 8 | 4.74, 1.40, 0.83, 0.45 | 2 | +0.80 † | +0.56 / +0.62 | +0.93 | +0.98 | +0.90 |
| eight published + ρ_F | 9 | 4.85, 1.99, 0.95, 0.47 | 2 | +0.95 | +0.67 / +0.51 | +0.83 | +0.99 | +0.92 |
| stage 2 with spread on graded rates | 11 | 5.21, 3.01, 1.60, 0.56 | 3 | +1.00 | +0.77 / +0.46 | +0.75 | +0.98 | +0.97 |
| stage 2 + AUFI, FI premium (accuracy aliases) | 13 | 5.50, 3.88, 1.71, 0.69 | 3 | +0.96 | +0.72 / +0.35 | +0.76 | +0.98 | +0.96 |
| all fourteen earlier metrics (factor_audit.md, n = 136) | 14 | 5.95, 2.87, 1.62, 0.94 | 3 | +0.90 | +0.64 / +0.48 | +0.82 | +0.98 | +0.93 |

Loadings marked † are supplementary projections; the others are active loadings. Without
F_max and F_min the published set has a single success metric, and a component needs several
indicators to be retained; with them, success is retained and formulation dependence is the
one factor the published metrics do not carry. With all fourteen earlier metrics Horn retains
three because AUFI (an exact alias of accuracy) and the FI premium add success indicators and
ρ_F a dependence indicator; that count is superseded by the two-stage result above.

## 8. Conventions

- Matrix: mean over the six model × level strata of the within-stratum pairwise Spearman
  correlation (recipe of `make_metric_corr.py`); correlations never pool across levels.
- Horn: 500 random normal data sets of n × p, 95th percentile per eigenvalue, seed 42; n = 150
  (every published metric is defined on all 150 cells of every stratum). For stage 2, ρ_F is
  complete-case (56 to 117 cells per stratum), so the count is also reported at n = 56 and 84.
- Rotation: varimax (Kaiser 1958) on the first k components; each component's sign is fixed so
  that its largest |loading| is positive. Supplementary loadings use the same rotation.
