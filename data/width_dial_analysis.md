# R6 — generator-width dial (positive control for ρ_F)

Planned comparisons P0–P5 are in the module docstring and the runbook, written
2026-08-07 before the arm data existed — self-attested: the repository history
(batch-committed 2026-08-14) does not independently timestamp them, so the paper
says "planned", not "preregistered".

## P0 — manipulation check (gates everything below)

**cells with full ordering narrow<medium<wide: 66%; means 4.6 < 8.4 < 9.2 (tokens); Wilcoxon narrow<wide p = 4.2e-18; ORDERED = True**

| arm | pairwise token dist | length CV | mean |U| |
|---|---|---|---|
| medium | 8.37 | 0.20 | 10.0 |
| narrow | 4.58 | 0.08 | 6.7 |
| swap | 7.55 | 0.16 | 10.0 |
| wide | 9.26 | 0.15 | 10.0 |

Gate censoring per arm (identical gates; differences are the gate reacting to G):

| arm | universes | nli_reject_rate | constraint_reject_rate | dedup_reject_rate | dropped_frac | fallback_nli_frac | note |
|---|---|---|---|---|---|---|---|
| narrow | 100.0 | 0.07 | 0.032 | 0.844 | 0.66 | 0.66 |  |
| medium |  |  |  |  |  |  | no sidecar (pre-R6 cache) |
| wide | 100.0 | 0.636 | 0.018 | 0.231 | 0.01 | 0.01 |  |
| swap | 100.0 | 0.546 | 0.02 | 0.297 | 0.0 | 0.01 |  |

## qwen_2_5_7b

universe sizes |U| per arm (unequal-N caveat): medium: mean 10.0, median 10; narrow: mean 6.7, median 7; wide: mean 10.0, median 10

- **P1 σ²_B** (increases): 0.0256 → 0.0258 → 0.0301 | narrow<wide one-sided p = 0.0542, two-sided p = 0.108 | Friedman p = 0.0991 | Δ(wide−narrow) = +0.0045 [-0.0062, +0.0145] | monotone cells 70% (n = 98)
- **P2 ρ_F (hier., ⚠ see caveat)** (increases): 0.6770 → 0.5034 → 0.5158 | narrow<wide one-sided p = 1, two-sided p = 4.73e-15 | Friedman p = 6.05e-23 | Δ(wide−narrow) = -0.1612 [-0.1864, -0.1339] | monotone cells 4% (n = 100)
- **P2b ρ_F (MoM, covered cells)** (increases): 0.3563 → 0.4631 → 0.5193 | narrow<wide one-sided p = 0.00351, two-sided p = 0.00701 | Friedman p = 0.067 | Δ(wide−narrow) = +0.1629 [+0.0627, +0.2783] | monotone cells 23% (n = 22)
- **P3 accuracy** (little change): 0.3994 → 0.4184 → 0.4109 | narrow<wide one-sided p = 0.262, two-sided p = 0.525 | Friedman p = 0.681 | Δ(wide−narrow) = +0.0116 [-0.0055, +0.0323] | monotone cells 55% (n = 100)
- **P4 H_sem** (little change): 0.3125 → 0.2868 → 0.2593 | narrow<wide one-sided p = 0.966, two-sided p = 0.0684 | Friedman p = 0.469 | Δ(wide−narrow) = -0.0532 [-0.1007, -0.0101] | monotone cells 26% (n = 100)
- **ρ_F (MoM) unpaired per-arm** (each arm's own covered cells): 0.3342 (n=32) → 0.3892 (n=42) → 0.3946 (n=40)
- **σ²_B unpaired per-arm** (each arm's own covered cells): 0.0256 (n=98) → 0.0253 (n=100) → 0.0295 (n=100)
- **P1 σ²_B [N-matched, seeds 0–4]**: seed-0 means 0.0256 → 0.0272 → 0.0350 | narrow<wide one-sided p median 0.0356, range [0.0287, 0.246] (n = 98)
- **P2 ρ_F (hier.) [N-matched, seeds 0–4]**: seed-0 means 0.6770 → 0.3525 → 0.7334 | narrow<wide one-sided p median 2.76e-07, range [4.72e-08, 1] (n = 100)
- **P2b ρ_F (MoM, covered) [N-matched, seeds 0–4]**: seed-0 means 0.3516 → 0.5109 → 0.5405 | narrow<wide one-sided p median 0.00739, range [0.00311, 0.0156] (n = 20)
- **P5 swap (OLMo medium)**: per-cell ρ_F Spearman = +0.413 (p = 1.9e-05, n = 100); means 0.503 (Phi-4) vs 0.360 (OLMo); σ²_B per-cell Spearman = +0.679

## llama_3_1_8b

universe sizes |U| per arm (unequal-N caveat): medium: mean 10.0, median 10; narrow: mean 6.7, median 7; wide: mean 10.0, median 10

- **P1 σ²_B** (increases): 0.0108 → 0.0140 → 0.0153 | narrow<wide one-sided p = 0.0212, two-sided p = 0.0424 | Friedman p = 0.426 | Δ(wide−narrow) = +0.0045 [+0.0008, +0.0082] | monotone cells 51% (n = 98)
- **P2 ρ_F (hier., ⚠ see caveat)** (increases): 0.0807 → 0.0851 → 0.1182 | narrow<wide one-sided p = 1.46e-08, two-sided p = 2.92e-08 | Friedman p = 2.02e-13 | Δ(wide−narrow) = +0.0375 [+0.0229, +0.0518] | monotone cells 20% (n = 100)
- **P2b ρ_F (MoM, covered cells)** (increases): 0.1130 → 0.1348 → 0.1382 | narrow<wide one-sided p = 0.0785, two-sided p = 0.157 | Friedman p = 0.519 | Δ(wide−narrow) = +0.0253 [-0.0152, +0.0663] | monotone cells 30% (n = 54)
- **P3 accuracy** (little change): 0.3104 → 0.2969 → 0.3029 | narrow<wide one-sided p = 0.686, two-sided p = 0.628 | Friedman p = 0.525 | Δ(wide−narrow) = -0.0075 [-0.0258, +0.0094] | monotone cells 39% (n = 100)
- **P4 H_sem** (little change): 1.1864 → 1.0936 → 1.1388 | narrow<wide one-sided p = 0.797, two-sided p = 0.407 | Friedman p = 0.572 | Δ(wide−narrow) = -0.0476 [-0.1495, +0.0500] | monotone cells 19% (n = 100)
- **ρ_F (MoM) unpaired per-arm** (each arm's own covered cells): 0.0984 (n=62) → 0.1312 (n=62) → 0.1144 (n=71)
- **σ²_B unpaired per-arm** (each arm's own covered cells): 0.0108 (n=98) → 0.0137 (n=100) → 0.0150 (n=100)
- **P1 σ²_B [N-matched, seeds 0–4]**: seed-0 means 0.0108 → 0.0130 → 0.0162 | narrow<wide one-sided p median 0.0316, range [0.0124, 0.0686] (n = 98)
- **P2 ρ_F (hier.) [N-matched, seeds 0–4]**: seed-0 means 0.0807 → 0.0887 → 0.1935 | narrow<wide one-sided p median 7.01e-10, range [2.64e-16, 1.56e-06] (n = 100)
- **P2b ρ_F (MoM, covered) [N-matched, seeds 0–4]**: seed-0 means 0.1109 → 0.1392 → 0.1503 | narrow<wide one-sided p median 0.0368, range [0.0214, 0.143] (n = 50)
- **P5 swap (OLMo medium)**: per-cell ρ_F Spearman = +0.273 (p = 0.0061, n = 100); means 0.085 (Phi-4) vs 0.129 (OLMo); σ²_B per-cell Spearman = +0.643

## mistral_7b_v03

universe sizes |U| per arm (unequal-N caveat): medium: mean 10.0, median 10; narrow: mean 6.7, median 7; wide: mean 10.0, median 10

- **P1 σ²_B** (increases): 0.0210 → 0.0205 → 0.0249 | narrow<wide one-sided p = 0.0596, two-sided p = 0.119 | Friedman p = 0.253 | Δ(wide−narrow) = +0.0039 [-0.0044, +0.0120] | monotone cells 61% (n = 98)
- **P2 ρ_F (hier., ⚠ see caveat)** (increases): 0.1907 → 0.2791 → 0.2373 | narrow<wide one-sided p = 6.23e-07, two-sided p = 1.25e-06 | Friedman p = 3.72e-17 | Δ(wide−narrow) = +0.0465 [+0.0276, +0.0660] | monotone cells 14% (n = 100)
- **P2b ρ_F (MoM, covered cells)** (increases): 0.2110 → 0.2303 → 0.2761 | narrow<wide one-sided p = 0.0506, two-sided p = 0.101 | Friedman p = 0.675 | Δ(wide−narrow) = +0.0652 [-0.0058, +0.1387] | monotone cells 28% (n = 46)
- **P3 accuracy** (little change): 0.3655 → 0.3798 → 0.3736 | narrow<wide one-sided p = 0.3, two-sided p = 0.6 | Friedman p = 0.537 | Δ(wide−narrow) = +0.0081 [-0.0159, +0.0322] | monotone cells 55% (n = 100)
- **P4 H_sem** (little change): 0.8977 → 0.7698 → 0.8218 | narrow<wide one-sided p = 0.792, two-sided p = 0.416 | Friedman p = 0.622 | Δ(wide−narrow) = -0.0759 [-0.1852, +0.0383] | monotone cells 32% (n = 100)
- **ρ_F (MoM) unpaired per-arm** (each arm's own covered cells): 0.1952 (n=51) → 0.1967 (n=62) → 0.2255 (n=65)
- **σ²_B unpaired per-arm** (each arm's own covered cells): 0.0210 (n=98) → 0.0201 (n=100) → 0.0244 (n=100)
- **P1 σ²_B [N-matched, seeds 0–4]**: seed-0 means 0.0210 → 0.0203 → 0.0234 | narrow<wide one-sided p median 0.159, range [0.0414, 0.298] (n = 98)
- **P2 ρ_F (hier.) [N-matched, seeds 0–4]**: seed-0 means 0.1907 → 0.2759 → 0.2976 | narrow<wide one-sided p median 1.71e-15, range [3.23e-18, 1] (n = 100)
- **P2b ρ_F (MoM, covered) [N-matched, seeds 0–4]**: seed-0 means 0.2205 → 0.2277 → 0.2694 | narrow<wide one-sided p median 0.113, range [0.0647, 0.293] (n = 44)
- **P5 swap (OLMo medium)**: per-cell ρ_F Spearman = +0.512 (p = 5.3e-08, n = 100); means 0.279 (Phi-4) vs 0.307 (OLMo); σ²_B per-cell Spearman = +0.800
