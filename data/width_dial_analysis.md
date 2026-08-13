# R6 — generator-width dial (positive control for ρ_F)

Preregistered predictions P0–P5 are in the module docstring and the runbook; they were
fixed before any arm data existed.

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

- **P1 σ²_B** (increases): 0.0256 → 0.0258 → 0.0301 | narrow<wide one-sided p = 0.0542 | Friedman p = 0.0991 | monotone cells 70% (n = 98)
- **P2 ρ_F (hier., ⚠ see caveat)** (increases): 0.6770 → 0.5034 → 0.5158 | narrow<wide one-sided p = 1 | Friedman p = 6.05e-23 | monotone cells 4% (n = 100)
- **P2b ρ_F (MoM, covered cells)** (increases): 0.3563 → 0.4631 → 0.5193 | narrow<wide one-sided p = 0.00351 | Friedman p = 0.067 | monotone cells 23% (n = 22)
- **P3 accuracy** (little change): 0.3994 → 0.4184 → 0.4109 | narrow<wide one-sided p = 0.262 | Friedman p = 0.681 | monotone cells 55% (n = 100)
- **P4 H_sem** (little change): 0.3125 → 0.2868 → 0.2593 | narrow<wide one-sided p = 0.966 | Friedman p = 0.469 | monotone cells 26% (n = 100)
- **P1 σ²_B [N-matched]**: 0.0256 → 0.0245 → 0.0348 | narrow<wide one-sided p = 0.0223 (n = 98)
- **P2 ρ_F (hier.) [N-matched]**: 0.6770 → 0.6453 → 0.7321 | narrow<wide one-sided p = 2.76e-08 (n = 100)
- **P5 swap (OLMo medium)**: per-cell ρ_F Spearman = +0.413 (p = 1.9e-05, n = 100); means 0.503 (Phi-4) vs 0.360 (OLMo); σ²_B per-cell Spearman = +0.679

## llama_3_1_8b

universe sizes |U| per arm (unequal-N caveat): medium: mean 10.0, median 10; narrow: mean 6.7, median 7; wide: mean 10.0, median 10

- **P1 σ²_B** (increases): 0.0108 → 0.0140 → 0.0153 | narrow<wide one-sided p = 0.0212 | Friedman p = 0.426 | monotone cells 51% (n = 98)
- **P2 ρ_F (hier., ⚠ see caveat)** (increases): 0.0807 → 0.0851 → 0.1182 | narrow<wide one-sided p = 1.46e-08 | Friedman p = 2.02e-13 | monotone cells 20% (n = 100)
- **P2b ρ_F (MoM, covered cells)** (increases): 0.1130 → 0.1348 → 0.1382 | narrow<wide one-sided p = 0.0785 | Friedman p = 0.519 | monotone cells 30% (n = 54)
- **P3 accuracy** (little change): 0.3104 → 0.2969 → 0.3029 | narrow<wide one-sided p = 0.686 | Friedman p = 0.525 | monotone cells 39% (n = 100)
- **P4 H_sem** (little change): 1.1864 → 1.0936 → 1.1388 | narrow<wide one-sided p = 0.797 | Friedman p = 0.572 | monotone cells 19% (n = 100)
- **P1 σ²_B [N-matched]**: 0.0108 → 0.0121 → 0.0151 | narrow<wide one-sided p = 0.0508 (n = 98)
- **P2 ρ_F (hier.) [N-matched]**: 0.0807 → 0.0877 → 0.0724 | narrow<wide one-sided p = 0.986 (n = 100)
- **P5 swap (OLMo medium)**: per-cell ρ_F Spearman = +0.273 (p = 0.0061, n = 100); means 0.085 (Phi-4) vs 0.129 (OLMo); σ²_B per-cell Spearman = +0.643

## mistral_7b_v03

universe sizes |U| per arm (unequal-N caveat): medium: mean 10.0, median 10; narrow: mean 6.7, median 7; wide: mean 10.0, median 10

- **P1 σ²_B** (increases): 0.0210 → 0.0205 → 0.0249 | narrow<wide one-sided p = 0.0596 | Friedman p = 0.253 | monotone cells 61% (n = 98)
- **P2 ρ_F (hier., ⚠ see caveat)** (increases): 0.1907 → 0.2791 → 0.2373 | narrow<wide one-sided p = 6.23e-07 | Friedman p = 3.72e-17 | monotone cells 14% (n = 100)
- **P2b ρ_F (MoM, covered cells)** (increases): 0.2110 → 0.2303 → 0.2761 | narrow<wide one-sided p = 0.0506 | Friedman p = 0.675 | monotone cells 28% (n = 46)
- **P3 accuracy** (little change): 0.3655 → 0.3798 → 0.3736 | narrow<wide one-sided p = 0.3 | Friedman p = 0.537 | monotone cells 55% (n = 100)
- **P4 H_sem** (little change): 0.8977 → 0.7698 → 0.8218 | narrow<wide one-sided p = 0.792 | Friedman p = 0.622 | monotone cells 32% (n = 100)
- **P1 σ²_B [N-matched]**: 0.0210 → 0.0199 → 0.0240 | narrow<wide one-sided p = 0.214 (n = 98)
- **P2 ρ_F (hier.) [N-matched]**: 0.1907 → 0.2735 → 0.2270 | narrow<wide one-sided p = 7.03e-06 (n = 100)
- **P5 swap (OLMo medium)**: per-cell ρ_F Spearman = +0.512 (p = 5.3e-08, n = 100); means 0.279 (Phi-4) vs 0.307 (OLMo); σ²_B per-cell Spearman = +0.800
