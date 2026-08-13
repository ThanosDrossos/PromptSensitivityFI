# R4b — Axis 1 reported honestly

## The convention-free headline

The AUFI drop is, underneath, a censoring statement that needs no clamp value:

| model | thresholds unreachable at L0 | at L1 |
|---|---|---|
| qwen_2_5_7b | 55.7% | 32.3% |
| llama_3_1_8b | 66.4% | 39.3% |
| mistral_7b_v03 | 61.9% | 36.2% |

> **Say this, not "−0.86 bits":** the fraction of quality thresholds no paraphrase
> reaches falls from ~56–67 % (ambiguous) to ~32–40 % (disambiguated).

## Cap-sensitivity of ΔAUFI (why the bits figure must carry its convention)

| model | log2(5+1)=2.585 | log2(N)=3.322 | log2(N+1)=3.459 (v3 convention) | log2(20+1)=4.392 |
|---|---|---|---|---|
| qwen_2_5_7b | -0.569 | -0.742 | -0.775 | -0.997 |
| llama_3_1_8b | -0.604 | -0.800 | -0.838 | -1.097 |
| mistral_7b_v03 | -0.628 | -0.818 | -0.854 | -1.098 |

Sign robust under every convention; magnitude is the convention. Never print the delta
without the cap.

## Stepped shape ("islands of function") — now with a null

Statistic: largest adjacent gap in the sorted per-paraphrase F values. Null: one true
rate per cell, F_i ~ Binomial(k, p̄)/k (pure decoding noise), 2000 sims/cell, BH at 5 %:

| model | cells | cells with significant islands | Spearman(max-gap, ρ_F) |
|---|---|---|---|
| qwen_2_5_7b | 299 | 23.1% | +0.926 |
| llama_3_1_8b | 299 | 4.0% | +0.832 |
| mistral_7b_v03 | 299 | 15.4% | +0.930 |

**Reading.** "Islands" beyond sampling noise *is* between-paraphrase overdispersion —
the same null ρ_F subtracts (the max-gap statistic tracks ρ_F, last column). The Hazen
stepped-shape analogue is therefore not a separate finding: where it holds, it is axis 2
restated. The binary curve's steps (76 % of cells: one step at k=0→0.05; 24 %: flat) carry
no shape information and are retired from the deliverable.
