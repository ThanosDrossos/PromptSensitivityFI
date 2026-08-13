# Probe-head redundancy audit (supervisor feedback, 2026-08-11)

Question: the measurement model has three axes, the probe suite has four heads. Is the fourth (underspecification / FI_spec) redundant?

All heads: same features (hidden state, last prompt token, layer 0.5), same question-grouped 5-fold CV, logistic C = 0.01.

## T1 - Is it the same direction in representation space?

|cosine| between the underspecification weight vector and each axis head:

| model | vs competence | vs dispersion | vs sensitivity |
|---|---|---|---|
| qwen_2_5_7b | 0.293 | 0.082 | 0.191 |
| llama_3_1_8b | 0.231 | 0.006 | 0.043 |
| mistral_7b_v03 | 0.193 | 0.044 | 0.079 |

## T2/T3 - Can the three axis heads replace it?

| model | direct head (AUROC) | reconstructed from 3 axis heads | from TRUE axis values | best single axis head | length only |
|---|---|---|---|---|---|
| qwen_2_5_7b | **0.860** | 0.626 | 0.636 | 0.606 | 0.756 |
| llama_3_1_8b | **0.866** | 0.686 | 0.700 | 0.663 | 0.756 |
| mistral_7b_v03 | **0.872** | 0.658 | 0.652 | 0.648 | 0.756 |

| model | sub: competence | sub: dispersion | sub: sensitivity |
|---|---|---|---|
| qwen_2_5_7b | 0.606 | 0.572 | 0.578 |
| llama_3_1_8b | 0.628 | 0.663 | 0.580 |
| mistral_7b_v03 | 0.648 | 0.576 | 0.636 |

Mean advantage of the direct head over the best reconstruction: **+0.209 AUROC**.
