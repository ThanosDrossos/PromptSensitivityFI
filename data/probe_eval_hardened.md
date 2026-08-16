# R7 — hardened probe evaluation

Controls at EVERY layer (null distributions, not single draws); layer selection inside
nested CV; text baselines under matched protocols; operating-point analysis on the OOD
holdout. Missing by design (needs generation → cluster): the ask-an-LLM baseline.

## qwen_2_5_7b — in-distribution (grouped by question)

### vagueness

| head | layer | score | null mean ± sd | p |
|---|---|---|---|---|
| massmean | 0.25 | 0.763 | 0.501 ± 0.014 | 0.005 |
| logistic | 0.25 | 0.821 | — | — |
| massmean | 0.5 | 0.802 | 0.500 ± 0.015 | 0.005 |
| logistic | 0.5 | 0.864 | — | — |
| massmean | 0.75 | 0.781 | 0.500 ± 0.022 | 0.005 |
| logistic | 0.75 | 0.831 | — | — |
| massmean | 1 | 0.648 | 0.501 ± 0.021 | 0.005 |
| logistic | 1 | 0.712 | — | — |
| nested_cv | — | 0.864 | — | — |
| logistic+null | 0.5 | 0.864 | 0.501 ± 0.017 | 0.005 |
| baseline_length | — | 0.756 | — | — |
| baseline_tfidf_word | — | 0.672 | — | — |
| baseline_tfidf_char | — | 0.708 | — | — |
| baseline_first_word | — | 0.584 | — | — |

### dispersion

| head | layer | score | null mean ± sd | p |
|---|---|---|---|---|
| massmean | 0.25 | 0.507 | 0.496 ± 0.055 | 0.443 |
| logistic | 0.25 | 0.556 | — | — |
| massmean | 0.5 | 0.476 | 0.505 ± 0.055 | 0.716 |
| logistic | 0.5 | 0.542 | — | — |
| massmean | 0.75 | 0.668 | 0.507 ± 0.052 | 0.005 |
| logistic | 0.75 | 0.688 | — | — |
| massmean | 1 | 0.633 | 0.501 ± 0.054 | 0.005 |
| logistic | 1 | 0.616 | — | — |
| nested_cv | — | 0.688 | — | — |
| logistic+null | 0.75 | 0.688 | 0.520 ± 0.049 | 0.038 |
| baseline_length | — | 0.545 | — | — |
| baseline_tfidf_word | — | 0.504 | — | — |
| baseline_tfidf_char | — | 0.492 | — | — |
| baseline_first_word | — | 0.479 | — | — |

### fragility

| head | layer | score | null mean ± sd | p |
|---|---|---|---|---|
| massmean | 0.25 | 0.567 | 0.491 ± 0.077 | 0.159 |
| logistic | 0.25 | 0.525 | — | — |
| massmean | 0.5 | 0.556 | 0.496 ± 0.078 | 0.204 |
| logistic | 0.5 | 0.516 | — | — |
| massmean | 0.75 | 0.550 | 0.498 ± 0.075 | 0.229 |
| logistic | 0.75 | 0.562 | — | — |
| massmean | 1 | 0.492 | 0.494 ± 0.081 | 0.537 |
| logistic | 1 | 0.502 | — | — |
| nested_cv | — | 0.525 | — | — |
| logistic+null | 0.25 | 0.525 | 0.487 ± 0.067 | 0.308 |
| baseline_length | — | 0.524 | — | — |
| baseline_tfidf_word | — | 0.462 | — | — |
| baseline_tfidf_char | — | 0.472 | — | — |
| baseline_first_word | — | 0.434 | — | — |

### reliability

| head | layer | score | null mean ± sd | p |
|---|---|---|---|---|
| ridge | 0.25 | 0.162 | -0.001 ± 0.080 | 0.038 |
| ridge | 0.5 | 0.237 | 0.020 ± 0.086 | 0.038 |
| ridge | 0.75 | 0.467 | 0.026 ± 0.087 | 0.038 |
| ridge | 1 | 0.465 | -0.008 ± 0.091 | 0.038 |
| nested_cv | — | 0.437 | — | — |

## llama_3_1_8b — in-distribution (grouped by question)

### vagueness

| head | layer | score | null mean ± sd | p |
|---|---|---|---|---|
| massmean | 0.25 | 0.767 | 0.501 ± 0.014 | 0.005 |
| logistic | 0.25 | 0.841 | — | — |
| massmean | 0.5 | 0.836 | 0.502 ± 0.020 | 0.005 |
| logistic | 0.5 | 0.850 | — | — |
| massmean | 0.75 | 0.742 | 0.502 ± 0.021 | 0.005 |
| logistic | 0.75 | 0.797 | — | — |
| massmean | 1 | 0.715 | 0.501 ± 0.021 | 0.005 |
| logistic | 1 | 0.736 | — | — |
| nested_cv | — | 0.850 | — | — |
| logistic+null | 0.5 | 0.850 | 0.500 ± 0.019 | 0.005 |
| baseline_length | — | 0.756 | — | — |
| baseline_tfidf_word | — | 0.672 | — | — |
| baseline_tfidf_char | — | 0.708 | — | — |
| baseline_first_word | — | 0.584 | — | — |

### dispersion

| head | layer | score | null mean ± sd | p |
|---|---|---|---|---|
| massmean | 0.25 | 0.486 | 0.515 ± 0.043 | 0.716 |
| logistic | 0.25 | 0.599 | — | — |
| massmean | 0.5 | 0.818 | 0.552 ± 0.043 | 0.005 |
| logistic | 0.5 | 0.770 | — | — |
| massmean | 0.75 | 0.841 | 0.521 ± 0.045 | 0.005 |
| logistic | 0.75 | 0.790 | — | — |
| massmean | 1 | 0.847 | 0.519 ± 0.046 | 0.005 |
| logistic | 1 | 0.765 | — | — |
| nested_cv | — | 0.772 | — | — |
| logistic+null | 1 | 0.765 | 0.523 ± 0.036 | 0.038 |
| baseline_length | — | 0.575 | — | — |
| baseline_tfidf_word | — | 0.603 | — | — |
| baseline_tfidf_char | — | 0.607 | — | — |
| baseline_first_word | — | 0.536 | — | — |

### fragility

| head | layer | score | null mean ± sd | p |
|---|---|---|---|---|
| massmean | 0.25 | 0.436 | 0.491 ± 0.094 | 0.711 |
| logistic | 0.25 | 0.413 | — | — |
| massmean | 0.5 | 0.623 | 0.503 ± 0.091 | 0.090 |
| logistic | 0.5 | 0.618 | — | — |
| massmean | 0.75 | 0.579 | 0.496 ± 0.092 | 0.209 |
| logistic | 0.75 | 0.554 | — | — |
| massmean | 1 | 0.554 | 0.495 ± 0.090 | 0.264 |
| logistic | 1 | 0.579 | — | — |
| nested_cv | — | 0.594 | — | — |
| logistic+null | 0.5 | 0.618 | 0.472 ± 0.090 | 0.115 |
| baseline_length | — | 0.538 | — | — |
| baseline_tfidf_word | — | 0.449 | — | — |
| baseline_tfidf_char | — | 0.402 | — | — |
| baseline_first_word | — | 0.458 | — | — |

### reliability

| head | layer | score | null mean ± sd | p |
|---|---|---|---|---|
| ridge | 0.25 | 0.072 | 0.033 ± 0.094 | 0.423 |
| ridge | 0.5 | 0.514 | 0.046 ± 0.069 | 0.038 |
| ridge | 0.75 | 0.520 | 0.034 ± 0.077 | 0.038 |
| ridge | 1 | 0.513 | 0.007 ± 0.070 | 0.038 |
| nested_cv | — | 0.505 | — | — |

## mistral_7b_v03 — in-distribution (grouped by question)

### vagueness

| head | layer | score | null mean ± sd | p |
|---|---|---|---|---|
| massmean | 0.25 | 0.737 | 0.501 ± 0.014 | 0.005 |
| logistic | 0.25 | 0.809 | — | — |
| massmean | 0.5 | 0.830 | 0.501 ± 0.021 | 0.005 |
| logistic | 0.5 | 0.872 | — | — |
| massmean | 0.75 | 0.727 | 0.502 ± 0.021 | 0.005 |
| logistic | 0.75 | 0.828 | — | — |
| massmean | 1 | 0.665 | 0.502 ± 0.020 | 0.005 |
| logistic | 1 | 0.745 | — | — |
| nested_cv | — | 0.872 | — | — |
| logistic+null | 0.5 | 0.872 | 0.501 ± 0.019 | 0.005 |
| baseline_length | — | 0.756 | — | — |
| baseline_tfidf_word | — | 0.672 | — | — |
| baseline_tfidf_char | — | 0.708 | — | — |
| baseline_first_word | — | 0.584 | — | — |

### dispersion

| head | layer | score | null mean ± sd | p |
|---|---|---|---|---|
| massmean | 0.25 | 0.407 | 0.495 ± 0.051 | 0.960 |
| logistic | 0.25 | 0.477 | — | — |
| massmean | 0.5 | 0.717 | 0.502 ± 0.054 | 0.005 |
| logistic | 0.5 | 0.686 | — | — |
| massmean | 0.75 | 0.787 | 0.496 ± 0.050 | 0.005 |
| logistic | 0.75 | 0.748 | — | — |
| massmean | 1 | 0.772 | 0.497 ± 0.052 | 0.005 |
| logistic | 1 | 0.731 | — | — |
| nested_cv | — | 0.748 | — | — |
| logistic+null | 0.75 | 0.748 | 0.501 ± 0.055 | 0.038 |
| baseline_length | — | 0.528 | — | — |
| baseline_tfidf_word | — | 0.582 | — | — |
| baseline_tfidf_char | — | 0.548 | — | — |
| baseline_first_word | — | 0.524 | — | — |

### fragility

| head | layer | score | null mean ± sd | p |
|---|---|---|---|---|
| massmean | 0.25 | 0.450 | 0.482 ± 0.074 | 0.642 |
| logistic | 0.25 | 0.545 | — | — |
| massmean | 0.5 | 0.495 | 0.488 ± 0.078 | 0.473 |
| logistic | 0.5 | 0.466 | — | — |
| massmean | 0.75 | 0.518 | 0.486 ± 0.074 | 0.378 |
| logistic | 0.75 | 0.441 | — | — |
| massmean | 1 | 0.549 | 0.487 ± 0.075 | 0.224 |
| logistic | 1 | 0.464 | — | — |
| nested_cv | — | 0.467 | — | — |
| logistic+null | 1 | 0.464 | 0.462 ± 0.063 | 0.462 |
| baseline_length | — | 0.532 | — | — |
| baseline_tfidf_word | — | 0.484 | — | — |
| baseline_tfidf_char | — | 0.464 | — | — |
| baseline_first_word | — | 0.446 | — | — |

### reliability

| head | layer | score | null mean ± sd | p |
|---|---|---|---|---|
| ridge | 0.25 | 0.182 | 0.001 ± 0.084 | 0.077 |
| ridge | 0.5 | 0.364 | 0.058 ± 0.068 | 0.038 |
| ridge | 0.75 | 0.362 | 0.014 ± 0.070 | 0.038 |
| ridge | 1 | 0.378 | -0.021 ± 0.084 | 0.038 |
| nested_cv | — | 0.342 | — | — |

## OOD — frozen heads on the annotator-labelled holdout

| model | AUROC head | PR-AUC (prev.) | length | TF-IDF w/c (frozen) | first-word (frozen) |
|---|---|---|---|---|---|
| qwen_2_5_7b | 0.667 | 0.694 (0.55) | 0.543 | 0.544 / 0.562 | 0.571 |
| llama_3_1_8b | 0.655 | 0.686 (0.55) | 0.543 | 0.544 / 0.562 | 0.571 |
| mistral_7b_v03 | 0.670 | 0.703 (0.55) | 0.543 | 0.544 / 0.562 | 0.571 |

### Operating point at the shipped threshold (0.65)

| model | flagged | precision | recall | TP/FP/FN/TN |
|---|---|---|---|---|
| qwen_2_5_7b | 69.9% | 0.629 | 0.796 | 814/481/208/349 |
| llama_3_1_8b | 65.1% | 0.627 | 0.739 | 755/450/267/380 |
| mistral_7b_v03 | 71.6% | 0.622 | 0.807 | 825/501/197/329 |

**Claim discipline:** the head's OOD advantage is over *frozen* baselines (zero-shot transfer / label efficiency). With in-domain labels, bag-of-words matches it — never claim detection quality beyond that. The fragility head's null table above is the honest version of the axis-2 probe result.