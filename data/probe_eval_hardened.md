# R7 — hardened probe evaluation

Controls at EVERY layer (null distributions, not single draws); layer selection inside
nested CV; text baselines under matched protocols; operating-point analysis on the OOD
holdout. Missing by design (needs generation → cluster): the ask-an-LLM baseline.

## qwen_2_5_7b — in-distribution (grouped by question)

### vagueness

| head | layer | score | null mean ± sd | p |
|---|---|---|---|---|
| massmean | 0.25 | 0.763 | 0.501 ± 0.014 | 0.005 |
| logistic | 0.25 | 0.836 | — | — |
| massmean | 0.5 | 0.804 | 0.500 ± 0.015 | 0.005 |
| logistic | 0.5 | 0.874 | — | — |
| massmean | 0.75 | 0.790 | 0.501 ± 0.021 | 0.005 |
| logistic | 0.75 | 0.848 | — | — |
| massmean | 1 | 0.652 | 0.501 ± 0.020 | 0.005 |
| logistic | 1 | 0.723 | — | — |
| nested_cv | — | 0.874 | — | — |
| logistic+null | 0.5 | 0.874 | 0.501 ± 0.017 | 0.005 |
| baseline_length | — | 0.756 | — | — |
| baseline_tfidf_word | — | 0.677 | — | — |
| baseline_tfidf_char | — | 0.707 | — | — |
| baseline_first_word | — | 0.584 | — | — |

### dispersion

| head | layer | score | null mean ± sd | p |
|---|---|---|---|---|
| massmean | 0.25 | 0.527 | 0.497 ± 0.058 | 0.328 |
| logistic | 0.25 | 0.537 | — | — |
| massmean | 0.5 | 0.488 | 0.507 ± 0.054 | 0.672 |
| logistic | 0.5 | 0.534 | — | — |
| massmean | 0.75 | 0.668 | 0.503 ± 0.052 | 0.005 |
| logistic | 0.75 | 0.671 | — | — |
| massmean | 1 | 0.624 | 0.500 ± 0.053 | 0.005 |
| logistic | 1 | 0.618 | — | — |
| nested_cv | — | 0.671 | — | — |
| logistic+null | 0.75 | 0.671 | 0.516 ± 0.047 | 0.038 |
| baseline_length | — | 0.545 | — | — |
| baseline_tfidf_word | — | 0.424 | — | — |
| baseline_tfidf_char | — | 0.421 | — | — |
| baseline_first_word | — | 0.471 | — | — |

### fragility

| head | layer | score | null mean ± sd | p |
|---|---|---|---|---|
| massmean | 0.25 | 0.555 | 0.492 ± 0.074 | 0.214 |
| logistic | 0.25 | 0.520 | — | — |
| massmean | 0.5 | 0.494 | 0.495 ± 0.072 | 0.542 |
| logistic | 0.5 | 0.508 | — | — |
| massmean | 0.75 | 0.467 | 0.499 ± 0.078 | 0.682 |
| logistic | 0.75 | 0.529 | — | — |
| massmean | 1 | 0.391 | 0.497 ± 0.080 | 0.905 |
| logistic | 1 | 0.497 | — | — |
| nested_cv | — | 0.500 | — | — |
| logistic+null | 0.25 | 0.520 | 0.497 ± 0.070 | 0.423 |
| baseline_length | — | 0.524 | — | — |
| baseline_tfidf_word | — | 0.421 | — | — |
| baseline_tfidf_char | — | 0.460 | — | — |
| baseline_first_word | — | 0.442 | — | — |

### reliability

| head | layer | score | null mean ± sd | p |
|---|---|---|---|---|
| ridge | 0.25 | 0.073 | 0.002 ± 0.073 | 0.192 |
| ridge | 0.5 | 0.185 | 0.033 ± 0.073 | 0.038 |
| ridge | 0.75 | 0.479 | 0.029 ± 0.066 | 0.038 |
| ridge | 1 | 0.476 | -0.007 ± 0.075 | 0.038 |
| nested_cv | — | 0.440 | — | — |

## llama_3_1_8b — in-distribution (grouped by question)

### vagueness

| head | layer | score | null mean ± sd | p |
|---|---|---|---|---|
| massmean | 0.25 | 0.768 | 0.501 ± 0.014 | 0.005 |
| logistic | 0.25 | 0.841 | — | — |
| massmean | 0.5 | 0.837 | 0.502 ± 0.020 | 0.005 |
| logistic | 0.5 | 0.873 | — | — |
| massmean | 0.75 | 0.743 | 0.502 ± 0.021 | 0.005 |
| logistic | 0.75 | 0.814 | — | — |
| massmean | 1 | 0.720 | 0.501 ± 0.021 | 0.005 |
| logistic | 1 | 0.750 | — | — |
| nested_cv | — | 0.873 | — | — |
| logistic+null | 0.5 | 0.873 | 0.500 ± 0.018 | 0.005 |
| baseline_length | — | 0.756 | — | — |
| baseline_tfidf_word | — | 0.677 | — | — |
| baseline_tfidf_char | — | 0.707 | — | — |
| baseline_first_word | — | 0.584 | — | — |

### dispersion

| head | layer | score | null mean ± sd | p |
|---|---|---|---|---|
| massmean | 0.25 | 0.572 | 0.516 ± 0.043 | 0.100 |
| logistic | 0.25 | 0.640 | — | — |
| massmean | 0.5 | 0.811 | 0.551 ± 0.040 | 0.005 |
| logistic | 0.5 | 0.785 | — | — |
| massmean | 0.75 | 0.828 | 0.520 ± 0.042 | 0.005 |
| logistic | 0.75 | 0.795 | — | — |
| massmean | 1 | 0.834 | 0.518 ± 0.043 | 0.005 |
| logistic | 1 | 0.758 | — | — |
| nested_cv | — | 0.758 | — | — |
| logistic+null | 1 | 0.758 | 0.526 ± 0.041 | 0.038 |
| baseline_length | — | 0.575 | — | — |
| baseline_tfidf_word | — | 0.616 | — | — |
| baseline_tfidf_char | — | 0.618 | — | — |
| baseline_first_word | — | 0.533 | — | — |

### fragility

| head | layer | score | null mean ± sd | p |
|---|---|---|---|---|
| massmean | 0.25 | 0.479 | 0.491 ± 0.089 | 0.572 |
| logistic | 0.25 | 0.495 | — | — |
| massmean | 0.5 | 0.643 | 0.501 ± 0.093 | 0.045 |
| logistic | 0.5 | 0.631 | — | — |
| massmean | 0.75 | 0.563 | 0.497 ± 0.089 | 0.234 |
| logistic | 0.75 | 0.537 | — | — |
| massmean | 1 | 0.558 | 0.494 ± 0.087 | 0.234 |
| logistic | 1 | 0.593 | — | — |
| nested_cv | — | 0.580 | — | — |
| logistic+null | 0.5 | 0.631 | 0.456 ± 0.063 | 0.038 |
| baseline_length | — | 0.538 | — | — |
| baseline_tfidf_word | — | 0.495 | — | — |
| baseline_tfidf_char | — | 0.442 | — | — |
| baseline_first_word | — | 0.449 | — | — |

### reliability

| head | layer | score | null mean ± sd | p |
|---|---|---|---|---|
| ridge | 0.25 | 0.040 | 0.024 ± 0.101 | 0.423 |
| ridge | 0.5 | 0.511 | 0.053 ± 0.065 | 0.038 |
| ridge | 0.75 | 0.509 | 0.026 ± 0.075 | 0.038 |
| ridge | 1 | 0.503 | 0.006 ± 0.075 | 0.038 |
| nested_cv | — | 0.505 | — | — |

## mistral_7b_v03 — in-distribution (grouped by question)

### vagueness

| head | layer | score | null mean ± sd | p |
|---|---|---|---|---|
| massmean | 0.25 | 0.734 | 0.501 ± 0.014 | 0.005 |
| logistic | 0.25 | 0.813 | — | — |
| massmean | 0.5 | 0.834 | 0.501 ± 0.021 | 0.005 |
| logistic | 0.5 | 0.873 | — | — |
| massmean | 0.75 | 0.721 | 0.502 ± 0.021 | 0.005 |
| logistic | 0.75 | 0.823 | — | — |
| massmean | 1 | 0.653 | 0.502 ± 0.020 | 0.005 |
| logistic | 1 | 0.722 | — | — |
| nested_cv | — | 0.873 | — | — |
| logistic+null | 0.5 | 0.873 | 0.501 ± 0.019 | 0.005 |
| baseline_length | — | 0.756 | — | — |
| baseline_tfidf_word | — | 0.677 | — | — |
| baseline_tfidf_char | — | 0.707 | — | — |
| baseline_first_word | — | 0.584 | — | — |

### dispersion

| head | layer | score | null mean ± sd | p |
|---|---|---|---|---|
| massmean | 0.25 | 0.465 | 0.495 ± 0.052 | 0.726 |
| logistic | 0.25 | 0.533 | — | — |
| massmean | 0.5 | 0.729 | 0.499 ± 0.055 | 0.005 |
| logistic | 0.5 | 0.669 | — | — |
| massmean | 0.75 | 0.792 | 0.493 ± 0.049 | 0.005 |
| logistic | 0.75 | 0.765 | — | — |
| massmean | 1 | 0.783 | 0.494 ± 0.048 | 0.005 |
| logistic | 1 | 0.753 | — | — |
| nested_cv | — | 0.765 | — | — |
| logistic+null | 0.75 | 0.765 | 0.497 ± 0.044 | 0.038 |
| baseline_length | — | 0.528 | — | — |
| baseline_tfidf_word | — | 0.594 | — | — |
| baseline_tfidf_char | — | 0.556 | — | — |
| baseline_first_word | — | 0.512 | — | — |

### fragility

| head | layer | score | null mean ± sd | p |
|---|---|---|---|---|
| massmean | 0.25 | 0.520 | 0.480 ± 0.079 | 0.353 |
| logistic | 0.25 | 0.527 | — | — |
| massmean | 0.5 | 0.567 | 0.488 ± 0.078 | 0.169 |
| logistic | 0.5 | 0.557 | — | — |
| massmean | 0.75 | 0.539 | 0.485 ± 0.075 | 0.249 |
| logistic | 0.75 | 0.450 | — | — |
| massmean | 1 | 0.556 | 0.485 ± 0.079 | 0.184 |
| logistic | 1 | 0.468 | — | — |
| nested_cv | — | 0.510 | — | — |
| logistic+null | 1 | 0.468 | 0.476 ± 0.066 | 0.500 |
| baseline_length | — | 0.532 | — | — |
| baseline_tfidf_word | — | 0.526 | — | — |
| baseline_tfidf_char | — | 0.511 | — | — |
| baseline_first_word | — | 0.509 | — | — |

### reliability

| head | layer | score | null mean ± sd | p |
|---|---|---|---|---|
| ridge | 0.25 | 0.139 | 0.008 ± 0.082 | 0.115 |
| ridge | 0.5 | 0.371 | 0.067 ± 0.071 | 0.038 |
| ridge | 0.75 | 0.363 | 0.017 ± 0.072 | 0.038 |
| ridge | 1 | 0.365 | -0.012 ± 0.066 | 0.038 |
| nested_cv | — | 0.350 | — | — |

## OOD — frozen heads on the annotator-labelled holdout

| model | AUROC head | PR-AUC (prev.) | length | TF-IDF w/c (frozen) | first-word (frozen) |
|---|---|---|---|---|---|
| qwen_2_5_7b | 0.678 | 0.725 (0.59) | 0.545 | 0.565 / 0.587 | 0.564 |
| llama_3_1_8b | 0.670 | 0.721 (0.59) | 0.545 | 0.565 / 0.587 | 0.564 |
| mistral_7b_v03 | 0.678 | 0.731 (0.59) | 0.545 | 0.565 / 0.587 | 0.564 |

### Operating point at the shipped threshold (0.65)

| model | flagged | precision | recall | TP/FP/FN/TN |
|---|---|---|---|---|
| qwen_2_5_7b | 71.8% | 0.666 | 0.817 | 957/481/215/349 |
| llama_3_1_8b | 67.3% | 0.666 | 0.766 | 898/450/274/380 |
| mistral_7b_v03 | 73.4% | 0.659 | 0.827 | 969/501/203/329 |

**Claim discipline:** the head's OOD advantage is over *frozen* baselines (zero-shot transfer / label efficiency). With in-domain labels, bag-of-words matches it — never claim detection quality beyond that. The fragility head's null table above is the honest version of the axis-2 probe result.