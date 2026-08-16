# Evidence-coverage audit — is the bundle biased toward the canonical reading?

Script: `prompt_sensitivity/scripts/evidence_coverage_audit.py` (AmbigQA `full`/validation via the pipeline's own loader; target seed 42).

## Funnel, reproduced from the dataset

2,002 validation rows → **1172** with ≥2 interpretations → **609** pass the target-in-evidence filter → first **150** in dataset order analysed (matches `data/run_manifest.json`).

| group | n | mean m0 | mean question chars | all-readings coverage |
|---|---|---|---|---|
| analysed 150 | 150 | 2.97 | 46.4 | 82.9% |
| eligible, passed filter, not analysed | 459 | 2.91 | 47.5 | 79.5% |
| eligible, failed filter | 563 | 3.44 | 46.7 | 23.9% |

## 1. Answer-in-bundle coverage by reading rank

| group | rank 0 | rank 1 | rank 2 | rank 3 | rank 4 |
|---|---|---|---|---|---|
| all eligible | 61.7% (n=1172) | 51.3% (n=1172) | 41.1% (n=613) | 39.6% (n=316) | 39.5% (n=172) |
| analysed 150 | 88.0% (n=150) | 82.7% (n=150) | 66.2% (n=65) | 51.6% (n=31) | 70.0% (n=20) |

Coverage of the TARGET reading in the analysed 150 is 100% by construction (the filter). The quantity union-gold scoring needs at L0 is the coverage of the OTHER readings:

- analysed 150, non-target readings covered: **72.9%** (rank 0 among them: 88.0%)

## 2. Does evidence coverage explain the reading-rank moderator?

Spearman of the per-question Δ_union with the evidence coverage of the non-target readings (and of all readings), plus the coverage by pinned group:

| model | n | ρ(Δ_union, non-target cov) | p | ρ(Δ_union, all-readings cov) | p | non-target cov, pinned=rank0 | pinned=later |
|---|---|---|---|---|---|---|---|
| qwen_2_5_7b | 150 | +0.076 | 0.36 | +0.081 | 0.32 | 64.4% | 78.5% |
| llama_3_1_8b | 150 | +0.080 | 0.33 | +0.101 | 0.22 | 64.4% | 78.5% |
| mistral_7b_v03 | 150 | +0.011 | 0.89 | +0.013 | 0.88 | 64.4% | 78.5% |

**Reading.** Three facts. (a) The bundle IS rank-biased in the eligible pool — coverage falls monotonically with reading rank — so the mechanism is plausible a priori. (b) On the analysed 150 the evidence channel would predict a NEGATIVE per-question association (poorer competitor coverage → harder L0 under union gold → larger Δ_union); the observed associations are ≈0 and slightly positive, so the moderator is NOT explained by evidence coverage. (c) The group-level coverage difference (pinned=rank0 lower than pinned=later) is a selection structure, not a substantive signal: for later-pinned questions the non-target set CONTAINS the well-covered rank-0 reading, for rank0-pinned questions it does not. The paper reports the moderator as a substantive finding and this audit as the check that rules out the evidence-artifact explanation.
