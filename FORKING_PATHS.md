# Forking paths — analysis decisions that changed results (R8)

Every analysis choice below was made during the project and materially changed a
number or a claim. We disclose them so a reader can judge which results are
decision-robust and which are decision-dependent. Source of adjusted statistics:
`data/stats_hygiene.md`.

| # | fork | option taken | option not taken | what changes |
|---|---|---|---|---|
| 1 | gold set | **union of all valid interpretations (primary since R1)** | pinned target interpretation | Δ accuracy for the specificity dial: +.06/+.13/+.12 vs +.22/+.24/+.25. The excess (47–72 %) is a grading lottery, not ability. Target-gold is retained only as the protocol comparison. |
| 2 | ρ_F estimator | **hierarchical beta-binomial EB (primary since R2)** | ANOVA method-of-moments | coverage 45–66 % → 100 %; MoM is undefined exactly on outcome extremes (all-right/all-wrong cells), making missingness outcome-dependent. Point estimates agree where both exist; all headline orderings hold under both. |
| 3 | degenerate cells in correlations | **complete-case + mechanical-coupling null** | impute ρ_F = 0 for degenerate cells | imputation manufactures ρ_F–accuracy correlations of +.26/+.48/+.41 (degenerate cells are mostly all-wrong). Retired 2026-08-06. |
| 4 | axis-1 headline | **graded accuracy F̄** | AUFI "bits" | AUFI ≡ accuracy (ρ = −.9997); the bits framing added a cap-dependent scale, no information. AUFI demoted to appendix presentation. |
| 5 | dispersion representative | **H_sem alone, with reduction proofs** | report S_τ, TVD-consistency, \|A_q\|, variation ratio, Var[FI_out] as convergent evidence | the "family" is one object (identities to 4.4e-16); counting it 5× inflated apparent convergence. FI_out_fixed = log₂m₀ − H_sem is a unit conversion — testing both double-counts one test. |
| 6 | dispersion aggregation | **fi_out_fixed (fixed m₀)** | fi_out_mean (per-paraphrase m varies) | the level effect on FI_out reverses sign vs fi_out_mean (llama Δ = −0.374, p = 2.2e-04; qwen Δ = −0.339, p = 3.1e-04): fi_out_mean confounds dispersion with valid-prompt count. Disclosed because the significant "FI_out rises" result under the old aggregation was reported in the Aug-3 deck. |
| 7 | reliability statistic | **disjoint-paraphrase split-half (200 splits, Spearman–Brown): .38/.52/.57** | k=10 vs k=20 correlation (.81/.92/.95) | the k=20 sample contains the k=10 sample; correlating a statistic with its superset is not reliability. The old numbers are retired. |
| 8 | richness estimator | **observed \|A_q\| with cap disclosure** | Chao1-corrected \|A_q\| | Chao1 dropped: at k = 10 draws with dedup-capped support the correction is dominated by singleton noise; robustness left as a cache-only Good–Turing backlog item. |
| 9 | "evidence dial" | **withdrawn** | evidence-tokens-as-dose regression | ESS_in range restriction (56–66 % of cells censored at ceiling) makes the dose curve unidentifiable in-data; replaced by the two-point manipulated variable + the R6 generator-width dial. |
| 10 | reformulation gain | **not reported as a finding** | "rephrasing recovers .98 of headroom" | the gain statistic is F_max − F̄, whose expectation rises mechanically with k (max of k draws); without a per-k null it is uninterpretable. Superseded by the out-of-sample payoff prediction (ρ_F(k=10) → payoff on disjoint k=20: +.61/+.39/+.70). |
| 11 | multiplicity | **declared 12-test family, Holm + BH (R8)** | per-test unadjusted p | qwen's primary Δ_union survives BH (.045) but not Holm (.15); wording is fixed at "BH-significant 3/3, Holm-robust 2/3". |
| 12 | model count as evidence | **models = correlated measurements; pooled per-question test** | "replicated in 3 models" | per-question deltas correlate ρ = .52–.64 across models — three models are ~1.2–1.5 effective replications, not 3. The pooled single-experiment tests: Δ_union p = 3.3e-05, ΔH_sem p = 6.1e-05. |

Decision-robust across all forks: the model ordering of ρ_F (qwen > mistral >
llama, share and σ²_B, both golds, both estimators, both paraphrase generators);
the flatness of ρ_F under the specificity dial; the direction of the union-gold
accuracy gain; the width-dial double dissociation.

Decision-dependent (and flagged in text): the *size* of the specificity effect
(fork 1), the significance pattern of H_sem's level response (forks 6, 11 —
Holm-robust in llama only), and any per-question use of ρ_F (fork 7).
