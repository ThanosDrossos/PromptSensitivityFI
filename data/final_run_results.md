# Final-run results (generated 2026-08-02, updated as arms land)

## Vagueness holdout — frozen heads on annotator-labeled unseen questions ✅

Frozen FeedbackModel bundles applied to `vagueness_holdout_<model>.parquet`
(one training-format prompt per AmbigQA question; the 830 annotator-labeled
NON-ambiguous rows vs ambiguous ones; all 150 v3 training questions excluded
→ n = 1,852 questions per model, 1,022 ambiguous / 830 specific):

| model | AUROC (frozen head) | length baseline |
|---|---|---|
| qwen_2_5_7b | **0.667** | 0.457 |
| llama_3_1_8b | **0.655** | 0.457 |
| mistral_7b_v03 | **0.670** | 0.457 |

Reading: genuine out-of-distribution transfer — the label here comes from a
DIFFERENT mechanism (annotator judgment "this NQ question has multiple
interpretations") than training (L0-vs-L1 rewrite pairs), on questions the
heads never saw. The signal drops from in-distribution .849–.874 to ~.66 —
an honest OOD gap to report — while the length baseline sits BELOW chance
(0.457), so the heads' signal is semantic, not a length artifact.
Results parquet: `data/vagueness_holdout_results.parquet`.

## Evidence dial — specificity × evidence interaction (qwen, same 50 q) ✅

f = 0.0 / 0.5 fresh runs; f = 1.0 sliced from the v3 parquet (100 matched
cells). All cells 100/100, 0 failed.

| evidence | acc L0 | acc L1 | Δacc | AUFI L0 | AUFI L1 | ΔFI (bits) | H_sem L0→L1 |
|---|---|---|---|---|---|---|---|
| 0.0 (closed book) | 0.106 | 0.126 | +0.02 | 2.94 | 2.90 | −0.05 | 0.59→0.58 |
| 0.5 | 0.311 | 0.409 | +0.10 | 2.22 | 1.91 | −0.31 | 0.28→0.25 |
| 1.0 | 0.301 | 0.536 | +0.24 | 2.29 | 1.47 | −0.83 | 0.35→0.22 |

Reading — the dial's payoff REQUIRES evidence and grows with it:
- **Closed book:** floor everywhere; disambiguation buys ~nothing (the model
  lacks the knowledge either way) — replicates the v1 floor finding.
- **L0 saturates at ~0.30 from half evidence onward** (ambiguity, not
  evidence, is the binding constraint on the ambiguous question), while
  **L1 keeps climbing with evidence** (0.13 → 0.41 → 0.54).
- In bits: paying 1.59 bits of question specificity converts to −0.05 /
  −0.31 / −0.83 bits of formulation-luck reduction at evidence 0 / ½ / full.
  Question-side information is only convertible when the answer is
  extractable — evidence raises the ceiling, specificity determines how much
  of it you reach.

## POSIX — all three models ✅ (completes the axis-3 literature row)

100/100 cells per model, `posix_psi` non-null everywhere, 0 failed (the first
attempt simply ran out of chain windows; one resubmit finished it). Same 50
questions × 2 levels as the other arms.

Within-arm Spearman of POSIX ψ (Chatterjee et al. 2024) against the stack:

| model | vs H_sem | vs S_τ | vs TVD-sens | vs accuracy | vs ρ_F | ψ L0→L1 |
|---|---|---|---|---|---|---|
| qwen_2_5_7b | +0.63 | +0.59 | **+0.69** | −0.26 | +0.24 | 0.61→0.35 |
| llama_3_1_8b | +0.43 | +0.35 | **+0.57** | −0.33 | +0.30 | 0.21→0.18 |
| mistral_7b_v03 | +0.61 | +0.53 | **+0.62** | −0.40 | +0.32 | 0.21→0.19 |

Reading — the "one construct, many costumes" claim now rests on three models,
not one: POSIX's dominant loadings are on the **output-dispersion** family
(TVD .57–.69, H_sem .43–.63, S_τ .35–.59), while its correlation with the
formulation-sensitivity axis stays weak (ρ_F .24–.32). So a published
prompt-sensitivity index measures dispersion, NOT phrasing-induced
variability — **ρ_F remains alone on axis 2**. ψ also falls with
disambiguation in all three models, the same direction as H_sem.
