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

## Evidence dial — WITHDRAWN as a result (2026-08-03), kept as design justification

The arm crossed specificity with 0 % / 50 % / 100 % of the evidence snippets.
**Dropped from the results flow**: withholding evidence manipulates
*answerability*, not context, and re-creates the knowledge floor that the
uniform-evidence design exists to remove — the same objection that killed the
VoI context arms. Concretely, halving the snippet list **deletes the target
answer outright for 11 of 50 questions** (answer present: 100 % at full,
78 % at half), so the middle point confounds "less context" with "no answer".

On the 39 questions whose answer survives the halving (same questions at all
three points), the pattern is monotone — Δaccuracy +0.03 / +0.15 / +0.19 at
0 / 50 / 100 % evidence — but it is reported as **the reason evidence is held
constant**, not as a second experimental axis. Figure retained as an appendix
ablation (`figures/supervisor_2026-08-03/dial.png`, clean subset).

Closed-book endpoint (Δaccuracy +0.02 on all 50) replicates the v1 floor
finding: 84 % of questions unanswerable without evidence.

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
