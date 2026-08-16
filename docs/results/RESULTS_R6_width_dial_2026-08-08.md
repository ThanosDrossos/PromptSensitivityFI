# R6 results — the generator-width dial + paraphraser-swap ablation (2026-08-08)

Run: 4 arms × 50 q × 2 levels × 3 models on bwUniCluster (900 new eval cells + 300 universes; medium
arm = existing v3 data). Preregistered P0–P5 fixed 2026-08-07 before any arm data existed
(`RUNBOOK_R6_width_dial_cluster.md`). Full tables: `data/width_dial_analysis.md`.

**Data quality:** all 9 eval parquets complete (100/100 cells, no duplicates, no nulls); script
arithmetic hand-verified (σ²_B means and Wilcoxon p reproduce exactly); the mid-run SQLite lock storm
cost only wall-time, no data (post-mortem in the runbook). **No cluster re-run is needed.**

---

## Verdict per preregistered prediction

| pred | verdict | evidence |
|---|---|---|
| **P0** manipulation check | **PASS, with disclosed compression** | realized width ordered 4.6 < 8.4 < 9.3 tokens (narrow/medium/wide), 66 % of cells fully ordered, Wilcoxon narrow<wide p = 4e-18. *But* the NLI gate censored the wide arm hard (64 % NLI rejections) — realized medium→wide gap is small, so the dial's usable range is mostly narrow→medium |
| **P1** σ²_B ↑ with width | **directional 3/3, marginal significance** | .0256→.0258→.0301 (qwen, p=.054) · .0108→.0140→.0153 (llama, p=.021) · .0210→.0205→.0249 (mistral, p=.060). Wide > narrow in every model; seed-stable direction |
| **P2** ρ_F ↑ with width | **PASS as a graded positive control** (see estimator note) | **MoM on covered cells: monotone in all three models** — .356→.463→.519 (qwen, p=.0035) · .113→.135→.138 (llama, p=.079) · .211→.230→.276 (mistral, p=.051). N-matched MoM seed-stable for qwen (p=.001–.013 over 5 seeds), marginal mistral, n.s. llama. Full-N narrow cells (no N argument possible) confirm qwen's increase (.415<.532<.573) |
| **P3** accuracy ~flat | **PASS, clean, 3/3** | Friedman p = .68/.53/.54; largest arm-mean gap ≤ .02 |
| **P4** H_sem ~flat | **PASS, clean, 3/3** | Friedman p = .47/.57/.62 |
| **P5** swap ablation | **PASS** | per-cell ρ_F Spearman +.41/+.27/+.51 (all p ≤ .006); σ²_B per-cell +.68/+.64/+.80; **model ordering preserved under OLMo: qwen .36 > mistral .31 > llama .13** |

## The headline: the double dissociation is now complete on both sides

| dial | competence | ρ_F / σ²_B | H_sem |
|---|---|---|---|
| **specificity** (L0→L1) | **+.06–.13, BH-sig 3/3** | flat (n.s. 3/3, both golds) | weak (Holm-robust 1/3) |
| **generator width** (narrow→wide) | flat (Friedman .53–.68) | **↑ directionally 3/3; sig. in qwen** | flat (Friedman .47–.62) |

Each axis responds to its own dial and only its own dial. The width side is *graded* rather than
uniform — and the grading itself is coherent: **the dial's effect size tracks how phrasing-sensitive
the model is** (qwen, ρ_F ≈ .5, responds clearly; llama, ρ_F ≈ .1, has almost no sensitivity for the
dial to amplify). State it exactly that way; do not claim uniform significance.

## The swap ablation (first paraphraser-swap ablation in this literature)

Same instructions + temperature, different-family generator (OLMo-2-13B, judge swapped with it):
- per-cell ρ_F agreement +.41/+.27/+.51 — **at or near the measurement-reliability ceiling**
  (split-half reliability of each measurement is ~.35–.58, so the maximum observable correlation
  between two independent measurements is about that; disattenuated agreement is therefore high);
- σ²_B agreement +.68/+.64/+.80 raw;
- the model ranking survives a change of generator family — ρ_F is not a Phi-4 artifact.
Mean levels shift modestly (qwen .50→.36 under OLMo; realized width 7.6 vs 8.4 tokens — OLMo's
accepted universes are slightly narrower), consistent with FI^G's generator-relativity: **levels are
G-relative, structure (ordering, per-cell ranking) is G-robust.** That sentence is the paper's claim.

## Problems found during verification (all analysis-layer; data is sound)

1. **The hierarchical per-arm estimator is unreliable on the narrow arm** — the report's raw P2
   "hier" row printed qwen narrow = .68 (ordering inverted). Diagnosis: weak identifiability of the
   per-arm empirical-Bayes fit when the arm has median |U| = 7 **and** a majority of fully-degenerate
   cells (all-0/all-1 at qwen's near-deterministic decoding): (ρ, μ) cannot be separated, and the fit
   chose the high-ρ explanation. A homogeneous-μ simulation shows **no** generic small-N bias
   (MoM/hier both ≈ unbiased at N = 4/7/10), confirming it is a regime problem, not a bias law.
   **Fix applied:** the analysis now prints a P2b MoM row + a caveat on the hier row; interpret P2
   via MoM / N-matched / full-N cells. Do NOT quote the .68.
2. **Single-seed N-matching was seed-lucky** — the report's N-matched p-values wobble across seeds
   (e.g. llama σ²_B .020–.569). Conclusions above use the 5-seed ranges; only qwen's ρ_F increase is
   seed-stable (.001–.013).
3. `tabulate` dependency and a cp1252 console crash — cosmetic, fixed.

## Consequences for the paper

- **C2 gets its positive control**, honestly worded: *"the width of the paraphrase distribution — the
  dial the construct itself predicts — raises the phrasing-attributable variance in every model,
  significantly so in the most phrasing-sensitive one, while leaving competence and per-prompt
  dispersion untouched."* Pair with the specificity dial for the full double dissociation.
- **The swap ablation is a contribution bullet** (per LITERATURE_INTEGRATION §4.4: no published
  prompt-sensitivity metric has one): ordering and per-cell structure survive a generator-family swap.
- **The gate-censoring finding is a methodological point in its own right**: the NLI equivalence gate,
  not the generator instructions, controls the upper end of realizable width (64 % rejection in the
  wide arm; narrow arm shaped by the dedup gate, 84 % rejection, and 66 % of its universes relaxed to
  NLI@0.85). Every NLI-filtered paraphrase evaluation inherits this — worth two sentences in Methods
  and one in Discussion.
- Provenance notes for Methods: mixed A100/H100 windows (content-hash caching makes this harmless);
  per-chain SQLite caches after the lock-storm post-mortem; narrow-arm |U| median 7 with the
  N-matched robustness pass.

## Suggested next steps

1. **Write the Results section** — all experimental arms are now complete (v3 + union-gold + width
   dial + swap + hardened probes). Nothing blocks the draft.
2. R8 (statistical hygiene) + R9 (reproducibility manifest) as the final pass over the assembled
   numbers — as agreed, last.
3. Optional cheap additions: the ask-an-LLM baseline for C3 (one small cluster arm); Good-Turing
   |A_q| robustness (cache-only cluster job) if a reviewer presses on k = 10 entropy bias.
4. Literature session verdicts (Żatuchin, Pecher) before the related-work text freezes.
