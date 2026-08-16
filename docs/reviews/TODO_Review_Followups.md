# Review follow-ups — deferred work queue

Source: `REVIEW_2026-08-06_Adversarial.md`. Decisions taken by Thanos, 2026-08-07.

## Status

| item | decision | status |
|---|---|---|
| **R1** union-gold control arm | **DO NOW** | **DONE (2026-08-07, cluster).** Δ_union = +0.064/+0.125/+0.119 (all BH-sig) — lottery was 47–72 % of the old headline; a real 6–13-pt effect survives. New headline = Δ_union; lottery decomposition → Methods |
| **R2** hierarchical ρ_F (fix coverage + reliability) | **DO NOW** | **DONE** — coverage 45–66 % → **100 %**; see `RESULTS_R1_R2_R3_2026-08-07.md` |
| **R3** equivalence tests / drop "orthogonal" if it fails across models | **DO NOW** | **DONE** — "orthogonal" **dropped**; replaced by bounds. ρ_F vs dispersion factor is 6/6 positive (p=.031) |
| **R4** costumes → analytic proofs | **DONE (2026-08-07)** | `scripts/metric_reductions.py` → `data/metric_reductions.md`: identities verified ≤ 4.4e-16, Williams table, S_τ degeneracy rule (13.8 %) |
| **R4b** graded FI_in curve + convention-free axis-1 headline | **DONE (2026-08-07)** | `scripts/axis1_graded_curve.py` → `data/axis1_graded_curve.md` + figure. Censoring 56–66 % → 32–39 %; cap table; islands test WITH null — and Spearman(max-gap, ρ_F) = **+.83/+.93**: the Hazen stepped shape *is* axis 2 restated |
| **R5** factual corrections + generator-relative FI framing | **DONE (2026-08-07)** | EXPLAINER (banner + 6 inline [R5] patches), final_run_results.md (.457→.543), make_kit_deck.py (5 slide strings), make_supervisor_figures.py footer, paper_analyses_b.py (70 %→75.5 % divisor bug + Horn), main_body.tex (FI^G, Hazen precedent, FI_out perplexity identity, cap disclosure, FI_out_fixed single-test note, union-gold dual scoring, hierarchical ρ_F + σ²_B, probes-baseline sentence). LaTeX compiles |
| **R6** generator-width dial + paraphraser-swap ablation | **DONE (2026-08-08, cluster run complete)** — see `RESULTS_R6_width_dial_2026-08-08.md`. **P0 ✓** (width ordered 4.6<8.4<9.3 tokens; NLI gate compresses the wide end, 64 % rejection). **P1/P2 ✓ as a graded positive control**: σ²_B up 3/3 (p .021–.060); ρ_F MoM monotone 3/3 (qwen p=.0035 seed-stable, mistral .051, llama .079) — effect tracks the model's own sensitivity level. **P3/P4 ✓ clean** (accuracy & H_sem flat, Friedman .47–.68) → **double dissociation complete on both sides**. **P5 ✓** swap ablation: ρ_F per-cell +.41/+.27/+.51 (≈ reliability ceiling), σ²_B +.64–.80, ordering preserved under OLMo → levels are G-relative, structure is G-robust. ⚠ hier estimator unreliable on the narrow arm (weak identifiability; P2b MoM row added to the analysis — never quote the .68) | **(spec below now historical.)** 4 arms (narrow/medium/wide Phi-4 + swap OLMo-2-13B); medium = existing v3 data (free); 50 q × both levels; gates identical, gate-censoring sidecars persisted; preregistered P0–P5 in `RUNBOOK_R6_width_dial_cluster.md`; analysis `scripts/width_dial_analysis.py`; +11 tests. Swap arm = generator AND judge swap (40 GB VRAM constraint, flagged). Payoff: the **double dissociation** (specificity dial moves competence not ρ_F; width dial should move ρ_F not competence) + first paraphraser-swap ablation in the literature |
| **R7** rebuild probe evaluation | **DONE (2026-08-07)** — full 3-model run complete | `scripts/probe_eval_hardened.py` (+7 tests): null *distributions* at every layer (200-draw massmean perms; flip control for vagueness), nested-CV layer selection, TF-IDF/first-word/length baselines both protocols, PR-AUC + confusion at threshold 0.65, fragility null reported; Kossen SEP cited in main_body.tex; fragility disclosure added to deck. **Still missing: the ask-an-LLM baseline (needs generation → cluster arm).** Output: `data/probe_eval_hardened*.md/parquet`. **Results:** vagueness .873–.874 SURVIVES nested-CV layer selection + flip-null (p=.005), beats all text baselines in-distribution (+.17 over TF-IDF); dispersion real but deep-layers-only (.67–.78, null p≤.04); fragility/ρ_F head = chance with full null distributions (nested .50, p=.42) — final honest negative; OOD head .670–.678 / PR-AUC .72–.73 vs frozen text .545–.587 (zero-shot margin ~+.09); ⚠ shipped threshold 0.65 flags 67–73 % of prompts at precision .66 vs base rate .59 — re-tune the threshold or present gauges as rankings, not binary flags |
| **R8** statistical hygiene | **DONE (2026-08-08)** | `scripts/stats_hygiene.py` → `data/stats_hygiene.md` (**the source of truth for Results numbers**): declared 12-test family with Holm+BH — qwen's Δ_union survives BH (.045) but NOT Holm (.15) → wording fixed at "BH-sig 3/3, Holm-robust 2/3"; question-clustered bootstrap CIs on every endpoint; cross-model delta correlations .52–.64 + pooled single-experiment tests (Δ_union +.103 p=3.3e-5, ΔH_sem −.232 p=6.1e-5); split-half reliability table **.38/.52/.57** (canonical). Forking-paths disclosure: `FORKING_PATHS.md` (12 forks, robust-vs-dependent split). main_body.tex §Aggregation rewritten (family, clustered CIs, correlated-models, reliability policy, FI_out_fixed exclusion) |
| **R9** reproducibility | **DONE (2026-08-08)** | `scripts/make_metric_corr.py` — canonical generator for `figures/v3_metric_corr.npy` (reproduces archived matrix to ≤.009; original archived as `*_archived_20260727.npy`; now 14 vars incl. spread → **factor numbers are 74.6 % top-3, Horn 3 — stop quoting 75.5 %**). `scripts/make_run_manifest.py` → `data/run_manifest.json` (funnel 2002→ambiguous→covered→150, seeds, models, thresholds, question IDs, sha256 of 12 key files). `fig_independence` nan_to_num(0) replaced by hard NaN check (a missing correlation must fail, not render as "independent"). Rejection-count persistence existed since R6 (sidecars); historical v3 gap documented in the manifest note |
| **R10** related work | folded into the separate literature session | see `LITERATURE_REVIEW_HANDOFF_2026-08-07.md` |
| **R11** framing | **DECIDED** ↓ | settled |

> **R1 outcome (2026-08-07, cluster run).** The old headline (+0.22–0.25) was 47–72 % grading lottery;
> the surviving union-gold effect is +0.06–0.13, BH-significant in all three models. ρ_F is gold-robust
> (cross-gold agreement .53–.69; ordering and dial-null replicate). The collision "natural experiment"
> did NOT survive (negative under union gold too) — dropped as evidence.
>
> **R2–R3 outcome (2026-08-07).** ρ_F survives as a distinct axis but "orthogonal" does not: the primary
> (hierarchical) estimates are all below |0.09|, yet the sample supports equivalence only to |ρ| < 0.33–0.45,
> and ρ_F is **consistently positively associated with the dispersion family** (6/6, sign test p = .031).
> Two things got *stronger*: "ρ_F does not respond to the dial" is now a properly powered null on n = 150
> (was n = 36–70), and the model ordering qwen > mistral > llama holds under all three estimators.
> Full detail: `RESULTS_R1_R2_R3_2026-08-07.md`.

## R11 — framing decision (settled, 2026-08-07)

The paper is a **measurement model**, and the analysis of the other published metrics is a **significant part of it** (not an aside).

**Findings = contributions 1, 2 and 4:**
1. ρ_F as a distinct axis — *conditional on it surviving R2/R3*. If the three-axis structure holds after the corrected analysis, that is a genuine contribution, especially ρ_F.
2. The FI measurement model / unifying ruler — reframed per R5 as a generator-relative instantiation.
4. The prompt-checker artefact.

**Contribution 3 (the lottery / grading-artifact result) moves to METHODS**, as the justification for the R1 union-gold design. It is not presented as a finding.

---

## R4 — Convert "many costumes" into analytic proofs · QUEUED

**Wrong.** Reported inter-metric agreements are presented as empirical convergence, but two are exact identities and three more are functionals of the same object.

**Verified facts to build on** (all recomputed over 900 rows):
- `s_tau_mean ≡ h_sem_mean / log₂(a_q)` — max abs diff **1.67e-16**. Errica's S_τ *is* normalised semantic entropy.
- `fi_out_fixed ≡ log₂(m₀) − h_sem_mean` — residual **0.0**; m₀ constant within question ⇒ the paired L0→L1 test on FI_out_fixed **is** the H_sem test (identical Wilcoxon p to 4 s.f.).
- `fi_out_var ≡ h_sem_var` — max abs diff **4.44e-16**. (Note: the *reported* .70 pairs Var[FI_out] against H_sem **mean**, +0.758 — a real correlation, not an identity.)
- `a_q`, `variation_ratio`, `consistency_mean` (1−TVD) are all functionals of the same pooled `cluster_assignments`.
- `s_tau_mean` is exactly 0 in **14.6 %** of cells, 94.7 % of which have `a_q ≤ 1` (degenerate normaliser).

**Do.** Replace the correlation-heatmap headline with a **reduction table** + a theorem-style proposition. Reserve *empirical* correlation claims for indices not re-implemented from a shared input — in practice POSIX only — and restate POSIX honestly (see below). Fix or exclude the S_τ = 0 degeneracy with a stated rule.

**POSIX must be restated.** On the common (ρ_F-covered) subset, POSIX~H_sem is **not** significantly larger than POSIX~ρ_F: Williams test p = **.42 / .63 / .054** (n = 42/62/62); POSIX~ρ_F CIs exclude zero for llama and mistral. The reported .63/.43/.61 used n=100 while the ρ_F comparison necessarily used the covered subset — apples to oranges. **Withdraw "the phrasing axis stays empty".**

**Target.** *"Proposition 1. S_τ, Var[FI_out], \|A_q\| and 1−TVD are deterministic functions of the semantic cluster distribution; consequently the reported inter-metric agreements are identities, not evidence."* Plus one honest empirical row for POSIX with its CI.

---

## R4b — Report axis 1 honestly · QUEUED

**Wrong.**
- The persisted `fi_in_curve_vals` is the **binary (T=0)** curve: ≤2 distinct values in all 900 cells, the only step ever firing is k=0→0.05 (688/900 = 76.4 %), 212/900 (23.6 %) completely flat. Since FI_in(q,0)=0 by construction it encodes exactly one free number, −log₂(f_mean), and `aufi_in = 0.975·(−log₂ f_mean)` exactly. **The Hazen "islands of function" claim has no support in that column.**
- ΔAUFI "in bits" scales with the arbitrary log₂(N+1) cap: Δ = −0.617 / −0.817 / −0.855 / −1.119 (llama) at caps 2.585 / 3.322 / 3.459 / 4.392. Sign robust, magnitude is a convention.

**Do.**
1. Persist and plot the **graded** FI_in curve (per-paraphrase F takes 11 values ⇒ a real curve).
2. Re-run the stepped-shape / islands test **on the graded curve against a permutation null** (permute F within cell; report how often the null yields as many steps).
3. Never print "Δ = −0.86 bits" without the cap convention. Prefer the convention-free statement: **the fraction of quality thresholds no paraphrase reaches falls from 56–67 % (L0) to 32–40 % (L1)**.
4. Report the censoring rate wherever AUFI appears.

**Target.** An axis-1 figure with a real curve + bootstrap band, a stepped-shape test with a null, and one convention-free headline number.

---

## R5 — Factual corrections + FI reframing · QUEUED

**Corrections (all verified):**

| where | says | must say |
|---|---|---|
| `EXPLAINER:195` | "Both levels are closed-book" | `uniform_evidence`, **900/900 rows**, mean 17.9 snippets (min 2, max 32) |
| `EXPLAINER:106-108` | ρ_u is "embedding geometry of the **prompts**" | **response** embeddings, from the *same* N×k generations as ρ_F. (The prompt-embedding quantity is ESS_in.) `main_body.tex` is already correct |
| deck slide 26 | "ΔH_sem < 0 — all 3 models, p ≤ 5e-9" | p ≤ 5e-9 holds for Δaccuracy/ΔAUFI only. ΔH_sem: llama 3.7e-07, qwen 0.011, **mistral 0.034 (BH q = .051)**. Under Holm over 57 tests ΔH_sem survives in **1 of 3** models |
| deck slide 30 / `final_run_results.md:14` | length baseline ".457, below chance" | **0.543** — .4569 is the sign-flipped orientation. Margin +0.124, not +0.21 |
| deck slide 31 | "First application of Szostak/Hazen to prompts" | Hazen 2007 §"The Functional Information of Letter Sequences" already did natural language (see literature hand-off §1.1) |
| independence figure | "70 % by top-3 eigenvalues" | **75.5 %**; and cite **Horn's parallel analysis** (retains exactly 3 factors: 5.77/2.44/1.60 vs random 1.69/1.50/1.38) — a much stronger test |
| `EXPLAINER:266` | cross-model ".27–.30" | ≈.2–.45 (already flagged internally) |
| `main_body.tex` §probes | "We report permuted-label and prompt-length baselines for every head" | not true in the data — controls exist only at the shallowest layer (see R7) |

**FI reframing.** Define `FI_in^G(q,k)` with the proposal distribution **G explicit**; state that bits are comparable only within a fixed G; cite Hazen's own requirement (all configurations + degree of function for every configuration) as the reason. Also state the identity **FI_out = −log₂(2^H_sem / \|A_q\|)** (surviving count = perplexity / Hill number of order 1), which puts FI_out genuinely on the ruler; and **stop claiming ρ_F is on the ruler** — the paper's own Table 1 already lists its unit as "share".

---

## R7 — Rebuild the probe evaluation · FUTURE

- `control_permuted` and `baseline_length` exist **only at the shallowest layer**; headline layers (21/24, AUROC .72–.83) have no control. Compute at every layer, with ≥100 permutations for a null *distribution*.
- Select the layer inside a nested CV loop (currently post-hoc over 4 layers × 2 head types).
- Add **TF-IDF word, TF-IDF char, first-word-only** baselines under **both** protocols. Verified numbers: frozen/no-holdout-labels 0.544 / 0.562 / — vs head **0.667**; in-domain 5-fold CV 0.655 / 0.663 / 0.650 vs head 0.667. ⇒ the head's advantage is **zero-shot label efficiency**, not detection quality.
- Report PR-AUC + confusion matrix at the shipped threshold (base rate 0.585), not only AUROC.
- Add the missing competitor: **ask an LLM "is this question ambiguous?"** in one call.
- Report the fragility/ρ_F head's near-chance result **in the deck** (best-over-layers: qwen .534/.555, llama .644, mistral .582).
- Cite/compare Kossen et al. 2024 semantic entropy probes.

---

## R8 — Statistical hygiene · DONE 2026-08-08 (spec below kept for the record)

Effect sizes with question-clustered CIs everywhere; Holm/BH across the declared family with failures reported; **treat model as a random effect** — per-question deltas correlate r ≈ .73 (accuracy) / .40 (H_sem) ⇒ ~1.2 / 1.7 effective replications of 3, so "significant in all three models" is close to one test reported three times; stop reporting `fi_out_fixed` and `H_sem` as two results; replace the k10-vs-k20 "stability" with the disjoint-paraphrase split-half reliability (**0.215/0.350/0.405**, Spearman-Brown **0.35/0.52/0.58**); add a **forking-paths appendix** (AUFI demoted at ρ=−1.00; reformulation gain rejected at .98; `fi_out_mean` → `fi_out_fixed`, which *reverses a significant result* — fi_out_mean: llama Δ=−0.374 p=2.2e-04, qwen Δ=−0.339 p=3.1e-04; Chao1 tried then dropped; evidence dial withdrawn; H_sem chosen as axis-3 representative).

---

## R9 — Reproducibility · DONE 2026-08-08 (spec below kept for the record)

Commit the script that generates `figures/v3_metric_corr.npy` (**there is none**); emit a run manifest with the 150 selected question IDs, seeds, model revisions, filter counts; persist paraphrase **rejection** counts per level (only accepted rows are stored, so the level-symmetry of the gate is unverifiable); remove `np.nan_to_num(nan=0.0)` from `fig_independence` or make it raise.
