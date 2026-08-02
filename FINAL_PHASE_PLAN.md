# Final-phase plan — probes status, the 3-metric defense, and the last cluster run

**Status:** PROPOSAL for review, 2026-08-02. Nothing here is implemented yet.
Every number in Part A/B was re-verified against the parquets on this date
(sources: `data/probe_results_*.parquet`, `data/feedback_verification.parquet`,
`figures/v3_metric_corr.npy`, `data/sensitivity_v2_k20_*`, `data/posix_arm_*`,
`load_ambigqa()` live, `cluster_logs/psf-v2arm-*`).

---

## Part A — What the probes/feedback model can already do (verified)

### A1. Split protocol — what "seen" and "unseen" mean today

There is **no random train/test split anywhere** (that would leak: paraphrases of
one question are near-duplicates). Protocol everywhere is **GroupKFold by
`question_id`** (`train_fi_probes.py:21,115`; feedback heads: question-grouped
OOF + isotonic calibration on OOF + full refit): every reported number is
**out-of-fold performance on questions the head never saw**. Additional
controls: question-permuted labels and a prompt-length-only baseline; ECE is
2-fold cross-fitted (both self-caught flaws — vacuous permutation control,
circular ECE — were fixed before these numbers).

So, precisely:
- **Seen questions:** not reported (train fit is trivially high and meaningless).
- **Unseen questions, same dataset (AmbigQA):** all numbers below.
- **Unseen dataset/distribution:** **ZERO evaluation exists today.** This is
  the single biggest gap and a core goal of the final run (Part C4).

### A2. The user-facing feedback model (4 heads on one TBG forward pass)

From `data/feedback_verification.parquet` (n=2,980 prompts / 149 questions per
model; heads: linear + isotonic on TBG hidden states at 50/75/100% depth):

| head | what it tells the user | qwen | llama | mistral | controls |
|---|---|---|---|---|---|
| **vagueness** | "your prompt is ambiguous/underspecified" | **.874** | **.860** | **.849** | flip .50–.52, length .755 |
| **dispersion** (P3) | "answers to this will scatter" | .688 | .785 | .780 | permuted .47–.57 |
| **reliability** (P2, Spearman ρ) | "likelihood the answer is right" | .486 | .523 | .407 | ECE .06–.08 |
| **fragility** (ρ_F, experimental) | "correctness depends on phrasing" | .516 | .587 | .511 | ≈ controls → flagged experimental |

Reading: **vagueness is the star head** (far above the length confound), which
is exactly the head the user story needs ("is my prompt too vague"). Dispersion
is solid. Reliability is a usable calibrated score, not a strong ranker.
Fragility does not beat controls — k=20 arm proved that's a true signal limit
(TBG states carry little pure-sensitivity information), not label noise; it
stays experimental-only.

### A3. The research probes (P1–P3, per-target best layer, OOF AUROC)

| target | qwen | llama | mistral | note |
|---|---|---|---|---|
| P2 f_graded (correctness) | .767 (L21) | .824 (L16) | .728 (L32) | best overall |
| P1 AUFI_graded | .752 (L21) | .807 (L16) | .715 (L32) | tracks P2 (labels ρ≈−.999) |
| P3 H_sem | .733 (L21) | .835 (L32) | .811 (L24) | SEP-analogue works |
| P1-pure (ρ_F) | .555 | .653 | .620 | weak; honest negative |
| controls | ≤.583 perm / ≤.593 length | | | |

### A4. Gaps the final run must close (answering "seen vs unseen" properly)

1. **Cross-distribution transfer**: frozen heads applied to non-AmbigQA data — currently missing.
2. **A genuinely held-out vagueness test**: AmbigQA's own **830 non-ambiguous
   NQ questions** (the rows `min_interpretations=2` drops; verified loadable)
   vs. ambiguous ones — labels from annotator judgment (a different mechanism
   than L0/L1 rewriting), questions never seen. Nearly free (one dump job).
3. **Frozen-holdout hygiene for the paper**: keep GroupKFold as the protocol
   (standard at n=149) but freeze one final fold assignment + seed, and report
   per-fold spread, not just pooled OOF.

---

## Part B — Paper: why all three metrics TOGETHER (the defense)

Four layers, from math to data. (Analyses marked 💻 are laptop-only, no cluster.)

### B1. One formalism, three measure spaces (derivation)

All quantities derive from Hazen/Szostak FI = −log₂ P(function ≥ threshold)
applied to a different (configuration space, function) pair — plus one variance
decomposition on the first space:

| quantity | space U | function F | formula |
|---|---|---|---|
| FI_in(q,k) | paraphrases U_q | graded correctness F(x) | −log₂(N_k/\|U_q\|) |
| ρ_F | same U_q | same F(x) | ICC(1) share of Var[F] due to phrasing |
| H_sem / FI_out | semantic answer space 𝒜_q | cluster probability | log₂\|𝒜_q\| − H_sem |
| FI_spec | interpretation set (size m₀) | admitted-by-wording | log₂(m₀/m_valid) |

The key derivation point for the paper: **FI_in and ρ_F are different
functionals of the same distribution** — FI_in(k) is determined by the marginal
distribution of F over U_q; ρ_F is determined by its decomposition into
between-paraphrase vs within-paraphrase (sampling) variance. One can move
freely while the other is held fixed (shown constructively in B2). FI_out lives
on the output measure, FI_spec on the dataset — neither is computable from the
others even in principle.

### B2. Independence — mathematical (constructive counterexamples) 💻

A half-page "identifiability box" + tiny simulation appendix. Construct
explicit cells (N=10 paraphrases, k=10 samples) showing each metric can vary
while the others are pinned:

1. Same accuracy + same FI_in curve, **different ρ_F**: rates
   (0.5,…,0.5) vs (1,0,1,0,…) — mean .5 both; ρ_F ≈ 0 vs ≈ 1.
2. Same accuracy + same ρ_F, **different H_sem**: wrong answers concentrated in
   ONE wrong cluster vs spread over many — correctness identical.
3. Same H_sem, **different accuracy**: confidently-wrong vs confidently-right
   (our real L0 finding: ambiguity shows as confident single-mode WRONG answers).
4. FI_spec independent of all: it contains no model quantity by construction.

This is cheap, rigorous, and reviewers love it. (Simulation: ~50 lines, seeded.)

### B3. Independence — empirical 💻

- **Correlation structure** (have): within-level Spearman — ρ_F ⊥ accuracy
  (.08), ⊥ H_sem (.03); H_sem family loadings .60–.94 ("one construct, many
  costumes" = the literature collapses into axis 3).
- **NEW: factor analysis** on the 14-metric within-level correlation matrix:
  show a 3-factor solution (+ ESS_in as an isolate) explains the structure;
  report variance explained + loadings table. One afternoon.
- **NEW: octant occupancy**: median-split the three axes → show real questions
  populate all 2×2×2 cells (with per-model counts + one named example per
  interesting octant, e.g. high-ability/high-sensitivity). Proves no axis is
  predictable from the others *in data*, not just in math.
- **Different derivatives w.r.t. the dial** (have): turn FI_spec and the axes
  respond differently — ability ↑ (p ≤ 5e-9 ×3 models), dispersion ↓
  (sig. ×3), sensitivity FLAT (.397→.403) — three different responses = three
  different constructs. A metric pair that were secretly one thing could not
  respond differently.

### B4. Per-metric validity (measures the right thing, correctly)

Structured as a construct-validity checklist per metric:

- **FI_in curve**: content validity = verbatim Hazen translation (§7.3);
  criterion = dose-response to the dial (CI bands separate at every k>0, all 3
  models); the honest limit = its scalar AUFI ≡ accuracy (ρ = −1.00) → appendix.
- **ρ_F**: convergent = ρ_u agreement .67 across two channels (behavior vs
  embeddings); discriminant = ⊥ accuracy/H_sem; reliability = k10↔k20
  .81/.92/.95; interpretation = (question × model) trait (cross-model .2–.45 vs
  .77–.82 for accuracy); honesty = coverage reported (undefined on zero-variance
  cells, principled). 💻 NEW: cluster-bootstrap CIs over paraphrases.
- **H_sem / FI_out_fixed**: lineage = Farquhar Nature 2024 + exact Errica
  rescaling on MC; criterion = family loadings incl. POSIX .60–.63; correctness
  rule = fixed-capacity yardstick (moving-|𝒜_q| trap documented); calibration
  reading = capacity share 11→41% / 38→49% / 74→82%.
- **FI_spec**: model-free by construction (identical across models in data);
  exact by definition (log₂(m₀/m_valid)); currently only a 2-point dial —
  **the final run upgrades it to a within-question dose-response (C1)**, the
  strongest possible evidence that the dial is causal.

---

## Part C — The last cluster run (arms, cost, gates)

Everything below reuses the existing driver/cells; new code is confined to two
builders + one harvester + sbatch files. Cost anchors (measured): POSIX ≈ 13
s/cell on cached generations; a 100-cell arm ≈ 2×30-min windows; v3 (900
cells, 3 models, k=10, N=10) ≈ one overnight of chained `gpu_a100_short`
windows.

### C1 — Multi-level specificity ladder (the dial becomes a dose-response) — P0

- **Pool**: 613 AmbigQA questions have m₀ ≥ 3 (verified live); after the 52%
  evidence filter ≈ 300; sample **~100 fresh questions with m₀ ∈ {3,4,5}**.
- **Levels**: L0 (ambiguous, m_valid=m₀) → **L_mid (partially disambiguated,
  m_valid=⌈m₀/2⌉)** → L1 (fully disambiguated, m_valid=1). FI_spec hits
  intermediate values *within the same question* → per-question dose-response
  curves of accuracy/FI_in/H_sem vs bits, not just a 2-point contrast.
- **New code**: `build_levels` extension — L_mid question text via Phi-4
  ("rewrite so it admits exactly interpretations {i,j} — e.g. 'either A or B'")
  + NLI gate that the rewrite admits exactly that subset + fixed-gold guardrail
  unchanged (gold = target interpretation's answer at every level; L_mid keeps
  multi-gold constraint over the admitted subset).
- **Risk (the one real design risk in this plan)**: L_mid rewrite validity.
  Mitigations: restrict to m₀∈{3,4,5}; NLI + rule gates; smoke gate includes a
  **10-question human eyeball** (you) before full launch; fallback = drop C1 to
  a fresh-question 2-level replication arm (still valuable: out-of-sample
  replication of v3).
- **Scope**: 100 q × 3 levels × 3 models = 900 cells ≈ v3-scale (one overnight).

### C2 — VoI context arms (Phase 2, per REBUILD_PLAN §14) — P1

- **Design**: closed-book cell + ONE context line, 5 conditions per question:
  {none, disambiguating-non-revealing, partially-relevant, distractor,
  misleading}. Lines harvested from AmbigQA interpretation clauses + the §14
  NLI non-revealing gate (NLI(line ⊨ gold) must be LOW — the line may resolve
  *which question*, never *the answer*).
- **Why it belongs in the paper**: measures **value-of-information in bits** —
  how much FI_in/accuracy a context *type* buys, including negative VoI
  (does a distractor RAISE FI_in?). Extends the dial family: question-side
  specificity (FI_spec) vs context-side information (VoI) as two currencies.
- **Scope**: reuse 50 v3 questions (paraphrase caches exist → cheap), **L0
  only**, qwen + llama, 5 × 50 = 250 cells/model. New code: line harvester +
  gate (reuses NLI stack).

### C3 — POSIX everywhere — P0 (trivial cost)

`--posix` backfill over the cached v3 generations for **all 3 models × 300
cells** (~1.5 h total). Closes the axis-3 family row: POSIX loading currently
rests on one qwen arm (.60–.63); after this it's a 3-model claim. Also emit
`h_sem_per_paraphrase` (persistence fix a8f62b1 postdates the v3 cluster tree)
so P3 becomes per-prompt like SEP.

### C4 — Cross-dataset verification — P0 (the "unseen data" answer)

- **C4a metric-structure transfer**: **HotpotQA + 2WikiMultihopQA** (loaders
  already in repo; gold paragraphs = native uniform evidence — zero new loader
  code). 50 q × 1 level × 3 models × 2 datasets = 600 cells + dumps. Analyses:
  does the 3-axis correlation structure replicate (ρ_F ⊥ accuracy; H_sem family
  coheres)? Do frozen P2/P3 heads transfer (OOD AUROC vs in-distribution)?
  Optionally retrain-on-top for a transfer-vs-retrain gap.
- **C4b frozen-head vagueness holdout (nearly free)**: dump TBG states for the
  **830 non-ambiguous NQ questions** + a matched ambiguous sample (1 prompt
  each, one short dump job), apply the **frozen** vagueness heads → AUROC on
  annotator-labeled, never-seen questions. This is the honest "unseen data"
  number for the user-facing story.
- Deliberately **excluded**: TriviaQA/NQ-open full arms (need new loaders +
  evidence design — cut as scope; HotpotQA/2Wiki cover the transfer claim),
  MuSiQue (generator-era baggage).

### C5 — Dumps + retraining (bundled with the above)

Every new cell dumps hidden states (`--dump-hidden` phase per arm). Laptop
afterwards: retrain feedback heads on v3 + C1 (3-level vagueness becomes
*ordinal* — head upgrade from binary to 3-class or rank loss), rerun probe
suite, C4 transfer tables.

### C6 — Evidence-dial arm (0 / half / full snippets × L0/L1, qwen, 50 q) — P2 stretch

Completes a 2D (specificity × evidence) surface. First thing to cut if the
queue is slow.

### Sequencing, gates, budget

1. **Laptop build week**: C1 builder + C2 harvester + sbatches + tests; B2/B3
   analyses (they need no new data — can go into the paper draft immediately).
2. **SMOKE gate** (one `gpu_a100_short` window): 5 q through C1 + C2 with all
   §11-style asserts + **human eyeball of 10 L_mid rewrites — HARD GATE**.
3. **Full submission** (existing chained-window pattern): C1 (3 chains) + C3 +
   C4a (3 chains) + C2 + dumps. Estimated ≈ 2,000–2,400 cells ≈ **1.5–2×
   the v3 run ≈ a weekend** of chained short-partition windows.
4. **Pull + backfill** (run.sh pull auto-backfills), retrain, analyses, paper
   tables. Everything checkpoint/resume-safe as before.

### Explicitly out of scope

Larger-N x* per-question geometry (stays population-level), MI300/H100
experiments, live-demo cluster deployment (separate 30-min task when needed),
any new metric.

---

## Decision points for the GO

1. **C1 L_mid mechanism** — approve the partial-disambiguation design (with
   human smoke gate + fallback)? It's the highest-value + highest-risk arm.
2. **C2 scope** — qwen+llama OK, or all 3 models (adds ~250 cells)?
3. **C4a datasets** — HotpotQA + 2Wiki confirmed (loaders exist), or is one of
   TriviaQA/NQ-open worth a new loader to you?
4. **C6** — include or pre-cut?
5. **Paper protocol** — freeze GroupKFold (seeded) + per-fold spread as the
   reported protocol, plus C4 as the OOD chapter — agreed?
