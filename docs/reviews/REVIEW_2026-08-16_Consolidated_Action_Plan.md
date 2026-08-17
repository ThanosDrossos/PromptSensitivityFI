# Consolidated review & action plan — 2026-08-16

> **IMPLEMENTATION STATUS (2026-08-16 evening, branch `fix/review-2026-08-16`
> in both repos):** everything below is **implemented** except the Horn-dedup
> print (M10, skipped per Thanos) — blockers §2.1–§2.5, major items M1–M16
> (M10 partial: assignment-not-count wording adopted, Q2 number not printed),
> new findings §4.2–§4.6b, decisions D2 (parquets committed) and D3
> (`papers/` gitignored). New committed analyses: `rho_f_recovery_sim`,
> `rho_f_construct_validity`, `evidence_coverage_audit`, extended
> `width_dial_analysis` / `stats_hygiene` / `metric_reductions` /
> `probe_eval_hardened` (decontaminated OOD, `probe_eval_ood.json`), figure
> pipeline reads artifacts only. Paper rewritten: all claims artifact-backed,
> content compressed 22 → **~10.7 pages** (22 total incl. statements,
> references, appendix; 0 unresolved refs; suite 382/0). Two notable
> substantive outcomes of the new analyses: **spread predicts rephrasing
> payoff at least as well as ρ_F** (the paper now frames ρ_F as measurement,
> not payoff forecasting), and the **fragility-head null now holds in all
> three models** on the macOS reproduction. Residual items for Thanos:
> the last ~0.7 page to strict ≤10 (options: cut the Terms ¶, move Fig. 1 to
> the appendix, or confirm KIT's counting excludes the title block); the
> "Submitted: 12 August 2026" date on the title page (stale for a revised
> hand-in); merging + pushing the two branches; D1's ICLR variant
> (anonymization) if ever submitted.

**Scope.** The paper (`SensitivityFunctionalInformationPaper` @ `4303886`, the
30-page 2026-08-12 build) against ICLR 2027 standards, consolidating
`REVIEW_2026-08-14_Paper_ICLR.md` (all of whose findings are still open — the
paper has not been touched since 08-12) with a fresh full read of the paper,
the codebase, the committed artifacts, the migrated parquets, and two
web-verification passes (novelty claims; bibliography) run 2026-08-16.

**How to read this.**
- §1 decisions you need to make before the work starts.
- §2 the five blockers, §3 the major items, §4 findings that are **new in this
  pass** (not in the 08-14 review). Every item: problem → concrete fix → cost.
- §5 literature/novelty status after the fresh web sweep, §6 the page-cut plan,
  §7 the recommended execution order, §8 what I verified first-hand.
- Cross-references like "(08-14 §2.3)" point into
  `docs/reviews/REVIEW_2026-08-14_Paper_ICLR.md`, which carries the full
  evidence; I re-verified its load-bearing claims rather than trusting them
  (§8 lists which and how).

**Bottom line.** The 08-14 verdict stands: reject as framed, clear path to a
defensible paper, no new data needed. Nothing I found overturns any of its
findings; I confirm them and add a venue-critical one (the real ICLR limit is
**9 pages**, not 10) plus a set of smaller repairs and open citations. The
single most important content fact: **all three abstract-level quantitative
claims (the 4.4e-16 reduction, the empty third axis, the 0.670–0.678 probe
transfer) are currently indefensible as written, and all three have honest
versions that are nearly as strong.**

---

## 1. Decisions needed before work starts

**D1 — What is the target: ICLR 2027 submission, seminar final, or both?**
This changes the mechanical work materially:

| | seminar build (current) | ICLR 2027 submission |
|---|---|---|
| page limit | you said 10 max (content) | **9 pages main text at submission** (10 only at rebuttal/camera-ready); over-limit = **desk reject** (official Author Guidelines, checked 2026-08-16) |
| anonymity | author + advisor block shown (`\iclrfinalcopy`) | double-blind: remove the seminar block, comment out `\iclrfinalcopy`, **anonymize the GitHub link** (the repo URL carries your name — use anonymous.4open.science or an anonymized fork) |
| statements | AI-use + reproducibility present ✓ | both allowed and excluded from the limit ✓; ethics statement optional |
| deadlines | — | abstract **2026-09-18 AoE**, paper **2026-09-25 AoE** (~5½ weeks) |

Recommendation: cut to **9 pages** regardless (satisfies both targets), and add
a `\seminartrue/false` toggle so one source produces both builds.

**D2 — Reproducibility strategy.** The Reproducibility statement is false as
written (08-14 §3.13): no per-cell files are committed. Now that all artifacts
are on this machine the fix is nearly free — I measured it: the **entire
analysis-level parquet set is 1.7 MB** (specificity_v3, union_gold, rho_f_hier,
width_*, posix_arm, sensitivity_v2_k20, width_dial_cells, mechanical_null,
independence_*, paraphrases_ambigqa + width sidecars, probe result tables).
Options: (a) `git add -f` those 1.7 MB — every table and figure then re-derives
from the repo alone — and put the six large probe inputs (`hidden_states_*`,
`vagueness_holdout_*`, 450 MB) plus the LLM cache on Zenodo with a DOI;
(b) everything on Zenodo; (c) soften the statement to match reality.
Recommendation: (a) — it converts the review's Reproducibility 3/10 into a
genuine strength for ~an hour of work. Either way `make_paper_figures.py` must
stop hardcoding fig1/fig3 numbers (08-14 §3.13.3).

**D3 — `papers/` (127 MB of PDFs, now in this repo untracked).** gitignore it,
track via Git LFS, or commit plain. Recommendation: add `papers/` to
`.gitignore`; it syncs via OneDrive/backup, not git.

---

## 2. Blockers (reject-level; all confirmed still present)

### 2.1 Probe holdout contains all 150 training questions *(08-14 §2.3; fix A1)*

**Problem.** §3.6 and §4.5 claim the zero-shot holdout labels "were never used
in training"; the numbers quoted (0.670–0.678, abstract + Fig. 3) come from
`probe_eval_hardened.py::eval_ood`, which evaluates on all 2,002 holdout
questions **including the 150 training questions, all of them positives**.
Verified three ways: the code has no exclusion (my read of
`probe_eval_hardened.py:336–366`); the confusion matrix in the committed
artifact sums to 2,002; the overlap join on the local parquets returns 150/150
ambiguous (§8). The decontaminated numbers **already exist** and are barely
worse: 0.667/0.655/0.670 (`data/final_run_results.md`, n = 1,852) — the two
committed artifacts currently contradict each other with no supersession note.

**Fix (half a day; step 1 already done).**
1. ✅ **Applied 2026-08-16**: `eval_ood` now excludes the training questions
   (mirrors `eval_vagueness_holdout.py`); I verified end-to-end through the
   frozen committed bundles that the exclusion turns exactly 0.678/0.670/0.678
   into exactly 0.667/0.655/0.670 (§8). Tests pass.
2. On your go: re-run the OOD block (local, CPU), regenerate
   `data/probe_eval_hardened.md` and Fig. 3, requote 0.667/0.655/0.670 (and
   the baselines/operating point on n = 1,852) in abstract, §4.5, §5.1, §5.2,
   Fig. 3 caption.
3. Add a supersession note to whichever artifact is retired.
4. The qualitative claim survives (margin over frozen baselines +0.08–0.10).

### 2.2 The "five indices agree to 4.4e-16" reduction is definitional *(08-14 §2.1; fix A3)*

**Problem.** S_τ is *implemented* as `H_sem / log2|A_q|`
(`metrics/errica.py:40–51` — I read it; the identity is the definition, and the
original token-entropy formulation was redefined onto the shared clustering
because logprobs were unavailable, which §4.1 does not disclose while
Appendix A.3 contradicts it). Of the "five published indices": two are this
paper's own constructs, one (variation ratio) is uncited in the text
(`lu2024prompts` sits unused in the bib — I checked: 76 entries, 61 cited),
and only two carry citations, from one source. Three of the five were never
checked numerically (`data/metric_reductions.md`: "by construction"). And
"functionals of one distribution" ≠ "one measurement": on the paper's own
primary test the indices give different answers (H_sem responds in Qwen at
p = .011, S_τ at p = .213 — recomputation confirmed, §8).

**Fix (writing + one table; the honest claim is nearly as strong).**
1. Restate as a scoped proposition: *the dispersion quantities we compute are
   functionals of one pooled cluster distribution; two are exact affine
   restatements of H_sem; correlations among them are therefore not convergent
   evidence.* (08-14 §2.1 has ready wording.)
2. Drop "$4.4\times10^{-16}$" and "five published" from the abstract; disclose
   the S_τ re-implementation in Methods; reconcile §4.1 with Appendix A.3;
   cite Lu et al. 2024 for the variation ratio.
3. Print the per-index paired-test table (the divergence is *support* for
   "report one representative", not a threat).

### 2.3 "The third axis is measured by no existing index" *(08-14 §2.2; fix A4)*

**Problem.** ρ_F correlates with Cox's ρ_u at +0.24…+0.53 and with Cao/Sclar's
spread at +0.18…+0.37 across all six strata under the primary estimator
(recomputation confirmed, §8) — the two strongest associations ρ_F has with
anything. §3.2.2 *promises* ρ_u "as a second channel on the same construct" and
§4 never reports it; Figure 2's own formulation-sensitivity block shows the
axis occupied by ρ_u, spread, ESS_in. The abstract's sentence is contradicted
by the paper's own figure.

**Fix (one afternoon; turns the paper's most exposed claim into support).**
1. Reword to: measured by no existing index *with the within-prompt sampling
   term removed* (that conjunction is verified clean — see §5).
2. Report ρ_F↔ρ_u and ρ_F↔spread in §4.4 as **convergent validity**.
3. Add the missing **incremental-validity test**: does ρ_F beat `spread` (a
   one-line statistic needing no ICC) and ρ_u at predicting the out-of-sample
   rephrasing payoff? Pure re-analysis of existing parquets. If ρ_F does *not*
   beat spread, that is a headline result and the paper must say it.

### 2.4 The width dial's evidence is thinner than claimed *(08-14 §2.4, §3.3c, §3.3d; fixes A2+A6)*

**Problem (verified from `data/width_dial_analysis.md` + code).**
(a) Table 4's ρ_F rows are complete-case MoM on cells covered in *all three*
arms: n = **22**/54/46 of 100 — the caption says "50 questions, both levels
pooled" and gives no n; Llama's monotonicity exists only in the paired subset.
(b) The preregistered estimator (P2, hierarchical) gives the **opposite**
significance pattern (Qwen p = 1 raw; N-matched: Qwen p = 2.8e-8 but **Llama
p = 0.986**), and the substitution is not labelled as a deviation;
`FORKING_PATHS.md` fork 2 asserts decision-robustness that its own R6 artifact
contradicts. (c) Methods' "all filters … identical across arms" is false in
realization: the NLI gate ran at the relaxed 0.85 threshold in **66%** of
narrow-arm universes vs 1% wide (adaptive fallback, `paraphrases/pipeline.py`
— I read it), the narrow arm's universes are a third smaller (|U| 6.7 vs 10),
and the production arm has **no rejection sidecar at all**. (d) The quoted
five-seed N-matched ρ_F range has no committed code path: the N-matching loop
covers only `sigma2_B` and `rho_f_hier`, never `rho_f_mom`
(`width_dial_analysis.py:362` — I read it), and there is no multi-seed loop.
(e) The σ²_B endpoint loses individual significance in 2/3 models under
N-matching (0.051 / 0.214), and that row is absent from the paper.

**Fix (one day: small analysis + writing).**
1. Rebuild Table 4: n per row; both estimators (MoM covered + N-matched
   hierarchical); unpaired means; state Llama's non-monotonicity.
2. Implement and commit the 5-seed N-matched **MoM** pass (or delete the
   sentence quoting it); commit the identifiability simulation or drop that
   sentence too.
3. Methods: disclose the adaptive NLI fallback (0.90→0.85) with per-arm rates,
   the |U| gap, and the missing production sidecar; note that |A_q|-family
   metrics are |U|-dependent, so the N-matched contrast is the primary one.
4. Label the estimator substitution as a preregistration deviation, show both
   rows; fix `FORKING_PATHS.md` fork 2 and add this fork to the list.
5. Add the N-matched σ²_B row with five-seed ranges ("direction holds 3/3;
   individual significance in 1/3").
6. Carry the "significantly in Qwen" qualifier into the abstract, Fig. 1b
   caption, Discussion, Conclusion (currently only §4.3 has it).

### 2.5 The "properly powered null" is not — the estimator cannot see the effect it excludes *(08-14 §2.5; fix A5)*

**Problem.** §4.2/§5.1 sell Δρ_F ≈ 0 under disambiguation as "a properly
powered null … CIs exclude changes larger than 0.04". The hierarchical
estimator pins 34–55% of cells to a near-constant prior value (degenerate
cells carry no information), so paired posterior-*mean* deltas are shrunk
toward zero by construction. My independent 3-seed re-run of the review's
recovery simulation through the pipeline's own estimator (§8): a **true Δρ_F
of +0.20 reports as ≈+0.04 in Qwen and Mistral** — at the very bound the paper
claims excludes it — and ≈+0.08 in Llama (whose null is therefore the most
informative of the three; the reported "0.04" corresponds to true effects of
roughly 0.2 / 0.10–0.15 / 0.2 on the estimand scale). Additionally, 41/22/37
of the paired deltas are exactly zero and SciPy's Wilcoxon discards them, so
the tests run on 109/128/113 pairs, not the stated n = 150 (recomputed, §8).

**Fix (one afternoon, analysis only).** In order of preference:
1. Run the recovery simulation properly (multi-seed) and report the
   estimator's **minimum detectable Δρ_F** next to the null.
2. Test on the multiple-imputation **draws** — `data/rho_f_hier_draws_*.npy`
   (400 draws × 300 cells) are already committed; a Rubin-pooled paired test is
   a short script.
3. Report the complete-case MoM paired test on the informative pairs
   alongside, with its CI.
Until then the sentence must read as the review puts it: *a null we cannot
distinguish from an underpowered one at this design size*. Discussion/
Conclusion inherit the change. Note the prediction itself is theoretically
right and the discriminant-validity framing survives — what changes is the
claimed evidential strength.

---

## 3. Major items (consolidated from 08-14 §3, deduplicated; all confirmed)

| # | problem (08-14 ref) | fix | cost |
|---|---|---|---|
| M1 | **Reading-rank moderator unreported** (§3.1): for Qwen the union-gold headline is +0.175 on first-listed-reading questions and −0.011 on the rest; computed in `data/seed_audit.md`, absent from the paper | Report as a moderator result (it also pre-empts the target-seed question); one table or two sentences + appendix table | half a day |
| M2 | **Evidence filter conditions on the L1 target** (§3.2): `target_in_evidence()` checks only the pinned target's answers (code-verified), so the retained 52% guarantees L1's gold in-context but not the other readings L0 needs under union gold; "no selection on model knowledge" answers the wrong objection; snippets are the annotators' search results for the *original* question — the most plausible alternative mechanism for M1 | Add the per-reading answer-presence check (data exists in the AmbigQA payloads); one Methods paragraph naming the selection correctly; characterize the 459 discarded eligible questions | half a day |
| M3 | **Asymmetric inference discipline** (§3.3, §3.3b): width-arm nulls rest on omnibus Friedman with no equivalence bounds while the paper's own nulls get bounds; one-sided tests for the predicted-to-move axis, omnibus for the predicted-flat axes (directional H_sem test in Qwen: p = .034 narrow→wide — a *smaller* p than the σ²_B result reported as positive); POSIX Williams test at unstated n = 42/62/62; "all inference follows the declared family" covers only the specificity arm | One test policy for the whole width table; equivalence bounds (or "no information") on every claimed null; state POSIX's n and restore the artifact's "at this sample size" qualifier; one sentence in §3.4 saying which claims carry error control | writing + small analysis |
| M4 | **Estimator/gold-set mixing in single sentences** (§3.3e, §3.5): the 6/6 sign-test sentence is complete-case dressed as primary (hierarchical is 4/6); Table 5's σ²_B and coverage rows are target-gold unlabelled; the ρ_F row of the union-primary family is computed on target gold (`stats_hygiene.py:95–106`, code-verified) and §4.2's "replicates under target gold" is backwards; "Spearman 0.64–0.72" reproduces from no artifact (true value 0.85–0.95 — recomputed, §8) | Label every gold set; one estimator per sentence; make union the computed primary for ρ_F (or relabel); fix the Spearman range (error runs in your favor) | half a day |
| M5 | **Out-of-sample checks never leave the paraphrase set** (§3.4): both payoff checks resample decoding draws over the same ten paraphrases — closer to split-half reliability than to "will rephrasing pay off" | Add the held-out-paraphrase version (estimate ρ_F on 5, predict payoff on the other 5); pure re-analysis; also justify predicting F_max−F̄, a statistic FORKING_PATHS #10 retired | one afternoon |
| M6 | **"24 correlations" is 6 × 4 assumed-ρ values** (§3.6): `data/mechanical_null.md` itself says 6/6 | Write "6/6, robust across four assumed true ρ" (stronger and shorter) | one line |
| M7 | **Table 5's three predictive rows have no committed scripts** (§3.7) and use the estimator §4.4 disowns (complete-case MoM, n = 42/62/62, unstated) | Commit the scripts (the dossier `docs/results/RHO_F_CONSTRUCT_VALIDITY_2026-08-07.md` documents the computations); re-run on hierarchical estimates; state n | one day |
| M8 | **FI_out^fixed < 0 in up to 42% of cells** (§3.8): 2^H_sem counts model answer clusters (3.6–15.8) while m₀ counts annotator readings (mean 2.7); a negative "surviving fraction" bit value is a formal breakdown, not "informative in itself" (Llama L0: 42% negative — recomputed, §8) | Reframe log₂m₀ as a unit-conversion reference constant; report the negative rate; fix Table 1's "semantic answer space" label for that row | writing only |
| M9 | **FI_spec's bits do no work; the dose test exists and is null** (§3.9): FI_spec ≡ level × log₂m₀; §5.3 claims a dose test "a two-point design cannot" do, while `data/seed_audit.md` already contains the between-question dose regression — Spearman(Δ_union, m₀) = −0.105/−0.126/−0.049, null and wrong-signed | Report the dose regression as an honest negative; delete "which a two-point design cannot"; state plainly that FI_spec is a relabelled binary indicator whose bit *scale* is not validated | one hour |
| M10 | **"Exactly three axes" overrides the paper's own computed counter-number** (§3.10): de-duplicated Horn retains 2 (`data/factor_audit.md` Q2); the supervisor's required restatement ("assignment, not count" — `docs/reviews/SUPERVISOR_FEEDBACK_2026-08-11.md`) was never carried into the text | Print Q2; adopt the assignment-not-count wording (the count then rests on the reductions + the dissociation, which is the stronger argument anyway) | writing only |
| M11 | **The dissociation halves are not commensurable** (§3.11): 150 vs 50 questions, hierarchical vs MoM, two-sided vs one-sided, family vs none | Recompute the specificity null on the width arm's 50 questions with MoM (costs nothing); present the halves in one table with the asymmetries visible | one afternoon |
| M12 | **Probe section misstatements** (§3.12): vagueness null is a flip control, not the "full refitted permutation null" Methods claims (code-verified: `_TARGETS` sets `perm_ok=False`); the 0.873 is a within-question L0-vs-L1 contrast (label *is* `spec_level==0`, code-verified) and reads as a population detection rate; Llama fragility 0.63 is the fixed-layer value (nested-CV = 0.580) and its p = 0.038 is the 1/26 floor of a 25-draw null (code-verified); the reliability head has no text baseline (gated behind `if binary:`); the 0.59 base rate is never stated next to precision 0.66; threshold 0.65 is a hardcoded literal, not a validated operating point | Five wording fixes + quote 0.580 + state draw counts and base rate; present gauges as rankings or re-tune the threshold | writing only |
| M13 | **Reproducibility statement false in three checkable ways** (§3.13) + empty Appendix C with a `%% TODO` rendering on p. 30 + promised "forking-paths appendix" missing while `FORKING_PATHS.md` exists | Execute D2; put the forking-paths table in the appendix; populate or delete Appendix C; make figures read artifacts | half a day |
| M14 | **Three `%% TODO-LIT` markers = three uncited literature claims** (§4) — including the LLM-paraphrase-diversity claim printed with no citation | All three records are now pinned (§5.2): cite Cegin 2023, Jiang & de Marneffe 2022, McCabe 2509.14478 (+ Nguyen 2025) | one hour |
| M15 | **Length control computed but unused** (§4): L1 questions are +43% longer; within-cell ρ(length,F) ≈ 0 with 95% of cells spanning more length variation than the manipulation; ρ(Δlength, Δaccuracy) ≈ 0 (`data/final_run_results.md`) | One Methods/Results paragraph; free credibility | one hour |
| M16 | Assorted §4 fixes: `fi_in.py` grid-epsilon bug (code-verified; numerically negligible but inconsistent with `sensitivity_v2.py`); §3.3.3 says the L0 gold filter is an AND while the code is an OR (code-verified, `constraint_filter.py`; the OR is right, the stated reason isn't); S_τ degeneracy "13.8%" should be the range 6.7–26.3% (Qwen worst); split-half 0.38/0.52/0.57 is target-gold (union: 0.363/0.509/0.495) used against union-gold quantities; Fig. 2 plots \|Spearman\| without saying so; 2026 arXiv cites lacking venues (now resolved, §5.2) | Batch of one-line edits | half a day |

---

## 4. New findings from this pass (not in the 08-14 review)

### 4.1 Venue mechanics (blocking for an ICLR submission)

- **The ICLR 2027 limit is 9 pages at submission** ("the main text should be
  9 pages or fewer … Papers with main text beyond the page limit will be
  desk-rejected"), 10 only at rebuttal/camera-ready. Your stated target of 10
  is safe for the rebuttal build but **not for submission**. §6 plans for 9.
- Anonymization is not just `\iclrfinalcopy`: the seminar block (advisor,
  examiner, matriculation number, submission date) and the **GitHub URL in the
  reproducibility statement** identify you. Use an anonymized repo link for
  submission.
- The AI-use statement is *required* at ICLR 2027 and yours is already
  compliant and well-written — keep it verbatim.

### 4.2 The public repo contradicts the paper *(new)*

The paper links `github.com/ThanosDrossos/PromptSensitivityFI`. Its
`README.md` (unchanged since 08-06) still asserts the two claims the R-series
retracted: "the three axes are empirically independent (ρ_F ⊥ accuracy .08,
⊥ H_sem .03)" and the target-gold headline "+0.22…+0.25 … accuracy roughly
doubles" as *the* result. A reviewer who follows the link reads a README that
contradicts the submission's own §4.1/§4.2. Also visible: `app/README.md`
publishes an ngrok basic-auth credential pair; `cluster/runbooks/` (if you
commit them) embed your cluster username. **Fix:** rewrite README to the
union-gold/bounded-non-redundancy story (one hour); scrub credentials.

### 4.3 The preregistration claim has no external timestamp *(new)*

§3.3.5: "Five predictions were fixed before any arm data existed." True per
the runbook (2026-08-08) — but the entire R-series was committed in one batch
on **2026-08-14**, *after* the arm data, so the public git history cannot
support the claim, and OneDrive file dates are not citable. Soften to
"specified in the analysis plan before the arms were analysed" or present
P0–P5 as planned comparisons — and never use the word "preregistered" in the
paper unless you can point at an external timestamp. (The 08-14 review §3.3c
already flags the P2→P2b deviation; this is the additional point that the
*registration itself* is only self-attested.)

### 4.4 Dead and unreported machinery in Methods *(new; page savings)*

- **ΔFI (Eq. 6) is defined and never reported anywhere in §4** — the 08-14
  review notes its axis assignment in passing, but no ΔFI number appears in
  the paper. Cut Eq. 6 or demote to one sentence + appendix (saves ~⅓ page).
- **The permissive scoring threshold (0.5) is promised as "a robustness
  check" (§3.4 Scoring) and never reported.** Either report one sentence of
  results or delete the promise.
- **`ESS_in` appears in Fig. 2 and nowhere in the text** (08-14 noted the 14
  metrics are never enumerated — the fix is one appendix table listing all 14
  with definitions and sources; that also grounds the factor analysis).

### 4.5 "Cross-model transfer 0.2–0.45" has no artifact *(new)*

§4.4/§5.1 quote per-question ρ_F cross-model Spearman "0.2 to 0.45". I could
not find this range in any committed artifact (like the Table 5 rows of 08-14
§3.7, it traces to an internal note). My recomputation from the committed
estimates (§8): hierarchical union pooled **+0.12 to +0.23**, all variants
tried (within-level, MoM complete-case, target gold) span **0.10–0.29** — no
variant reaches 0.45. Replace the printed range with a scripted number in the
M7 batch; the "question×model trait" reading gets *stronger*.

### 4.6 The k=10 richness bias defense is thinner than the text implies *(carried from the verified literature review, still unaddressed)*

§3.4 (Clustering) discloses the downward bias of observed |A_q| and argues the
cross-level comparison is conservative. The verified literature review
(§6.4/§8.1) flags that the *fix it points to* (Good–Turing at small k, McCabe
2509.14478; Nguyen Findings-ACL 2025 works at exactly k = 10) is available and
that a sensitivity analysis is expected — but the raw cluster assignments
needed for it are not persisted, which the paper's own TODO-LIT comment admits.
Honest options: (i) cite both and state the bias direction argument as the
defense (current text, plus the two citations — minimum); (ii) persist
assignments in a future run (out of scope now). Do (i).

### 4.6b Figure-level fixes *(new; from rendering all three figure PDFs)*

- **Fig. 1a** plots three endpoints with different units (accuracy points,
  bits, a variance share) on one numeric axis labelled "change from ambiguous
  to disambiguated". Annotate units per row (or use per-row axes); the
  caption's "intervals that exclude changes larger than 0.04" must go with A5.
- **Fig. 1b** shows three cleanly rising MoM lines with **no uncertainty, no
  n** — precisely the n = 22/54/46 problem of §2.4 in visual form. After A2 it
  needs error bars, the n, and (ideally) the N-matched hierarchical version as
  the companion panel; the caption's "raises ρ_F in every model … stay flat"
  needs the Qwen qualifier.
- **Fig. 2**: the absolute value is only visible in the colorbar label — say
  "\|Spearman\|" in the caption (08-14 §4) and, better, visually mark the
  provable-identity pairs (S_τ–H_sem, FI_out^fixed–H_sem, Var[FI_out]–H_sem)
  so "arithmetic vs empirical" is legible in the figure itself; the near-solid
  dispersion block is the reduction's own illustration.
- **Fig. 3**: no CIs on any AUROC bar; regenerate with decontaminated numbers
  (A1) + bootstrap CIs; the in-distribution bars (0.873) should carry the M12
  caveat (within-question contrast) in the caption.

### 4.7 Scope defense for ICLR *(new framing item)*

Three 7–8B models, one dataset, 150 questions, factoid QA — a standard ICLR
objection. No new data is possible, so the defense must be framing: (a) the
central contributions (the reduction proposition, the estimand definition, the
protocol with union-gold + guardrails, the dissociation *design*) are
dataset- and scale-independent; (b) the limitations section already says the
right things — add one sentence positioning the study explicitly as a
measurement-methodology paper whose empirical part is a worked demonstration,
not a survey of models; (c) expect and pre-empt "why no frontier models" with
the hidden-state requirement (probes + in-process decoding control need open
weights). Do not promise scale you cannot deliver.

---

## 5. Literature and novelty status (fresh web verification, 2026-08-16)

Two verification agents ran today; full details in their reports (§8 notes
method). Everything below is checked against live sources.

### 5.1 Novelty claims

| claim | verdict | action |
|---|---|---|
| "First use of functional information to evaluate LLMs" | **CLEAN** — nothing in arXiv/venues applies Szostak/Hazen FI to LLM evaluation | Keep (hedged, as now). Add a one-line distinction from "Functional Entropy" (arXiv 2605.28500, code-UQ — name collision only) |
| "No published prompt-sensitivity metric reports a paraphraser-swap ablation" | **CLEAN with two required footnotes** — PTEB (arXiv 2510.06730) compares three paraphrasers on *quality*, runs its evaluation with one, and names the swap as future work; AUGMENT (arXiv 2505.03563) uses two generators but pools them and draws no generator-dependence conclusion | Keep the claim; cite both as near-misses (quoting PTEB's limitation strengthens the gap). Verify AUGMENT §6/Figs 4–5 against the pooled reading when citing (PDF now in `papers/`) |
| "No formal reduction has been published" (gap 1) | **HOLDS** — chen2026position (confirmed ICML 2026 Position Track) is prose-only, touches none of the specific indices; Semantic Volume (2502.21239) and KLE *generalize upward*, not collapse downward | Keep, with the "informally argued, we supply the reduction" framing already in the text; add the upward-vs-downward positioning sentence the bib comments prescribe (nikitin2024kernel) |
| ρ_F estimand (per-question, gold-referenced success, noise-corrected) | **UNCLAIMED as of 2026-08-16** — a targeted sweep of May–Aug 2026 found no equivalent; Cox descendants (12 citing works, all checked) and BrittleBench (0 citations) don't occupy it | Keep the narrow conjunction wording; do not reintroduce estimator novelty (G-theory/MS_within is IR practice since 2007 — already correctly attributed to Urbano/Bodoff in the text) |
| Gap 3 ("specificity not manipulated as an IV") | Already conceded in-text via keluskar2024llms, contradicting the unqualified gap sentence 27 lines earlier (08-14 §4) | Narrow the gap sentence; cite `kim2025detail` (in bib, uncited) |

### 5.2 Bibliography: pending records now pinned (agent-verified from ACL Anthology / publisher pages)

Ready to paste into `references.bib` (all fields confirmed; corrections vs the
current comments **bolded**):

- **cegin2023chatgpt**: Cegin, Simko, Brusilovsky. "ChatGPT to Replace Crowdsourcing of Paraphrases for Intent Classification: Higher Diversity and **Comparable** Model Robustness." EMNLP 2023, pp. 1889–1905. DOI 10.18653/v1/2023.emnlp-main.117. → resolves TODO-LIT #2.
- **jiang2022investigating**: **Nan-Jiang Jiang** (one person) & de Marneffe. "Investigating Reasons for Disagreement in Natural Language Inference." TACL 10 (2022), pp. 1357–1374. DOI 10.1162/tacl_a_00523. → resolves TODO-LIT #1.
- **mccabe2025alphabet**: McCabe, Melamed, Hartvigsen, Huang. "Estimating Semantic Alphabet Size for LLM Uncertainty Quantification." arXiv:2509.14478 (still preprint). → resolves TODO-LIT #3, with **nguyen2025beyond**: Nguyen, Payani, Mirzasoleiman. "Beyond Semantic Entropy…" Findings ACL 2025, pp. 4530–4540. DOI 10.18653/v1/2025.findings-acl.234 (check the "small-k saturation" gloss against the paper before citing it that way).
- **bowyer2025clt**: **Bowyer, Aitchison, and Ivanova (three authors)**, "Position: Don't Use the CLT…" **ICML 2025, PMLR 267:81143–81184 (Spotlight)** — not a bare arXiv preprint.
- **huber2013programmatic**: POQ **77(1)** (not S1), pp. 385–397.
- **fitelson1999hownot**: Fitelson, **Stephens**, Sober. Philosophy of Science 66(3), pp. 472–488.
- **plank2022problem** pp. 10671–10682; **webergenzel2024varierr** pp. 2256–2269 (Weber-Genzel, Peng, de Marneffe, Plank).
- Venue updates for cited preprints: **kirchhof2025position** = ICML 2025 Position track, PMLR 267:81665–81677; **chen2026position** = accepted, ICML 2026 Position track; **taparia2026anatomy** = ICBINB Workshop @ ICLR 2026; **razavi2025benchmarking** = ECIR 2025, LNCS pp. 303–313, DOI 10.1007/978-3-031-88714-7_29; **zhang2025sparse** = EMNLP 2025 main, pp. 16081–16099 — and the mechanism is sparse **neurons**, not sparse autoencoders (check the paper's description; the related-work sentence says "internal activations", which is fine). **kunievsky2026measuring** changed title to "Measuring Intent Comprehension in LLMs" (v3). **messing2026hidden**, **zatuchin2026noise** (= arXiv:2607.13304, Żatuchin), **mehta2026format** (arXiv:2607.09665), **romanou2026brittlebench**, **pecher2026revisiting**, **staliunaite2026role**, **mustahsan2025stochasticity**, **tomov2025illusion** remain preprints (tomov's ICLR 2026 withdrawal is unconfirmed — don't assert it). **cencerrado2025noanswer** venue note stays accurate.

### 5.3 New papers worth engaging (PDFs added to `papers/` today)

Must-position (2): **PRIG** (arXiv 2606.05486, prompt-ambiguity probes +
attribution — the newest neighbor of the underspecification head; differentiate
as localization vs variance-prediction) and **PTEB** (above).
Should-cite as support (pick per page budget): Żatuchin 2607.13304 (already
your `zatuchin2026noise` — no action beyond the venue check), DBPA (arXiv
2412.00868, NeurIPS 2024 workshop — the hypothesis-testing alternative to
variance decomposition; currently uncited), ParaEval (2606.10657), judge-swap
auditing (2607.08535, instrument-relativity support), conditional G-theory
(2607.11981), clinical stability metrics (2605.30646 — three new dispersion
indices; one sentence noting they also fall under the reduction's scope
*if* they are cluster-distribution functionals — check before claiming),
Kostiuk & Enevoldsen (2605.22544, leaderboard instability one-liner).

---

## 6. The page plan: 22 → 9 pages of main text

Current build: Intro to p. 2, Background to p. 5, **Methods pp. 5–14**,
Results pp. 14–19, Discussion pp. 19–21, Conclusion p. 22. The cut is
feasible without losing any result because Methods carries ~5 pages of
material that is either historical, duplicated in the appendix, or never used
in Results.

| section | now | target | how |
|---|---|---|---|
| 1 Intro | ~1.7 | 1.0 | keep the two-source problem + RQ + contributions; cut the second worked example and the FI preview paragraph (it repeats §3.1) |
| 2 Background | ~3.3 | 1.5 | §2.2 (FI history) shrinks to one paragraph folded into Methods; related work keeps the variance-decomposition + underspecification groups as prose, compresses groups 1–3 to citation clusters; the three-gaps paragraph stays (narrowed per §5.1) |
| 3 Methods | ~9 | 3.5 | §3.1 → ½ page (Eq. 1–2, generator-relativity; Hazen/Corona/Wong lineage → appendix); §3.2 → 1¼ (cut ΔFI, compress FI_out to Eq. 8–9 + the fixed-m₀ paragraph, keep Table 1); §3.3 → 1¼ (levels+guardrails+union-gold decomposition tightened ~40%; paraphrase pipeline to one paragraph + appendix; width-dial design ½ col; Hazen restatements deduplicated — the review counts three); §3.4 → ½ (scoring/clustering/inference-family; conventions already in appendix); §3.6 probes → ¼ |
| 4 Results | ~5.5 | 2.5 | keep all five subsections and all tables (rebuilt per §2); cut the per-subsection narrative repetition and the italicized conclusion sentences (or shrink to one clause); islands paragraph → appendix; POSIX → two sentences + appendix table |
| 5 Discussion | ~2.3 | 0.75 | findings ¶ stops restating §4 numbers; contributions ¶ 3 sentences each; limitations keeps reference-class + design-scope + the three conceded threats, drops the rest to appendix |
| 6 Conclusion | ~0.7 | 0.25 | four sentences |
| **total** | **~22** | **~9** | |

Appendix gains (allowed, unlimited): FI lineage; the 14-metric enumeration
table (fixes M16/4.4); full probe tables; forking-paths table (fixes M13);
persona/prompt templates (fills the empty Appendix C); islands + POSIX
detail; per-cap AUFI table; moderator table (M1).
Figures: all three stay (fig2 at 0.6\textwidth); regenerate fig1b/fig3 after
§2 fixes.

Mechanical note: the cut interacts with the fixes — do the §2/§3 content fixes
*first*, then cut; otherwise you compress sentences you are about to rewrite.

---

## 6b. The abstract, sentence by sentence (where the fixes surface first)

The abstract is where a reviewer meets every overstatement; after Phase 1 it
must change as follows (the Conclusion mirrors several of these):

| current sentence (gist) | action |
|---|---|
| "…growing set of indices in incompatible units…" (framing) | keep |
| "…common ruler taken from functional information…" | keep |
| "fourteen candidate quantities resolve into three axes" | keep, backed by the M10 assignment-not-count wording in §4.1 |
| "Two of the three require no new metric… monotone transform of accuracy… semantic entropy up to a fixed offset" | keep — both identities are exact and verified |
| "five published dispersion indices are arithmetic restatements of the latter, agreeing to within 4.4×10⁻¹⁶" | **rewrite (A3):** e.g. "the dispersion indices we compute are functionals of one pooled clustering — two of them exact restatements of semantic entropy — so their mutual agreement is not convergent evidence" |
| "The third axis is measured by no existing index and needs the decoding noise subtracted…" | **rewrite (A4):** "…is isolated by no existing index once decoding noise is removed — the subtraction a single sample per prompt cannot supply"; report ρ_u/spread convergence in §4.4 |
| "disambiguating … raises competence by 6 to 13 accuracy points and leaves the phrasing share unchanged" | **soften (A5):** "…does not detectably move the phrasing share" + state the estimator's minimum detectable change once computed; drop any "properly powered" language |
| "widening the paraphrase generator raises the phrasing share and leaves competence and dispersion flat" | **hedge (A2/A6):** "raises the phrasing share — significantly in the most phrasing-sensitive model — while competence and dispersion show no comparable response" |
| "transfers … at an AUROC of 0.670 to 0.678, against matched text baselines of 0.545 to 0.587" | **requote (A1)** with the decontaminated values from the rerun (head ≈ 0.655–0.670; take the exact baseline band from the regenerated artifact) |

---

## 7. Recommended execution order

**Phase 0 — decisions (you):** D1 venue, D2 reproducibility, D3 papers/.

**Phase 1 — analysis reruns (2–3 days, all local, no cluster):**
A1 holdout decontamination rerun + Fig. 3 (§2.1) · A5 recovery-sim MDE +
draws-based test (§2.5) · A4 convergent + incremental validity (§2.3) · A2
five-seed N-matched MoM + unpaired means (§2.4) · M11 commensurable null ·
M5 5-vs-5 payoff · M7 commit Table-5 scripts + rerun on hierarchical + the
§4.5 cross-model number · M2 per-reading evidence coverage · M1 moderator
table. Each writes a committed artifact so every paper number re-traces.

**Phase 2 — the writing pass (2–3 days):** §2.2/§2.3 claim rewrites (abstract
first — every abstract sentence flagged here changes) · §2.4/§2.5 hedging ·
M3/M4/M6/M8/M9/M10/M12 · M14–M16 · §4.2 README rewrite · §4.3 preregistration
wording · bib updates from §5.2.

**Phase 3 — the cut (§6; 2 days).**

**Phase 4 — release hygiene:** D2 execution, figure-script de-hardcoding,
anonymization if ICLR, final `pdflatex` + check p. 30 TODO gone.

**If you do only four things** (unchanged from 08-14, still right): A1, A2,
A3, A4 — plus, from this pass, the 9-page constraint if the target is ICLR.

---

## 8. Verification log (what I checked first-hand today)

**Artifacts read in full:** stats_hygiene.md, width_dial_analysis.md,
probe_eval_hardened.md, metric_reductions.md, mechanical_null.md,
independence_union.md, seed_audit.md, factor_audit.md, axis1_graded_curve.md,
probe_redundancy_audit.md, final_run_results.md, FORKING_PATHS.md,
run_manifest.json; the full paper (main.tex, main_body.tex 1,563 lines,
appendix.tex, references.bib structure); REVIEW_2026-08-14 (991 lines);
REVIEW_2026-08-06 skim; SUPERVISOR_FEEDBACK; PAPER_PROPOSAL;
LITERATURE_REVIEW_VERIFIED (§6–§9 in full); both runbooks.

**Code read at the cited lines:** probe_eval_hardened.py (eval_ood 336–384,
_TARGETS 57–62, n_perm 179–216), errica.py (30–65), build_levels.py (184–200),
paraphrases/pipeline.py (345–365), constraint_filter.py (330–345), fi_in.py
(35–45, 94–100), width_dial_analysis.py (N-match loop 362, load_arm_cells
160–200), stats_hygiene.py (90–110), feedback/heads.py (labels 82, thresholds
236). Every code-level claim of the 08-14 review that I relied on reproduced
exactly.

**Recomputed from the migrated parquets (this machine, 2026-08-16; every one
reproduces the 08-14 review's numbers digit-for-digit):**

| check | result |
|---|---|
| Holdout contamination (§2.1 here) | 150 training questions; holdout 2,002; **overlap 150, all labelled ambiguous** |
| Dispersion indices diverge on the primary test (§2.2) | Qwen: H_sem p=.011 vs S_τ p=.213 · var-ratio p=.006 · 1−TVD p=.002 · \|A_q\| p=8.9e-5; Mistral: H_sem p=.034 vs var-ratio p=.483, 1−TVD p=.163, S_τ p=.057; Llama: all respond. Confirms "one clustering ≠ one measurement" |
| ρ_F(hier) ↔ ρ_u (Cox) per stratum (§2.3) | +0.455/+0.240 (Qwen L0/L1), +0.432/+0.529 (Llama), +0.517/+0.505 (Mistral); ↔ spread +0.213/+0.175, +0.244/+0.370, +0.282/+0.339 — vs ↔ accuracy ≤\|0.16\| and ↔ H_sem ≤\|0.22\|. The "empty axis" is occupied by the two published indices |
| Width dial paired vs unpaired MoM (§2.4) | paired n = **22**/54/46; paired means = Table 4's exactly; unpaired per-arm means: Llama **0.098→0.131→0.114 (non-monotone)**, Qwen 0.334→0.389→0.395, Mistral 0.195→0.197→0.226; per-arm coverage Qwen 32/42/40 |
| Table 5 provenance (M4) | union coverage **59.7/82.7/77.7%** (paper prints target 45.3/66.0/57.3 unlabelled); Spearman(MoM, hier) union 0.903/0.945/0.848, target 0.867/0.893/0.856 — the printed "0.64–0.72" reproduces from **neither**; σ²_B union 0.0374/0.0179/0.0283 vs target 0.0307/0.0143/0.0226 (paper prints target unlabelled); hier means match Table 5's labelled rows |
| FI_out^fixed negative rates (M8) | 2.3% / **31.7%** / 23.3% of cells (L0: 2.7/**42.0**/26.7%), minima −1.21/−2.07/−2.16 |
| Zero paired ρ_F deltas (§2.5) | 41/22/37 of 150 exactly zero → Wilcoxon effectively runs on n = 109/128/113 |
| Question lengths (M15) | L0 46.4 → L1 66.5 chars (+43%) |
| Hierarchical-estimator recovery sim (§2.5) | Re-run independently, 3 seeds, pipeline's own `fit_hierarchical_rho_f`: true Δρ_F **+0.20** reports as **+0.040/+0.040/+0.043** (Qwen), **+0.074/+0.097/+0.080** (Llama), **+0.047/+0.045/+0.036** (Mistral); true **+0.10** reports as +0.015–0.029 / +0.038–0.041 / +0.017–0.040. The "0.04" bound is a statement about the shrunken statistic; on the estimand scale it corresponds to roughly **+0.2 (Qwen, Mistral)** and **+0.10–0.15 (Llama)**. Confirms 08-14 §2.5 (Llama attenuates less in my seeds than in theirs — its null is the most informative of the three; the conclusion is unchanged) |
| Cross-model ρ_F transfer (§4.5) | Paper prints "0.2 to 0.45"; committed artifacts support: hierarchical union pooled **+0.162/+0.117/+0.225**, within-level 0.12–0.29, complete-case MoM 0.23–0.29, target-gold hier 0.10–0.28. **No variant reaches 0.45.** The qualitative claim (weak transfer → question×model trait) survives and is if anything stronger; the printed range needs replacing with a scripted number |
| Decontaminated holdout AUROC (§2.1) | Frozen committed bundles, my exclusion mirroring `eval_vagueness_holdout.py`: contaminated n=2,002 → **0.678/0.670/0.678** (exactly the paper's numbers); decontaminated n=1,852 → **0.667/0.655/0.670** (exactly `final_run_results.md`). The A1 story is verified end-to-end |

**Code fixes already applied (working tree, uncommitted):** the `eval_ood`
exclusion in `probe_eval_hardened.py` (+ explanatory comment) and the
`_EPS = 1e-9` threshold guard in `metrics/fi_in.py` (+ a regression test in
`tests/test_fi_in.py` that fails before the fix). **Full test suite passes: 377
tests, 0 failures.** Not done yet (waits for your go, since it rewrites a
committed artifact + a paper figure): regenerating `data/probe_eval_hardened.md`
and Fig. 3 via the fixed script, and an integration test for `eval_ood`
(needs synthetic hidden-state fixtures).

**Web-verified (two agents, live sources, 2026-08-16):** ICLR 2027 CFP +
Author Guidelines (page limit, deadlines, statements, double-blind); the five
novelty claims of §5.1; every pending bib record and every 2026 venue status
of §5.2; the Cox/BrittleBench citation graphs.

**Cross-checks:** bib 76 entries / 61 cited / 15 uncited (matches 08-14);
paper repo in sync with Overleaf remote (no unpulled edits); OneDrive Code/
and Paper/ git-identical to the live repos (inventory agent, git-verified);
data/ + caches + decks + logs migrated (505 MB; `docs/MIGRATION_2026-08-16.md`).
