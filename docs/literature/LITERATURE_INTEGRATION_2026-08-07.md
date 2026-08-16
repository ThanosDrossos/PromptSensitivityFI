# Literature → research integration (2026-08-07, afternoon)

Companion to `LITERATURE_REVIEW_2026-08-07_VERIFIED.md` (the evidence) and
`PAPER_PROPOSAL_2026-08-07.md` (the claim structure). This file records what was **changed today**
in response to the verified literature, and what remains as recommendations with owners.

Everything below was done after reading the post-R1–R3 state (`RESULTS_R1_R2_R3_2026-08-07.md`,
`data/independence_{target,union}.md`, `data/metric_reductions.md`, `data/axis1_graded_curve.md`,
the corrected `EXPLAINER` and `final_run_results.md`).

---

## 1. APPLIED — new analysis: the mechanical-coupling null (Gulliford check)

**Finding applied:** ICC on binary outcomes is mechanically coupled to outcome prevalence
(Gulliford et al. 2005; no NLP paper accounts for it). This is the rival explanation for the
complete-case ρ_F ~ accuracy associations that R3 could not bound below |ρ| < 0.42.

**What was built:** `prompt_sensitivity/scripts/mechanical_coupling_null.py`
(+ `tests/test_mechanical_coupling_null.py`, 5 tests, passing; `metrics/` untouched — the estimator
is re-implemented vectorised and equivalence-tested against `sensitivity_v2.rho_f` to 1e-12,
including the NaN rule).

Protocol: hold each cell's mean success rate at its observed value, set the TRUE ρ to one constant
for all cells (grid 0.05/0.2/0.5 + each model's hierarchical population mean), simulate the full
10×10 grid from the beta-binomial, re-estimate, and compare the observed complete-case
Spearman(ρ̂_F, accuracy) against the null band. Seeded, 500 sims per combo.

**Result (`data/mechanical_null.md` / `.parquet`):**

> **24/24 observed complete-case correlations sit inside the 90 % mechanical null band** — including
> llama L0's +0.37, the value the adversarial review treated as the orthogonality failure. At
> llama's own population ρ = 0.13, mechanics alone produce a null band of [+0.12, +0.49] at L0.
> The coverage-selection artifact is also reproduced by mechanics (null coverage matches observed
> within a few points everywhere). Recomputed per-level coverages (.37/.54, .54/.79, .47/.68)
> reproduce the review's pooled 45/66/57 % exactly.

**What this changes for the paper:** the complete-case ρ_F ~ accuracy association is
**uninformative about the constructs at this design size** — in both directions. It neither
supports orthogonality nor contradicts it; it is the size the estimator produces on its own. This
(a) retires the last live use of the review's imputed-0 counter-numbers, (b) gives the hierarchical
estimator a second, independent justification, and (c) is a small method contribution in itself:
first application of the prevalence-coupling null to a prompt-sensitivity metric. Wired into
`main_body.tex` §ρ_F (see §3 below).

---

## 2. APPLIED — bibliography (`references.bib`)

- **Fixed a blocking defect:** `mustahsan2025stochasticity` carried **four fabricated given names**
  behind a `% VERIFIED` marker (Mohammad/Sungjin/Ankit/Sarthak → Zairah/Abel/Megna/Saahil; only
  McCann was right; re-verified against the arXiv abs page first-hand). Also cs.CL → cs.AI.
- **Added GROUP H** — 23 new entries whose fields were verified during the sweep, each with the
  provenance and a usage note: Corona 2010 · Root-Bernstein 2024 + Wong reply · Elsberry & Shallit ·
  Dembski & Marks (caution-flagged) · Urbano 2013 · Bodoff & Li 2007 · Bayerl & Paul 2007 ·
  Messing · Żatuchin · Kunievsky & Evans · Fel (Sobol) · Kirchhof ICML 2025 · Pecher · Zhang
  EMNLP 2025 · Cole 2023 (Gillick, not "Bulian") · Malaviya TACL 2025 · Keluskar · Staliūnaitė ·
  Hou ICML 2024 · Taparia · Kobalczyk (first author verified, list TODO) · Chen ICML 2026 Position ·
  Nikitin KLE · Razavi ECIR 2025 · Yang ICML 2026.
- **Added a PENDING block** (fully commented, cannot render): Gulliford 2005 (record unpinned!),
  Jiang & de Marneffe TACL 2022, Bowyer & Aitchison, Cencerrado, McCabe, Tomov, Cegin 2023,
  Plank 2022, VariErr, Fitelson 1999 (pages), Huber & Paris (pages). **Complete these fields from
  primary sources, then uncomment.**
- **Build verified:** bibtex + 2× pdflatex exit 0, no undefined citations.

## 3. APPLIED — `main_body.tex` (surgical; compiles clean)

| where | change |
|---|---|
| §Formalization | Wong-law "contested" note (Root-Bernstein + reply); **Corona 2010 finite-U formulation** adopted explicitly (the R5 FI^G reframing is now the *established* formulation, not an ad-hoc dodge); reference-class relativity named with the peer-reviewed citation (Elsberry & Shallit) and pointed at Limitations |
| §Related work | G-theory lineage paragraph (Bodoff 2007 → Urbano 2013 → Bayerl & Paul 2007 → Messing/Żatuchin/Kunievsky/Taparia); new **fifth group: underspecification** (Kirchhof, Pecher, Hou, Keluskar, Cole, Staliūnaitė); gap 1 now cites Chen (ICML 2026 Position) as the informal precedent for the one-construct thesis; gap 2 restated honestly — "the estimator is standard (Urbano); what is not available is its application to gold-referenced task success, per question, with the paraphrase set as the grouping factor" |
| §ρ_F | contribution clause corrected: **neither the decomposition nor the estimator** (standard since Bodoff/Urbano) — the **estimand**; Sobol-index positioning sentence (Fel); new paragraph handling the two binary-outcome pathologies: the **mechanical null result** (§1 above) and Urbano's small-sample instability, with the consequences (aggregate claims only, no per-question point claims) |
| §H_sem | H_sem framed as **one instantiation** of dispersion, special case of KLE (Nikitin) |
| §Sampling | k = 10 justified by the ICC convergence range n ≈ 8–16 (Mustahsan) — "adequate rather than generous" |
| §Clustering | Chao1 removal defended properly (undefined without doubletons); |A_q| and H_sem stated as **lower bounds at k = 10**, with the direction-of-bias argument that cross-level comparisons are conservative; TODO-LIT for McCabe/Nguyen cites |
| §Levels/guardrails | union-gold protocol positioned in the verified literature: any-match is the field's "common practice" (Staliūnaitė, Cole), Keluskar is the single-gold precedent that never states its selection rule (our seeded hash is more explicit), Malaviya's "ill-posed" argument (correct word, not "invalid") |
| §Paraphrase universes | three instrument bounds: paraphrase = mildest perturbation family ⇒ FI_in is a **lower bound**, effects conservative (BrittleBench); NLI-on-questions = high-precision heuristic, ill-defined in principle (TODO-LIT Jiang & de Marneffe); **diversity-collapse hedge dropped** (Cegin 2023 finds LLM paraphrases MORE diverse — the opposite); paraphraser-swap ablation noted as a field-wide gap |
| §Probes | full prior-art positioning: Kossen (dispersion head, same token position), **Zhang EMNLP 2025 (ambiguity head, on AmbigQA, with transfer)**, Razavi (the task name); the surviving claim narrowed to the **paraphrase-distribution targets** (ρ_F / surviving-fraction heads) + the zero-shot transfer evaluation |
| §Limitations | TODO block rewritten with the 9 literature-anchored limitations, ordered by severity, each with its citation status |

## 4. RECOMMENDED — decisions for Thanos (not applied; they change claims or need compute)

1. **C2's novelty sentence must lose the word "estimator."** `PAPER_PROPOSAL_2026-08-07.md` §3 C2
   says "…with an estimator that works." Keep that phrasing only if it means the *hierarchical*
   estimator; the MoM formula itself is textbook (Urbano eqs. 1–3). Suggested wording: *"the
   standard variance-components estimator, applied to a new estimand, with a hierarchical version
   that is defined everywhere."* Also update §6 risk 3 — the literature verification is now DONE;
   verdicts: Żatuchin real-but-different-estimand (apparatus taken, estimand safe), Pecher
   real-and-scoops-the-prose-thesis (differentiate: T=0, instruction-side, classification).
2. **ρ_u name collision needs one explicit sentence** where ρ_F is introduced: Cox et al. call ρ_u
   "the prompt sensitivity ratio," per question. The Methods paragraph already distinguishes them
   technically; add the naming sentence so a reviewer cannot claim surprise. (Renaming ρ_F is the
   alternative; not recommended this late.)
3. **Metric-collapse contribution: read LM-Polygraph before asserting the negative.** The claim "no
   published study correlates prompt-sensitivity indices" survives every check EXCEPT LM-Polygraph
   (arXiv:2406.15627, TACL 2025), which could not be fetched (no HTML build, MIT Press 403).
   Download the PDF, read the results section, then either keep the clean negative or add one
   sentence positioning. **This is the one loose thread under the paper's cleanest contribution.**
4. **Reconsider R6 (second paraphrase generator), demoted from "robustness check" to
   "contribution".** No published prompt-sensitivity metric has a paraphraser-swap ablation (PTEB
   lists it as open future work). One rule-based or different-family generator arm on a 30-question
   subset would be the first such ablation in the literature. Cost: one cluster run.
5. **Good-Turing robustness pass for |A_q| / H_sem** (answers McCabe/SENECA/SHADE/Nguyen). Blocked
   locally: raw cluster assignments are not persisted in the v3 parquets. Feasible as an R1-style
   cache-only job on the cluster (re-cluster the cached generations, emit f1 singleton counts,
   apply |S|^GT = kn/(n−f1)). Cost: one cluster job, no generation. Alternative: keep the
   lower-bound + conservative-direction argument now in Methods (§3) and defer.
6. **Probe deliverable (C3) framing:** lead with the FI_PROBES P1 target (predicting the paraphrase
   universe's sensitivity from one member) — it is the only probe target with no published
   counterpart (Zhang do not link ambiguity to answer-side quantities; Kossen's App. A.12 already
   tried and dropped continuous SE targets; Cencerrado predicts single-answer correctness). The
   honest OOD comparator is Cencerrado's 0.53–0.88 band, where 0.667 sits mid-band — use that,
   never the QPP correlation literature (different statistic).
7. **Deck/talk-track sweep for the four ChatGPT-artifact mis-attributions** ("Bowen et al.",
   "Renduchintala et al.", "Lu et al. 2023", "Nitsure et al."): `Research_Design_v3`,
   `Research_Synthesis`, and `Section_7` say "Polo-Nitsure" / "Polo, Nitsure et al." — the
   PromptEval author list contains **no Nitsure** (verified). Those are superseded design docs, so
   fix on next touch; nothing in `references.bib` or `main_body.tex` is affected.
8. **Intro framing gift:** Kirchhof et al. (ICML 2025) name underspecification uncertainty as
   research direction #1 and cite AmbigQA's ~half-ambiguous figure. Exactly one citing work
   operationalises it (Matsnev, self-report scalar). The intro can open on that gap and claim:
   *first external, model-free measurement model answering that call.* (Do not claim "first to
   answer the call" unqualified.)
9. **Optional one-liners now available:** Kostiuk & Enevoldsen's "any model can be promoted to
   first place by choosing prompts favorably" (embedding models — cite as analogous); Bowyer &
   Aitchison + Messing as the two-citation case against naive CLT error bars at n = 149 (pending
   verification of Bowyer).

## 5. What was deliberately NOT done

- No changes to `metrics/` (contract), no re-generation, no new model calls — everything today is
  re-analysis of persisted data.
- No prose written into Results/Discussion (Thanos's sections; the TODO scaffolding now carries the
  literature anchors).
- No entry added to the bib whose fields were not verified this session (they are in the commented
  PENDING block instead).
