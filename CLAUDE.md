# CLAUDE.md — project context for coding agents

## What this is

Measuring **LLM prompt sensitivity in bits** on a functional-information
ruler. KIT/KSRI seminar project by Thanos Drossos (advisor Moritz Diener,
examiner Prof. Satzger), now being reframed as an **ICLR 2027 submission with
additional contributors**.

The seminar paper was **submitted 2026-08-26**. Data collection finished in
August 2026 on bwUniCluster and **no further cluster runs are planned** — the
remaining work is the ICLR reframe and whatever analysis it needs, all of which
re-runs locally from the committed parquets.

## The frame — use this vocabulary

The published paper measures **three axes** and manipulates **two variables**:

| | what it is | how it is measured / set |
|---|---|---|
| **Competence** | how often the model is right | graded accuracy; the FI_in(q,θ) curve |
| **Formulation sensitivity** | how much of success is decided by *which phrasing* | **ρ_F**, a one-way ICC with decoding noise subtracted; absolute σ²_B alongside |
| **Output dispersion** | how scattered the answers are | **H_sem** (the sole representative) |
| **Specificity** — *content variable*, manipulated | how narrowly the question fixes what is asked | **FI_spec** = log₂(m₀/m_valid), from annotations only |
| **Width** — *form variable*, manipulated | how broadly the wording varies at fixed meaning | generator configuration: narrow / production / wide (+ swap arm) |

Terminology that is **retired** and must not come back: "three axes + one
dial" and the word *dial* generally (say *the specificity intervention* /
*the width intervention*); FI_spec as a fourth measurement (it quantifies the
setting of a manipulated variable, and its bit scale is **not** a validated
dose); "reliability probe" (the four probes are **underspecification,
dispersion, fragility, correctness**); "vagueness" for the underspecification
probe or its holdout. A **cell** is one question at one specificity level for
one model.

## Where truth lives (in order of authority)

1. **The submitted paper** (`../SensitivityFunctionalInformationPaper`, branch
   `main`) — every number in it was traced to an artifact in the 2026-08-27
   audit.
2. `data/stats_hygiene.md` (+ `.json`) — the declared 12-test primary family.
   Then `data/metric_reductions.md`, `width_dial_analysis.md`,
   `probe_eval_hardened.md` + `probe_eval_ood.json`, `rho_f_recovery_sim.md`,
   `rho_f_construct_validity.md`, `independence_{union,target}.md`,
   `evidence_coverage_audit.md`, `mechanical_null.md`,
   `{factor,seed,probe_redundancy}_audit.md`, `data/run_manifest.json`.
3. `FORKING_PATHS.md` — the 13 analysis forks and which results depend on them.
4. `docs/PROJECT_STATE_2026-08-27.md` — current status snapshot.
5. `docs/design/Section_7_Functional_Information_for_Prompts.md` (formulas) and
   `docs/design/Research_Design_v2_Specificity_FI.md` (design) — historical
   authorities; anything v3–v6 is superseded.
6. `docs/literature/LITERATURE_REVIEW_2026-08-07_VERIFIED.md` — citation source
   of truth. `references.bib` in the paper repo keeps per-entry verification
   comments; preserve them when editing.

Everything under `docs/reviews/`, `docs/results/`, `docs/talks/` and
`docs/archive/` is a **dated record of a past moment**. Read them as history,
do not update them, and do not treat their claims as current.

## Corrections from the 2026-08-27 numbers audit

About 360 quantitative claims were traced to artifacts and adversarially
re-verified; 13 corrections landed in the paper. Older documents and drafts
still carry the superseded versions, so watch for these:

- The **|ρ| ≤ 0.14 cross-axis bound applies only to ρ_F** against the other two
  axes, under the hierarchical estimator. Competence and output dispersion are
  clearly negatively associated within strata (Spearman −0.34 to −0.72, union
  gold). The old blanket "no cross-axis association exceeds 0.14 / equivalence
  bounds only to 0.34" claim is **wrong**.
- The grid is 900 cells but **89,730 scored responses**, not 90,000: one
  universe retains a single paraphrase, so it is "up to ten paraphrases".
- The **Qwen > Mistral > Llama** ordering holds under both gold sets and both
  estimators on the share and on σ²_B, and survives the generator swap **on the
  share only** — under the swap generator σ²_B reorders to Mistral > Qwen >
  Llama.
- ρ_F specificity null: p ≥ .41 union gold, p ≥ .22 target-gold replication;
  Rubin bracket **±0.08**; a true change of 0.20 is **more than half** the
  between-model range, not larger than all of it.
- Width arm: competence CIs within **4** accuracy points (max +0.033).
- Spread beats or matches ρ_F at predicting rephrasing payoff in **2 of 3**
  models (k=20) and in all three on disjoint paraphrase sets — the disjoint
  check does **not** reproduce the same per-model ordering.
- Chao1 was dropped because at k=10 the correction is dominated by singleton
  noise, **not** because it is "undefined without doubletons".
- Probe numbers are the macOS reproduction: underspecification 0.85–0.87
  in-distribution, **0.655–0.670 zero-shot** on the decontaminated holdout
  (n = 1,852) vs frozen text baselines 0.543–0.571; fragility is a clean
  negative in all three models.

## Practical notes

- Python via `uv sync --extra app --extra dev` (both extras, or ruff/pytest
  disappear); run modules as `uv run python -m prompt_sensitivity.scripts.<name>`.
  Tests: `uv run pytest -q` — **399 tests**, CPU-only except one gated NLI test.
- 50 result parquets and 18 `data/*.md` artifacts are **committed**; the large
  probe inputs (`hidden_states_*`, `vagueness_holdout_*`, ~450 MB) and
  `papers/` are gitignored. `data/cache/llm_cache.sqlite` is only the June
  smoke cache — the full final-run cache is cluster-only, so anything needing
  new model calls is cluster work; everything else re-runs locally.
- Analysis scripts have no Makefile targets; invoke modules directly.
- Paper figures: `uv run python -m prompt_sensitivity.scripts.make_paper_figures
  --out ../SensitivityFunctionalInformationPaper/1_Figures`. The script reads
  committed artifacts only — there are no hardcoded numbers, so never
  hand-patch a figure.
- `data/*.md` artifacts are **script-generated**: fix the generating script,
  not the markdown, or the next run reverts the edit.
- `cluster/runbooks/*` contain the bwUniCluster username — scrub before making
  the repo public (relevant now that the repo link goes into a submission).

## The paper repo and Overleaf

`../SensitivityFunctionalInformationPaper` syncs with Overleaf on **`main`**
(no `master`). **Always `git pull` before editing the tex** — Thanos edits in
Overleaf between sessions. When both sides change, Overleaf pushes an
`overleaf-YYYY-MM-DD-HHMM` branch and asks for a manual merge: diff it against
its merge base, merge keeping the newer revision plus any genuine Overleaf
edit, compile with `tectonic main.tex`, push `main`, delete the remote branch,
and tell Thanos to pull in Overleaf. Grep for `<<<<<<<` before committing any
merge.

## ICLR 2027 (re-verified 2026-08-27)

**9 pages** of main text at submission, 10 at rebuttal; over-limit is a desk
reject. **Abstract 2026-09-18 AoE, paper 2026-09-25 AoE.** Double-blind; the
AI-use statement is mandatory and excluded from the limit, as are the optional
reproducibility and ethics statements.

The current build is the **non-anonymous seminar version**: 24 pages total with
content on pp. 1–13. Reaching submission needs ~4 pages of content cut,
`\iclrfinalcopy` commented out (which restores anonymity and line numbers),
removal of the author block, matriculation number, seminar header, advisor and
examiner names and submission date, and an anonymized repo link in the
reproducibility statement. The cut plan in
`docs/reviews/REVIEW_2026-08-16_Consolidated_Action_Plan.md` §6 predates the
two-variable reframe — treat it as input, not as the plan.
