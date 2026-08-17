# Project state — 2026-08-16 (post-migration snapshot)

One-page orientation for anyone (human or agent) joining now. Details:
`CLAUDE.md` (conventions), `docs/MIGRATION_2026-08-16.md` (what moved where),
`docs/reviews/REVIEW_2026-08-14_Paper_ICLR.md` (paper review of record).

## Timeline so far

- **Mar–Jun 2026** — metric design eras (ladders, MuSiQue, HotpotQA) — all superseded.
- **Jul 2026** — AmbigQA specificity pivot (`REBUILD_PLAN_AmbigQA_Specificity.md`); v3 design frozen.
- **2026-08-02/03** — final cluster run complete (3 models × 150 q × 2 levels × 10 paraphrases × 10 samples, + POSIX, k=20 extension, vagueness holdout, evidence dial [withdrawn]); KSRI deck.
- **2026-08-06** — first adversarial review (4/10) → work programme R1–R11.
- **2026-08-07/08** — R1–R9 executed on the cluster + locally: union-gold rescore, hierarchical ρ_F, reductions, width dial + paraphraser swap (P0–P5 preregistered in runbook), hardened probes, stats hygiene, reproducibility manifest.
- **2026-08-09** — KSRI seminar talk delivered (deck in `decks/`, track in `docs/talks/`).
- **2026-08-11** — supervisor feedback → factor/seed/probe-redundancy audits.
- **2026-08-12** — paper rewritten as a measurement-framework paper (ICLR 2027 format, 30-page build).
- **2026-08-14** — second adversarial review (`REVIEW_2026-08-14_Paper_ICLR.md`): 4/10, five reject-level findings, A1–C5 fix list. **None of it is implemented yet** — the paper tex is untouched since 08-12; the code repo HEAD (f76833f) is exactly the state that review examined.
- **2026-08-14 → 16** — Windows → Mac migration; OneDrive snapshot frozen; all data artifacts + docs transferred into this repo (2026-08-16).

## Current status by area

| area | status |
|---|---|
| Data collection | **Complete.** No more cluster runs planned. All 81 result parquets + hidden states + holdout dumps local under `data/` (gitignored). Full LLM cache is cluster-only; local cache = June smoke. |
| Analysis code | HEAD f76833f + two uncommitted 2026-08-16 bug fixes: probe `eval_ood` holdout exclusion (review §2.3) and the `fi_in.py` grid epsilon (§4), both with tests — suite passes 377/0. Still missing: the N-matched MoM/5-seed pass for the width table (§2.4c) and the artifact/figure regeneration behind the eval_ood fix (awaits triage). |
| Results of record | `data/stats_hygiene.md` + the R-series `data/*.md` (all committed). Headlines: Δ_union +.064/+.125/+.119 (BH-sig 3/3, Holm 2/3, pooled p=3.3e-5); ΔH_sem −.124/−.433/−.139 (Holm-robust in Llama); Δρ_F ≈ 0; width dial: σ²_B up 3/3 (p .021–.060), ρ_F MoM monotone 3/3 (Qwen p=.0035); swap arm: structure preserved (+.41/+.27/+.51 per-cell, ordering intact); probes: vagueness .873 in-dist / .655–.670 OOD decontaminated (.670–.678 contaminated numbers currently in the paper), fragility head = null. |
| Paper | `../SensitivityFunctionalInformationPaper`, branch `fix/review-2026-08-16` (2026-08-16 evening). **All 08-14 review findings implemented** (minus the Horn-dedup print, per Thanos): rewritten body, artifact-backed numbers, ~10.7 content pages (22 total with statements/refs/appendix), 0 unresolved refs. Seminar identity kept. ICLR 2027 would additionally need: ≤9 pages, anonymization, de-identified repo link. |
| Literature | Verified through 2026-08-16: novelty claims re-checked clean (FI-first; swap-first needs PTEB/AUGMENT footnotes); pending bib records all pinned (see the 2026-08-16 consolidated review); 10 new PDFs added to `papers/`. |
| Reviews pipeline | 08-06 review → R1–R9 all landed. 08-14 review → **open**, consolidated with fresh findings in `docs/reviews/REVIEW_2026-08-16_Consolidated_Action_Plan.md` (the working to-do for the paper revision). |

## The three claims the paper currently makes (abstract), and their real status

1. "Five published dispersion indices are arithmetic restatements … 4.4e-16" —
   overstated as written; survives as a *scoped* reduction proposition (review §2.1).
2. "The third axis is measured by no existing index" — contradicted by the
   paper's own ρ_u/spread correlations; survives as "no existing index measures
   it *with the sampling term removed*", with convergent validity reported (§2.2).
3. Probe transfers zero-shot at .670–.678 — contaminated holdout; the honest
   decontaminated numbers are .655–.670 and the margin over baselines survives (§2.3).

Plus the double dissociation (§2.4/§2.5): direction correct, both halves need
the disclosed-estimator/power repairs before they carry the weight the text
puts on them.
