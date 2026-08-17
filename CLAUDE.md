# CLAUDE.md — project context for coding agents

## What this is

KIT seminar → ICLR 2027 paper project (Thanos Drossos, advisor Moritz Diener,
examiner Prof. Satzger): **measuring LLM prompt sensitivity in bits** via a
functional-information ruler. Three measured axes — competence (FI_in/accuracy),
formulation sensitivity (ρ_F, a noise-corrected ICC over paraphrases), output
dispersion (H_sem) — plus one manipulated variable (FI_spec, question
specificity on AmbigQA L0/L1). Data collection is **complete** (bwUniCluster,
3×7–8B models, 150 questions × 2 levels × 10 paraphrases × 10 samples + width/
swap/POSIX/k20/holdout arms). **No further cluster runs are planned.** The
remaining work is analysis fixes and the paper.

## The three repos/locations

| location | role |
|---|---|
| this repo (`PromptSensitivityFI`) | code, data, analysis, docs — the working repo |
| `../SensitivityFunctionalInformationPaper` | the paper (ICLR 2027 tex, synced with Overleaf via GitHub — never edit without pulling first) |
| OneDrive `…/Backup KIT/Claude/Seminar Prompt Sensitivity Metric` | frozen Windows snapshot (2026-08-14), **read-only**; fully migrated 2026-08-16 (see `docs/MIGRATION_2026-08-16.md`) |

## Where truth lives (in order of authority)

1. `data/stats_hygiene.md` — the declared 12-test primary family; source of
   truth for Results numbers. Then: `data/metric_reductions.md`,
   `data/width_dial_analysis.md`, `data/probe_eval_hardened.md`,
   `data/independence_{union,target}.md`, `data/mechanical_null.md`,
   `data/{factor,seed,probe_redundancy}_audit.md`, `data/run_manifest.json`.
2. `FORKING_PATHS.md` — the 12 analysis forks and which results depend on them.
3. `docs/reviews/REVIEW_2026-08-14_Paper_ICLR.md` — the adversarial review of
   record for the paper (says explicitly to prefer it over older result docs).
4. `docs/design/Section_7_Functional_Information_for_Prompts.md` (formulas) and
   `docs/design/Research_Design_v2_Specificity_FI.md` (design) — historical
   authorities; anything v3–v6 is superseded.
5. `docs/literature/LITERATURE_REVIEW_2026-08-07_VERIFIED.md` — citation source
   of truth (four-tier verification). `references.bib` in the paper repo keeps
   per-entry verification comments — preserve them when editing.

2026-08-16 evening: the review's fix list is **implemented** on branch
`fix/review-2026-08-16` in both repos (code: new analyses R-series++, probe
decontamination, committed parquets; paper: rewritten to ~10.7 content pages
with every claim artifact-backed). `README.md` is rewritten to the corrected
frame; `data/probe_eval_hardened.md` and `data/final_run_results.md` now
agree (.667/.655/.670, n = 1,852, macOS reproduction).
`CODEBASE_WALKTHROUGH.md` mechanics are right, framing predates the axes
pivot. Neither branch is merged or pushed — that is Thanos's call.

## Practical notes

- Python via `uv sync --extra app --extra dev`; run things as
  `uv run python -m prompt_sensitivity.scripts.<name>`. Tests: `uv run pytest -q`
  (376 tests, CPU-only except one gated NLI test).
- All parquets/caches in `data/` are **gitignored**; they exist locally (copied
  from OneDrive) and on the cluster. Do not commit them without deciding the
  reproducibility strategy first; never regenerate committed `data/*.md`
  artifacts without saying so.
- `data/cache/llm_cache.sqlite` is only the June smoke cache — the full
  final-run LLM cache lives on bwUniCluster. Anything needing new model calls
  or cache re-scores is cluster-only; **everything else re-runs locally from
  the parquets.**
- Analysis scripts have no Makefile targets; invoke modules directly.
- Paper figures: `uv run python -m prompt_sensitivity.scripts.make_paper_figures
  --out ../SensitivityFunctionalInformationPaper/1_Figures` (beware: fig1/fig3
  numbers are hardcoded literals inside the script; fig2 reads
  `figures/v3_metric_corr.npy`).
- ICLR 2027: **9 pages main text at submission** (10 at rebuttal), desk-reject
  over limit; abstract deadline 2026-09-18, paper 2026-09-25 AoE; double-blind
  (the current tex is the non-anonymous seminar build).
- `cluster/runbooks/*` contain the bwUniCluster username — scrub before making
  the repo public.
