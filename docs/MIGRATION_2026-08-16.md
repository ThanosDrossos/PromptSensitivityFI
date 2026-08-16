# Windows → Mac migration record (2026-08-16)

What was transferred from the OneDrive snapshot
(`OneDrive-Persönlich/Dokumente/01-KIT/Backup KIT/Claude/Seminar Prompt Sensitivity Metric`,
taken on the Windows machine 2026-08-14 ~18:20, i.e. **after** the last commit
`f76833f` and after `REVIEW_2026-08-14_Paper_ICLR.md` was written) into this
repo. OneDrive was treated read-only throughout.

## Verified before copying

- `Code/PromptSensitivityFI` in the snapshot is **git-identical** to this repo
  (same HEAD `f76833f`, clean tree, no stashes; file diffs are CRLF-only).
- `Paper/SensitivityFunctionalInformationPaper` is git-identical to the live
  paper repo (HEAD `4303886`, clean).
- Therefore nothing code- or tex-side needed transfer. Everything that did need
  transfer is content `.gitignore` excludes — i.e. the experimental data — plus
  root-level project documents that were never inside either repo.

## Copied (destination ← source)

| destination | contents | git status |
|---|---|---|
| `data/` | **81 result parquets (~452 MB)** incl. `specificity_v3_*`, `union_gold_*`, `rho_f_hier_*`, `width_*`, `posix_arm_*`, `sensitivity_v2_k20_*`, `hidden_states_*` (~90 MB each), `vagueness_holdout_*` (~60 MB each), `width_dial_cells`, `mechanical_null`, plus `data/cache/llm_cache.sqlite` (43 MB LLM response cache — required for any cache-only re-score), `data/plots*/`, `data/slides/`. Copied with `--ignore-existing`: git-tracked files were never overwritten. | gitignored (invisible) |
| `.env` | the project env file (absent from the fresh clone). Contents not inspected (secret); expected variables per `.env.example`: `LITELLM_API_KEY` (gateway was retired 2026-06-23, key may be dead), `LITELLM_BASE_URL`, `HF_TOKEN`, `SPEND_LIMIT_USD`. | gitignored |
| `docs/reviews/` | `REVIEW_2026-08-14_Paper_ICLR.md` (review of record, 4/10, A1–C5 fix list) · `REVIEW_2026-08-06_Adversarial.md` + `_Appendix_Recomputations.md` (defined R1–R11) · `SUPERVISOR_FEEDBACK_2026-08-11.md` · `TODO_Review_Followups.md` (ledger mapping the 08-06 review to commits) · `main_2026-08-12_reviewed_build.pdf` (the exact 30-page compiled paper the 08-14 review examined) | untracked |
| `docs/results/` | `RESULTS_R1_R2_R3_2026-08-07.md`, `RESULTS_R6_width_dial_2026-08-08.md`, `RHO_F_CONSTRUCT_VALIDITY_2026-08-07.md` (narrative result write-ups; the committed `data/*.md` files remain the tabular source of truth) | untracked |
| `docs/literature/` | `LITERATURE_REVIEW_2026-08-07_VERIFIED.md` (**citation source of truth**, four-tier verification down to ❌-does-not-exist) · `LITERATURE_INTEGRATION_2026-08-07.md` · `REFERENCES_KSRI_DECK_2026-08-10.md` (27 refs re-verified) · `new_papers.md` (novelty-threat scan; found Cox ρ_u) · `LITERATURE_REVIEW_HANDOFF_2026-08-07.md` (⚠ superseded in part — kept for its negatives/open questions) | untracked |
| `docs/design/` | `Section_7_Functional_Information_for_Prompts.md` (**formula authority** — README points here) · `Research_Design_v2_Specificity_FI.md` (research-cycle authority) · `PAPER_PROPOSAL_2026-08-07.md` (post-R1–R3 claim structure the paper was rewritten to) | untracked |
| `docs/talks/` | `PLAYBOOK_KSRI_TALK_2026-08-10.md` (talk track for the delivered deck, incl. Q&A defences) | untracked |
| `cluster/runbooks/` | `RUNBOOK_R1_union_gold_cluster.md`, `RUNBOOK_R6_width_dial_cluster.md` (R6 contains the **P0–P5 preregistration**, fixed before arm data existed). ⚠ Both embed the bwUniCluster username and a Windows SSH-key path — scrub before making the repo public. | untracked |
| `docs/archive/` | `Implementation_Prompt_FullPilot_2026-06-26.md` (only surviving copy; the `Implementation_Prompt_*.md` gitignore rule had kept it out of git) | gitignored by that rule |
| `decks/` | the delivered KSRI seminar deck: `20260809_..._final.pptx`, `_final_copy.pptx` (the 33-slide version the playbook is keyed to), `_final_gapfix.pptx` — manual post-generation edits not reproducible from `make_ksri_deck.py` | gitignored (`*.pptx`) |
| `logs/` | 24 local run logs; last Windows activity 2026-08-14 02:51 (`run_specificity.log`, `smoke_metrics.log`) | gitignored |
| `cluster_logs/` | 668 SLURM out/err logs (21 MB) — provenance for every cluster run (job IDs, the R6 SQLite-lock post-mortem) | gitignored |
| `papers/` | **58 reference PDFs (127 MB)** — the project's paper library matched to `references.bib` and audited by the VERIFIED literature review. New papers found during the 2026-08-16 review are added here too. | untracked |

## Deliberately skipped (still in OneDrive only)

- `⛔ OUTDATED`-bannered docs: `Dataset_Evaluation_v4/v5/v6`, `Research_Design_v3_Context_Ladder.md`, `Research_Proposal_v6_State_and_Literature.md` (partially outdated), the four executed `Implementation_Prompt_*` specs, `Research_Synthesis_Prompt_Sensitivity_Metric.md` (Apr era).
- Old decks (`Prompt_Sensitivity_*` pptx series, supervisor decks) and May slide JPGs — presentation history, superseded by `decks/` + `figures/supervisor_2026-08-03/`.
- June `.docx` memos (metrics glossary, smoke-run analysis, supervisor briefing) — pre-pivot era.
- Windows `.venv` (2.3 GB), LaTeX build intermediates, `.claude/` settings (Windows paths).

## Notes / decisions for Thanos

1. **Nothing was committed.** Untracked additions are listed above; `git add`
   what you want tracked. Suggested: track everything under `docs/` and
   `cluster/runbooks/`.
2. **`papers/` (127 MB of PDFs) is untracked.** Options: add `papers/` to
   `.gitignore` (keep the library machine-local + OneDrive), or track via
   Git LFS. Committing 127 MB of binaries to a plain GitHub repo that the
   paper links publicly is not recommended.
3. The review's reproducibility finding (§3.13/C1) still stands for third
   parties: parquets remain gitignored, so the GitHub repo alone cannot re-run
   the analyses. This machine now can.
4. `data/final_run_results.md` (committed) and `data/probe_eval_hardened.md`
   (committed) still disagree on the holdout AUROC (.667/.655/.670 vs
   .678/.670/.678) — that is review item A1, not a migration artifact.
