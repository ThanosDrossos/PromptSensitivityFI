# ⛔ Archive — outdated documents, kept only for history

**Nothing in this folder describes the current design.** These files belong to
superseded eras of the project (HotpotQA smoke → MuSiQue dual-ladder →
pilot-prep). They are retained so past decisions remain traceable.

**Do not use these as a source of truth. Do not cite their numbers.**

| file | era | superseded by |
|---|---|---|
| `Implementation_Prompt_FullPilot_2026-06-26.md` | pilot-prep hardening (branch `pilot-prep-2026-06-26`, since archived as a tag) | the AmbigQA pivot plan and the FI-probes design note (both executed, removed 2026-08-27) |
| `REPORT_musique_pilot.md` | MuSiQue dual-ladder pilot (63 cells, 3 questions) | `data/final_run_results.md` |
| `REPORT_hotpotqa_smoke.md` | HotpotQA smoke (18 cells, 3 questions) | `data/final_run_results.md` |

## Where the current documentation lives

- **`CLAUDE.md`** — the frame, the fixed vocabulary, and where truth lives
- **`docs/PROJECT_STATE_2026-08-27.md`** — current status and headline results
- **`PIPELINE_WALKTHROUGH.md`** — the pipeline end to end, for newcomers
- **`data/stats_hygiene.md`** — the declared test family and the primary numbers
- **`FORKING_PATHS.md`** — every analysis decision that moved a number

On 2026-08-27 six superseded root documents were removed (the pre-pivot
codebase walkthrough and three-axes explainer, and the executed metric,
probe, final-phase and AmbigQA-pivot plans). They remain in git history;
recover one with `git log --diff-filter=D --name-only` to find the deleting
commit, then `git show <commit>^:<path>`.
