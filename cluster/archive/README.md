# Archived cluster scripts

Superseded scripts, kept for reference only — **do not use**.

| file | superseded by | why |
|------|---------------|-----|
| `run_full.sh` | `cluster/submit_full_pilot.sh` | old single-orchestrator full run (built a SLURM array). |
| `full_run.sbatch` | `cluster/full_eval_<model>.sbatch` | per-model eval is now three separate `--dependency=afterok` jobs, not a SLURM array (the array wrote one shared parquet → per-cell write races). |
| `full_prep.sbatch` | `cluster/full_paraphrase_prep.sbatch` | paraphrase-prep job; the new one uses the Phi-4 generator + 17 q/hop-stratum. |

The current full-run flow is one command — `bash cluster/submit_full_pilot.sh` — see `cluster/README.md`.
