# Runbook — R1 union-gold arm on bwUniCluster

Push → probe → run → pull → analyse. Everything below is copy-pasteable from **Git Bash on the laptop**, in `Code/PromptSensitivityFI/`.

> **The probe is not optional.** The arm re-scores generations that must still be in the cluster's
> `data/cache/llm_cache.sqlite`. If they are gone, this stops being a cheap re-scoring job and becomes a
> full re-run. The probe answers that in seconds, without a GPU. **Do not skip to step 3.**

---

## 0. One-time per shell — credentials

```bash
export BWUC_USER=<bwuc-username>
export BWUC_SSH_KEY=<path-to-ssh-key>
```

Key must be unlocked via bwIDM (OTP + service password, 8-hour window). Sanity check:

```bash
bash cluster/run.sh check
```

Expect `CONNECTED as ... on uc3nX`, `sbatch`/`squeue` paths, and `repo present`.

---

## 1. Push the laptop working tree

```bash
bash cluster/run.sh push
```

Ships the new files: `scripts/rescore_union_gold.py`, `analysis/rho_f_hierarchical.py`, `scripts/fit_rho_f_hierarchical.py`, `scripts/independence_analysis.py`, `cluster/union_gold_arm.sbatch`, `cluster/submit_union_gold.sh`.
`data/` is excluded from push by design — the cluster keeps its own cache and parquets.

---

## 2. PROBE — is the cache still there?

```bash
ssh -i "$BWUC_SSH_KEY" -o IdentitiesOnly=yes "$BWUC_USER@uc3.scc.kit.edu" \
  "cd PromptSensitivityFI && bash cluster/submit_union_gold.sh probe"
```

Three 1-window jobs (`psf-ugold-probe-<model>`). When they finish:

```bash
ssh -i "$BWUC_SSH_KEY" -o IdentitiesOnly=yes "$BWUC_USER@uc3.scc.kit.edu" \
  "cd PromptSensitivityFI && grep -h 'cache' cluster_logs/union_gold_probe_*.log"
```

**Read the result:**

| output | meaning | action |
|---|---|---|
| `cache 32901/32901 hits (100.0%)` … `skipped 0` | generations intact | go to step 3 |
| high hits but some `skipped` | a few cells incomplete | fine — those cells are dropped, not regenerated. Note the count for the paper |
| `cache 0/32901 hits (0.0%)` … `skipped 300` | **cache gone** | **STOP.** Do not run `full`. Re-scope: R1 becomes a full re-run |

(32 901 = 300 cells × 10 paraphrases × 11 requests, minus the one singleton cell.)

---

## 3. RUN the arm

```bash
ssh -i "$BWUC_SSH_KEY" -o IdentitiesOnly=yes "$BWUC_USER@uc3.scc.kit.edu" \
  "cd PromptSensitivityFI && bash cluster/submit_union_gold.sh full 3"
```

Three singleton chains (`psf-ugold-<model>`), 3 windows × 30 min each. Per-cell resume: each window picks up where the last stopped, and a surplus window is a clean no-op. If 3 windows are not enough, just submit more — same command.

Monitor:

```bash
ssh -i "$BWUC_SSH_KEY" -o IdentitiesOnly=yes "$BWUC_USER@uc3.scc.kit.edu" "squeue --me"
ssh -i "$BWUC_SSH_KEY" -o IdentitiesOnly=yes "$BWUC_USER@uc3.scc.kit.edu" \
  "cd PromptSensitivityFI && tail -20 cluster_logs/union_gold_qwen_2_5_7b.log"
```

Done when each log shows `wrote 300 cells -> data/union_gold_<model>.parquet` plus the
`=== R1 union-gold arm ===` summary block. Stop a chain: `scancel --name=psf-ugold-<model>`.

---

## 4. Pull

```bash
bash cluster/run.sh pull
```

`union_gold_*.parquet` is now in `sync.sh`'s `PULL_FILES`, so both the rsync and the tar path fetch it.
The post-pull sensitivity-v2 / collision backfill runs automatically and does not touch the new files.

---

## 5. Analyse on the laptop

```bash
uv run python -m prompt_sensitivity.scripts.fit_rho_f_hierarchical --scoring union
uv run python -m prompt_sensitivity.scripts.independence_analysis --scoring union
```

Outputs: `data/rho_f_hier_union_{model}.parquet`, `data/independence_union.parquet`, `data/independence_union.md`.

Compare against the target-gold run already on disk (`data/independence_target.md`).

**The number that decides the headline** is printed by the arm itself and repeated per model:

```
decomposition: Delta_target +0.24 = Delta_union +X + Delta_targeting +Y
```

- **Δ_union ≈ 0** → the entire effect was the grading lottery. Say so plainly; FI_spec measures resolution of
  referential ambiguity *for the grader*, not model improvement. That is a cleaner claim than the current one,
  and per the agreed framing it belongs in **Methods**, not Findings.
- **Δ_union > 0 and significant** → that residual is the real ability effect and becomes the new headline,
  with the lottery share quantified and removed.

Either way, also report the collision split (already computable, no new compute): non-collision Δ = **+0.268**
[+0.208, +0.328] vs collision Δ = **−0.044** [−0.135, +0.035], group difference p = 3.8e-07.

---

## Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `repo NOT pushed yet` | first run | `bash cluster/run.sh push` |
| jobs pending forever | `--dependency=singleton` serialises same-named jobs | expected — chains run one at a time per model |
| `Cannot stat … transport endpoint shutdown` on pull | Lustre OST damage on the cluster copy (see `sync.sh:skipped_report`) | the pull reports whether you already hold each file locally; only files listed `NOT LOCAL` are actually lost |
| probe shows 100 % but `full` skips cells | a paraphrase universe changed since the run | universes come from `data/paraphrases_ambigqa.parquet`; do not regenerate it before this arm |
| walltime kill mid-run | expected | resume is automatic; submit another window |

## What NOT to do

- **Do not run `full` before the probe.** A 0 %-cache `full` run wastes three chains and produces nothing.
- **Do not regenerate paraphrases** before this arm — the cache keys depend on the exact prompt text.
- **Do not delete `data/cache/llm_cache.sqlite` on the cluster.** It is the only copy of the v3 generations,
  and R4b/R7 may need it again.
