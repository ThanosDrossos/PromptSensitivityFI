# bwUniCluster 3.0 — smoke roundtrip

Smallest possible end-to-end cluster workflow: verify **SSH key auth, file
sync, SLURM scheduling, the Python env, gateway reachability, and a GPU-backed
DeBERTa pass** — before moving any real compute to the cluster. This is a
roundtrip on a handful of files, **not** the pipeline.

bwUniCluster 3.0 is a SLURM-managed HPC at KIT (`uc3.scc.kit.edu`). Login uses
an SSH key registered through bwIDM (not `~/.ssh/authorized_keys`). Login nodes
are for setup only; compute goes through `sbatch`. The smoke targets the
**`dev_gpu_a100_il`** development queue (30 min cap, immediate access, 1 node,
4× A100).

---

## 1. Manual prerequisites (you do these once, by hand)

The Makefile/agent does **not** run these.

1. **Generate an ed25519 key locally** (skip if you already have one):
   ```bash
   ssh-keygen -t ed25519 -C "thanos@kit"
   ```
2. **Register the public key** at <https://login.bwidm.de/user/ssh-keys.xhtml>
   as an **Interactive** key for bwUniCluster 3.0. Validity is 180 days; the
   8-hour unlock needs OTP + service password.
3. **Test login from a BelWü-connected network** (campus or KIT VPN):
   ```bash
   ssh ka_<username>@uc3.scc.kit.edu
   ```
   First login this session asks for OTP + service password to unlock the key;
   afterwards it's key-only for an 8-hour window.
4. **Workspace note.** For now the sync writes to `$HOME/PromptSensitivityFI`.
   When data grows, switch to a Lustre workspace (`ws_allocate`) instead of
   `$HOME` — out of scope for this smoke.

---

## 2. One-time cluster-side setup (on the login node)

The login node has internet and no time limit, so do the heavy one-time work
there. The compute node reuses `$HOME` (shared filesystem), including the
`.venv` and the HuggingFace cache.

After your **first** `make cluster-push` (section 4):

```bash
ssh ka_<username>@uc3.scc.kit.edu
cd PromptSensitivityFI

# (a) Find the real Python module name, then edit cluster/smoke.sbatch's
#     `module load ...` line to match, and commit the change locally.
module avail python          # confirmed on uc3: devel/miniforge/25.3.1-python-3.12

module purge && module load <the-python-module-you-found>

# (b) Build the uv env once (downloads torch+CUDA; minutes, but no time limit).
pip install --user uv
~/.local/bin/uv sync --frozen

# (c) Pre-cache the DeBERTa NLI weights so the compute node doesn't download
#     them inside the 20-min job (also avoids any compute-node egress limits).
~/.local/bin/uv run python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; m='MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli'; AutoTokenizer.from_pretrained(m); AutoModelForSequenceClassification.from_pretrained(m); print('DeBERTa cached')"

# (d) Create the gateway credentials file (NEVER synced, NEVER committed):
cat > "$HOME/.psf_env" <<'EOF'
LITELLM_API_KEY=sk-...your-key...
# LITELLM_BASE_URL=https://ai-gateway.dsi-experimente.de/v1   # only if non-default
EOF
chmod 600 "$HOME/.psf_env"
```

`$HOME/.psf_env` lives **only on the cluster**. It is in `.gitignore` and in
the rsync excludes — it is never pushed from local nor committed.

---

## 3. Env vars the Makefile expects

| Var | Required | Default | Meaning |
|---|---|---|---|
| `BWUC_USER` | yes | — | your cluster login, e.g. `ka_jc8392` |
| `BWUC_HOST` | no | `uc3.scc.kit.edu` | login host |
| `BWUC_SSH_KEY` | no | (default `~/.ssh`) | private-key path if the key is NOT in `~/.ssh` |

Set them inline per command, or `export` them once for the session.

**This project's values** (login `ka_jc8392`, key at `C:\Users\thano\ssh_key_thanoskit`).
Run from **Git Bash** (rsync/ssh are not in PowerShell). In Git Bash a Windows
path `C:\Users\thano\ssh_key_thanoskit` is written `/c/Users/thano/ssh_key_thanoskit`:

```bash
export BWUC_USER=ka_jc8392
export BWUC_SSH_KEY=/c/Users/thano/ssh_key_thanoskit
```

### Windows: lock down the key permissions once

OpenSSH refuses a private key that is group/world-readable
("UNPROTECTED PRIVATE KEY FILE"). Fix it once in **PowerShell**:

```powershell
icacls C:\Users\thano\ssh_key_thanoskit /inheritance:r
icacls C:\Users\thano\ssh_key_thanoskit /grant:r "$($env:USERNAME):(R)"
```

---

## 4. The roundtrip

> **No `make`?** Git Bash on Windows usually lacks GNU make. Every step below
> has an identical make-free form: replace `make cluster-<cmd>` with
> `bash cluster/run.sh <cmd>` (same env vars). e.g. `bash cluster/run.sh check`.
>
> **No `rsync`?** Git Bash also lacks rsync. `sync.sh` auto-detects this and
> falls back to **tar-over-ssh** (ssh + tar only, both present) — an overlay
> copy with the same excludes, just no `--delete`. Nothing extra to install.

### Step 0 — minimal connectivity test (do this FIRST)

Before pushing anything, confirm the key + login work and SLURM is visible.
This is the smallest possible "is my setup wired up" check — no sync, no job:

```bash
make cluster-check
```

Expected: `CONNECTED as ka_jc8392 on uc3nXXXX`, paths to `sbatch`/`squeue`,
and `repo NOT pushed yet` (until you run `cluster-push`). The first connection
this session prompts for OTP + service password to unlock the key.

### Steps 1-3 — the actual roundtrip

```bash
# push code + submit the smoke job in one shot
make cluster-smoke

# watch the queue (R = running, PD = pending, empty = finished)
make cluster-status

# once it leaves the queue, pull logs + the result parquet back
make cluster-pull
```

(With `BWUC_USER` / `BWUC_SSH_KEY` exported as in §3, you don't repeat them
each time. Otherwise prefix each command, e.g.
`BWUC_USER=ka_jc8392 BWUC_SSH_KEY=/c/Users/thano/ssh_key_thanoskit make cluster-check`.)

`make cluster-smoke` = `cluster-push` + `cluster-submit`. Individual targets:

| Target | Does |
|---|---|
| `cluster-check` | `ssh` in, print hostname/whoami, confirm `sbatch`/`squeue` + whether the repo is pushed. No sync, no job — the minimal connectivity test |
| `cluster-push` | `rsync` the repo to `$HOME/PromptSensitivityFI` (excludes `.venv`, `data/`, `logs/`, `.git/`, `cluster_logs/`, `.env*`, `.psf_env`) |
| `cluster-submit` | `ssh` + `sbatch cluster/smoke.sbatch`, prints the job id |
| `cluster-status` | `ssh` + `squeue --me` |
| `cluster-pull` | `rsync` `cluster_logs/` and `data/cluster_smoke.parquet` back |

After `cluster-pull`, inspect locally:

```bash
cat cluster_logs/psf-smoke-<jobid>.out
uv run python -m prompt_sensitivity.scripts.show_results --in data/cluster_smoke.parquet
```

### Windows

`rsync` is not native to PowerShell. Run the `make cluster-*` targets from
**Git Bash** or **WSL** (both ship `make`, `rsync`, `ssh`). `make api-check`
and the other local targets still work from PowerShell via `.\tasks.ps1`.

---

## 5. What the smoke job actually runs (`cluster/smoke.sbatch`)

1. `module load` Python, build/reuse the `uv` `.venv`, `source $HOME/.psf_env`.
2. Copy the committed MuSiQue fixture
   (`cluster/fixtures/musique_smoke.jsonl`) into the path the loader checks by
   default — so the smoke needs **no HuggingFace dataset download**.
3. `api_check` — pings the gateway and lists models (non-fatal: a flaky model
   won't abort the job).
4. `e2e_smoke` — the smallest self-contained cell:
   ```
   --musique-direct 1 --singleton --fast \
   --families context --ladders random --levels 0,10 \
   --k-samples 1 --models gpt_4o --out data/cluster_smoke.parquet
   ```
   `--singleton` skips paraphrase generation (1-element universe); `--fast`
   skips H_sem clustering. The **F-call goes to the gateway**; the **DeBERTa
   chain-scoring runs on the GPU** — which is why we request `gres=gpu:1` even
   though `api_check` itself needs no GPU.
5. `show_results` prints a one-line sanity of the parquet.

### Deviations from the original task skeleton (intentional, documented)

- **`--models gpt_4o`, not `kit.gpt-4.1`.** `--models` takes the *config key*;
  `gpt_4o` maps to model_id `kit.gpt-4.1` on the gateway. Passing
  `kit.gpt-4.1` would error with "unknown model_key".
- **`--musique-direct 1 --singleton --fast`, not `--n-questions 1`.** The
  `--n-questions` path reads a paraphrase parquet from `data/`, which is
  excluded from sync and absent on the cluster. The musique-direct + fixture
  path is fully self-contained (no `data/` dependency, no dataset download).

---

## 6. Acceptance checklist

- [ ] `make cluster-push` succeeds; repo appears at `$HOME/PromptSensitivityFI`.
- [ ] `make cluster-submit` returns a SLURM job id.
- [ ] Job completes within 30 min on `dev_gpu_a100_il`, exit code 0
      (`cluster_logs/psf-smoke-<jobid>.out` ends with `DONE <timestamp>`).
- [ ] `make cluster-pull` brings back the job `.out`, `api_check.log`,
      `e2e_smoke.log`, and `data/cluster_smoke.parquet`.
- [ ] The parquet has ≥1 row with `n_paraphrases >= 1` and a non-null `f_mean`.

---

## 6b. Local in-process models (Sprint 6 — the real pilot, NO gateway)

The smoke above used the LiteLLM gateway. The pilot drops it entirely: the three
eval models (Llama-3.1-8B, Mistral-7B-v0.3, Qwen2.5-7B) and the Qwen paraphrase
generator load **in-process via HF transformers** on the GPU (`provider: local`,
`models/local_hf.py`). This is the only path that yields token logprobs, exact
teacher-forced scoring (POSIX), and last-layer hidden states (ESS_in^own) at
once. `LITELLM_*` / `$HOME/.psf_env` are **not needed** for this path.

### Manual prerequisites (one-time, login node — the job does NOT do these)

1. **Accept the gated-model licenses** (browser, logged into your HF account):
   <https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct> and
   <https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3>. Qwen2.5-7B is open.
2. **Authenticate HF on the cluster** so gated downloads work:
   ```bash
   huggingface-cli login        # paste a token from https://huggingface.co/settings/tokens
   ```
   (Token lands in `~/.cache/huggingface/`, on the shared `$HOME` — the compute
   node reuses it. Alternatively put `HF_TOKEN=hf_...` in `$HOME/.psf_env`.)
3. **Pre-cache the weights** on the login node (internet, no time limit) so the
   job never downloads inside its walltime:
   ```bash
   ~/.local/bin/uv run python - <<'PY'
   from transformers import AutoModelForCausalLM, AutoTokenizer
   for m in ["meta-llama/Llama-3.1-8B-Instruct",
             "mistralai/Mistral-7B-Instruct-v0.3",
             "Qwen/Qwen2.5-7B-Instruct",
             "microsoft/phi-4"]:                       # P3-3: paraphrase generator + judge
       AutoTokenizer.from_pretrained(m); AutoModelForCausalLM.from_pretrained(m)
       print("cached", m)
   PY
   ```
   (Pre-cache DeBERTa NLI the same way — see §2c.) **Phi-4** (`microsoft/phi-4`,
   the paraphrase generator+judge) is **MIT-licensed — no token gating**; you can
   also grab it with `huggingface-cli download microsoft/phi-4`. It's ~28 GB bf16
   and runs alongside DeBERTa (~1.6 GB) inside a 40 GB A100 for the prep job.
4. **Materialise the real MuSiQue dev data** at the path the loader checks:
   `data/raw/musique/musique_ans_v1.0_dev.jsonl` (MuSiQue-Ans dev, via the
   official `StonyBrookNLP/musique` `download_data.sh` or an HF mirror). The toy
   `cluster/fixtures/` file is for wiring only — `e2e_local.sbatch` does NOT copy
   it and aborts if the real jsonl is missing.

### Run

```bash
bash cluster/run.sh push                 # code only (data/ excluded)
ssh $BWUC_USER@uc3.scc.kit.edu
cd PromptSensitivityFI
sbatch cluster/e2e_local.sbatch          # gpu_a100_il, 1 GPU, up to 6h
squeue --me
```

The job runs a **preflight** (`local_check` — generate + logprobs + teacher-forced
score + hidden states for all 3 models, aborts on any failure), then a paraphrase
pre-pass, the 3 eval passes (both ladders, `--own-encoder`), an optional exact
POSIX probe, and `show_results` + `plot_pilot`. Pull results back:

```bash
bash cluster/run.sh pull                 # fetches cluster_e2e_musique.parquet + plots
uv run python -m prompt_sensitivity.scripts.show_results --in data/cluster_e2e_musique.parquet
```

---

## 6c. Full production run (scaled, parallel across models)

Once the §6b pilot is green, the full run scales it up: **stratified questions**
(equal numbers of 2-/3-/4-hop), **both context ladders** (`random` + `gold_first`,
so the context-vs-reasoning comparison is fair), **all context levels**
(0,2,4,6,8,10), both families, all 3 models, `--own-encoder`. Same manual
prerequisites as §6b (they're already done if the pilot ran).

**Design:** a SLURM dependency chain — one paraphrase-prep job
(`full_paraphrase_prep.sbatch`, Phi-4 generator) builds the SHARED paraphrase
universe once (FI_in is only comparable across models when `|U_q|` is identical),
then **three per-model eval jobs** (`full_eval_<model>.sbatch`) run in parallel,
one GPU each, gated `--dependency=afterok`. Each writes its own
`data/full_<model>.parquet` (no write races); you merge locally after pulling.
(The older single-orchestrator scripts `run_full.sh` / `full_run.sbatch` /
`full_prep.sbatch` are superseded — see `cluster/archive/`.)

```bash
# 1) push, then on the cluster:
bash cluster/submit_full_pilot.sh   # prep (51 q, N<=30) -> 3 per-model eval jobs (afterok)
squeue --me                         # psf-fp-prep, then psf-fe-{llama,mistral,qwen}

# 2) when all three eval jobs show DONE, from your laptop:
bash cluster/run.sh pull            # fetches data/full_*.parquet + inspect_*.md + logs
uv run python -m prompt_sensitivity.scripts.merge_results          # -> data/full_run.parquet
uv run python -m prompt_sensitivity.scripts.show_results --in data/full_run.parquet
uv run python -m prompt_sensitivity.scripts.plot_pilot   --in data/full_run.parquet --out data/plots
```

**Walltime / cost:** the e2e **checkpoints every cell**, so a timed-out job resumes
on resubmit (`sbatch cluster/full_eval_llama_3_1_8b.sbatch` re-runs just Llama;
done cells skip). NOTE: at the current settings (k=10 H_sem samples × N≤30
paraphrases × 6 levels × 2 families) the per-cell cost is far higher than the k=2
smoke (~340 s/cell vs ~135 s) and the full run is likely **out of scope** without
trimming — see the efficiency notes / cut proposals before launching.

---

## 7. Out of scope (next sprints)

No automated 2FA (you unlock the SSH key by hand every 8 hours). The Sprint-6
local path uses **transformers generation**, which is slow at scale — the next
step for a scaled run is batched generation or a vLLM **offline** (`LLM.generate`,
in-process, still no API) generator, keeping transformers only for the
hidden-state passes. Full-vocab `S_tau` (true token entropy) is now reachable but
deferred; the pilot keeps the MC-over-clusters estimate.

---

## Sources

- [bwUniCluster 3.0 login](https://wiki.bwhpc.de/e/BwUniCluster3.0/Login)
- [Registration / SSH keys](https://wiki.bwhpc.de/e/Registration/SSH)
- [Batch queues and partitions](https://wiki.bwhpc.de/e/BwUniCluster3.0/Batch_Queues)
- [bwIDM SSH key portal](https://login.bwidm.de/user/ssh-keys.xhtml)
