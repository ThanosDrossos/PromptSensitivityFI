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
module avail python          # e.g. devel/python/3.11 — pick the 3.11+ one

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
| `BWUC_USER` | yes | — | your cluster login, e.g. `ka_xxxxx` |
| `BWUC_HOST` | no | `uc3.scc.kit.edu` | login host |

Set `BWUC_USER` inline per command, or `export` it for the session.

---

## 4. The roundtrip (three commands, run locally)

```bash
# push code + submit the smoke job in one shot
BWUC_USER=ka_xxxxx make cluster-smoke

# watch the queue (R = running, PD = pending)
BWUC_USER=ka_xxxxx make cluster-status

# once it leaves the queue, pull logs + the result parquet back
BWUC_USER=ka_xxxxx make cluster-pull
```

`make cluster-smoke` = `cluster-push` + `cluster-submit`. Individual targets:

| Target | Does |
|---|---|
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

## 7. Out of scope (next sprints)

No vLLM, no model-weight downloads, no open-weight models on the cluster yet;
no headline pilot on the cluster; no automated 2FA (you unlock the SSH key by
hand every 8 hours). Once this is green, the next sprint adds a vLLM
open-weight smoke (Llama-3.1-8B on `gpu_h100`), then the pilot moves over.

---

## Sources

- [bwUniCluster 3.0 login](https://wiki.bwhpc.de/e/BwUniCluster3.0/Login)
- [Registration / SSH keys](https://wiki.bwhpc.de/e/Registration/SSH)
- [Batch queues and partitions](https://wiki.bwhpc.de/e/BwUniCluster3.0/Batch_Queues)
- [bwIDM SSH key portal](https://login.bwidm.de/user/ssh-keys.xhtml)
