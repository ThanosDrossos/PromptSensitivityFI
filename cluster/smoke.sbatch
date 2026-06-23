#!/bin/bash
# bwUniCluster 3.0 smoke roundtrip. Verifies SSH/sync/SLURM/env + gateway
# reachability + GPU-backed DeBERTa, WITHOUT moving any real compute yet.
# See cluster/README.md for the manual prerequisites and the one-time
# `module avail python` step.
#
#SBATCH --job-name=psf-smoke
#SBATCH --partition=dev_gpu_a100_il      # dev GPU queue: 30 min cap, immediate, 1 node, 4x A100
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                      # keep a GPU even though api_check needs none:
                                          # the e2e step runs DeBERTa NLI on it, which is
                                          # exactly the GPU-provisioning we want to confirm.
#SBATCH --output=cluster_logs/%x-%j.out
#SBATCH --error=cluster_logs/%x-%j.err

set -euo pipefail

cd "$HOME/PromptSensitivityFI"
mkdir -p cluster_logs data data/raw/musique

# --- Python module ---------------------------------------------------------
# Confirmed on uc3 (2026-06-23): miniforge ships Python 3.12 (within our
# requires-python >=3.11,<3.13).
module purge
module load devel/miniforge/25.3.1-python-3.12

# --- uv-based env (reused across runs; .venv is NOT synced from local) ------
# First-time setup is best done on the LOGIN node (internet, no time limit);
# see the README. This block only rebuilds if .venv is missing.
if [[ ! -d .venv ]]; then
  pip install --user uv
  "$HOME/.local/bin/uv" sync --frozen
fi
source .venv/bin/activate

# --- Gateway credentials ---------------------------------------------------
# $HOME/.psf_env is created BY HAND on the cluster and never synced from local.
# It must define LITELLM_API_KEY (see README for the template).
set -a
source "$HOME/.psf_env"
set +a

# --- Self-contained MuSiQue fixture ----------------------------------------
# The smoke does NOT depend on a HuggingFace dataset download: it copies a tiny
# committed fixture into the path the loader checks by default.
cp -f cluster/fixtures/musique_smoke.jsonl data/raw/musique/musique_ans_v1.0_dev.jsonl

echo "=== api_check (gateway reachability + model list) ==="
# Non-fatal: a single flaky model must not abort the job; the e2e step below is
# the real acceptance test. Tee to a log we pull back.
python -m prompt_sensitivity.scripts.api_check 2>&1 | tee cluster_logs/api_check.log || true

echo "=== e2e smoke: 1 MuSiQue question, gpt_4o (=kit.gpt-4.1), random ladder, levels 0+10 ==="
# --musique-direct 1 --singleton --fast is the smallest self-contained cell:
#   * singleton -> no paraphrase generation (1-element universe)
#   * fast      -> no H_sem clustering / embeddings (FI_in + f_mean only)
#   * DeBERTa chain-scoring runs on the GPU; the F-call hits the gateway.
# NOTE: --models takes the config KEY (gpt_4o), which maps to model_id
#       kit.gpt-4.1 on the gateway. `kit.gpt-4.1` is NOT a valid --models value.
python -m prompt_sensitivity.scripts.e2e_smoke \
  --musique-direct 1 \
  --singleton --fast \
  --families context \
  --ladders random \
  --levels 0,10 \
  --k-samples 1 \
  --max-paraphrases 2 \
  --models gpt_4o \
  --out data/cluster_smoke.parquet \
  2>&1 | tee cluster_logs/e2e_smoke.log

echo "=== quick parquet sanity ==="
python -m prompt_sensitivity.scripts.show_results --in data/cluster_smoke.parquet 2>&1 \
  | tee cluster_logs/show_results.log || true

echo "DONE $(date -Iseconds)"
