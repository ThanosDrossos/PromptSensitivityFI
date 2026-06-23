#!/bin/bash
# bwUniCluster 3.0 — full MuSiQue end-to-end on LOCAL in-process models.
# No gateway, no API: the 3 eval models (Llama-3.1-8B, Mistral-7B-v0.3,
# Qwen2.5-7B) and the Qwen generator run via HF transformers on the GPU; DeBERTa
# NLI + (optional) own-encoder hidden states run in the same process.
#
# Sequential per model (one model resident at a time) so a single A100 suffices.
# Prereqs (one-time, login node — see cluster/README.md): accept Llama-3.1 +
# Mistral-7B HF licenses, `huggingface-cli login`, pre-cache the 3 LLMs + DeBERTa,
# and materialise data/raw/musique/musique_ans_v1.0_dev.jsonl.
#
#SBATCH --job-name=psf-e2e-local
#SBATCH --partition=gpu_a100_il          # 48h cap, 4x A100/node (dev_gpu_a100_il = 30min, bring-up only)
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=cluster_logs/%x-%j.out
#SBATCH --error=cluster_logs/%x-%j.err

set -euo pipefail

cd "$HOME/PromptSensitivityFI"
mkdir -p cluster_logs data

# --- Python module ---------------------------------------------------------
# Confirmed on uc3 (2026-06-23): miniforge ships Python 3.12, within our
# requires-python >=3.11,<3.13. torch ships its own CUDA runtime, so no cuda
# module is needed for the sdpa path (we don't build flash-attn).
module purge
module load devel/miniforge/25.3.1-python-3.12

# --- uv env (NOT --frozen here: pyproject gained `accelerate`, so let uv
#     re-resolve the lock rather than fail on a stale --frozen check) ----------
pip install --user -q uv || true
export PATH="$HOME/.local/bin:$PATH"
uv sync                                   # installs accelerate (now a pyproject dep)
source .venv/bin/activate

# --- Optional secrets (HF_TOKEN can live here; LITELLM_* no longer needed) --
# huggingface-cli login (done by hand on the login node) is the primary auth;
# this is only a fallback if you put HF_TOKEN in $HOME/.psf_env.
if [[ -f "$HOME/.psf_env" ]]; then set -a; source "$HOME/.psf_env"; set +a; fi
export TOKENIZERS_PARALLELISM=false

echo "=== GPU ==="
nvidia-smi || true

# --- MuSiQue dev data sanity (must be materialised by hand; NOT the fixture) -
if [[ ! -s data/raw/musique/musique_ans_v1.0_dev.jsonl ]]; then
  echo "ERROR: data/raw/musique/musique_ans_v1.0_dev.jsonl missing or empty."
  echo "       Download the MuSiQue-Ans dev jsonl on the login node first (README §6b)."
  exit 1
fi

echo "=== preflight: local_check (generate + logprobs + score + hidden) ==="
if ! python -m prompt_sensitivity.scripts.local_check 2>&1 | tee cluster_logs/local_check.log; then
  echo "ABORT: local_check failed — at least one model is not fully capable."
  exit 1
fi

echo "=== paraphrase pre-pass (generator = qwen_2_5_7b) ==="
python -m prompt_sensitivity.scripts.e2e_smoke \
  --musique-direct 3 --paraphrase-only --max-paraphrases 6 \
  2>&1 | tee cluster_logs/paraphrase_prepass.log

echo "=== eval passes (3 models, both ladders, own-encoder) ==="
for M in llama_3_1_8b mistral_7b_v03 qwen_2_5_7b; do
  echo "--- eval model: $M ---"
  python -m prompt_sensitivity.scripts.e2e_smoke \
    --musique-direct 3 --families context,reasoning --ladders random \
    --levels 0,4,8 --k-samples 2 --max-paraphrases 6 \
    --models "$M" --own-encoder \
    --out data/cluster_e2e_musique.parquet \
    2>&1 | tee "cluster_logs/e2e_${M}.log"
done

echo "=== POSIX probe (exact teacher forcing now possible) ==="
python -m prompt_sensitivity.scripts.e2e_smoke \
  --musique-direct 1 --families context --ladders random --levels 0,8 \
  --k-samples 1 --max-paraphrases 3 --include-posix \
  --models llama_3_1_8b --out data/cluster_posix_probe.parquet \
  2>&1 | tee cluster_logs/posix_probe.log || echo "(POSIX probe non-fatal)"

echo "=== results ==="
python -m prompt_sensitivity.scripts.show_results --in data/cluster_e2e_musique.parquet \
  2>&1 | tee cluster_logs/show_results.log || true
python -m prompt_sensitivity.scripts.plot_pilot --in data/cluster_e2e_musique.parquet \
  --out data/plots 2>&1 | tee cluster_logs/plot_pilot.log || true

echo "DONE $(date -Iseconds)"
