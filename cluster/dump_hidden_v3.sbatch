#!/bin/bash
# TBG hidden-state dump chain (FI probes features; FI_PROBES_PLAN.md §2-3).
# Cache-only + resumable: safe to run before/alongside/after eval chains — it
# never generates paraphrases, and skipped-missing universes are picked up on
# the next window once prep has written them.
#
# Env (set by submit script):
#   MODEL        eval model key (one chain per model, own job name)
#   N_QUESTIONS  50 = backfill of the finished v2 set (probe prototyping data);
#                150 = full v3 feature set (default)
#
#SBATCH --partition=gpu_a100_short
#SBATCH --time=00:30:00
#SBATCH --dependency=singleton
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=cluster_logs/%x-%j.out
#SBATCH --error=cluster_logs/%x-%j.err

set -euo pipefail
cd "$HOME/PromptSensitivityFI"
mkdir -p cluster_logs data

module purge
module load devel/miniforge/25.3.1-python-3.12
pip install --user -q uv || true
export PATH="$HOME/.local/bin:$PATH"
uv sync
source .venv/bin/activate
if [[ -f "$HOME/.psf_env" ]]; then set -a; source "$HOME/.psf_env"; set +a; fi
export TOKENIZERS_PARALLELISM=false

MODEL="${MODEL:-qwen_2_5_7b}"
N_QUESTIONS="${N_QUESTIONS:-150}"

nvidia-smi | head -12 || true

echo "=== hidden-state dump [${MODEL}]: ${N_QUESTIONS} q x 2 levels, TBG @ 4 layers ==="
python -m prompt_sensitivity.scripts.dump_hidden_states \
  --n-questions "${N_QUESTIONS}" --models "${MODEL}" \
  2>&1 | tee -a "cluster_logs/dump_hidden_${MODEL}.log"
echo "WINDOW DONE $(date -Iseconds)"
