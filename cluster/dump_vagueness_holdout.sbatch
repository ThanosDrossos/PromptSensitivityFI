#!/bin/bash
# Vagueness-holdout dump (FINAL_PHASE_PLAN, approved scope): TBG states for
# ONE prompt per AmbigQA question INCLUDING the ~830 annotator-labeled
# NON-ambiguous rows the main pipeline drops. 2,002 prompts x MODEL, training
# prompt format (uniform evidence), no paraphrases, no scoring, no gold.
# Post-pull laptop eval: scripts/eval_vagueness_holdout.py (frozen heads).
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

nvidia-smi | head -12 || true

echo "=== vagueness-holdout dump [${MODEL}] ==="
python -m prompt_sensitivity.scripts.dump_vagueness_holdout \
  --models "${MODEL}" \
  2>&1 | tee -a "cluster_logs/vh_dump_${MODEL}.log"
echo "WINDOW DONE $(date -Iseconds)"
