#!/bin/bash
# Full v2 specificity run (AmbigQA pivot, UNIFORM-EVIDENCE design): 50 questions
# x 2 levels x qwen — every cell gets the question's own annotator-search snippets
# as an identical context block; answerability from reading, not recall.
# Original header:
# k=10, 10 paraphrases/universe, inspection bundle for the first 5 questions.
#
# DESIGN — singleton chain on gpu_a100_short: this job is submitted MANY times
# with the same --job-name and --dependency=singleton (see
# cluster/submit_specificity_full.sh), so SLURM runs the copies strictly one
# after another on the fast 30-min partition (which schedules in minutes, vs
# days of queueing for a single long gpu_a100_il job). Progress is monotonic
# across windows: the paraphrase cache persists PER UNIVERSE and the eval
# checkpoints PER CELL, so each window resumes where the previous died and
# surplus windows no-op in ~2 min. Expected active windows: ~9-12
# (~2.5 h Phi-4 prep for the 90 uncached universes + ~2 h eval for 100 cells).
#
#SBATCH --job-name=psf-spec-full
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

nvidia-smi | head -12 || true

echo "=== specificity v2 full (uniform evidence): 50 q x 2 levels x qwen (k=10, N=10, inspect 5) ==="
python -m prompt_sensitivity.scripts.run_specificity \
  --n-questions 50 --models qwen_2_5_7b \
  --k-samples 10 --max-paraphrases 10 --inspect-n 5 \
  --out data/specificity_v2_metrics.parquet \
  2>&1 | tee -a cluster_logs/spec_full_v2.log

echo "=== level pivot ==="
python -m prompt_sensitivity.scripts.show_specificity \
  --in data/specificity_v2_metrics.parquet \
  2>&1 | tee cluster_logs/show_specificity_full_v2.log || true
echo "WINDOW DONE $(date -Iseconds)"
