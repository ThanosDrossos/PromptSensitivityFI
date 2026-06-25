#!/bin/bash
# Full pilot, step 2/2 (P1-5): qwen_2_5_7b. See full_eval_llama_3_1_8b.sbatch for
# the design notes (per-model parquet to avoid the concurrent-write race;
# resumable; calibrate walltime with the dev smoke first).
#
#SBATCH --job-name=psf-fe-qwen
#SBATCH --partition=gpu_a100_il
#SBATCH --time=08:00:00
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

M=qwen_2_5_7b
nvidia-smi || true

echo "=== preflight: local_check (${M}) ==="
if ! python -m prompt_sensitivity.scripts.local_check --models "$M" \
     2>&1 | tee "cluster_logs/full_check_${M}.log"; then
  echo "ABORT: local_check failed for ${M}"; exit 1
fi

echo "=== full eval: ${M} (N<=30, k=10, POSIX, own-encoder) ==="
python -m prompt_sensitivity.scripts.e2e_smoke \
  --musique-strata 17 --families context,reasoning --ladders random \
  --levels 0,2,4,6,8,10 --k-samples 10 --max-paraphrases 30 \
  --models "$M" --own-encoder --include-posix \
  --out "data/full_${M}.parquet" \
  2>&1 | tee "cluster_logs/full_eval_${M}.log"

echo "DONE ${M} $(date -Iseconds)"
