#!/bin/bash
# Full run, step 2/2: per-model SLURM array — one model per GPU, in parallel.
# Each task writes its OWN parquet (data/full_<model>.parquet) so the concurrent
# tasks never touch the same file. Merge locally after pull (merge_results.py).
#
# Depends on full_prep (shared paraphrases). Submit via cluster/run_full.sh.
#
# Walltime sizing (pilot: Llama ~135s/cell, Mistral ~80s, Qwen ~48s; ~15 cells
# per question with full levels + 2 ladder variants): at PSF_STRATA=10 (30 q,
# ~450 cells/model) Llama ~17h < 24h. Bump --time or PSF_STRATA with care; the
# e2e checkpoints every cell, so a timed-out task RESUMES on resubmit (done
# cells are skipped). To resume one model:  sbatch --array=<idx> full_run.sbatch
#
#SBATCH --job-name=psf-full
#SBATCH --partition=gpu_a100_il
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --array=0-2
#SBATCH --output=cluster_logs/%x-%A_%a.out
#SBATCH --error=cluster_logs/%x-%A_%a.err

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

MODELS=(llama_3_1_8b mistral_7b_v03 qwen_2_5_7b)
M="${MODELS[$SLURM_ARRAY_TASK_ID]}"
STRATA="${PSF_STRATA:-10}"
echo "=== array task ${SLURM_ARRAY_TASK_ID} -> model ${M}, strata ${STRATA} ==="
nvidia-smi || true

echo "=== preflight: local_check (${M} only) ==="
if ! python -m prompt_sensitivity.scripts.local_check --models "$M" \
     2>&1 | tee "cluster_logs/full_check_${M}.log"; then
  echo "ABORT: local_check failed for ${M}"; exit 1
fi

echo "=== full e2e: ${M} (context+reasoning, random+gold_first, levels 0..10, own-encoder) ==="
python -m prompt_sensitivity.scripts.e2e_smoke \
  --musique-strata "$STRATA" \
  --families context,reasoning \
  --ladders random,gold_first \
  --levels 0,2,4,6,8,10 \
  --k-samples 2 --max-paraphrases 6 \
  --models "$M" --own-encoder \
  --out "data/full_${M}.parquet" \
  2>&1 | tee "cluster_logs/full_${M}.log"

echo "DONE ${M} $(date -Iseconds)"
