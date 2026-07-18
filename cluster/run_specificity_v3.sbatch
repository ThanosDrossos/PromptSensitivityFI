#!/bin/bash
# v3 full run (FI_PROBES_PLAN.md §4): 150 q x 2 levels, k=10, N=10, uniform
# evidence, GRADED-F track + run-time fi_out_fixed + ESS_in fix (commit 3147bb1).
#
# ONE parametrized sbatch, driven by the submit script via environment:
#   MODEL      eval model key (default qwen_2_5_7b); each model gets its OWN
#              chain (submit-time --job-name) writing its OWN out parquet, so
#              three model chains run in PARALLEL with no write races.
#   PREP_ONLY  "1" -> --prep-only: build the paraphrase-universe cache (Phi-4)
#              and exit before loading any eval model. Run the prep chain to
#              completion BEFORE the eval chains: eval chains treat the cache
#              as read-only only when every universe already exists.
#
# Singleton-chain pattern as v2 (same job name + --dependency=singleton on
# gpu_a100_short): strictly serial 30-min windows, per-universe + per-cell
# resume, surplus windows no-op. See cluster/submit_specificity_v3.sh.
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

if [[ "${PREP_ONLY:-0}" == "1" ]]; then
  echo "=== v3 PREP: ${N_QUESTIONS} q x 2 levels of paraphrase universes (Phi-4) ==="
  python -m prompt_sensitivity.scripts.run_specificity \
    --n-questions "${N_QUESTIONS}" --prep-only --max-paraphrases 10 \
    2>&1 | tee -a cluster_logs/spec_v3_prep.log
else
  echo "=== v3 EVAL [${MODEL}]: ${N_QUESTIONS} q x 2 levels (k=10, N=10, graded, inspect 5) ==="
  python -m prompt_sensitivity.scripts.run_specificity \
    --n-questions "${N_QUESTIONS}" --models "${MODEL}" \
    --k-samples 10 --max-paraphrases 10 --inspect-n 5 \
    --out "data/specificity_v3_${MODEL}.parquet" \
    2>&1 | tee -a "cluster_logs/spec_v3_${MODEL}.log"

  echo "=== level pivot [${MODEL}] ==="
  python -m prompt_sensitivity.scripts.show_specificity \
    --in "data/specificity_v3_${MODEL}.parquet" \
    2>&1 | tee "cluster_logs/show_specificity_v3_${MODEL}.log" || true
fi
echo "WINDOW DONE $(date -Iseconds)"
