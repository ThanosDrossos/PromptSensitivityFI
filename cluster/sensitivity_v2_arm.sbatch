#!/bin/bash
# Sensitivity-v2 validation arm (METRIC_PROPOSALS.md §4) — overnight-sized.
# The new metrics (rho_F, fi_premium) are already BACKFILLED into v3 from the
# stored per-paraphrase rates; what still needs fresh compute is:
#
#   ARM=k20    50 q x 2 levels at k=20 samples/paraphrase, fresh out file
#              data/sensitivity_v2_k20_${MODEL}.parquet.
#              Purpose: rho_F precision — its noise term has df = N(k-1), so
#              doubling k halves the MSW noise; we test (a) k10-vs-k20
#              stability on the same 50 questions, (b) whether cross-model
#              reliability (0.33 at k=10) rises. The first 10 H_sem samples
#              per paraphrase are generation-cached from v3 -> only 10 new
#              samples/paraphrase actually generate (~1.5-2.5 min/cell,
#              ~100 cells/model: one evening per chain, 3 chains in parallel).
#
#   ARM=posix  qwen x 50 q with --posix on a separate out file
#              data/posix_arm_${MODEL}.parquet — completes the LITERATURE
#              comparison row (Chatterjee 2024) missing from the report card.
#              N^2=100 teacher-forced echo passes/cell (~2-3 min extra),
#              ~100 cells: one evening.
#
# Singleton chains on gpu_a100_short (job name set by the submit script).
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

ARM="${ARM:-k20}"
MODEL="${MODEL:-qwen_2_5_7b}"
N_QUESTIONS="${N_QUESTIONS:-50}"

nvidia-smi | head -12 || true

if [[ "${ARM}" == "k20" ]]; then
  echo "=== v2-arm k20 [${MODEL}]: ${N_QUESTIONS} q x 2 levels, k=20, N=10 ==="
  python -m prompt_sensitivity.scripts.run_specificity \
    --n-questions "${N_QUESTIONS}" --models "${MODEL}" \
    --k-samples 20 --max-paraphrases 10 \
    --out "data/sensitivity_v2_k20_${MODEL}.parquet" \
    2>&1 | tee -a "cluster_logs/v2arm_k20_${MODEL}.log"
elif [[ "${ARM}" == "posix" ]]; then
  echo "=== v2-arm POSIX [${MODEL}]: ${N_QUESTIONS} q x 2 levels, k=10, --posix ==="
  python -m prompt_sensitivity.scripts.run_specificity \
    --n-questions "${N_QUESTIONS}" --models "${MODEL}" \
    --k-samples 10 --max-paraphrases 10 --posix \
    --out "data/posix_arm_${MODEL}.parquet" \
    2>&1 | tee -a "cluster_logs/v2arm_posix_${MODEL}.log"
else
  echo "unknown ARM=${ARM}" >&2; exit 2
fi
echo "WINDOW DONE $(date -Iseconds)"
