#!/bin/bash
# R6 — generator-width dial (rho_F positive control) + paraphraser-swap ablation.
#
# ARMS (prompts.WIDTH_ARMS; gates NLI@0.9 + gold judge + dedup identical, only
# the proposal distribution G changes):
#   narrow  3 minimal-edit personas, Phi-4 @ T=0.5
#   medium  == the production v3 universes (NO new compute; the middle point)
#   wide    8 register/syntax/indirectness personas, Phi-4 @ T=1.0
#   swap    medium personas under OLMo-2-13B-Instruct (generator AND gold judge
#           swap together: judge==generator is the production rule, and the
#           40 GB A100 cannot hold OLMo + Phi-4 at once)
#
# PHASES per arm (v3 topology: ONE prep chain, then per-model eval chains):
#   PHASE=prep   build the arm's paraphrase universes (writes the arm cache +
#                the _reject_stats gate-censoring sidecar). Singleton chain.
#   PHASE=eval   run one MODEL over the arm's cached universes. Per-cell resume.
#
# Scope decision 2026-08-07: N_QUESTIONS=50 (the k20/POSIX subset), both levels.
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

ARM="${ARM:?set ARM=narrow|wide|swap}"
PHASE="${PHASE:-prep}"
MODEL="${MODEL:-qwen_2_5_7b}"
N_QUESTIONS="${N_QUESTIONS:-50}"

# One LLM-cache DB per serial chain (2026-08-08 post-mortem): nine concurrent
# eval chains sharing data/cache/llm_cache.sqlite ACROSS NODES over Lustre
# produced "database is locked" (93 failed cells/window) and SIGBUS/exit-135
# crashes — SQLite WAL's shm mmap is unsafe on network filesystems with
# multiple nodes. Each singleton chain is serial, so a per-(phase,arm[,model])
# DB has exactly one writer ever. Prep already completed on the shared DB.
if [[ "${PHASE}" == "prep" ]]; then
  export PSF_CACHE_DB="data/cache/llm_cache_width_prep_${ARM}.sqlite"
else
  export PSF_CACHE_DB="data/cache/llm_cache_width_${ARM}_${MODEL}.sqlite"
fi
echo "PSF_CACHE_DB=${PSF_CACHE_DB}"

nvidia-smi | head -12 || true

if [[ "${PHASE}" == "prep" ]]; then
  echo "=== R6 width-dial PREP [${ARM}]: ${N_QUESTIONS} q x 2 levels ==="
  python -m prompt_sensitivity.scripts.run_specificity \
    --n-questions "${N_QUESTIONS}" --width-arm "${ARM}" \
    --max-paraphrases 10 --prep-only \
    --out "data/width_${ARM}_prep_dummy.parquet" \
    2>&1 | tee -a "cluster_logs/width_${ARM}_prep.log"
else
  echo "=== R6 width-dial EVAL [${ARM} / ${MODEL}] ==="
  python -m prompt_sensitivity.scripts.run_specificity \
    --n-questions "${N_QUESTIONS}" --width-arm "${ARM}" --models "${MODEL}" \
    --k-samples 10 --max-paraphrases 10 \
    --out "data/width_${ARM}_${MODEL}.parquet" \
    2>&1 | tee -a "cluster_logs/width_${ARM}_${MODEL}.log"
fi
echo "WINDOW DONE $(date -Iseconds)"
