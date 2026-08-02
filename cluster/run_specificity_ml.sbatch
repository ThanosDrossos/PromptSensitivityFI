#!/bin/bash
# Multi-level ladder FULL run (FINAL_PHASE_PLAN C1): 100 q (m0 in 3..5) x 3
# levels (L0 / L_mid / L_top = spec_level 0/1/2), k=10, N=10, uniform evidence,
# graded track. LAUNCH ONLY AFTER the human review of data/ml_review_sample.md
# (the smoke's HARD GATE) was approved.
#
# Parametrized like v3, driven by cluster/submit_specificity_ml.sh:
#   MODEL      eval model key; per-model chains in parallel, own out parquet.
#   PREP_ONLY  "1" -> generate L_mid rewrites + the ML paraphrase cache
#              (data/paraphrases_ambigqa_ml.parquet — SEPARATE from the
#              two-level cache) and exit. Run prep to completion first.
#   DUMP_ONLY  "1" -> TBG hidden-state dump of the ML cells (cache-only,
#              never generates; per-model out data/hidden_states_ml_<M>.parquet).
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
N_QUESTIONS="${N_QUESTIONS:-100}"

nvidia-smi | head -12 || true

if [[ "${PREP_ONLY:-0}" == "1" ]]; then
  echo "=== ML PREP: ${N_QUESTIONS} q -> L_mid rewrites + 3-level universes (Phi-4) ==="
  python -m prompt_sensitivity.scripts.run_specificity \
    --ladder multilevel --n-questions "${N_QUESTIONS}" --prep-only \
    --max-paraphrases 10 \
    2>&1 | tee -a cluster_logs/spec_ml_prep.log
  echo "=== ML PREP: refresh the human-review file (full pool) ==="
  python -m prompt_sensitivity.scripts.show_midlevel \
    --n 10 --out data/ml_review_sample.md \
    2>&1 | tee -a cluster_logs/spec_ml_prep.log || true
elif [[ "${DUMP_ONLY:-0}" == "1" ]]; then
  echo "=== ML DUMP [${MODEL}]: TBG states of the 3-level cells ==="
  python -m prompt_sensitivity.scripts.dump_hidden_states \
    --ladder multilevel --n-questions "${N_QUESTIONS}" --models "${MODEL}" \
    --out-template "data/hidden_states_ml_{model}.parquet" \
    2>&1 | tee -a "cluster_logs/dump_ml_${MODEL}.log"
else
  echo "=== ML EVAL [${MODEL}]: ${N_QUESTIONS} q x 3 levels (k=10, N=10, graded) ==="
  python -m prompt_sensitivity.scripts.run_specificity \
    --ladder multilevel --n-questions "${N_QUESTIONS}" --models "${MODEL}" \
    --k-samples 10 --max-paraphrases 10 --inspect-n 5 \
    --out "data/specificity_ml_${MODEL}.parquet" \
    2>&1 | tee -a "cluster_logs/spec_ml_${MODEL}.log"

  echo "=== level pivot [${MODEL}] ==="
  python -m prompt_sensitivity.scripts.show_specificity \
    --in "data/specificity_ml_${MODEL}.parquet" \
    2>&1 | tee "cluster_logs/show_specificity_ml_${MODEL}.log" || true
fi
echo "WINDOW DONE $(date -Iseconds)"
