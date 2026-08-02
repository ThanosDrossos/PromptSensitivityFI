#!/bin/bash
# SHELVED (2026-08-02, user decision): see run_specificity_ml.sbatch.
# Multi-level ladder SMOKE gate (FINAL_PHASE_PLAN C1): 5 questions x 3 levels
# x qwen. Validates, before the full ML run burns compute:
#   1. L_mid generation + admits-exactly-subset gate end-to-end on Phi-4
#      (coverage printed: accepted/failed of 5);
#   2. fi_spec takes an INTERMEDIATE value at spec_level 1 (log2(m0/|S|));
#   3. guardrails hold on the new path (fixed gold + identical evidence
#      across 3 levels — asserted by the driver's existing machinery);
#   4. the separate ML paraphrase cache fills (no two-level cache contact);
#   5. data/ml_review_sample.md rendered for the HUMAN HARD GATE.
#
# After the pull: READ data/ml_review_sample.md AND APPROVE IT before
# submitting the full run. The full run is not gated technically on this —
# the gate is you.
#
#SBATCH --job-name=psf-ml-smoke
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

echo "=== ML smoke: 5 q x 3 levels x qwen (k=10, N=10, graded) ==="
python -m prompt_sensitivity.scripts.run_specificity \
  --ladder multilevel --n-questions 5 --models qwen_2_5_7b \
  --k-samples 10 --max-paraphrases 10 --inspect-n 2 \
  --out data/smoke_specificity_ml.parquet \
  2>&1 | tee -a cluster_logs/smoke_ml.log

echo "=== ML smoke: level pivot (expect 3 rows/question, fi_spec 0 < mid < top) ==="
python -m prompt_sensitivity.scripts.show_specificity \
  --in data/smoke_specificity_ml.parquet \
  2>&1 | tee cluster_logs/show_smoke_ml.log || true

echo "=== ML smoke: HUMAN REVIEW FILE ==="
python -m prompt_sensitivity.scripts.show_midlevel \
  --n 10 --out data/ml_review_sample.md \
  2>&1 | tee -a cluster_logs/smoke_ml.log

echo "WINDOW DONE $(date -Iseconds)"
