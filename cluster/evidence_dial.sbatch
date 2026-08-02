#!/bin/bash
# Evidence-dial arm (FINAL_PHASE_PLAN C6): how much does the EVIDENCE side
# buy, holding the specificity dial fixed? qwen x first 50 filtered questions
# x 2 levels at evidence fractions {0.0, 0.5}; the f=1.0 point comes FREE from
# the v3 qwen parquet (same questions, same cells).
#
#   FRACTION   "0.0" (closed-book) or "0.5" (first half of the snippets);
#              each fraction writes its OWN out parquet — the driver resume
#              key does not include the fraction.
#
# Paraphrase universes are fully cached from v3 (same (qid, level) texts) ->
# no Phi-4; generation partially cache-hits v1's closed-book cells where
# prompts coincide. ~200 cells total across both fractions.
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

FRACTION="${FRACTION:-0.5}"
MODEL="${MODEL:-qwen_2_5_7b}"
N_QUESTIONS="${N_QUESTIONS:-50}"
TAG="${FRACTION/./}"   # 0.0 -> 00, 0.5 -> 05 (parquet-name safe)

nvidia-smi | head -12 || true

echo "=== evidence-dial [${MODEL}] f=${FRACTION}: ${N_QUESTIONS} q x 2 levels ==="
python -m prompt_sensitivity.scripts.run_specificity \
  --n-questions "${N_QUESTIONS}" --models "${MODEL}" \
  --k-samples 10 --max-paraphrases 10 \
  --evidence-fraction "${FRACTION}" \
  --out "data/evidence_dial_f${TAG}_${MODEL}.parquet" \
  2>&1 | tee -a "cluster_logs/evidence_dial_f${TAG}_${MODEL}.log"
echo "WINDOW DONE $(date -Iseconds)"
