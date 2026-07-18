#!/bin/bash
# v3 SMOKE gate (FI_PROBES_PLAN.md §3.1 + §5): 5 questions x 2 levels x ALL 3
# eval models + a TBG hidden-state dump of the same cells. Validates, before
# the full run burns compute:
#   1. graded-F columns populated WITH VARIANCE (the P1-probe prerequisite —
#      binary-F v2 had rho(f, AUFI) = -1.000);
#   2. ESS_in != 0 (paraphrase-text embedding fix);
#   3. H_sem clustering sane on llama + mistral (fix only ever validated on qwen);
#   4. chat_hidden_states end-to-end: dump parquet readable, right shapes;
#   5. model-major eval loop + between-model VRAM free (3 models, one process).
#
# Universes for the first 5 questions already exist in the cache -> no Phi-4.
# Submit as a 3-4 window singleton chain (job psf-v3-smoke); each step resumes.
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

MODELS="qwen_2_5_7b,llama_3_1_8b,mistral_7b_v03"

nvidia-smi | head -12 || true

echo "=== v3 smoke: 5 q x 2 levels x {${MODELS}} (k=10, N=10, graded) ==="
python -m prompt_sensitivity.scripts.run_specificity \
  --n-questions 5 --models "${MODELS}" \
  --k-samples 10 --max-paraphrases 10 --inspect-n 2 \
  --out data/smoke_specificity_v3.parquet \
  2>&1 | tee -a cluster_logs/smoke_v3.log

echo "=== v3 smoke: TBG hidden-state dump (same 5 q, all 3 models) ==="
# Same out files as the full dump — these cells resume-skip later, so the
# smoke's forward passes are not wasted.
python -m prompt_sensitivity.scripts.dump_hidden_states \
  --n-questions 5 --models "${MODELS}" \
  2>&1 | tee -a cluster_logs/smoke_v3_dump.log

echo "=== level pivot ==="
python -m prompt_sensitivity.scripts.show_specificity \
  --in data/smoke_specificity_v3.parquet \
  2>&1 | tee cluster_logs/show_smoke_v3.log || true
echo "WINDOW DONE $(date -Iseconds)"
