#!/bin/bash
# FAST acceptance smoke on the SHORT A100 partition. Use this when gpu_a100_il is
# backlogged: gpu_a100_short = Cascade-Lake A100, 12 nodes (vs 9 on _il), 30-min
# walltime cap, no daytime reservation -> short jobs backfill almost immediately.
#
# Trimmed to 1 MuSiQue question (levels 0,8, k=1) so the whole thing — 4-model
# local_check + Phi-4 paraphrase smoke + 1-q dual-ladder eval — fits well under
# 30 min. This is a PLUMBING gate (verifies Phi-4 loads, local_check gating works,
# and the new columns are populated), NOT the real run. For the real pilot use
# cluster/submit_full_pilot.sh (gpu_a100_il, off-peak).
#
#   sbatch cluster/smoke_short.sbatch
#   squeue --me
#
#SBATCH --job-name=psf-smoke-short
#SBATCH --partition=gpu_a100_short
#SBATCH --time=00:30:00
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

nvidia-smi || true
if [[ ! -s data/raw/musique/musique_ans_v1.0_dev.jsonl ]]; then
  echo "ERROR: data/raw/musique/musique_ans_v1.0_dev.jsonl missing (README §6b)."; exit 1
fi

echo "=== gate 2: local_check (3 eval models all-caps + phi_4_14b generate-only) ==="
if ! python -m prompt_sensitivity.scripts.local_check 2>&1 | tee cluster_logs/smoke_local_check.log; then
  echo "ABORT: local_check failed (check Phi-4 load / chat template)."; exit 1
fi

echo "=== gate 3: Phi-4 paraphrase smoke (1 q) ==="
python -m prompt_sensitivity.scripts.e2e_smoke \
  --musique-direct 1 --paraphrase-only --max-paraphrases 6 \
  2>&1 | tee cluster_logs/smoke_paraphrase.log

echo "=== gate 4: 1-q dual-ladder eval (llama, levels 0,8, k=1, own-encoder + POSIX) ==="
python -m prompt_sensitivity.scripts.e2e_smoke \
  --musique-direct 1 --families context,reasoning --ladders random \
  --levels 0,8 --k-samples 1 --max-paraphrases 6 \
  --models llama_3_1_8b --own-encoder --include-posix \
  --out data/smoke_short.parquet \
  2>&1 | tee cluster_logs/smoke_eval.log

echo "=== results (expect non-null posix_psi/fi_out_var, 21-pt fi_in_curve_vals + CI) ==="
python -m prompt_sensitivity.scripts.show_results --in data/smoke_short.parquet \
  2>&1 | tee cluster_logs/smoke_show.log || true
echo "DONE $(date -Iseconds)"
