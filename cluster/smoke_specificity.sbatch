#!/bin/bash
# Specificity-pivot verification gate (pivot spec §11; v2 uniform-evidence) on gpu_a100_short.
#
# v2: cells now carry the question's own AmbigNQ evidence snippets as a UNIFORM
# context block (config.specificity.context_mode). New question set (evidence-
# coverage filter) => fresh Phi-4 paraphrase prep on first submission; resubmit
# on timeout as before.
#
# ~5 AmbigQA questions x 2 spec levels x qwen, --max-paraphrases 6 --k-samples 3.
# The Phi-4 paraphrase prep for 10 universes is the slow part (~17 min measured),
# so the whole gate may exceed the 30-min cap on the FIRST submission. That is
# fine: the paraphrase parquet AND the per-cell checkpoint persist, so simply
# RESUBMIT — the second run skips finished work and completes in minutes.
#
# VRAM note (job 5762430 post-mortem): gpu_a100_short cards are 40 GB. The
# driver frees the Phi-4 generator (28 GiB) via _free_vram() before loading the
# eval model — without that, Phi-4 + Qwen OOM'd every cell on this partition.
#
#   sbatch cluster/smoke_specificity.sbatch      # (resubmit once if it times out)
#   squeue --me
#
# Prereqs (once, login node): microsoft/phi-4 + Qwen2.5-7B + DeBERTa cached
# (cluster/README.md §prep) and the ambig_qa dataset downloadable (login node has
# internet; run e.g.:
#   python -c "from datasets import load_dataset; load_dataset('ambig_qa','light')"
# so the compute node reads it from the HF cache).
#
#SBATCH --job-name=psf-spec-gate
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
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"

nvidia-smi || true

echo "=== specificity gate: 5 q x 2 levels x qwen (max-paraphrases 6, k=3) ==="
python -m prompt_sensitivity.scripts.smoke_specificity \
  2>&1 | tee cluster_logs/smoke_specificity.log

echo "=== level pivot ==="
python -m prompt_sensitivity.scripts.show_specificity --in data/smoke_specificity_v2.parquet \
  2>&1 | tee cluster_logs/show_specificity.log || true
echo "DONE $(date -Iseconds)"
