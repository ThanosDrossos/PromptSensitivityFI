#!/bin/bash
# R1 — union-gold control arm (review 2026-08-06 §2.1).
#
# Re-scores the EXISTING v3 generations against the union of every
# interpretation's answers, at BOTH specificity levels, so that
#
#     Delta_target = Delta_union + Delta_targeting
#
# separates "the model read better" from "the grader was told which reading it
# wanted". AmbigQA's own protocol (Min et al. 2020 §3.2) scores an ambiguous
# question against ALL interpretations; scoring both levels against the union
# keeps the gold set IDENTICAL across levels, so the fixed-gold guardrail holds.
#
# THIS IS A RE-SCORING JOB, NOT A RE-RUN. No eval model is loaded and no
# generation is issued: every response is recovered from data/cache/
# llm_cache.sqlite by rebuilding the byte-identical LLMRequest. Only DeBERTa
# (the NLI scorer) runs. A cache miss is reported and the cell is SKIPPED --
# never silently regenerated.
#
# Singleton chain on gpu_a100_short: 30-min windows, per-cell resume, a surplus
# window is a clean no-op. One MODEL per chain. Submit via
# cluster/submit_union_gold.sh (which sets --job-name, required by singleton).
#
# ARM=probe   count cache coverage only. No GPU work, no DeBERTa. DO THIS FIRST.
# ARM=full    the actual re-scoring (default).
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

ARM="${ARM:-full}"
MODEL="${MODEL:-qwen_2_5_7b}"
N_QUESTIONS="${N_QUESTIONS:-150}"   # the v3 grid; enumeration verified to match exactly

nvidia-smi | head -12 || true

if [[ "${ARM}" == "probe" ]]; then
  echo "=== R1 union-gold CACHE PROBE [${MODEL}] (no DeBERTa, no generation) ==="
  python -m prompt_sensitivity.scripts.rescore_union_gold \
    --probe --n-questions "${N_QUESTIONS}" --models "${MODEL}" \
    2>&1 | tee -a "cluster_logs/union_gold_probe_${MODEL}.log"
else
  echo "=== R1 union-gold arm [${MODEL}]: ${N_QUESTIONS} q x 2 levels ==="
  python -m prompt_sensitivity.scripts.rescore_union_gold \
    --n-questions "${N_QUESTIONS}" --models "${MODEL}" \
    --out "data/union_gold_${MODEL}.parquet" \
    2>&1 | tee -a "cluster_logs/union_gold_${MODEL}.log"
fi
echo "WINDOW DONE $(date -Iseconds)"
