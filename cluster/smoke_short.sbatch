#!/bin/bash
# FAST smoke on the SHORT A100 partition (gpu_a100_short: Cascade-Lake A100,
# 12 nodes, 30-min cap, no daytime reservation -> backfills almost immediately).
#
# Validates the pilot-prep + H_sem-clustering-collapse fix on the model that
# collapsed worst (Qwen2.5-7B had a_q==1 -> H_sem=0 in 100% of cells on 23-Jun).
# Scoped to FIT 30 min: Qwen only, 1 MuSiQue question, context ladder levels 0,8,
# k=10, new defaults (cluster_on=answer, cluster_criterion=label). Exercises the
# Phi-4 generator (live paraphrases), local_check flag-gating, the a_q column +
# degeneracy warning, and the per-sample audit sidecar.
#
# The full 3-model / dual-ladder / k=10 validation is the gpu_a100_il job (run the
# 3-question command in the PR notes, or cluster/e2e_local.sbatch with k bumped).
#
#   bash cluster/run.sh push          # FROM LAPTOP FIRST — the fix must reach the cluster
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

# Fail-fast capability check for the two models this smoke uses: Qwen (eval, all
# caps) and Phi-4 (generator+judge, generate-only via the P3-3b flag gating).
echo "=== local_check: qwen_2_5_7b + phi_4_14b ==="
if ! python -m prompt_sensitivity.scripts.local_check --models qwen_2_5_7b,phi_4_14b \
     2>&1 | tee cluster_logs/smoke_local_check.log; then
  echo "ABORT: local_check failed (check Phi-4 load / chat template)."; exit 1
fi

# H_sem-fix validation. --musique-direct generates the paraphrase universe live
# via the Phi-4 generator, then evaluates Qwen at k=10. --dump-hsem-samples is on
# by default -> data/hsem_samples_smoke_hsem.parquet for the audit.
echo "=== Qwen @ k=10, answer-level + label clustering ==="
python -m prompt_sensitivity.scripts.e2e_smoke \
  --musique-direct 1 --families context --ladders random --levels 0,8 \
  --k-samples 10 --max-paraphrases 6 --models qwen_2_5_7b --own-encoder \
  --out data/smoke_hsem.parquet \
  2>&1 | tee cluster_logs/smoke_hsem_eval.log

# The OUTPUT-CLUSTER DEGENERACY block is the headline: Qwen should now have at
# least one cell with a_q>1 (h_sem_mean>0). If it stays a_q==1, the sidecar tells
# you whether the samples are genuinely identical or still over-merging.
echo "=== results ==="
python -m prompt_sensitivity.scripts.show_results --in data/smoke_hsem.parquet \
  2>&1 | tee cluster_logs/smoke_hsem_show.log || true
echo "DONE $(date -Iseconds)"
