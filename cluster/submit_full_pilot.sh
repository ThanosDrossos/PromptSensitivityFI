#!/usr/bin/env bash
# Submit the full pilot (P1-5): paraphrase prep -> 3 per-model eval jobs, each
# gated on the prep with --dependency=afterok. Run ON the cluster after
# `bash cluster/run.sh push` from your laptop:
#
#   bash cluster/submit_full_pilot.sh
#
# Each eval writes data/full_<model>.parquet (no shared-file race). After all
# three finish, pull and merge on your laptop (see the printed commands).

set -euo pipefail
cd "$HOME/PromptSensitivityFI"

PREP=$(sbatch --parsable cluster/full_paraphrase_prep.sbatch)
echo "prep job: ${PREP}"

for M in llama_3_1_8b mistral_7b_v03 qwen_2_5_7b; do
  J=$(sbatch --parsable --dependency=afterok:"${PREP}" "cluster/full_eval_${M}.sbatch")
  echo "eval ${M}: ${J}  (afterok:${PREP})"
done

echo
echo "watch :  squeue --me"
echo "after :  (laptop) bash cluster/run.sh pull"
echo "         (laptop) uv run python -m prompt_sensitivity.scripts.merge_results"
echo "         (laptop) uv run python -m prompt_sensitivity.scripts.show_results --in data/full_run.parquet"
