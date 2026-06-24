#!/usr/bin/env bash
# Submit the full run as a dependency chain: paraphrase prep -> per-model array.
# Run this ON the cluster (after `bash cluster/run.sh push` from your laptop):
#
#   bash cluster/run_full.sh            # default 10 questions / hop-stratum
#   bash cluster/run_full.sh 16         # 16 / stratum (heavier; watch walltime)
#
# The array only starts if prep succeeds (afterok). Each model writes its own
# data/full_<model>.parquet; merge locally after pulling.

set -euo pipefail
cd "$HOME/PromptSensitivityFI"

STRATA="${1:-10}"
EXPORT="ALL,PSF_STRATA=${STRATA}"

echo ">> paraphrase prep (strata=${STRATA}) ..."
PREP=$(sbatch --parsable --export="${EXPORT}" cluster/full_prep.sbatch)
echo "   prep job id: ${PREP}"

echo ">> per-model array (runs afterok:${PREP}) ..."
ARR=$(sbatch --parsable --dependency=afterok:"${PREP}" --export="${EXPORT}" cluster/full_run.sbatch)
echo "   array job id: ${ARR}  (tasks ${ARR}_0 llama, ${ARR}_1 mistral, ${ARR}_2 qwen)"

echo
echo "watch :  squeue --me"
echo "logs  :  tail -f cluster_logs/psf-full-${ARR}_0.out"
echo "after :  (laptop) bash cluster/run.sh pull"
echo "         (laptop) uv run python -m prompt_sensitivity.scripts.merge_results"
echo "         (laptop) uv run python -m prompt_sensitivity.scripts.show_results --in data/full_run.parquet"
