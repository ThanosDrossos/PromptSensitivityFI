#!/usr/bin/env bash
# Submit the full v1 specificity run as a SINGLETON CHAIN of 30-min
# gpu_a100_short windows (see run_specificity_full.sbatch for why).
#
#   bash cluster/submit_specificity_full.sh [N_WINDOWS]   # default 14 (= 7 h budget)
#
# All copies share --job-name + --dependency=singleton -> SLURM runs them
# strictly serially; each resumes from the checkpoint; surplus windows no-op in
# ~2 min. If the run is still unfinished after all windows (watch
# cluster_logs/spec_full_v2.log), just submit more the same way. To stop early:
#   scancel --name=psf-spec-full
set -euo pipefail
cd "$HOME/PromptSensitivityFI"
mkdir -p cluster_logs data

N="${1:-14}"
echo "submitting ${N} singleton windows of psf-spec-full:"
for _ in $(seq 1 "$N"); do
  sbatch --parsable cluster/run_specificity_full.sbatch
done
echo
squeue --me --name=psf-spec-full
echo
echo "watch    : squeue --me --name=psf-spec-full"
echo "progress : tail -f cluster_logs/spec_full_v2.log   (look for 'cell N done' / 'universe i/n')"
echo "when done: (laptop) bash cluster/run.sh pull -> data/specificity_v2_metrics.parquet"
echo "           + data/inspect_specificity_v2_metrics.md (audit bundle)"
