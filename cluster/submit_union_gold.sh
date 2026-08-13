#!/usr/bin/env bash
# R1 union-gold arm — submit orchestration. Same singleton-chain pattern as
# submit_final_run.sh: N windows of 30 min per model, per-cell resume, surplus
# windows no-op.
#
#   bash cluster/submit_union_gold.sh probe        # <- ALWAYS DO THIS FIRST
#   bash cluster/submit_union_gold.sh full   [N]   # default 3 windows/model
#
# WHY PROBE FIRST. The arm only works if the v3 generations are still in
# data/cache/llm_cache.sqlite on the cluster. The probe counts cache hits without
# loading DeBERTa and finishes in seconds. If coverage is not ~100%, STOP: the
# arm would become a full re-run, which is a different (much larger) decision --
# do not just launch the full chain and hope.
#
# Check the probe result:
#   grep "cache" cluster_logs/union_gold_probe_*.log
# You want lines reading `cache 11000/11000 hits (100.0%)`.
#
# Stop a chain:  scancel --name=psf-ugold-<model>
set -euo pipefail
cd "$HOME/PromptSensitivityFI"
mkdir -p cluster_logs data

PHASE="${1:-}"
ALL_MODELS=(qwen_2_5_7b llama_3_1_8b mistral_7b_v03)

chain() {  # chain <n_windows> <job-name> <sbatch> [--export ...]
  local n="$1" name="$2" sbatch_file="$3"; shift 3
  echo ">> ${name}: ${n} window(s)"
  for _ in $(seq 1 "$n"); do
    sbatch --parsable --job-name="$name" "$@" "$sbatch_file" >/dev/null
  done
}

case "$PHASE" in
  probe)
    for m in "${ALL_MODELS[@]}"; do
      chain 1 "psf-ugold-probe-${m}" cluster/union_gold_arm.sbatch \
        --export=ALL,ARM=probe,MODEL="$m"
    done
    echo
    echo "when they finish:  grep -h 'cache' cluster_logs/union_gold_probe_*.log"
    echo "expect ~100% hits. If not, STOP and re-scope before running 'full'."
    ;;
  full)
    N="${2:-3}"
    for m in "${ALL_MODELS[@]}"; do
      chain "$N" "psf-ugold-${m}" cluster/union_gold_arm.sbatch \
        --export=ALL,ARM=full,MODEL="$m"
    done
    echo
    echo "watch: tail -f cluster_logs/union_gold_<model>.log"
    ;;
  *)
    echo "usage: bash cluster/submit_union_gold.sh {probe|full} [N_WINDOWS]" >&2
    exit 2
    ;;
esac
echo
squeue --me | grep -E "psf-ugold|JOBID" || true
