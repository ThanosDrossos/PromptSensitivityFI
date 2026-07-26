#!/usr/bin/env bash
# Overnight sensitivity-v2 validation arm (METRIC_PROPOSALS.md §4).
#
#   bash cluster/submit_sensitivity_v2.sh all   [N]   # k20 x 3 models + posix (default N=10 windows each)
#   bash cluster/submit_sensitivity_v2.sh k20   [N]   # just the k=20 chains
#   bash cluster/submit_sensitivity_v2.sh posix [N]   # just the POSIX arm (qwen)
#
# NOTE: rho_F + fi_premium need NO new run for v3 itself — they are already
# backfilled from stored per-paraphrase rates. These arms add (a) k=20 for
# rho_F precision/stability, (b) POSIX for the literature-comparison row.
# All universes + the first k=10 samples come from cache: only the increments
# compute. Everything fits one night on gpu_a100_short chains.
set -euo pipefail
cd "$HOME/PromptSensitivityFI"
mkdir -p cluster_logs data

PHASE="${1:-all}"
N="${2:-10}"
MODELS=(qwen_2_5_7b llama_3_1_8b mistral_7b_v03)

chain() {  # chain <n> <job-name> <export...>
  local n="$1" name="$2"; shift 2
  echo ">> ${name}: ${n} windows"
  for _ in $(seq 1 "$n"); do
    sbatch --parsable --job-name="$name" "$@" cluster/sensitivity_v2_arm.sbatch >/dev/null
  done
}

case "$PHASE" in
  k20|all)
    for m in "${MODELS[@]}"; do
      chain "$N" "psf-v2arm-k20-${m}" --export=ALL,ARM=k20,MODEL="$m"
    done
    ;;&
  posix|all)
    chain "$N" "psf-v2arm-posix" --export=ALL,ARM=posix,MODEL=qwen_2_5_7b
    ;;
esac
[[ "$PHASE" =~ ^(k20|posix|all)$ ]] || { echo "usage: $0 {all|k20|posix} [N]" >&2; exit 2; }
echo
squeue --me | grep -E "psf-v2arm|JOBID" || true
echo "watch: cluster_logs/v2arm_*.log ; pull when quiet -> laptop analysis:"
echo "  k10-vs-k20 rho_F stability + reliability, POSIX report-card row."
