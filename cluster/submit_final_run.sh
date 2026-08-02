#!/usr/bin/env bash
# FINAL RUN orchestration — approved scope as of 2026-08-02:
#   * NO constructed ladder (the multi-level L_mid code is SHELVED — see
#     run_specificity_ml.sbatch header) and NO second dataset. The 2-level
#     AmbigQA design stays as-is; target_collision flags are collected
#     dataset-side (driver emits them; run.sh pull backfills old parquets).
#
# Phases (all independent — submit together, e.g. one evening):
#
#   bash cluster/submit_final_run.sh posix    [N]  # default 3/model — llama+mistral POSIX,
#                                                  # same 50-q subset as the done qwen arm
#   bash cluster/submit_final_run.sh dial     [N]  # default 4/fraction — qwen evidence dial
#                                                  # f=0.0 + f=0.5 (f=1.0 = the v3 parquet)
#   bash cluster/submit_final_run.sh holdout  [N]  # default 4/model — vagueness-holdout TBG
#                                                  # dump (2,002 prompts/model, incl. the 830
#                                                  # annotator-labeled non-ambiguous rows)
#
# After completion: bash cluster/run.sh pull  (auto-backfills sensitivity-v2 +
# collision columns), then laptop analyses:
#   python -m prompt_sensitivity.scripts.eval_vagueness_holdout   # frozen heads, OOD
#
# Every phase = singleton chain(s) on gpu_a100_short (30-min windows, per-cell
# resume, surplus windows no-op). Stop a chain: scancel --name=<job-name>
set -euo pipefail
cd "$HOME/PromptSensitivityFI"
mkdir -p cluster_logs data

PHASE="${1:-}"
ALL_MODELS=(qwen_2_5_7b llama_3_1_8b mistral_7b_v03)
POSIX_MODELS=(llama_3_1_8b mistral_7b_v03)   # qwen arm already done 2026-07-26

chain() {  # chain <n_windows> <job-name> <sbatch> [--export ...]
  local n="$1" name="$2" sbatch_file="$3"; shift 3
  echo ">> ${name}: ${n} windows"
  for _ in $(seq 1 "$n"); do
    sbatch --parsable --job-name="$name" "$@" "$sbatch_file" >/dev/null
  done
}

case "$PHASE" in
  posix)
    N="${2:-3}"
    for m in "${POSIX_MODELS[@]}"; do
      chain "$N" "psf-posix-${m}" cluster/sensitivity_v2_arm.sbatch \
        --export=ALL,ARM=posix,MODEL="$m"
    done
    echo "watch: cluster_logs/v2arm_posix_<model>.log"
    ;;
  dial)
    N="${2:-4}"
    for f in 0.0 0.5; do
      tag="${f/./}"
      chain "$N" "psf-dial-f${tag}" cluster/evidence_dial.sbatch \
        --export=ALL,FRACTION="$f"
    done
    echo "watch: cluster_logs/evidence_dial_f*_qwen_2_5_7b.log"
    ;;
  holdout)
    N="${2:-4}"
    for m in "${ALL_MODELS[@]}"; do
      chain "$N" "psf-vh-${m}" cluster/dump_vagueness_holdout.sbatch \
        --export=ALL,MODEL="$m"
    done
    echo "watch: cluster_logs/vh_dump_<model>.log"
    ;;
  *)
    echo "usage: bash cluster/submit_final_run.sh {posix|dial|holdout} [N_WINDOWS]" >&2
    exit 2
    ;;
esac
echo
squeue --me | grep -E "psf-(posix|dial|vh)|JOBID" || true
