#!/usr/bin/env bash
# FINAL RUN orchestration (FINAL_PHASE_PLAN, user-approved scope 2026-08-02):
# C1 multi-level ladder + C3 POSIX x3 models + C6 evidence dial. Phases:
#
#   bash cluster/submit_final_run.sh ml-smoke  [N]  # default 4 windows; 5q x 3 lvl x qwen
#       -> PULL, then READ + APPROVE data/ml_review_sample.md  (HARD GATE)
#   bash cluster/submit_final_run.sh ml-prep   [N]  # default 14 — L_mid + universes (Phi-4)
#   bash cluster/submit_final_run.sh ml-eval   [N]  # default 14/model — AFTER prep + human gate
#   bash cluster/submit_final_run.sh ml-dump   [N]  # default 4/model — TBG states of ML cells
#   bash cluster/submit_final_run.sh posix     [N]  # default 3/model — llama+mistral POSIX
#   bash cluster/submit_final_run.sh dial      [N]  # default 4/fraction — qwen f=0.0 + f=0.5
#
# GATES (do not skip):
#   * ml-smoke analysis on the laptop AND the human review of
#     data/ml_review_sample.md BEFORE ml-prep/ml-eval.
#   * ml-eval only after cluster_logs/spec_ml_prep.log prints
#     "PREP DONE ... missing=0".
#   * posix + dial are independent of the ML phases — submit anytime
#     (they reuse v3 caches; run them the same night as ml-prep).
#
# Every phase = singleton chain(s) on gpu_a100_short (30-min windows, per-cell
# resume, surplus windows no-op). Stop a chain: scancel --name=<job-name>
set -euo pipefail
cd "$HOME/PromptSensitivityFI"
mkdir -p cluster_logs data

PHASE="${1:-}"
ML_MODELS=(qwen_2_5_7b llama_3_1_8b mistral_7b_v03)
POSIX_MODELS=(llama_3_1_8b mistral_7b_v03)   # qwen arm already done 2026-07-26

chain() {  # chain <n_windows> <job-name> <sbatch> [--export ...]
  local n="$1" name="$2" sbatch_file="$3"; shift 3
  echo ">> ${name}: ${n} windows"
  for _ in $(seq 1 "$n"); do
    sbatch --parsable --job-name="$name" "$@" "$sbatch_file" >/dev/null
  done
}

case "$PHASE" in
  ml-smoke)
    N="${2:-4}"
    chain "$N" psf-ml-smoke cluster/smoke_specificity_ml.sbatch
    echo "watch: tail -f cluster_logs/smoke_ml.log"
    echo "THEN: pull + read data/ml_review_sample.md — the full run is NO-GO until approved."
    ;;
  ml-prep)
    N="${2:-14}"
    chain "$N" psf-ml-prep cluster/run_specificity_ml.sbatch --export=ALL,PREP_ONLY=1
    echo "watch: tail -f cluster_logs/spec_ml_prep.log — eval after 'PREP DONE ... missing=0'"
    ;;
  ml-eval)
    N="${2:-14}"
    if ! grep -q "PREP DONE.*missing=0" cluster_logs/spec_ml_prep.log 2>/dev/null; then
      echo "WARNING: prep not finished (no 'PREP DONE ... missing=0' in spec_ml_prep.log)."
      read -r -p "Submit anyway? [yes/NO] " ans
      [[ "${ans}" == "yes" ]] || { echo "aborted."; exit 1; }
    fi
    for m in "${ML_MODELS[@]}"; do
      chain "$N" "psf-ml-${m}" cluster/run_specificity_ml.sbatch --export=ALL,MODEL="$m"
    done
    echo "watch: squeue --me | grep psf-ml ; logs: cluster_logs/spec_ml_<model>.log"
    ;;
  ml-dump)
    N="${2:-4}"
    for m in "${ML_MODELS[@]}"; do
      chain "$N" "psf-ml-dump-${m}" cluster/run_specificity_ml.sbatch \
        --export=ALL,MODEL="$m",DUMP_ONLY=1
    done
    echo "watch: cluster_logs/dump_ml_<model>.log"
    ;;
  posix)
    N="${2:-3}"
    for m in "${POSIX_MODELS[@]}"; do
      chain "$N" "psf-posix-${m}" cluster/sensitivity_v2_arm.sbatch \
        --export=ALL,ARM=posix,MODEL="$m"
    done
    echo "watch: cluster_logs/v2arm_posix_<model>.log (same 50-q subset as the qwen arm)"
    ;;
  dial)
    N="${2:-4}"
    for f in 0.0 0.5; do
      tag="${f/./}"
      chain "$N" "psf-dial-f${tag}" cluster/evidence_dial.sbatch \
        --export=ALL,FRACTION="$f"
    done
    echo "watch: cluster_logs/evidence_dial_f*_qwen_2_5_7b.log (f=1.0 = the v3 parquet)"
    ;;
  *)
    echo "usage: bash cluster/submit_final_run.sh {ml-smoke|ml-prep|ml-eval|ml-dump|posix|dial} [N_WINDOWS]" >&2
    exit 2
    ;;
esac
echo
squeue --me | grep -E "psf-(ml|posix|dial)|JOBID" || true
