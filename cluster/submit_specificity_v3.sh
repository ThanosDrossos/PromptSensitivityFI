#!/usr/bin/env bash
# v3 launch orchestration (FI_PROBES_PLAN.md §3-4). Phases, in order:
#
#   bash cluster/submit_specificity_v3.sh smoke    [N]   # default 4 windows
#   bash cluster/submit_specificity_v3.sh backfill [N]   # default 3 — v2 probe data (can run anytime)
#   bash cluster/submit_specificity_v3.sh prep     [N]   # default 16 — Phi-4 universes for 150 q
#   bash cluster/submit_specificity_v3.sh eval     [N]   # default 12 per model — AFTER prep completes!
#   bash cluster/submit_specificity_v3.sh dump     [N]   # default 4 per model — full 150-q features
#
# GATES (do not skip):
#   * smoke must pass analysis on the laptop (graded-F variance, ESS_in != 0,
#     llama/mistral H_sem sane, hidden parquet readable) BEFORE prep/eval.
#   * eval only after the prep chain prints "PREP DONE ... missing=0"
#     (grep cluster_logs/spec_v3_prep.log) — eval chains read the paraphrase
#     cache in parallel and must never need to write it.
#
# Every phase is a singleton chain on gpu_a100_short: same job name +
# --dependency=singleton -> strictly serial windows, resume, surplus no-ops.
# Stop any chain early: scancel --name=<job-name>
set -euo pipefail
cd "$HOME/PromptSensitivityFI"
mkdir -p cluster_logs data

PHASE="${1:-}"
MODELS=(qwen_2_5_7b llama_3_1_8b mistral_7b_v03)

chain() {  # chain <n_windows> <job-name> <sbatch> [--export KEY=V,...]
  local n="$1" name="$2" sbatch_file="$3"; shift 3
  echo ">> ${name}: ${n} windows"
  for _ in $(seq 1 "$n"); do
    sbatch --parsable --job-name="$name" "$@" "$sbatch_file" >/dev/null
  done
}

case "$PHASE" in
  smoke)
    N="${2:-4}"
    chain "$N" psf-v3-smoke cluster/smoke_specificity_v3.sbatch
    echo "watch: tail -f cluster_logs/smoke_v3.log ; then PULL and analyze before prep/eval."
    ;;
  backfill)
    N="${2:-3}"
    # v2 probe-prototype features: qwen, first 50 questions (universes complete).
    chain "$N" psf-v3-dump-qwen_2_5_7b cluster/dump_hidden_v3.sbatch \
      --export=ALL,MODEL=qwen_2_5_7b,N_QUESTIONS=50
    echo "watch: tail -f cluster_logs/dump_hidden_qwen_2_5_7b.log"
    ;;
  prep)
    N="${2:-16}"
    chain "$N" psf-v3-prep cluster/run_specificity_v3.sbatch --export=ALL,PREP_ONLY=1
    echo "watch: tail -f cluster_logs/spec_v3_prep.log — eval only after 'PREP DONE ... missing=0'"
    ;;
  eval)
    N="${2:-12}"
    if ! grep -q "PREP DONE.*missing=0" cluster_logs/spec_v3_prep.log 2>/dev/null; then
      echo "WARNING: no 'PREP DONE ... missing=0' in cluster_logs/spec_v3_prep.log yet."
      echo "         Eval chains would generate missing universes concurrently (cache race)."
      read -r -p "Submit anyway? [yes/NO] " ans
      [[ "${ans}" == "yes" ]] || { echo "aborted."; exit 1; }
    fi
    for m in "${MODELS[@]}"; do
      chain "$N" "psf-v3-${m}" cluster/run_specificity_v3.sbatch --export=ALL,MODEL="$m"
    done
    echo "watch: squeue --me | grep psf-v3 ; logs: cluster_logs/spec_v3_<model>.log"
    ;;
  dump)
    N="${2:-4}"
    for m in "${MODELS[@]}"; do
      chain "$N" "psf-v3-dump-${m}" cluster/dump_hidden_v3.sbatch \
        --export=ALL,MODEL="$m",N_QUESTIONS=150
    done
    echo "watch: cluster_logs/dump_hidden_<model>.log"
    ;;
  *)
    echo "usage: bash cluster/submit_specificity_v3.sh {smoke|backfill|prep|eval|dump} [N_WINDOWS]" >&2
    exit 2
    ;;
esac
echo
squeue --me | grep -E "psf-v3|JOBID" || true
