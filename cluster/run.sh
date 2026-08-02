#!/usr/bin/env bash
# Make-free runner for the bwUniCluster smoke roundtrip. Use this when `make`
# is not installed (e.g. plain Git Bash on Windows). Every subcommand mirrors
# the corresponding `make cluster-*` target.
#
#   export BWUC_USER=ka_jc8392
#   export BWUC_SSH_KEY=/c/Users/thano/ssh_key_thanoskit   # if key not in ~/.ssh
#   bash cluster/run.sh check     # ssh in, print hostname + tooling (no job)
#   bash cluster/run.sh push      # rsync repo to the cluster
#   bash cluster/run.sh submit    # sbatch the smoke job, print job id
#   bash cluster/run.sh status    # squeue --me
#   bash cluster/run.sh pull      # fetch cluster_logs/ + cluster_smoke.parquet
#   bash cluster/run.sh smoke     # push + submit in one shot
#
# Requires ssh + rsync (both ship with Git Bash). See cluster/README.md.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BWUC_HOST="${BWUC_HOST:-uc3.scc.kit.edu}"

if [[ -z "${BWUC_USER:-}" ]]; then
  echo "ERROR: set BWUC_USER (e.g. export BWUC_USER=ka_jc8392)" >&2
  exit 2
fi
REMOTE="$BWUC_USER@$BWUC_HOST"

# ssh transport as an array so a key path is passed safely. (ControlMaster
# multiplexing was tried but Git-Bash/MSYS ssh resets the mux socket and aborts,
# so it's off; each run.sh subcommand authenticates once.)
SSH_CMD=(ssh)
if [[ -n "${BWUC_SSH_KEY:-}" ]]; then
  SSH_CMD=(ssh -i "$BWUC_SSH_KEY" -o IdentitiesOnly=yes)
fi

check() {
  "${SSH_CMD[@]}" "$REMOTE" \
    'echo "CONNECTED as $(whoami) on $(hostname)"; \
     which sbatch squeue; \
     ls -d PromptSensitivityFI >/dev/null 2>&1 && echo "repo present" || echo "repo NOT pushed yet"'
}

submit() {
  "${SSH_CMD[@]}" "$REMOTE" \
    "cd PromptSensitivityFI && mkdir -p cluster_logs data && sbatch cluster/smoke.sbatch"
}

status() {
  "${SSH_CMD[@]}" "$REMOTE" "squeue --me"
}

push() { bash "$HERE/sync.sh" push; }
pull() {
  bash "$HERE/sync.sh" pull
  # Post-pull backfill (2026-07-27 gotcha): cluster parquets never carry the
  # laptop-side sensitivity-v2 columns, so a pull OVERWRITES any locally
  # backfilled rho_f/fi_premium. Recompute right away — idempotent, seconds,
  # and byte-identical to the driver path. Best-effort: a missing venv or file
  # must never fail the pull itself.
  local py=""
  [[ -x "$HERE/../.venv/Scripts/python.exe" ]] && py="$HERE/../.venv/Scripts/python.exe"
  [[ -z "$py" && -x "$HERE/../.venv/bin/python" ]] && py="$HERE/../.venv/bin/python"
  if [[ -n "$py" ]]; then
    local files=()
    for f in "$HERE"/../data/specificity_v3_*.parquet \
             "$HERE"/../data/sensitivity_v2_k20_*.parquet \
             "$HERE"/../data/posix_arm_*.parquet; do
      [[ -f "$f" ]] && files+=("$f")
    done
    if [[ ${#files[@]} -gt 0 ]]; then
      echo ">> post-pull: backfilling sensitivity-v2 columns (${#files[@]} parquet(s))"
      (cd "$HERE/.." && PYTHONUTF8=1 "$py" -m prompt_sensitivity.scripts.backfill_sensitivity_v2 "${files[@]}") \
        || echo "   WARNING: backfill failed — run backfill_sensitivity_v2 manually before analysis"
      echo ">> post-pull: backfilling target_collision flags"
      (cd "$HERE/.." && PYTHONUTF8=1 "$py" -m prompt_sensitivity.scripts.backfill_collisions "${files[@]}") \
        || echo "   WARNING: collision backfill failed — run backfill_collisions manually"
    fi
  else
    echo "   (no local venv found — skip sensitivity-v2 backfill; run it manually)"
  fi
}
smoke() { push; submit; }

case "${1:-}" in
  check)  check ;;
  push)   push ;;
  submit) submit ;;
  status) status ;;
  pull)   pull ;;
  smoke)  smoke ;;
  *)
    echo "usage: bash cluster/run.sh {check|push|submit|status|pull|smoke}" >&2
    exit 2
    ;;
esac
