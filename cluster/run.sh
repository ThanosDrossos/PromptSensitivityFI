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
pull() { bash "$HERE/sync.sh" pull; }
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
