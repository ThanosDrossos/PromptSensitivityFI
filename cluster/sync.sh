#!/usr/bin/env bash
# Thin rsync wrapper to push the repo to bwUniCluster 3.0 and pull results back.
#
#   BWUC_USER=ka_xxxxx bash cluster/sync.sh push
#   BWUC_USER=ka_xxxxx bash cluster/sync.sh pull
#
# Host defaults to uc3.scc.kit.edu; override with BWUC_HOST.
# The remote repo lives at $HOME/PromptSensitivityFI on the cluster.
#
# Requires rsync + ssh. On Windows run this from Git Bash or WSL (PowerShell
# has no rsync). See cluster/README.md.

set -euo pipefail

BWUC_HOST="${BWUC_HOST:-uc3.scc.kit.edu}"
REMOTE_DIR="PromptSensitivityFI"   # relative to the cluster $HOME

if [[ -z "${BWUC_USER:-}" ]]; then
  echo "ERROR: set BWUC_USER (e.g. BWUC_USER=ka_xxxxx)" >&2
  exit 2
fi

REMOTE="$BWUC_USER@$BWUC_HOST"

# Exclusions: never sync the venv, large/regenerable data, logs, git, or any
# secret. cluster/ ITSELF is synced (it holds the sbatch + fixture), but
# cluster_logs/ (job output) is pulled, not pushed.
PUSH_EXCLUDES=(
  --exclude=.venv
  --exclude=data
  --exclude=logs
  --exclude=.git
  --exclude=cluster_logs
  --exclude='.env*'
  --exclude=.psf_env
  --exclude=__pycache__
  --exclude='*.pyc'
  --exclude=.pytest_cache
  --exclude=.ruff_cache
)

push() {
  echo ">> push  $PWD/  ->  $REMOTE:$REMOTE_DIR/"
  rsync -avz --delete "${PUSH_EXCLUDES[@]}" \
    ./ "$REMOTE:$REMOTE_DIR/"
}

pull() {
  echo ">> pull  cluster_logs/  +  data/cluster_smoke.parquet"
  mkdir -p ./cluster_logs ./data
  rsync -avz "$REMOTE:$REMOTE_DIR/cluster_logs/" ./cluster_logs/
  # The parquet may not exist yet (e.g. job still queued) — don't fail the pull.
  rsync -avz "$REMOTE:$REMOTE_DIR/data/cluster_smoke.parquet" \
    ./data/cluster_smoke.parquet || \
    echo "   (data/cluster_smoke.parquet not present yet — has the job finished?)"
}

case "${1:-}" in
  push) push ;;
  pull) pull ;;
  *)
    echo "usage: BWUC_USER=ka_xxxxx bash cluster/sync.sh {push|pull}" >&2
    exit 2
    ;;
esac
