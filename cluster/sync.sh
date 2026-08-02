#!/usr/bin/env bash
# Thin rsync wrapper to push the repo to bwUniCluster 3.0 and pull results back.
#
#   BWUC_USER=ka_xxxxx bash cluster/sync.sh push
#   BWUC_USER=ka_xxxxx bash cluster/sync.sh pull
#
# Host defaults to uc3.scc.kit.edu; override with BWUC_HOST.
# Custom SSH key: set BWUC_SSH_KEY to the private-key path (e.g. a key NOT in
# ~/.ssh). In Git Bash a Windows path like C:\Users\thano\ssh_key_thanoskit is
# written /c/Users/thano/ssh_key_thanoskit.
# The remote repo lives at $HOME/PromptSensitivityFI on the cluster.
#
# Transport: uses rsync if available (incremental + --delete). Git Bash on
# Windows ships ssh + tar but NOT rsync, so when rsync is absent it falls back
# to tar-over-ssh (same excludes; overlay copy, no remote deletion). Both paths
# need only ssh + tar. See cluster/README.md.

set -euo pipefail

BWUC_HOST="${BWUC_HOST:-uc3.scc.kit.edu}"
REMOTE_DIR="PromptSensitivityFI"   # relative to the cluster $HOME

if [[ -z "${BWUC_USER:-}" ]]; then
  echo "ERROR: set BWUC_USER (e.g. BWUC_USER=ka_xxxxx)" >&2
  exit 2
fi

REMOTE="$BWUC_USER@$BWUC_HOST"

# Build the ssh transport rsync uses. With BWUC_SSH_KEY set, force that key
# (IdentitiesOnly stops the agent from offering other keys first).
#
# NOTE: SSH ControlMaster multiplexing was tried (2026-06-29) to share one auth
# across connections, but Git-Bash/MSYS ssh fails the mux socket
# ("mux_client_request_session: read from master failed: Connection reset by
# peer") and aborts, so it stays OFF. The one-OTP-per-pull win comes instead from
# pull_tar streaming everything over a SINGLE connection (below).
SSH_CMD="ssh"
if [[ -n "${BWUC_SSH_KEY:-}" ]]; then
  SSH_CMD="ssh -i $BWUC_SSH_KEY -o IdentitiesOnly=yes"
fi

# Exclusions: never sync the venv, large/regenerable data, logs, git, or any
# secret. cluster/ ITSELF is synced (it holds the sbatch + fixture), but
# cluster_logs/ (job output) is pulled, not pushed.
#
# ANCHORING MATTERS: a BARE name like `data` matches at ANY depth in both tar
# and rsync, so it also drops the SOURCE package prompt_sensitivity/data/ (the
# 2026-06-23 cluster ImportError: "No module named prompt_sensitivity.data").
# So anchor the regenerable top-level dirs to the repo root, and keep only the
# genuinely any-depth junk (__pycache__, *.pyc, .env*) matching everywhere.
TOPLEVEL_DIRS=(data logs cluster_logs .venv .git .pytest_cache .ruff_cache .psf_env)
ANYDEPTH=(__pycache__)

RSYNC_EXCLUDES=()
TAR_EXCLUDES=(--no-anchored)            # any-depth section
for n in "${ANYDEPTH[@]}"; do
  RSYNC_EXCLUDES+=(--exclude="$n")
  TAR_EXCLUDES+=(--exclude="$n")
done
RSYNC_EXCLUDES+=(--exclude='.env*' --exclude='*.pyc')
TAR_EXCLUDES+=(--exclude='.env*' --exclude='*.pyc')
TAR_EXCLUDES+=(--anchored)              # top-level-only section
for n in "${TOPLEVEL_DIRS[@]}"; do
  RSYNC_EXCLUDES+=(--exclude="/$n")     # rsync: leading / anchors to the transfer root
  TAR_EXCLUDES+=(--exclude="./$n")      # tar: ./ + --anchored anchors to the archive root
done

HAVE_RSYNC=0
command -v rsync >/dev/null 2>&1 && HAVE_RSYNC=1

# Result parquets to fetch on `pull` (best-effort — absent until a job runs).
# cluster_smoke = smoke roundtrip; cluster_e2e_musique = the local e2e pilot;
# cluster_posix_probe = the optional exact-POSIX probe.
PULL_FILES=(
  data/cluster_smoke.parquet
  data/cluster_e2e_musique.parquet
  data/cluster_posix_probe.parquet
  data/paraphrases_musique.parquet
  data/smoke_hsem.parquet
  data/hsem_samples_smoke_hsem.parquet
  data/paraphrases_ambigqa.parquet
  data/smoke_specificity.parquet
  data/specificity_metrics.parquet
  data/smoke_specificity_v2.parquet
  data/specificity_v2_metrics.parquet
  data/smoke_specificity_v3.parquet
  data/specificity_v3_qwen_2_5_7b.parquet
  data/specificity_v3_llama_3_1_8b.parquet
  data/specificity_v3_mistral_7b_v03.parquet
  data/hidden_states_qwen_2_5_7b.parquet
  data/hidden_states_llama_3_1_8b.parquet
  data/hidden_states_mistral_7b_v03.parquet
  data/sensitivity_v2_k20_qwen_2_5_7b.parquet
  data/sensitivity_v2_k20_llama_3_1_8b.parquet
  data/sensitivity_v2_k20_mistral_7b_v03.parquet
  data/posix_arm_qwen_2_5_7b.parquet
  data/posix_arm_llama_3_1_8b.parquet
  data/posix_arm_mistral_7b_v03.parquet
  data/smoke_specificity_ml.parquet
  data/specificity_ml_qwen_2_5_7b.parquet
  data/specificity_ml_llama_3_1_8b.parquet
  data/specificity_ml_mistral_7b_v03.parquet
  data/hidden_states_ml_qwen_2_5_7b.parquet
  data/hidden_states_ml_llama_3_1_8b.parquet
  data/hidden_states_ml_mistral_7b_v03.parquet
  data/midlevel_questions.parquet
  data/ml_review_sample.md
  data/evidence_dial_f00_qwen_2_5_7b.parquet
  data/evidence_dial_f05_qwen_2_5_7b.parquet
  data/vagueness_holdout_qwen_2_5_7b.parquet
  data/vagueness_holdout_llama_3_1_8b.parquet
  data/vagueness_holdout_mistral_7b_v03.parquet
  data/full_llama_3_1_8b.parquet
  data/full_mistral_7b_v03.parquet
  data/full_qwen_2_5_7b.parquet
  data/full_run.parquet
)

# ---- rsync transport (preferred) ------------------------------------------

push_rsync() {
  rsync -avz --delete -e "$SSH_CMD" "${RSYNC_EXCLUDES[@]}" ./ "$REMOTE:$REMOTE_DIR/"
}

pull_rsync() {
  rsync -avz -e "$SSH_CMD" "$REMOTE:$REMOTE_DIR/cluster_logs/" ./cluster_logs/
  for f in "${PULL_FILES[@]}"; do
    rsync -avz -e "$SSH_CMD" "$REMOTE:$REMOTE_DIR/$f" "./$f" \
      || echo "   ($f not present yet)"
  done
  rsync -avz -e "$SSH_CMD" "$REMOTE:$REMOTE_DIR/data/plots/" ./data/plots/ \
    || echo "   (data/plots/ not present yet)"
}

# ---- tar-over-ssh transport (fallback when rsync is absent) ---------------
# Overlay copy (no remote deletion). Needs only ssh + tar, both in Git Bash.

push_tar() {
  echo "   (rsync not found -> tar-over-ssh; overlay copy, no --delete)"
  tar czf - "${TAR_EXCLUDES[@]}" -C "$PWD" . \
    | $SSH_CMD "$REMOTE" "mkdir -p '$REMOTE_DIR' && tar xzf - -C '$REMOTE_DIR'"
}

pull_tar() {
  # ONE ssh connection (=> one OTP prompt) for everything: cluster_logs + every
  # data/*.parquet + data/*.md + data/plots, tarred remotely and streamed back.
  #
  # NOTE the `;` after the items= assignment — it is load-bearing. The
  # assignment's exit status is ls's, and `ls -d A B C` exits NON-ZERO when ANY
  # glob has no match (e.g. no data/*.md yet). With `&&` that aborted the whole
  # chain before tar even ran ("nothing pulled yet" despite parquets existing —
  # the 2026-07-04 bug); with `;` we ignore ls's status and gate on the captured
  # list being non-empty instead.
  # --ignore-failed-read: a file the remote FS cannot read (stripe damage from a
  # Lustre incident — e.g. the three June leftovers that failed 'Cannot stat' on
  # every 2026-07-06 pull) is SKIPPED with a warning instead of failing the whole
  # archive's exit status. Everything readable still transfers.
  $SSH_CMD "$REMOTE" "cd '$REMOTE_DIR' 2>/dev/null || exit 1; items=\$(ls -d cluster_logs data/*.parquet data/*.md data/*.jsonl data/plots 2>/dev/null); [ -n \"\$items\" ] || exit 1; tar czf - --ignore-failed-read \$items" \
    | tar xzf - -C . 2>/dev/null \
    && echo "   pulled cluster_logs + data/*.parquet + data/*.md + data/plots (any
   'Cannot stat' warnings above = damaged remote files that were skipped)" \
    || echo "   PULL FAILED — remote has none of cluster_logs / data/*.parquet /
   data/*.md / data/plots, or the connection died before the stream started."
}

# ---- dispatch -------------------------------------------------------------

push() {
  echo ">> push  $PWD/  ->  $REMOTE:$REMOTE_DIR/"
  if [[ $HAVE_RSYNC -eq 1 ]]; then push_rsync; else push_tar; fi
}

pull() {
  echo ">> pull  cluster_logs/  +  result parquets  +  data/plots/"
  mkdir -p ./cluster_logs ./data ./data/plots
  if [[ $HAVE_RSYNC -eq 1 ]]; then pull_rsync; else pull_tar; fi
}

case "${1:-}" in
  push) push ;;
  pull) pull ;;
  *)
    echo "usage: BWUC_USER=ka_xxxxx bash cluster/sync.sh {push|pull}" >&2
    exit 2
    ;;
esac
