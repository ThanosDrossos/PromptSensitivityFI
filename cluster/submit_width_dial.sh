#!/usr/bin/env bash
# R6 width dial — submit orchestration (singleton chains, 30-min windows).
#
#   bash cluster/submit_width_dial.sh prep [N]    # default 4 windows/arm — build
#                                                 # narrow+wide (Phi-4) and swap
#                                                 # (OLMo) universes. RUN FIRST.
#   bash cluster/submit_width_dial.sh eval [N]    # default 3 windows per
#                                                 # (arm x model) chain — 9 chains
#   bash cluster/submit_width_dial.sh check       # progress: universes + cells
#
# Prep MUST finish before eval (one prep chain per arm builds every universe;
# eval chains read the cache read-only — same race-avoidance as the v3 run).
# The medium arm needs NOTHING: it is the existing v3 data.
#
# After eval completes: laptop `bash cluster/run.sh pull`, then
#   uv run python -m prompt_sensitivity.scripts.width_dial_analysis
set -euo pipefail
cd "$HOME/PromptSensitivityFI"
mkdir -p cluster_logs data

PHASE="${1:-}"
ARMS=(narrow wide swap)
MODELS=(qwen_2_5_7b llama_3_1_8b mistral_7b_v03)

chain() {  # chain <n_windows> <job-name> [--export ...]
  local n="$1" name="$2"; shift 2
  echo ">> ${name}: ${n} window(s)"
  for _ in $(seq 1 "$n"); do
    sbatch --parsable --job-name="$name" "$@" cluster/width_dial.sbatch >/dev/null
  done
}

case "$PHASE" in
  prep)
    N="${2:-4}"
    for a in "${ARMS[@]}"; do
      chain "$N" "psf-wd-prep-${a}" --export=ALL,ARM="$a",PHASE=prep
    done
    echo "watch: tail -f cluster_logs/width_<arm>_prep.log"
    echo "done when each log prints: PREP DONE universes=100 missing=0"
    ;;
  eval)
    N="${2:-3}"
    for a in "${ARMS[@]}"; do
      for m in "${MODELS[@]}"; do
        chain "$N" "psf-wd-${a}-${m}" --export=ALL,ARM="$a",PHASE=eval,MODEL="$m"
      done
    done
    echo "watch: tail -f cluster_logs/width_<arm>_<model>.log"
    ;;
  check)
    # Login nodes have no system pandas — use the repo venv that `uv sync`
    # builds inside every sbatch window. Falls back to a raw listing if no
    # job has created the venv yet.
    PY_BIN="python"
    [[ -x .venv/bin/python ]] && PY_BIN=".venv/bin/python"
    "$PY_BIN" - <<'PY' || { echo "(venv/pandas unavailable — raw file listing)"; ls -la data/paraphrases_width_*.parquet data/width_*_*.parquet 2>/dev/null || echo "  nothing yet"; }
import pandas as pd, pathlib
all_prep_done = True
for arm in ("narrow", "wide", "swap"):
    p = pathlib.Path(f"data/paraphrases_width_{arm}.parquet")
    n = len(pd.read_parquet(p).groupby(["question_id","spec_level"])) if p.exists() else 0
    if n < 100:
        all_prep_done = False
    print(f"prep {arm:7s}: {n}/100 universes")
    for m in ("qwen_2_5_7b","llama_3_1_8b","mistral_7b_v03"):
        q = pathlib.Path(f"data/width_{arm}_{m}.parquet")
        c = len(pd.read_parquet(q)) if q.exists() else 0
        print(f"  eval {m:15s}: {c}/100 cells")
print()
print("READY FOR EVAL" if all_prep_done else
      "prep incomplete -- do NOT submit eval yet (eval chains would race to "
      "generate missing universes into the same cache)")
PY
    ;;
  *)
    echo "usage: bash cluster/submit_width_dial.sh {prep|eval|check} [N_WINDOWS]" >&2
    exit 2
    ;;
esac
echo
squeue --me | grep -E "psf-wd|JOBID" || true
