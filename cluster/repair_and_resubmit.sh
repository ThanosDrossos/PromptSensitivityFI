#!/usr/bin/env bash
# One-command top-up for the v2 specificity run (login node):
#
#   bash cluster/repair_and_resubmit.sh [N_WINDOWS]      # default 6
#
# Does the venv boilerplate itself, then:
#   1. DRY-RUN of repair_spec_universes (shows which questions get redone:
#      the ones whose paraphrase universes are below the N=10 target — their
#      unequal FI_in ceilings made the pair deltas artifacts),
#   2. asks for confirmation,
#   3. applies the repair (drops those universes from the cache + their cells
#      from data/specificity_v2_metrics.parquet),
#   4. resubmits the singleton chain, which regenerates the universes at full
#      N (Phi-4, ~30 min) and re-evaluates the affected cells (fast — the
#      generation cache still holds most samples). The inspection bundle is
#      now written too (jsonl fix), so the pull brings the audit md along.
set -euo pipefail
cd "$HOME/PromptSensitivityFI"

N="${1:-6}"
module purge
module load devel/miniforge/25.3.1-python-3.12
pip install --user -q uv || true
export PATH="$HOME/.local/bin:$PATH"
uv sync
source .venv/bin/activate

echo "================ DRY RUN ================"
python -m prompt_sensitivity.scripts.repair_spec_universes \
  --metrics data/specificity_v2_metrics.parquet

echo
read -r -p "Apply the repair and resubmit ${N} chain windows? [yes/NO] " ans
if [[ "${ans}" != "yes" ]]; then
  echo "aborted — nothing changed."
  exit 0
fi

python -m prompt_sensitivity.scripts.repair_spec_universes \
  --metrics data/specificity_v2_metrics.parquet --apply

bash cluster/submit_specificity_full.sh "${N}"
