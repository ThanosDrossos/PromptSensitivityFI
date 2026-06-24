#!/bin/bash
# Full run, step 1/2: build the SHARED paraphrase universe once.
# All 3 per-model array tasks must reuse the SAME paraphrases (FI_in is only
# comparable across models when |U_q| is identical), so generation happens here,
# once, before the array — not concurrently inside it (which would race on
# data/paraphrases_musique.parquet).
#
# Submit via cluster/run_full.sh (sets PSF_STRATA + the array dependency).
#
#SBATCH --job-name=psf-full-prep
#SBATCH --partition=gpu_a100_il
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=cluster_logs/%x-%j.out
#SBATCH --error=cluster_logs/%x-%j.err

set -euo pipefail
cd "$HOME/PromptSensitivityFI"
mkdir -p cluster_logs data

module purge
module load devel/miniforge/25.3.1-python-3.12
pip install --user -q uv || true
export PATH="$HOME/.local/bin:$PATH"
uv sync
source .venv/bin/activate
if [[ -f "$HOME/.psf_env" ]]; then set -a; source "$HOME/.psf_env"; set +a; fi
export TOKENIZERS_PARALLELISM=false

STRATA="${PSF_STRATA:-10}"     # questions PER hop-stratum (2/3/4-hop)

if [[ ! -s data/raw/musique/musique_ans_v1.0_dev.jsonl ]]; then
  echo "ERROR: data/raw/musique/musique_ans_v1.0_dev.jsonl missing (README §6b)."; exit 1
fi

echo "=== paraphrase prep: $STRATA questions / hop-stratum (generator = qwen_2_5_7b) ==="
python -m prompt_sensitivity.scripts.e2e_smoke \
  --musique-strata "$STRATA" --paraphrase-only --max-paraphrases 6 \
  2>&1 | tee cluster_logs/full_prep.log

echo "DONE $(date -Iseconds)"
