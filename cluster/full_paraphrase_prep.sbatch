#!/bin/bash
# Full pilot, step 1/2 (P1-5): build the SHARED paraphrase universe ONCE.
# Generator = config.paraphrases.generator_model (phi_4_14b after P3-3). The three
# eval jobs reuse this universe — FI_in is comparable across models only at a
# fixed |U_q| — so it must be built here, not concurrently inside the eval jobs.
# P0-5: --max-paraphrases 30.  P0-7: --musique-strata 17 (17 per 2/3/4-hop = 51 q).
#
#SBATCH --job-name=psf-fp-prep
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

if [[ ! -s data/raw/musique/musique_ans_v1.0_dev.jsonl ]]; then
  echo "ERROR: data/raw/musique/musique_ans_v1.0_dev.jsonl missing (README §6b)."; exit 1
fi

echo "=== paraphrase prep: 17 questions / hop-stratum (=51), N<=30 paraphrases ==="
python -m prompt_sensitivity.scripts.e2e_smoke \
  --musique-strata 17 --paraphrase-only --max-paraphrases 30 \
  2>&1 | tee cluster_logs/full_paraphrase_prep.log

echo "DONE $(date -Iseconds)"
