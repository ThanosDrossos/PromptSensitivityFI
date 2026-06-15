# PromptSensitivityFI — Makefile. Sprint-by-sprint entry points.
# Use `uv` for everything; each `make` target maps to one Sprint-level deliverable.

.PHONY: install test lint sample sprint1-verify list-models clean \
        paraphrases paraphrases-smoke export-annotation compute-kappa \
        diagnose-paraphrases build-ladders smoke-metrics e2e-smoke \
        pilot plot-pilot paraphrases-extend-5 pilot-full \
        pilot-musique pilot-musique-dry pilot-musique-fast \
        show-results e2e-smoke-dry \
        cluster-push cluster-submit cluster-status cluster-pull cluster-smoke

install:
	uv sync --all-extras

test:
	uv run pytest -q

lint:
	uv run ruff check prompt_sensitivity tests
	uv run ruff format --check prompt_sensitivity tests

# --- Sprint 1 entry points ---

# 1.2a — list models exposed by the LiteLLM gateway (run this BEFORE api-check
#        to confirm config.yaml model_ids match what the supervisor registered).
list-models:
	uv run python -m prompt_sensitivity.scripts.list_models

# 1.2b — round-trip determinism + logprob probe on every configured model.
api-check:
	uv run python -m prompt_sensitivity.scripts.api_check

# 1.3 — download HotpotQA + 2WikiMultihopQA validation splits and snapshot.
data-download:
	uv run python -m prompt_sensitivity.scripts.download_datasets

# 1.4 — stratified sample of 100 HotpotQA + 50 2Wiki questions, write data/sample_v1.json.
sample:
	uv run python -m prompt_sensitivity.scripts.sample_questions

# --- Sprint 2 entry points ---

# 2.1-2.3 — full paraphrase pipeline on all 150 sampled questions.
#   Writes data/paraphrases_v1.parquet (accepted + rejected rows).
paraphrases:
	uv run python -m prompt_sensitivity.scripts.generate_paraphrases --resume

# Smoke test: first 3 questions only, useful to verify the gateway path
# before committing to a full ~$5 generator spend.
paraphrases-smoke:
	uv run python -m prompt_sensitivity.scripts.generate_paraphrases --limit 3 --out data/paraphrases_smoke.parquet

# 2.4a — pick 20 questions × 30 paraphrases, write CSV for Thanos + Diener
#        to fill in `thanos` / `diener` columns (yes/no per paraphrase).
export-annotation:
	uv run python -m prompt_sensitivity.scripts.export_annotation_sample

# 2.4b — Cohen's κ. Gate passes iff κ >= 0.8.
compute-kappa:
	uv run python -m prompt_sensitivity.scripts.compute_kappa

# 2.x — diagnostic: read the paraphrase parquet and report rejection
#       breakdown, NLI score distributions, threshold counterfactuals.
diagnose-paraphrases:
	uv run python -m prompt_sensitivity.scripts.diagnose_paraphrases

# --- Sprint 4 entry points ---

# 4.x — orchestrator smoke test on hand-built inputs. No network calls.
#       Sprint-4 gate: must return a plausible 11-scalar MetricTuple.
smoke-metrics:
	uv run python -m prompt_sensitivity.scripts.smoke_metrics

# Cross-sprint E2E smoke: reads cached paraphrases for N questions, builds
# ladders on demand, runs the full metric pipeline against the gateway.
# Defaults: 5 q, 1 model (gpt_4o = kit.gpt-4.1), random ladder, levels {0,4,10},
# k=3 H_sem samples, max 10 paraphrases per question. ~$0.30 + ~15 min.
# Override knobs by calling the module directly with --flags (see --help).
e2e-smoke:
	uv run python -m prompt_sensitivity.scripts.e2e_smoke

e2e-smoke-dry:
	uv run python -m prompt_sensitivity.scripts.e2e_smoke --dry-run

# --- Pilot for supervisor presentation ---

# Extends the paraphrase cache to the first 5 sampled questions so the
# `pilot` target has paraphrases for q4 and q5 (q1-3 are cached from
# `paraphrases-smoke`). With --resume, the existing 3 questions are
# skipped so this only costs the 2 new questions: ~50 min, ~$1.
paraphrases-extend-5:
	uv run python -m prompt_sensitivity.scripts.generate_paraphrases \
		--limit 5 \
		--out data/paraphrases_smoke.parquet \
		--resume

# Full three-ladder × three-level pilot on the first 5 cached questions
# × 2 models (kit.gpt-4.1 + Llama-3.1-8B). k=3 H_sem samples + max 8
# paraphrases per question for tractability on CPU.
# Output: data/pilot_metrics.parquet.
# Run paraphrases-extend-5 FIRST.
# Cost ~$5-6, time ~12-15 h overnight on CPU.
pilot:
	uv run python -m prompt_sensitivity.scripts.e2e_smoke \
		--n-questions 5 \
		--ladders "random,gold_first,distractor_first" \
		--levels "0,4,10" \
		--models "gpt_4o,llama_3_1_8b" \
		--k-samples 3 \
		--max-paraphrases 8 \
		--out data/pilot_metrics.parquet

# Convenience: do the paraphrase extension + the pilot in one command.
pilot-full: paraphrases-extend-5 pilot plot-pilot

# v6 MuSiQue dual-ladder pilot. Samples 5 MuSiQue questions directly (live
# paraphrase generation, cached), runs BOTH the context ladder (random) and
# the reasoning ladder, scores with graded chain-completion F. The headline
# v6 experiment: compare the two ladders' FI_in curves on the same questions.
# Requires MuSiQue available (data/raw/musique/ jsonl or the configured HF
# mirror). Output: data/pilot_musique.parquet.
pilot-musique:
	uv run python -m prompt_sensitivity.scripts.e2e_smoke \
		--musique-direct 5 \
		--families "context,reasoning" \
		--ladders "random,gold_first,distractor_first" \
		--levels "0,4,10" \
		--models "gpt_4o" \
		--k-samples 3 \
		--max-paraphrases 8 \
		--out data/pilot_musique.parquet

pilot-musique-dry:
	uv run python -m prompt_sensitivity.scripts.e2e_smoke \
		--musique-direct 5 --families "context,reasoning" \
		--ladders "random,gold_first,distractor_first" --levels "0,4,10" \
		--models "gpt_4o" --dry-run

# FAST first-look on CPU: 3 MuSiQue questions, ONE context ladder + reasoning,
# --singleton (no paraphrase gen) + --fast (no H_sem clustering). Finishes in
# minutes on CPU. Shows the graded chain-F dual-ladder curve. Resumable +
# checkpointed, so it's safe to Ctrl-C. Output: data/pilot_musique.parquet.
pilot-musique-fast:
	uv run python -m prompt_sensitivity.scripts.e2e_smoke \
		--musique-direct 3 \
		--families "context,reasoning" \
		--ladders "random" \
		--levels "0,4,10" \
		--models "gpt_4o" \
		--singleton --fast \
		--out data/pilot_musique.parquet

# Print whatever results exist (works mid-run thanks to checkpointing) + a PNG.
show-results:
	uv run python -m prompt_sensitivity.scripts.show_results --plot

# Read data/e2e_metrics.parquet (default) or data/pilot_metrics.parquet
# and generate plots + REPORT.md under data/plots/.
plot-pilot:
	uv run python -m prompt_sensitivity.scripts.plot_pilot --in data/pilot_metrics.parquet --out data/plots

# Same plot path but on the e2e_smoke 9-cell output (already on disk).
plot-smoke:
	uv run python -m prompt_sensitivity.scripts.plot_pilot --in data/e2e_metrics.parquet --out data/plots_smoke

# Convenience target: run all Sprint-1 deliverables that don't need API keys.
sprint1-no-api: install test data-download sample

# Convenience target: full Sprint-1 verification (needs .env).
sprint1-verify: install test list-models data-download sample api-check

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__
	rm -f logs/*.log

# --- bwUniCluster 3.0 smoke roundtrip (see cluster/README.md) ---------------
# Requires BWUC_USER (e.g. ka_xxxxx); BWUC_HOST defaults to uc3.scc.kit.edu.
# Needs rsync + ssh; on Windows run from Git Bash or WSL.

cluster-push:   ## rsync repo to bwUniCluster ($HOME/PromptSensitivityFI)
	@bash cluster/sync.sh push

cluster-submit: ## submit the smoke sbatch and print the SLURM job id
	@ssh "$$BWUC_USER@$${BWUC_HOST:-uc3.scc.kit.edu}" \
	  "cd PromptSensitivityFI && mkdir -p cluster_logs data && sbatch cluster/smoke.sbatch"

cluster-status: ## squeue for the user
	@ssh "$$BWUC_USER@$${BWUC_HOST:-uc3.scc.kit.edu}" "squeue --me"

cluster-pull:   ## rsync cluster_logs + data/cluster_smoke.parquet back
	@bash cluster/sync.sh pull

cluster-smoke: cluster-push cluster-submit ## one-shot push + submit
