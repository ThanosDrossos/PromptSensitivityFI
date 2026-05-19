# PromptSensitivityFI — Makefile. Sprint-by-sprint entry points.
# Use `uv` for everything; each `make` target maps to one Sprint-level deliverable.

.PHONY: install test lint sample sprint1-verify list-models clean \
        paraphrases paraphrases-smoke export-annotation compute-kappa \
        diagnose-paraphrases build-ladders

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

# Convenience target: run all Sprint-1 deliverables that don't need API keys.
sprint1-no-api: install test data-download sample

# Convenience target: full Sprint-1 verification (needs .env).
sprint1-verify: install test list-models data-download sample api-check

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__
	rm -f logs/*.log
