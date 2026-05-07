# PromptSensitivityFI — Makefile. Sprint-by-sprint entry points.
# Use `uv` for everything; each `make` target maps to one Sprint-level deliverable.

.PHONY: install test lint sample sprint1-verify clean

install:
	uv sync --all-extras

test:
	uv run pytest -q

lint:
	uv run ruff check prompt_sensitivity tests
	uv run ruff format --check prompt_sensitivity tests

# --- Sprint 1 entry points ---

# 1.2 — round-trip determinism: hits every configured API at temperature=0 twice.
api-check:
	uv run python -m prompt_sensitivity.scripts.api_check

# 1.3 — download HotpotQA + 2WikiMultihopQA validation splits and snapshot.
data-download:
	uv run python -m prompt_sensitivity.scripts.download_datasets

# 1.4 — stratified sample of 100 HotpotQA + 50 2Wiki questions, write data/sample_v1.json.
sample:
	uv run python -m prompt_sensitivity.scripts.sample_questions

# Convenience target: run all Sprint-1 deliverables that don't need API keys.
sprint1-no-api: install test data-download sample

# Convenience target: full Sprint-1 verification (needs .env).
sprint1-verify: install test data-download sample api-check

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__
	rm -f logs/*.log
