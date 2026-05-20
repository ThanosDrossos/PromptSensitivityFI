# Windows PowerShell equivalent of the Makefile. Use if you don't want
# to install GNU make. Run as: `.\tasks.ps1 <target>`.
#
# Each target mirrors the corresponding rule in `Makefile`.

param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet(
        "install", "test", "lint",
        "api-check", "list-models", "data-download", "sample",
        "paraphrases", "paraphrases-smoke", "export-annotation", "compute-kappa",
        "diagnose-paraphrases", "build-ladders", "smoke-metrics",
        "e2e-smoke", "e2e-smoke-dry",
        "sprint1-no-api", "sprint1-verify",
        "clean"
    )]
    [string]$Target
)

$ErrorActionPreference = "Stop"

function Run-UvSync   { uv sync --all-extras; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE } }
function Run-Pytest   { uv run pytest -q;     if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE } }
function Run-Lint {
    uv run ruff check prompt_sensitivity tests
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    uv run ruff format --check prompt_sensitivity tests
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
function Run-Module([string]$mod) {
    uv run python -m $mod
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

switch ($Target) {
    "install"      { Run-UvSync }
    "test"         { Run-Pytest }
    "lint"         { Run-Lint }
    "api-check"    { Run-Module "prompt_sensitivity.scripts.api_check" }
    "list-models"  { Run-Module "prompt_sensitivity.scripts.list_models" }
    "data-download"{ Run-Module "prompt_sensitivity.scripts.download_datasets" }
    "sample"       { Run-Module "prompt_sensitivity.scripts.sample_questions" }
    "paraphrases" {
        uv run python -m prompt_sensitivity.scripts.generate_paraphrases --resume
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    "paraphrases-smoke" {
        uv run python -m prompt_sensitivity.scripts.generate_paraphrases --limit 3 --out data/paraphrases_smoke.parquet
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    "export-annotation"{ Run-Module "prompt_sensitivity.scripts.export_annotation_sample" }
    "compute-kappa"    { Run-Module "prompt_sensitivity.scripts.compute_kappa" }
    "diagnose-paraphrases" { Run-Module "prompt_sensitivity.scripts.diagnose_paraphrases" }
    "build-ladders"    { Run-Module "prompt_sensitivity.scripts.build_ladders" }
    "smoke-metrics"    { Run-Module "prompt_sensitivity.scripts.smoke_metrics" }
    "e2e-smoke"        { Run-Module "prompt_sensitivity.scripts.e2e_smoke" }
    "e2e-smoke-dry" {
        uv run python -m prompt_sensitivity.scripts.e2e_smoke --dry-run
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    "sprint1-no-api" {
        Run-UvSync; Run-Pytest
        Run-Module "prompt_sensitivity.scripts.download_datasets"
        Run-Module "prompt_sensitivity.scripts.sample_questions"
    }
    "sprint1-verify" {
        Run-UvSync; Run-Pytest
        Run-Module "prompt_sensitivity.scripts.list_models"
        Run-Module "prompt_sensitivity.scripts.download_datasets"
        Run-Module "prompt_sensitivity.scripts.sample_questions"
        Run-Module "prompt_sensitivity.scripts.api_check"
    }
    "clean" {
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue `
            .pytest_cache, .ruff_cache, **\__pycache__
        Remove-Item -Force -ErrorAction SilentlyContinue logs\*.log
    }
}
