# PromptSensitivityFI demo app

Streamlit research demo: project overview, metric findings (v3), data
explorer, and the interactive 4-head prompt-feedback model — including LIVE
evaluation of unseen prompts on this laptop.

## One-command launch (Streamlit + browser + ngrok tunnel)

```powershell
powershell -ExecutionPolicy Bypass -File app\launch.ps1
```

Starts the app, opens http://localhost:8501, starts the ngrok tunnel with
basic auth **kit / promptsensitivity**, prints the public URL and copies it
to the clipboard. Ctrl+C stops both.

## Manual run

```bash
uv sync --extra app --extra dev    # BOTH extras, or ruff/pytest disappear
uv run streamlit run app/streamlit_app.py
```

## Live custom prompts on this laptop (no cluster needed)

The custom-prompt mode runs the real target model locally for one forward
pass. Requirements, all already handled in this repo:

* CUDA torch build on Windows — pinned via `[tool.uv.sources]` (cu128 index),
  installed by `uv sync`; the RTX 2060 (6 GB) runs the 7B in **4-bit NF4**
  (bitsandbytes, `PSF_4BIT=1`, set automatically by the app; ~5–15 s/analysis).
* Model weights come from the HuggingFace cache/download (Qwen-2.5-7B is
  already mostly cached), **not** from the cluster.
* No GPU at all → automatic CPU fallback (`PSF_DEVICE=cpu`, needs ~15 GB free
  RAM, 30–90 s/analysis). Llama-3.1 is licence-gated on HF; use Qwen locally.

Honesty notes surfaced in the UI: bare prompts are off-distribution vs the
evidence-context training format, and NF4 perturbs hidden states slightly
relative to the bf16 training features.

The tunnel exposes the app publicly while it runs; the app only reads local
parquets/joblib bundles (no write paths).

## Modularity

* All artifact paths live in `PATHS` at the top of `streamlit_app.py` —
  data moved or renamed ⇒ edit one dict.
* Each section checks its artifact and shows the regeneration command when
  missing; new runs (e.g. a v4) only require pointing `PATHS` at new files.
* Caches key on file mtime; the sidebar's "Reload data" clears them after a
  pull without restarting the server.
