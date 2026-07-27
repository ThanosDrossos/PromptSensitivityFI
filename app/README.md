# PromptSensitivityFI demo app

Streamlit research demo: project overview, metric findings (v3), data
explorer, and the interactive 4-head prompt-feedback model.

## Run (localhost)

```bash
uv sync --extra app          # one-time: installs streamlit into the venv
uv run streamlit run app/streamlit_app.py
```

Opens at http://localhost:8501.

## Share with the supervisor (ngrok)

```bash
ngrok http 8501
```

Send the printed `https://….ngrok-free.app` URL. Recommended: protect it —

```bash
ngrok http 8501 --basic-auth "diener:choose-a-password"
```

Notes: the tunnel exposes the app publicly while it runs; the app only reads
local parquets/joblib bundles (no write paths). The "Custom prompt" mode
needs the target LLM (GPU) and will show a clear error on the laptop — the
dataset-prompt mode is fully functional everywhere.

## Modularity

* All artifact paths live in `PATHS` at the top of `streamlit_app.py` —
  data moved or renamed ⇒ edit one dict.
* Each section checks its artifact and shows the regeneration command when
  missing; new runs (e.g. a v4) only require pointing `PATHS` at new files.
* Caches key on file mtime; the sidebar's "Reload data" clears them after a
  pull without restarting the server.
