"""Preflight for the in-process `local` (transformers) backend — Sprint 6.

Replaces `api_check` on the cluster path. For every `provider: local` model in
config it verifies the four capabilities the metric stack depends on, loading
ONE model at a time and freeing its VRAM before the next (so a 40 GB A100 is
enough even with three 8B models configured):

  1. generate           — greedy `complete` returns non-empty text
  2. chat logprobs       — `complete(logprobs=True)` returns per-token logprobs
  3. teacher-forced score — `score_continuation` returns per-token logprobs (POSIX)
  4. hidden states        — `embed_hidden` returns a finite (1, D) vector

Prints a PASS/FAIL table and exits non-zero if ANY model fails ANY check — the
gate the sbatch checks before spending walltime on the e2e.

Run: `python -m prompt_sensitivity.scripts.local_check`
"""

from __future__ import annotations

import argparse
import gc
import sys
from dataclasses import dataclass

import numpy as np
from loguru import logger

from ..config import load_config
from ..logging_setup import configure_logging
from ..models import LLMRequest
from ..models.registry import get_client, reset_clients
from ..models.schemas import CompletionRequest


PROBE_USER_MESSAGE = "Reply with exactly the single word: pong"
SCORE_PROMPT = "The capital of France is Paris."
HIDDEN_PROBE = "hidden state probe"


@dataclass
class LocalCheckRow:
    model_key: str
    model_id: str
    generate_ok: bool
    logprobs_ok: bool | None   # None = not required (capability flag off)
    score_ok: bool | None
    hidden_ok: bool | None
    hidden_dim: int | None
    text: str
    error: str | None = None

    @property
    def passed(self) -> bool:
        # P3-3b: a capability fails the gate only if it was REQUIRED (flag on ->
        # not None) and came back False. A generator (all flags off) passes on
        # generate alone.
        return (
            self.error is None
            and self.generate_ok
            and self.logprobs_ok is not False
            and self.score_ok is not False
            and self.hidden_ok is not False
        )


def _okstr(v: bool | None) -> str:
    return "n/a" if v is None else ("yes" if v else "NO")


def _free_vram() -> None:
    """Drop the loaded weights so the next model starts from a clean GPU."""
    try:
        from ..models.local_hf import _load_model

        _load_model.cache_clear()
    except Exception:  # noqa: BLE001
        pass
    reset_clients()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def _probe(model_key: str) -> LocalCheckRow:
    config = load_config()
    entry = config.models[model_key]
    try:
        client = get_client(model_key, config)

        # 1. greedy generate
        gen = client.complete(LLMRequest(
            provider=entry.provider,  # type: ignore[arg-type]
            model_id=entry.model_id,
            messages=[{"role": "user", "content": PROBE_USER_MESSAGE}],
            temperature=0.0, top_p=1.0, max_tokens=8, seed=42,
            purpose="local_check_generate",
        ))
        generate_ok = bool(gen.text.strip())

        # 2. chat logprobs — only required if the model claims it (P3-3b: a
        #    generator like Phi-4 has the flag off, so this is skipped/n-a).
        logprobs_ok: bool | None = None
        if entry.chat_logprobs:
            lp = client.complete(LLMRequest(
                provider=entry.provider,  # type: ignore[arg-type]
                model_id=entry.model_id,
                messages=[{"role": "user", "content": PROBE_USER_MESSAGE}],
                temperature=0.0, top_p=1.0, max_tokens=8, seed=42,
                logprobs=True, top_logprobs=5,
                purpose="local_check_logprobs",
            ))
            logprobs_ok = bool(lp.token_logprobs) and len(lp.token_logprobs[0].top_logprobs) > 0

        # 3. exact teacher-forced score (POSIX) — only if echo_completions.
        score_ok: bool | None = None
        if entry.echo_completions:
            sc = client.score_continuation(CompletionRequest(
                provider=entry.provider,  # type: ignore[arg-type]
                model_id=entry.model_id,
                prompt=SCORE_PROMPT, max_tokens=0, echo=True, logprobs=1, temperature=0.0,
                purpose="local_check_score",
            ))
            score_ok = bool(sc.token_logprobs) and all(
                np.isfinite(t.logprob) for t in sc.token_logprobs
            )

        # 4. hidden states — only if has_hidden.
        hidden_ok: bool | None = None
        hidden_dim: int | None = None
        if entry.has_hidden:
            vec = client.embed_hidden([HIDDEN_PROBE])
            hidden_ok = (
                isinstance(vec, np.ndarray)
                and vec.shape[0] == 1
                and vec.ndim == 2
                and vec.shape[1] > 0
                and bool(np.isfinite(vec).all())
            )
            hidden_dim = int(vec.shape[1]) if vec.ndim == 2 and vec.shape[0] else None

        return LocalCheckRow(
            model_key=model_key, model_id=entry.model_id,
            generate_ok=generate_ok, logprobs_ok=logprobs_ok,
            score_ok=score_ok, hidden_ok=hidden_ok, hidden_dim=hidden_dim,
            text=gen.text.strip()[:40],
        )
    except Exception as exc:  # noqa: BLE001 — surface to operator
        logger.exception("local_check failed for {}", model_key)
        return LocalCheckRow(
            model_key=model_key, model_id=entry.model_id,
            generate_ok=False, logprobs_ok=False, score_ok=False,
            hidden_ok=False, hidden_dim=None, text="", error=str(exc)[:200],
        )
    finally:
        _free_vram()


def main() -> int:
    ap = argparse.ArgumentParser(description="Preflight for provider:local models.")
    ap.add_argument("--models", default=None,
                    help="comma list of model keys to check (default: all provider:local).")
    args = ap.parse_args()

    configure_logging("local_check")
    config = load_config()
    reset_clients()

    local_keys = [k for k, e in config.models.items() if e.provider == "local"]
    if args.models:
        want = {m.strip() for m in args.models.split(",") if m.strip()}
        local_keys = [k for k in local_keys if k in want]
    if not local_keys:
        logger.error("no provider:local models to check (filter={!r})", args.models)
        return 1
    logger.info("checking {} local models: {}", len(local_keys), local_keys)

    rows = [_probe(k) for k in local_keys]

    print()
    print("=" * 96)
    header = (
        f"{'model_key':<18} {'gen':>4} {'logp':>5} {'score':>6} {'hidden':>7} "
        f"{'dim':>5} {'PASS':>5}  text"
    )
    print(header)
    print("-" * 96)
    n_pass = 0
    for r in rows:
        if r.error:
            print(f"{r.model_key:<18} {'ERR':>4} {'ERR':>5} {'ERR':>6} {'ERR':>7} "
                  f"{'-':>5} {'NO':>5}  {r.error[:40]}")
            continue
        n_pass += int(r.passed)
        print(
            f"{r.model_key:<18} "
            f"{('yes' if r.generate_ok else 'NO'):>4} "
            f"{_okstr(r.logprobs_ok):>5} "
            f"{_okstr(r.score_ok):>6} "
            f"{_okstr(r.hidden_ok):>7} "
            f"{(str(r.hidden_dim) if r.hidden_dim else '-'):>5} "
            f"{('yes' if r.passed else 'NO'):>5}  {r.text!r}"
        )
    print("=" * 96)
    print(f"PASS: {n_pass}/{len(rows)} local models meet their required capabilities "
          "(generate always; logprobs/score/hidden only where the config flag is on)")
    return 0 if n_pass == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
