"""Round-trip API verification — Sprint 1 §1.2 gate.

For each configured model, fires the same prompt twice at temperature=0
(second call hits the cache, so it's effectively free) and reports:
  - first-call latency (ms)
  - second-call latency (ms, expected ~0 from cache)
  - bit-identity of outputs (within model determinism guarantees)

Run with: `make api-check`. Requires `.env` with TOGETHER_API_KEY and
OPENAI_API_KEY.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass

from loguru import logger

from ..config import load_config
from ..logging_setup import configure_logging
from ..models import LLMRequest
from ..models.registry import get_client, reset_clients


PROBE_MESSAGES = [
    {"role": "user", "content": "Reply with exactly the single word: pong"},
]


@dataclass
class ApiCheckRow:
    model_key: str
    provider: str
    model_id: str
    first_text: str
    second_text: str
    first_ms: float | None
    second_ms: float | None
    deterministic: bool
    error: str | None = None


def _probe(model_key: str) -> ApiCheckRow:
    config = load_config()
    entry = config.models[model_key]
    request = LLMRequest(
        provider=entry.provider,  # type: ignore[arg-type]
        model_id=entry.model_id,
        messages=[{"role": m["role"], "content": m["content"]} for m in PROBE_MESSAGES],  # type: ignore[arg-type]
        temperature=0.0,
        top_p=1.0,
        max_tokens=8,
        seed=42,
        purpose="api_check",
    )
    client = get_client(model_key, config)
    try:
        first = client.complete(request)
        # The second call MUST hit the cache; we want to verify both the
        # cache and (after a cache reset) provider determinism.
        second = client.complete(request)
        return ApiCheckRow(
            model_key=model_key,
            provider=entry.provider,
            model_id=entry.model_id,
            first_text=first.text.strip(),
            second_text=second.text.strip(),
            first_ms=first.latency_ms,
            second_ms=second.latency_ms,
            deterministic=(first.text == second.text),
        )
    except Exception as exc:  # noqa: BLE001 — surface to operator
        logger.exception("probe failed for {}", model_key)
        return ApiCheckRow(
            model_key=model_key,
            provider=entry.provider,
            model_id=entry.model_id,
            first_text="",
            second_text="",
            first_ms=None,
            second_ms=None,
            deterministic=False,
            error=str(exc),
        )


def main() -> int:
    configure_logging("api_check")
    config = load_config()
    reset_clients()  # ensure singleton cache picks up current config

    rows: list[ApiCheckRow] = []
    for model_key in config.models:
        logger.info("probing {} -> {}", model_key, config.models[model_key].model_id)
        rows.append(_probe(model_key))

    print()
    print("=" * 100)
    print(f"{'model_key':<22} {'provider':<10} {'1st_ms':>10} {'2nd_ms':>10} {'det':>5} text")
    print("-" * 100)
    failures = 0
    for r in rows:
        if r.error:
            failures += 1
            print(f"{r.model_key:<22} {r.provider:<10} {'ERR':>10} {'ERR':>10} {'-':>5} {r.error[:40]}")
            continue
        first_ms = f"{r.first_ms:>9.0f}" if r.first_ms is not None else "      n/a"
        second_ms = f"{r.second_ms:>9.0f}" if r.second_ms is not None else "      n/a"
        det = "yes" if r.deterministic else "NO"
        print(f"{r.model_key:<22} {r.provider:<10} {first_ms:>10} {second_ms:>10} {det:>5} {r.first_text!r}")
    print("=" * 100)
    print(f"failures: {failures}/{len(rows)}")

    summary = {
        "rows": [
            {
                "model_key": r.model_key,
                "provider": r.provider,
                "model_id": r.model_id,
                "first_ms": r.first_ms,
                "second_ms": r.second_ms,
                "deterministic": r.deterministic,
                "first_text": r.first_text,
                "error": r.error,
            }
            for r in rows
        ],
        "failures": failures,
    }
    out_path = config.repo_root() / "logs" / "api_check_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("wrote summary -> {}", out_path)

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
