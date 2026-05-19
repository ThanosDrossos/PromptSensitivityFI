"""Round-trip API verification — Sprint 1 §1.2 gate + Sprint-5 echo probe.

For each configured model on the LiteLLM gateway:
  1. Fires the same prompt twice at temperature=0. First call hits the gateway;
     second is served from the local SQLite cache.
  2. Re-fires once more with `logprobs=true, top_logprobs=5` (bypassing the
     cache via a different `purpose` so we actually exercise the API path)
     to verify the gateway returns logprobs for this model.
  3. For models flagged `echo_completions: true` in config, fires a
     `/v1/completions` echo probe to verify POSIX's prerequisite path. This
     catches Sprint-5 blockers BEFORE the pilot run spends real money.

Reports: latency, deterministic-output match, whether logprobs came back,
whether echo+logprobs returned per-token logprobs.

Run with `make api-check`.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict

from loguru import logger

from ..config import load_config
from ..logging_setup import configure_logging
from ..models import LLMRequest
from ..models.schemas import CompletionRequest
from ..models.registry import get_client, list_gateway_models, reset_clients


PROBE_USER_MESSAGE = "Reply with exactly the single word: pong"


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
    logprobs_returned: bool
    logprobs_topk: int | None
    echo_supported_actual: bool | None      # None = not probed; True/False = result of /v1/completions echo
    echo_error: str | None
    on_gateway: bool
    error: str | None = None


def _probe(model_key: str, gateway_models: set[str]) -> ApiCheckRow:
    config = load_config()
    entry = config.models[model_key]
    on_gateway = entry.model_id in gateway_models if gateway_models else True
    base_request_kwargs: dict = dict(
        provider=entry.provider,
        model_id=entry.model_id,
        messages=[{"role": "user", "content": PROBE_USER_MESSAGE}],
        temperature=0.0,
        top_p=1.0,
        max_tokens=8,
        seed=42,
    )

    client = get_client(model_key, config)
    try:
        # Round-trip 1+2: identical request, second pulled from cache.
        rt_request = LLMRequest(**base_request_kwargs, purpose="api_check_roundtrip")
        first = client.complete(rt_request)
        second = client.complete(rt_request)
        # Round-trip 3: same prompt but with logprobs. Different `purpose`
        # so this lands as a fresh cache miss and we actually probe the API.
        lp_request = LLMRequest(
            **base_request_kwargs,
            logprobs=True,
            top_logprobs=5,
            purpose="api_check_logprobs",
        )
        lp_response = client.complete(lp_request)
        logprobs_returned = bool(lp_response.token_logprobs)
        logprobs_topk = (
            len(lp_response.token_logprobs[0].top_logprobs)
            if logprobs_returned
            else None
        )

        # Round-trip 4 (Sprint-5 prerequisite): echo probe via /v1/completions.
        # Only for models whose config flags `echo_completions: true`. We
        # try, and if the gateway rejects it we record the failure here
        # rather than discovering it during the pilot.
        echo_ok: bool | None = None
        echo_err: str | None = None
        if entry.echo_completions:
            echo_req = CompletionRequest(
                provider=entry.provider,  # type: ignore[arg-type]
                model_id=entry.model_id,
                prompt="The capital of France is Paris.",
                max_tokens=0,
                echo=True,
                logprobs=1,
                temperature=0.0,
                purpose="api_check_echo",
            )
            try:
                echo_resp = client.score_continuation(echo_req)
                echo_ok = bool(echo_resp.token_logprobs)
            except Exception as e:  # noqa: BLE001
                echo_ok = False
                echo_err = str(e)[:200]
                logger.warning(
                    "echo probe failed for {} -> {}: {}",
                    model_key,
                    entry.model_id,
                    echo_err,
                )

        return ApiCheckRow(
            model_key=model_key,
            provider=entry.provider,
            model_id=entry.model_id,
            first_text=first.text.strip(),
            second_text=second.text.strip(),
            first_ms=first.latency_ms,
            second_ms=second.latency_ms,
            deterministic=(first.text == second.text),
            logprobs_returned=logprobs_returned,
            logprobs_topk=logprobs_topk,
            echo_supported_actual=echo_ok,
            echo_error=echo_err,
            on_gateway=on_gateway,
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
            logprobs_returned=False,
            logprobs_topk=None,
            echo_supported_actual=None,
            echo_error=None,
            on_gateway=on_gateway,
            error=str(exc),
        )


def main() -> int:
    configure_logging("api_check")
    config = load_config()
    reset_clients()

    # Pull the gateway model catalogue once so we can flag config drift.
    try:
        gateway_models = set(list_gateway_models(config))
        logger.info("gateway exposes {} models", len(gateway_models))
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not list gateway models ({}); continuing", exc)
        gateway_models = set()

    rows: list[ApiCheckRow] = []
    for model_key in config.models:
        logger.info(
            "probing {} -> {} (expects chat_logprobs={}, echo={})",
            model_key,
            config.models[model_key].model_id,
            config.models[model_key].chat_logprobs,
            config.models[model_key].echo_completions,
        )
        rows.append(_probe(model_key, gateway_models))

    # Table.
    print()
    print("=" * 120)
    header = (
        f"{'model_key':<22} {'1st_ms':>8} {'2nd_ms':>8} {'det':>4} "
        f"{'lp':>4} {'topK':>4} {'echo':>5} {'gw':>4}  text"
    )
    print(header)
    print("-" * 120)
    failures = 0
    logprob_mismatch = 0
    echo_mismatch = 0
    for r in rows:
        if r.error:
            failures += 1
            print(
                f"{r.model_key:<22} {'ERR':>8} {'ERR':>8} {'-':>4} {'-':>4} {'-':>4} "
                f"{'-':>5} {'-':>4}  {r.error[:50]}"
            )
            continue
        first_ms = f"{r.first_ms:>7.0f}" if r.first_ms is not None else "    n/a"
        second_ms = f"{r.second_ms:>7.0f}" if r.second_ms is not None else "    n/a"
        det = "yes" if r.deterministic else " NO"
        lp = "yes" if r.logprobs_returned else " no"
        topk = str(r.logprobs_topk) if r.logprobs_topk is not None else "   -"
        if r.echo_supported_actual is None:
            echo = "  n/a"  # not probed (echo_completions=false in config)
        elif r.echo_supported_actual:
            echo = "  yes"
        else:
            echo = "   NO"
        gw = "yes" if r.on_gateway else " NO"
        print(
            f"{r.model_key:<22} {first_ms:>8} {second_ms:>8} {det:>4} "
            f"{lp:>4} {topk:>4} {echo:>5} {gw:>4}  {r.first_text!r}"
        )
        # Soft-fails: configured to expect a capability that the gateway didn't deliver.
        if config.models[r.model_key].chat_logprobs and not r.logprobs_returned:
            logprob_mismatch += 1
        if (
            config.models[r.model_key].echo_completions
            and r.echo_supported_actual is False
        ):
            echo_mismatch += 1

    print("=" * 120)
    print(
        f"failures: {failures}/{len(rows)}, "
        f"logprob mismatches: {logprob_mismatch}/{len(rows)}, "
        f"echo mismatches: {echo_mismatch}/{len(rows)}"
    )
    if echo_mismatch:
        print(
            "WARNING: at least one model configured echo_completions=true but the "
            "gateway rejected /v1/completions echo. POSIX will be unavailable for "
            "those models in Sprint 5 — investigate before the pilot."
        )

    summary = {
        "rows": [asdict(r) for r in rows],
        "failures": failures,
        "logprob_mismatch": logprob_mismatch,
        "echo_mismatch": echo_mismatch,
        "gateway_models_count": len(gateway_models),
    }
    out_path = config.repo_root() / "logs" / "api_check_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("wrote summary -> {}", out_path)

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
