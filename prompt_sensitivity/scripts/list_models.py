"""Print every model the LiteLLM gateway exposes — `make list-models`.

Run this BEFORE `make api-check`. The gateway's `model_id` aliases are admin-
configured and may differ from LiteLLM's canonical names baked into our
`config.yaml`. If a configured `model_id` is missing from this list, edit
`config.yaml.models.<key>.model_id` to match one of the printed IDs.
"""

from __future__ import annotations

import sys

from loguru import logger

from ..config import load_config
from ..logging_setup import configure_logging
from ..models.registry import list_gateway_models


def main() -> int:
    configure_logging("list_models")
    config = load_config()

    ids = list_gateway_models(config)
    logger.info("{} models registered on gateway", len(ids))

    configured = {key: entry.model_id for key, entry in config.models.items()}
    missing = {k: mid for k, mid in configured.items() if mid not in ids}

    print()
    print(f"Gateway exposes {len(ids)} models:")
    print("-" * 80)
    for mid in ids:
        print(f"  {mid}")
    print("-" * 80)
    print()
    print("Configured (config.yaml -> models.*.model_id):")
    for key, mid in configured.items():
        mark = "OK" if mid in ids else "MISSING"
        print(f"  [{mark:>8}] {key:<22} -> {mid}")

    if missing:
        print()
        print("FIX: update config.yaml so each model_id matches a row above.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
