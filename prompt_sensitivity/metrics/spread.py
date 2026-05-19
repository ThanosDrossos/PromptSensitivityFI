"""Performance spread — Sclar 2023 / Cao 2024.

`spread(scores) = max(scores) - min(scores)` over paraphrases.

One-liner; the wrapper exists so the orchestrator can fail loud when fed
an empty input rather than returning a misleading 0.
"""

from __future__ import annotations

from collections.abc import Sequence


def spread(scores: Sequence[float]) -> float:
    """Range of F(x) across paraphrases."""
    if not scores:
        raise ValueError("scores must be non-empty")
    return float(max(scores) - min(scores))
