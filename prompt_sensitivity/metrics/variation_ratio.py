"""Variation ratio — Lu et al. 2023 ACL 2024.

For free-form output we operate on semantic cluster IDs (the natural
analogue of "answers" — Farquhar 2024 §Methods).

    variation_ratio = 1 - mode_count / N

Lower = more agreement (the modal answer dominates). Higher = more variation.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence


def variation_ratio(items: Sequence) -> float:
    """1 - mode_count / N. Works on cluster IDs, class labels, or raw strings."""
    if not items:
        raise ValueError("items must be non-empty")
    counts = Counter(items)
    mode_count = max(counts.values())
    return float(1.0 - mode_count / len(items))
