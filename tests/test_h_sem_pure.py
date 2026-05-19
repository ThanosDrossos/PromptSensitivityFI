"""H_sem entropy + cluster helpers — pure math (no DeBERTa load)."""

from __future__ import annotations

import math

import pytest

from prompt_sensitivity.metrics.h_sem import (
    entropy_from_assignment,
    h_sem,
    n_unique_clusters,
)


def test_entropy_uniform_two_clusters_is_one_bit():
    """[0,0,1,1] -> H = -2*(0.5*log2 0.5) = 1 bit."""
    assert entropy_from_assignment([0, 0, 1, 1]) == pytest.approx(1.0)


def test_entropy_single_cluster_is_zero():
    """All responses in one semantic cluster -> H = 0."""
    assert entropy_from_assignment([3, 3, 3, 3]) == 0.0


def test_entropy_empty_is_zero():
    assert entropy_from_assignment([]) == 0.0


def test_entropy_three_balanced_clusters():
    """3 equal clusters -> H = log2(3)."""
    assert entropy_from_assignment([0, 1, 2]) == pytest.approx(math.log2(3))


def test_n_unique_clusters():
    assert n_unique_clusters([0, 0, 1, 1, 2]) == 3
    assert n_unique_clusters([]) == 0


def test_h_sem_accepts_precomputed_clusters():
    """Bypass DeBERTa by passing precomputed_clusters."""
    responses = ["yes", "yes", "no"]
    entropy, assignment = h_sem(responses, precomputed_clusters=[0, 0, 1])
    assert assignment == [0, 0, 1]
    # H = -(2/3)log2(2/3) - (1/3)log2(1/3) ~ 0.918
    expected = -(2/3) * math.log2(2/3) - (1/3) * math.log2(1/3)
    assert entropy == pytest.approx(expected)


def test_h_sem_precomputed_length_mismatch_raises():
    with pytest.raises(ValueError):
        h_sem(["a", "b"], precomputed_clusters=[0, 0, 0])


def test_cluster_responses_pooled_returns_per_prompt_with_shared_ids(monkeypatch):
    """Pooled-clustering contract: cluster IDs must be shared across paraphrases.

    We monkey-patch `cluster_responses` with a deterministic fake so the test
    doesn't load DeBERTa. The pooled wrapper must concatenate, call the
    inner once, and slice back into the per-prompt shape.
    """
    # Get the MODULE. Note: `from .h_sem import h_sem` in metrics/__init__.py
    # shadows the submodule attribute with the function of the same name, so
    # neither `from prompt_sensitivity.metrics import h_sem as M` nor
    # `import prompt_sensitivity.metrics.h_sem as M` returns the module.
    # `sys.modules` is the unshadowed lookup path.
    import sys
    h_sem_module = sys.modules["prompt_sensitivity.metrics.h_sem"]

    captured: list[list[str]] = []

    def fake_cluster_responses(responses, *, config=None, threshold=None):
        captured.append(list(responses))
        # Deterministic fake: first char of each response is its cluster id.
        # All responses starting with 'a' -> cluster 0, 'b' -> 1, etc.
        seen: dict[str, int] = {}
        out: list[int] = []
        for r in responses:
            ch = r[0] if r else ""
            if ch not in seen:
                seen[ch] = len(seen)
            out.append(seen[ch])
        return out

    monkeypatch.setattr(h_sem_module, "cluster_responses", fake_cluster_responses)

    inputs = {
        0: ["apple-1", "apple-2", "banana-1"],
        1: ["banana-2", "apple-3", "cherry-1"],
    }
    pooled = h_sem_module.cluster_responses_pooled(inputs)

    # The inner cluster_responses was called ONCE on the flattened pool.
    assert len(captured) == 1
    assert captured[0] == ["apple-1", "apple-2", "banana-1", "banana-2", "apple-3", "cherry-1"]

    # Output shape mirrors input.
    assert set(pooled.keys()) == {0, 1}
    assert len(pooled[0]) == 3
    assert len(pooled[1]) == 3

    # IDs are shared: paraphrase 0's 'banana-1' and paraphrase 1's 'banana-2'
    # must land in the same cluster (the whole point of pooled clustering).
    assert pooled[0][2] == pooled[1][0]  # both 'banana...'
    assert pooled[0][0] == pooled[1][1]  # both 'apple...'


def test_cluster_responses_pooled_empty_input():
    from prompt_sensitivity.metrics.h_sem import cluster_responses_pooled

    assert cluster_responses_pooled({}) == {}
