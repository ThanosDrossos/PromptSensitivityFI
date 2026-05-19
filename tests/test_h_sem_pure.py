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
