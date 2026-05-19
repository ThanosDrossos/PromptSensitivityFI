"""spread + variation_ratio (the one-liners)."""

import pytest

from prompt_sensitivity.metrics.spread import spread
from prompt_sensitivity.metrics.variation_ratio import variation_ratio


def test_spread_binary_F_is_one():
    assert spread([1.0, 0.0, 1.0, 0.0]) == 1.0


def test_spread_uniform_is_zero():
    assert spread([0.7, 0.7, 0.7]) == 0.0


def test_spread_empty_raises():
    with pytest.raises(ValueError):
        spread([])


def test_variation_ratio_all_same_is_zero():
    """Modal answer covers everyone -> variation_ratio = 0."""
    assert variation_ratio([1, 1, 1, 1, 1]) == 0.0


def test_variation_ratio_all_distinct():
    """5 distinct answers in 5 votes -> mode_count=1 -> VR = 4/5."""
    assert variation_ratio([1, 2, 3, 4, 5]) == pytest.approx(0.8)


def test_variation_ratio_strings_and_cluster_ids():
    """Works on any hashable items."""
    assert variation_ratio(["a", "a", "b"]) == pytest.approx(1 / 3)


def test_variation_ratio_empty_raises():
    with pytest.raises(ValueError):
        variation_ratio([])
