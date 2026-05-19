"""POSIX (Chatterjee 2024) ψ — matrix formula tests."""

from __future__ import annotations

import numpy as np
import pytest

from prompt_sensitivity.metrics.posix import (
    build_log_p_matrix,
    log_p_from_token_logprobs,
    posix,
)


def test_posix_zero_when_no_paraphrase_disagreement():
    """If every prompt gives identical logprobs for every response, ψ = 0."""
    n = 4
    log_p = np.full((n, n), -3.0)
    lengths = np.full(n, 5.0)
    assert posix(log_p, lengths) == 0.0


def test_posix_hand_computed_two_prompt_case():
    """
    N=2, log_p = [[-1, -3], [-3, -1]] (off-diag prompt scores y_j worse than natural).
    L_{y_0}=2, L_{y_1}=2.
    ψ = (1/(2*1)) * [(1/L_y0)|log_p[1,0] - log_p[0,0]| + (1/L_y1)|log_p[0,1] - log_p[1,1]|]
      = 0.5 * [(1/2)*|-3 - -1| + (1/2)*|-3 - -1|]
      = 0.5 * (1 + 1) = 1.0
    """
    log_p = np.array([[-1.0, -3.0], [-3.0, -1.0]])
    lengths = np.array([2.0, 2.0])
    assert posix(log_p, lengths) == pytest.approx(1.0)


def test_posix_single_prompt_is_zero():
    """N < 2 -> no pairs -> ψ = 0."""
    log_p = np.array([[-1.0]])
    lengths = np.array([3.0])
    assert posix(log_p, lengths) == 0.0


def test_posix_shape_validation():
    with pytest.raises(ValueError):
        posix(np.zeros((3, 4)), np.ones(3))
    with pytest.raises(ValueError):
        posix(np.zeros((3, 3)), np.ones(2))


def test_posix_rejects_zero_length():
    log_p = np.array([[-1.0, -2.0], [-2.0, -1.0]])
    lengths = np.array([1.0, 0.0])
    with pytest.raises(ValueError):
        posix(log_p, lengths)


def test_log_p_from_token_logprobs():
    """Sum of last y_token_count per-token logprobs."""
    prompt_logprobs = [-0.1, -0.2, -0.3, -0.4, -0.5]
    # last 2 -> -0.4 + -0.5 = -0.9
    assert log_p_from_token_logprobs(prompt_logprobs, y_token_count=2) == pytest.approx(-0.9)


def test_log_p_zero_y_returns_zero():
    assert log_p_from_token_logprobs([-1.0, -2.0], y_token_count=0) == 0.0


def test_build_log_p_matrix_validates_squareness():
    with pytest.raises(ValueError):
        build_log_p_matrix([[-1.0, -2.0], [-3.0]], lengths=[1, 1])
    with pytest.raises(ValueError):
        build_log_p_matrix([[-1.0, -2.0], [-3.0, -4.0]], lengths=[1, 1, 1])
