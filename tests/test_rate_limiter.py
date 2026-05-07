"""Token-bucket sanity checks."""

from __future__ import annotations

import time

import pytest

from prompt_sensitivity.models.rate_limiter import TokenBucket


def test_first_acquire_is_immediate():
    bucket = TokenBucket(rate=10.0, capacity=10.0)
    t0 = time.monotonic()
    bucket.acquire(1.0)
    assert time.monotonic() - t0 < 0.05


def test_burst_then_throttle():
    """After draining the bucket, the next acquire should wait ~ 1/rate seconds."""
    bucket = TokenBucket(rate=20.0, capacity=2.0)
    bucket.acquire(1.0)
    bucket.acquire(1.0)
    t0 = time.monotonic()
    bucket.acquire(1.0)  # bucket is empty, must refill
    elapsed = time.monotonic() - t0
    # 1 token at 20 tokens/sec ~= 50ms. Allow generous slack on slow CI hardware.
    assert 0.03 <= elapsed <= 0.5


def test_invalid_rate():
    with pytest.raises(ValueError):
        TokenBucket(rate=0.0)


def test_request_exceeds_capacity_raises():
    bucket = TokenBucket(rate=5.0, capacity=2.0)
    with pytest.raises(ValueError):
        bucket.acquire(5.0)
