"""Token-bucket rate limiter shared per provider.

We could lean on the SDK's built-ins but keeping our own bucket means we can
respect the provider QPS budget across threads/processes consistently.
"""

from __future__ import annotations

import threading
import time


class TokenBucket:
    """Classic token-bucket. Refills at `rate` tokens/sec up to `capacity`."""

    def __init__(self, rate: float, capacity: float | None = None) -> None:
        if rate <= 0:
            raise ValueError(f"rate must be positive, got {rate}")
        self.rate = rate
        self.capacity = capacity if capacity is not None else max(rate, 1.0)
        self._tokens = self.capacity
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, n: float = 1.0) -> None:
        """Block until `n` tokens are available, then consume them."""
        if n > self.capacity:
            raise ValueError(f"requested {n} > capacity {self.capacity}")
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last
                self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
                self._last = now
                if self._tokens >= n:
                    self._tokens -= n
                    return
                deficit = n - self._tokens
                wait = deficit / self.rate
            time.sleep(wait)
