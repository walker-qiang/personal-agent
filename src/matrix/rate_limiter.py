"""Token bucket rate limiter for LLM API calls.

Prevents exceeding provider rate limits by throttling requests before
they reach the network layer. Uses a simple token bucket algorithm
with thread safety.
"""

from __future__ import annotations

import threading
import time


class TokenBucketRateLimiter:
    """Thread-safe token bucket rate limiter.

    Tokens refill at a constant rate. Each call to `acquire()` consumes
    one token. If no token is available, the caller blocks until one
    becomes available.

    Args:
        rate: Maximum sustained calls per second.
        burst: Maximum burst size (number of tokens the bucket can hold).
    """

    def __init__(self, rate: float, burst: int = 5) -> None:
        if rate <= 0:
            raise ValueError("rate must be positive")
        self._rate = rate
        self._burst = burst
        self._tokens = float(burst)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
        self._last_refill = now

    def acquire(self, timeout: float | None = None) -> bool:
        """Acquire one token, blocking if necessary.

        Args:
            timeout: Maximum seconds to wait. None means wait forever.

        Returns:
            True if a token was acquired, False if timeout expired.
        """
        deadline = time.monotonic() + timeout if timeout is not None else None
        with self._lock:
            while True:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return True
                # Not enough tokens — calculate wait time
                wait = (1.0 - self._tokens) / self._rate
                if timeout is not None and time.monotonic() + wait > deadline:
                    return False
                # Release lock briefly to allow other threads to refill
                self._lock.release()
                time.sleep(min(wait, 0.1))
                self._lock.acquire()

    @property
    def available_tokens(self) -> float:
        with self._lock:
            self._refill()
            return self._tokens