"""Tiny in-process sliding-window rate limiter.

Per-key (we key on session id) with a bounded number of tracked keys, so an
attacker minting fresh session ids can't grow memory without bound. Single
process only — a multi-replica deployment would move this to a shared store.
"""

import time
from collections import deque

MAX_TRACKED_KEYS = 2000


class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: float):
        self._max_requests = max_requests
        self._window = window_seconds
        self._hits: dict[str, deque[float]] = {}

    def allow(self, key: str) -> bool:
        """Record one request for ``key``; False when over the limit."""
        now = time.monotonic()
        window = self._hits.get(key)
        if window is None:
            if len(self._hits) >= MAX_TRACKED_KEYS:
                self._evict(now)
            window = self._hits[key] = deque()
        while window and now - window[0] > self._window:
            window.popleft()
        if len(window) >= self._max_requests:
            return False
        window.append(now)
        return True

    def _evict(self, now: float) -> None:
        """Drop fully-expired keys; if none were, drop the oldest-created one."""
        expired = [k for k, w in self._hits.items() if not w or now - w[-1] > self._window]
        for key in expired:
            del self._hits[key]
        if not expired and self._hits:
            del self._hits[next(iter(self._hits))]
