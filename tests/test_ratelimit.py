import app.ratelimit as ratelimit
from app.ratelimit import RateLimiter


def test_blocks_over_limit_and_recovers_after_window(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(ratelimit.time, "monotonic", lambda: now[0])
    limiter = RateLimiter(max_requests=2, window_seconds=60)

    assert limiter.allow("s1")
    assert limiter.allow("s1")
    assert not limiter.allow("s1")

    now[0] = 61.0  # window rolled past the first hits
    assert limiter.allow("s1")


def test_sessions_are_limited_independently():
    limiter = RateLimiter(max_requests=1, window_seconds=60)

    assert limiter.allow("a")
    assert not limiter.allow("a")
    assert limiter.allow("b")


def test_tracked_keys_are_bounded(monkeypatch):
    monkeypatch.setattr(ratelimit, "MAX_TRACKED_KEYS", 3)
    limiter = RateLimiter(max_requests=5, window_seconds=60)

    for key in ("a", "b", "c", "d", "e"):
        assert limiter.allow(key)

    assert len(limiter._hits) <= 3
