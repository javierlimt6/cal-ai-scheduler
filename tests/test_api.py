import httpx
import pytest
import respx

from app.main import app

CAL_BASE = "https://api.cal.com/v2"


@pytest.fixture
async def api_client():
    async with httpx.ASGITransport(app=app) as transport:
        # ASGITransport doesn't run lifespan; drive it manually
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                yield client


async def test_health(api_client):
    response = await api_client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_index_serves_chat_ui(api_client):
    response = await api_client.get("/")
    assert response.status_code == 200
    assert "Scheduling Assistant" in response.text


@respx.mock
async def test_chat_round_trip_via_mock_provider(api_client):
    respx.get(f"{CAL_BASE}/bookings").mock(
        return_value=httpx.Response(
            200,
            json={
                "status": "success",
                "data": [
                    {"uid": "abc123def", "title": "Intro call", "start": "2026-06-11T14:00:00Z"}
                ],
            },
        )
    )

    response = await api_client.post(
        "/api/chat", json={"session_id": "test-session", "message": "What's on my calendar?"}
    )

    assert response.status_code == 200
    body = response.json()
    assert "Intro call" in body["reply"]
    assert body["tool_activity"] == [{"name": "list_bookings", "ok": True}]


async def test_chat_validates_input(api_client):
    response = await api_client.post("/api/chat", json={"session_id": "", "message": ""})
    assert response.status_code == 422


async def test_whitespace_only_message_is_rejected(api_client):
    response = await api_client.post("/api/chat", json={"session_id": "s", "message": "   "})
    assert response.status_code == 422


async def test_explicit_invalid_timezone_fails_startup(monkeypatch):
    """A user-set TIMEZONE typo should crash loudly at boot, not degrade."""
    from zoneinfo import ZoneInfoNotFoundError

    from app.config import get_settings

    monkeypatch.setenv("TIMEZONE", "Not/AZone")
    get_settings.cache_clear()

    with pytest.raises(ZoneInfoNotFoundError):
        async with app.router.lifespan_context(app):
            pass  # pragma: no cover — startup must raise before yielding


@respx.mock
async def test_destructive_action_needs_explicit_confirmation(api_client):
    cancel_route = respx.post(f"{CAL_BASE}/bookings/abc123def/cancel").mock(
        return_value=httpx.Response(
            200, json={"status": "success", "data": {"status": "cancelled"}}
        )
    )

    first = await api_client.post(
        "/api/chat", json={"session_id": "s-confirm", "message": "Cancel booking abc123def"}
    )

    assert first.status_code == 200
    body = first.json()
    assert body["tool_activity"] == []  # nothing executed yet
    pending = body["pending_action"]
    assert pending is not None and "abc123def" in pending["summary"]
    assert not cancel_route.called

    second = await api_client.post(
        "/api/chat/confirm",
        json={"session_id": "s-confirm", "action_id": pending["id"], "approved": True},
    )

    assert second.status_code == 200
    assert cancel_route.called
    confirmed = second.json()
    assert "cancelled" in confirmed["reply"].lower()
    assert confirmed["tool_activity"] == [{"name": "cancel_booking", "ok": True}]


async def test_confirm_with_unknown_action_is_409(api_client):
    response = await api_client.post(
        "/api/chat/confirm", json={"session_id": "sX", "action_id": "nope", "approved": True}
    )
    assert response.status_code == 409


async def test_chat_is_rate_limited_per_session(api_client):
    from app.main import app as main_app
    from app.ratelimit import RateLimiter

    main_app.state.rate_limiter = RateLimiter(max_requests=2, window_seconds=60)

    assert (
        await api_client.post("/api/chat", json={"session_id": "rl", "message": "help"})
    ).status_code == 200
    assert (
        await api_client.post("/api/chat", json={"session_id": "rl", "message": "help"})
    ).status_code == 200
    blocked = await api_client.post("/api/chat", json={"session_id": "rl", "message": "help"})
    assert blocked.status_code == 429

    other = await api_client.post("/api/chat", json={"session_id": "rl2", "message": "help"})
    assert other.status_code == 200  # per-session, not global


@respx.mock
async def test_username_resolved_from_me_when_unset(monkeypatch):
    """With a key but no CAL_USERNAME, startup pulls the username from /me."""
    from app.config import get_settings

    monkeypatch.setenv("CAL_API_KEY", "cal_test_key")
    get_settings.cache_clear()

    respx.get(f"{CAL_BASE}/me").mock(
        return_value=httpx.Response(
            200,
            json={
                "status": "success",
                "data": {"username": "resolved-user", "timeZone": "Europe/London"},
            },
        )
    )
    events_route = respx.get(f"{CAL_BASE}/event-types").mock(
        return_value=httpx.Response(
            200,
            json={
                "status": "success",
                "data": [{"id": 1, "title": "Intro", "lengthInMinutes": 30}],
            },
        )
    )

    async with httpx.ASGITransport(app=app) as transport:
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/api/chat",
                    json={"session_id": "s", "message": "What event types do I have?"},
                )

    assert response.status_code == 200
    assert "username=resolved-user" in str(events_route.calls.last.request.url)


@respx.mock
async def test_invalid_profile_timezone_degrades_instead_of_crashing_startup(monkeypatch):
    from app.config import get_settings

    monkeypatch.setenv("CAL_API_KEY", "cal_test_key")
    get_settings.cache_clear()

    respx.get(f"{CAL_BASE}/me").mock(
        return_value=httpx.Response(
            200,
            json={
                "status": "success",
                "data": {"username": "u", "timeZone": "Pacific Time"},  # not IANA
            },
        )
    )

    async with httpx.ASGITransport(app=app) as transport:
        async with app.router.lifespan_context(app):  # must not raise
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get("/api/health")

    assert response.status_code == 200
