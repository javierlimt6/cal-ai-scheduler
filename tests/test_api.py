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
                "data": [{"uid": "abc123def", "title": "Intro call", "start": "2026-06-11T14:00:00Z"}],
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
