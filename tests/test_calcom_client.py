import json

import httpx
import pytest
import respx

from app.calcom import CalComClient, CalComError

BASE = "https://api.cal.com/v2"


@pytest.fixture
async def client():
    c = CalComClient(api_key="cal_test_key", base_url=BASE)
    yield c
    await c.aclose()


@respx.mock
async def test_list_bookings_sends_auth_and_version_headers(client):
    route = respx.get(f"{BASE}/bookings").mock(
        return_value=httpx.Response(200, json={"status": "success", "data": [{"uid": "abc"}]})
    )

    data = await client.list_bookings(status="upcoming")

    assert data == [{"uid": "abc"}]
    request = route.calls.last.request
    assert request.headers["Authorization"] == "Bearer cal_test_key"
    assert request.headers["cal-api-version"] == "2026-05-01"
    assert "status=upcoming" in str(request.url)
    assert "limit=50" in str(request.url)  # cal.com 2026-05-01 paginates via limit/cursor
    # None params must be omitted
    assert "afterStart" not in str(request.url)


@respx.mock
async def test_list_bookings_follows_pagination_cursor(client):
    route = respx.get(f"{BASE}/bookings").mock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "status": "success",
                    "data": [{"uid": "page1"}],
                    "pagination": {"nextCursor": "cur2", "hasMore": True},
                },
            ),
            httpx.Response(
                200,
                json={
                    "status": "success",
                    "data": [{"uid": "page2"}],
                    "pagination": {"nextCursor": None, "hasMore": False},
                },
            ),
        ]
    )

    data = await client.list_bookings()

    assert [b["uid"] for b in data] == ["page1", "page2"]
    assert "cursor" not in str(route.calls[0].request.url)
    assert "cursor=cur2" in str(route.calls[1].request.url)


@respx.mock
async def test_list_bookings_stops_at_max_pages(client):
    respx.get(f"{BASE}/bookings").mock(
        return_value=httpx.Response(
            200,
            json={
                "status": "success",
                "data": [{"uid": "x"}],
                "pagination": {"nextCursor": "always-more", "hasMore": True},
            },
        )
    )

    data = await client.list_bookings(max_pages=2)

    assert len(data) == 2  # bounded, not infinite


@respx.mock
async def test_create_booking_payload_and_version(client):
    route = respx.post(f"{BASE}/bookings").mock(
        return_value=httpx.Response(201, json={"status": "success", "data": {"uid": "new1"}})
    )

    data = await client.create_booking(
        event_type_id=123,
        start="2026-06-11T14:00:00Z",
        attendee_name="Ada Lovelace",
        attendee_email="ada@example.com",
        time_zone="Europe/London",
        length_in_minutes=30,
    )

    assert data["uid"] == "new1"
    request = route.calls.last.request
    assert request.headers["cal-api-version"] == "2026-02-25"
    body = json.loads(request.content)
    assert body == {
        "start": "2026-06-11T14:00:00Z",
        "eventTypeId": 123,
        "attendee": {
            "name": "Ada Lovelace",
            "email": "ada@example.com",
            "timeZone": "Europe/London",
        },
        "lengthInMinutes": 30,
    }


@respx.mock
async def test_cancel_booking(client):
    route = respx.post(f"{BASE}/bookings/uid123/cancel").mock(
        return_value=httpx.Response(
            200, json={"status": "success", "data": {"status": "cancelled"}}
        )
    )

    data = await client.cancel_booking("uid123", reason="Conflict")

    assert data["status"] == "cancelled"
    assert json.loads(route.calls.last.request.content) == {"cancellationReason": "Conflict"}


@respx.mock
async def test_reschedule_booking(client):
    route = respx.post(f"{BASE}/bookings/uid123/reschedule").mock(
        return_value=httpx.Response(200, json={"status": "success", "data": {"uid": "uid124"}})
    )

    data = await client.reschedule_booking("uid123", "2026-06-12T10:00:00Z", reason="Running late")

    assert data["uid"] == "uid124"
    assert json.loads(route.calls.last.request.content) == {
        "start": "2026-06-12T10:00:00Z",
        "reschedulingReason": "Running late",
    }


@respx.mock
async def test_booking_uid_is_url_quoted(client):
    # An LLM-supplied uid must not be able to reshape the request path
    route = respx.post(f"{BASE}/bookings/ab%2F..%2Fcd/cancel").mock(
        return_value=httpx.Response(200, json={"status": "success", "data": {}})
    )

    await client.cancel_booking("ab/../cd")

    assert route.called


@respx.mock
async def test_list_event_types_sends_username_and_version(client):
    route = respx.get(f"{BASE}/event-types").mock(
        return_value=httpx.Response(200, json={"status": "success", "data": [{"id": 1}]})
    )

    data = await client.list_event_types("javier")

    assert data == [{"id": 1}]
    request = route.calls.last.request
    assert request.headers["cal-api-version"] == "2024-06-14"
    assert "username=javier" in str(request.url)


@respx.mock
async def test_get_me_unwraps_profile(client):
    route = respx.get(f"{BASE}/me").mock(
        return_value=httpx.Response(
            200, json={"status": "success", "data": {"username": "j", "timeZone": "UTC"}}
        )
    )

    me = await client.get_me()

    assert me == {"username": "j", "timeZone": "UTC"}
    assert route.calls.last.request.headers["cal-api-version"] == "2024-06-14"


@respx.mock
async def test_get_slots_uses_slots_api_version(client):
    route = respx.get(f"{BASE}/slots").mock(
        return_value=httpx.Response(
            200,
            json={"status": "success", "data": {"2026-06-11": [{"start": "2026-06-11T14:00:00Z"}]}},
        )
    )

    data = await client.get_slots(
        start="2026-06-11T00:00:00Z",
        end="2026-06-12T00:00:00Z",
        event_type_id=123,
        time_zone="Europe/London",
    )

    assert "2026-06-11" in data
    request = route.calls.last.request
    assert request.headers["cal-api-version"] == "2024-09-04"
    assert "eventTypeId=123" in str(request.url)


@respx.mock
async def test_error_response_raises_calcom_error(client):
    respx.get(f"{BASE}/bookings").mock(
        return_value=httpx.Response(
            401, json={"status": "error", "error": {"message": "Invalid API key"}}
        )
    )

    with pytest.raises(CalComError) as exc_info:
        await client.list_bookings()

    assert exc_info.value.status_code == 401
    assert "Invalid API key" in str(exc_info.value)


@respx.mock
async def test_no_auth_header_when_api_key_empty():
    route = respx.get(f"{BASE}/bookings").mock(
        return_value=httpx.Response(200, json={"status": "success", "data": []})
    )
    keyless = CalComClient(api_key="", base_url=BASE)
    try:
        await keyless.list_bookings()
    finally:
        await keyless.aclose()

    assert "Authorization" not in route.calls.last.request.headers


@respx.mock
async def test_string_error_body_is_surfaced(client):
    respx.get(f"{BASE}/bookings").mock(
        return_value=httpx.Response(403, json={"status": "error", "error": "Forbidden resource"})
    )

    with pytest.raises(CalComError) as exc_info:
        await client.list_bookings()

    assert "Forbidden resource" in str(exc_info.value)


@respx.mock
async def test_network_error_becomes_calcom_error(client):
    respx.get(f"{BASE}/bookings").mock(side_effect=httpx.ConnectError("dns failure"))

    with pytest.raises(CalComError) as exc_info:
        await client.list_bookings()

    assert exc_info.value.status_code == 503
    assert "Could not reach cal.com" in str(exc_info.value)


@respx.mock
async def test_non_json_error_body_is_handled(client):
    respx.get(f"{BASE}/bookings").mock(return_value=httpx.Response(502, text="Bad Gateway"))

    with pytest.raises(CalComError) as exc_info:
        await client.list_bookings()

    assert exc_info.value.status_code == 502
