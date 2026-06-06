"""Async client for the cal.com v2 REST API.

Note: cal.com versions its v2 endpoints individually via the
``cal-api-version`` header, so each method pins the version documented
for its endpoint.
"""

from typing import Any

import httpx

BOOKINGS_LIST_API_VERSION = "2026-05-01"
BOOKINGS_WRITE_API_VERSION = "2026-02-25"
SLOTS_API_VERSION = "2024-09-04"
EVENT_TYPES_API_VERSION = "2024-06-14"
ME_API_VERSION = "2024-06-14"


class CalComError(Exception):
    """Raised when the cal.com API returns an error response."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"cal.com API error {status_code}: {message}")


class CalComClient:
    def __init__(self, api_key: str, base_url: str = "https://api.cal.com/v2"):
        self._http = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30.0,
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _request(
        self,
        method: str,
        path: str,
        *,
        api_version: str,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> Any:
        # httpx serializes None params/body values as empty rather than
        # omitting them, so strip optional fields here in one place.
        response = await self._http.request(
            method,
            path,
            headers={"cal-api-version": api_version},
            params={k: v for k, v in (params or {}).items() if v is not None},
            json={k: v for k, v in json.items() if v is not None} if json is not None else None,
        )
        if response.is_error:
            try:
                message = response.json().get("error", {}).get("message", response.text)
            except ValueError:
                message = response.text
            raise CalComError(response.status_code, message)
        body = response.json()
        return body.get("data", body)

    async def list_bookings(
        self,
        status: str | None = None,
        after_start: str | None = None,
        before_end: str | None = None,
        limit: int = 50,
    ) -> Any:
        """List bookings. ``status``: upcoming | past | cancelled | unconfirmed | recurring."""
        return await self._request(
            "GET",
            "/bookings",
            api_version=BOOKINGS_LIST_API_VERSION,
            params={
                "status": status,
                "afterStart": after_start,
                "beforeEnd": before_end,
                "take": limit,
            },
        )

    async def create_booking(
        self,
        event_type_id: int,
        start: str,
        attendee_name: str,
        attendee_email: str,
        time_zone: str,
        length_in_minutes: int | None = None,
        guests: list[str] | None = None,
    ) -> Any:
        return await self._request(
            "POST",
            "/bookings",
            api_version=BOOKINGS_WRITE_API_VERSION,
            json={
                "start": start,
                "eventTypeId": event_type_id,
                "attendee": {
                    "name": attendee_name,
                    "email": attendee_email,
                    "timeZone": time_zone,
                },
                "lengthInMinutes": length_in_minutes,
                "guests": guests or None,
            },
        )

    async def cancel_booking(self, booking_uid: str, reason: str | None = None) -> Any:
        return await self._request(
            "POST",
            f"/bookings/{booking_uid}/cancel",
            api_version=BOOKINGS_WRITE_API_VERSION,
            json={"cancellationReason": reason},
        )

    async def reschedule_booking(
        self, booking_uid: str, new_start: str, reason: str | None = None
    ) -> Any:
        return await self._request(
            "POST",
            f"/bookings/{booking_uid}/reschedule",
            api_version=BOOKINGS_WRITE_API_VERSION,
            json={"start": new_start, "reschedulingReason": reason},
        )

    async def get_slots(
        self,
        start: str,
        end: str,
        event_type_id: int | None = None,
        time_zone: str | None = None,
        duration: int | None = None,
    ) -> Any:
        """Available slots, keyed by date: ``{"YYYY-MM-DD": [{"start": ...}, ...]}``."""
        return await self._request(
            "GET",
            "/slots",
            api_version=SLOTS_API_VERSION,
            params={
                "start": start,
                "end": end,
                "eventTypeId": event_type_id,
                "timeZone": time_zone,
                "duration": duration,
            },
        )

    async def list_event_types(self, username: str) -> Any:
        return await self._request(
            "GET",
            "/event-types",
            api_version=EVENT_TYPES_API_VERSION,
            params={"username": username},
        )

    async def get_me(self) -> Any:
        """Profile of the authenticated user (id, username, email, timeZone)."""
        return await self._request("GET", "/me", api_version=ME_API_VERSION)
