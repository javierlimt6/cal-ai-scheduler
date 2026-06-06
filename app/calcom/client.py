"""Async client for the cal.com v2 REST API.

Note: cal.com versions its v2 endpoints individually via the
``cal-api-version`` header, so each method pins the version documented
for its endpoint.
"""

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

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
        # Omit the header entirely when no key is configured: cal.com then
        # returns a clear 401 message (vs. an opaque protocol error for an
        # empty Bearer value), and public endpoints still work.
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.AsyncClient(base_url=base_url, headers=headers, timeout=30.0)

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
        envelope: bool = False,
    ) -> Any:
        """Make one API call. Returns the unwrapped ``data`` payload, or the
        full ``{"status", "data", "pagination", ...}`` body when ``envelope``
        is set (for endpoints whose metadata matters, e.g. cursor pagination).
        """
        # httpx serializes None params/body values as empty rather than
        # omitting them, so strip optional fields here in one place.
        try:
            response = await self._http.request(
                method,
                path,
                headers={"cal-api-version": api_version},
                params={k: v for k, v in (params or {}).items() if v is not None},
                json={k: v for k, v in json.items() if v is not None} if json is not None else None,
            )
        except httpx.HTTPError as exc:
            raise CalComError(503, f"Could not reach cal.com ({exc.__class__.__name__})") from exc
        if response.is_error:
            try:
                error = response.json().get("error")
                # The error payload is usually {"message": ...} but can be a bare string
                message = error.get("message") if isinstance(error, dict) else error
            except ValueError:
                message = None
            raise CalComError(response.status_code, str(message) if message else response.text)
        body = response.json()
        if envelope:
            return body
        return body.get("data", body)

    async def list_bookings(
        self,
        status: str | None = None,
        after_start: str | None = None,
        before_end: str | None = None,
        limit: int = 50,
        max_pages: int = 4,
    ) -> Any:
        """List bookings. ``status``: upcoming | past | cancelled | unconfirmed | recurring.

        Follows ``pagination.nextCursor`` for up to ``max_pages`` pages, so a
        busy calendar isn't silently truncated at the first ``limit`` results.
        """
        bookings: list[Any] = []
        cursor: str | None = None
        for _ in range(max_pages):
            body = await self._request(
                "GET",
                "/bookings",
                api_version=BOOKINGS_LIST_API_VERSION,
                params={
                    "status": status,
                    "afterStart": after_start,
                    "beforeEnd": before_end,
                    "limit": limit,
                    "cursor": cursor,
                },
                envelope=True,
            )
            data = body.get("data", [])
            bookings.extend(data if isinstance(data, list) else [data])
            cursor = (body.get("pagination") or {}).get("nextCursor")
            if not cursor:
                break
        else:
            logger.warning(
                "list_bookings stopped after %d pages (%d bookings); more pages exist",
                max_pages,
                len(bookings),
            )
        return bookings

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
