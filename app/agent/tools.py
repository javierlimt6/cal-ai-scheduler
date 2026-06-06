"""Tool schemas exposed to the LLM and their dispatch onto the calendar client."""

import json
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from app.llm import ToolDef


class SchedulingClient(Protocol):
    """What the tool layer needs from a calendar backend.

    ``CalComClient`` satisfies this structurally; tests satisfy it with an
    in-memory fake. Keeping the dependency structural means the agent package
    never imports the concrete HTTP client.
    """

    async def list_bookings(
        self,
        status: str | None = None,
        after_start: str | None = None,
        before_end: str | None = None,
        limit: int = 50,
        max_pages: int = 4,
    ) -> Any: ...

    async def list_event_types(self, username: str) -> Any: ...

    async def get_slots(
        self,
        start: str,
        end: str,
        event_type_id: int | None = None,
        time_zone: str | None = None,
        duration: int | None = None,
    ) -> Any: ...

    async def create_booking(
        self,
        event_type_id: int,
        start: str,
        attendee_name: str,
        attendee_email: str,
        time_zone: str,
        length_in_minutes: int | None = None,
        guests: list[str] | None = None,
    ) -> Any: ...

    async def cancel_booking(self, booking_uid: str, reason: str | None = None) -> Any: ...

    async def reschedule_booking(
        self, booking_uid: str, new_start: str, reason: str | None = None
    ) -> Any: ...


TOOLS = [
    ToolDef(
        name="list_bookings",
        description=(
            "List the user's bookings. Use this to answer 'what's on my calendar', to find a "
            "booking's uid before cancelling/rescheduling, or to review past events."
        ),
        parameters={
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["upcoming", "past", "cancelled", "unconfirmed", "recurring"],
                    "description": "Filter bookings by status. Defaults to all.",
                },
                "after_start": {
                    "type": "string",
                    "description": "Only bookings starting after this ISO 8601 UTC datetime.",
                },
                "before_end": {
                    "type": "string",
                    "description": "Only bookings ending before this ISO 8601 UTC datetime.",
                },
            },
        },
    ),
    ToolDef(
        name="list_event_types",
        description=(
            "List the user's bookable event types (id, title, duration). Call this first when "
            "booking so you can pick the right event_type_id for what the user asked for."
        ),
        parameters={"type": "object", "properties": {}},
    ),
    ToolDef(
        name="get_available_slots",
        description=(
            "Get open time slots for an event type within a window. Always check availability "
            "before creating a booking."
        ),
        parameters={
            "type": "object",
            "properties": {
                "event_type_id": {"type": "integer", "description": "The event type's numeric id."},
                "start": {"type": "string", "description": "Window start, ISO 8601 UTC."},
                "end": {"type": "string", "description": "Window end, ISO 8601 UTC."},
                "time_zone": {
                    "type": "string",
                    "description": "IANA timezone for the returned slots, e.g. America/New_York.",
                },
                "duration": {
                    "type": "integer",
                    "description": (
                        "Slot length in minutes. Only for event types with multiple allowed "
                        "durations — pass the same value you'll use for length_in_minutes."
                    ),
                },
            },
            "required": ["event_type_id", "start", "end"],
        },
    ),
    ToolDef(
        name="create_booking",
        description=(
            "Create a new booking. Gather the event type, a start time that is actually "
            "available (verify with get_available_slots), and the attendee's name and email "
            "before calling this."
        ),
        parameters={
            "type": "object",
            "properties": {
                "event_type_id": {"type": "integer"},
                "start": {"type": "string", "description": "Start time, ISO 8601 UTC."},
                "attendee_name": {"type": "string"},
                "attendee_email": {"type": "string"},
                "time_zone": {"type": "string", "description": "Attendee's IANA timezone."},
                "length_in_minutes": {
                    "type": "integer",
                    "description": "Only for event types with multiple allowed durations.",
                },
            },
            "required": ["event_type_id", "start", "attendee_name", "attendee_email", "time_zone"],
        },
    ),
    ToolDef(
        name="cancel_booking",
        description=(
            "Cancel a booking by its uid (find it via list_bookings if you don't have it). "
            "Calling this does not cancel immediately: the app holds the action and asks the "
            "user to confirm via a card in the UI."
        ),
        parameters={
            "type": "object",
            "properties": {
                "booking_uid": {"type": "string"},
                "reason": {"type": "string", "description": "Optional cancellation reason."},
            },
            "required": ["booking_uid"],
        },
    ),
    ToolDef(
        name="reschedule_booking",
        description=(
            "Move an existing booking to a new start time (check it's available with "
            "get_available_slots first). Calling this does not reschedule immediately: the app "
            "holds the action and asks the user to confirm via a card in the UI."
        ),
        parameters={
            "type": "object",
            "properties": {
                "booking_uid": {"type": "string"},
                "new_start": {"type": "string", "description": "New start time, ISO 8601 UTC."},
                "reason": {"type": "string", "description": "Optional rescheduling reason."},
            },
            "required": ["booking_uid", "new_start"],
        },
    ),
]

ToolFunc = Callable[..., Awaitable[Any]]


def build_dispatch(client: SchedulingClient, username: str) -> dict[str, ToolFunc]:
    """Map tool names to client calls. Each returns JSON-serializable data."""
    return {
        "list_bookings": client.list_bookings,
        # Tolerate stray kwargs from an LLM (the schema declares none)
        "list_event_types": lambda **_: client.list_event_types(username),
        "get_available_slots": client.get_slots,
        "create_booking": client.create_booking,
        "cancel_booking": client.cancel_booking,
        "reschedule_booking": client.reschedule_booking,
    }


async def execute_tool(dispatch: dict[str, ToolFunc], name: str, arguments: dict[str, Any]) -> str:
    func = dispatch.get(name)
    if func is None:
        raise KeyError(f"Unknown tool: {name}")
    result = await func(**arguments)
    return json.dumps(result, default=str)
