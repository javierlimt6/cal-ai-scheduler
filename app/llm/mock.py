"""Deterministic mock LLM provider.

Maps user-message keywords to scripted tool calls so the full
UI -> agent loop -> cal.com path runs without any LLM API key. A real
provider adapter (Anthropic/OpenAI) drops in behind the same
``LLMProvider`` protocol later.
"""

import json
import re
import uuid
from datetime import UTC, datetime, timedelta

from app.llm.base import LLMResponse, Message, ToolCall, ToolDef

ISO_DATETIME = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:\d{2})?")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
BOOKING_UID = re.compile(r"\b[a-zA-Z0-9]{8,}\b")
EVENT_TYPE_ID = re.compile(r"\b\d{2,}\b")

HELP_TEXT = (
    "I'm running in mock mode (no LLM connected). I understand simple phrasings:\n"
    "- 'What's on my calendar?' — list upcoming bookings\n"
    "- 'What event types do I have?' — list event types\n"
    "- 'Show slots for event type 123' — check availability\n"
    "- 'Book event type 123 at 2026-06-12T10:00:00Z for ada@example.com' — create a booking\n"
    "- 'Cancel booking <uid>' — cancel\n"
    "- 'Reschedule booking <uid> to 2026-06-12T10:00:00Z' — reschedule"
)


def _call(name: str, **arguments) -> LLMResponse:
    return LLMResponse(
        tool_calls=[ToolCall(id=f"mock_{uuid.uuid4().hex[:8]}", name=name, arguments=arguments)]
    )


class MockProvider:
    async def complete(
        self, system: str, messages: list[Message], tools: list[ToolDef]
    ) -> LLMResponse:
        last = messages[-1]
        if last.role == "tool":
            return self._summarize(messages)
        return self._route(last.content)

    def _route(self, text: str) -> LLMResponse:
        lowered = text.lower()
        iso = ISO_DATETIME.search(text)
        email = EMAIL.search(text)
        # Strip datetimes/emails before extracting ids, so the year in
        # "2026-06-12T10:00:00Z" or an email local part like "ada123" can't be
        # mistaken for an event type id or booking uid.
        plain = EMAIL.sub(" ", ISO_DATETIME.sub(" ", text))
        event_type_id = EVENT_TYPE_ID.search(plain)

        if any(k in lowered for k in ("slot", "availab", "free", "open")):
            if not event_type_id:
                return LLMResponse(
                    text="Which event type? Tell me its numeric ID (ask me to list your event types if unsure)."
                )
            now = datetime.now(UTC)
            return _call(
                "get_available_slots",
                event_type_id=int(event_type_id.group(0)),
                start=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                end=(now + timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            )

        if "cancel" in lowered:
            uid = self._find_uid(plain)
            if not uid:
                return _call("list_bookings", status="upcoming")
            return _call("cancel_booking", booking_uid=uid, reason="Requested via assistant")

        if any(k in lowered for k in ("reschedule", "move", "push")):
            uid = self._find_uid(plain)
            if uid and iso:
                return _call("reschedule_booking", booking_uid=uid, new_start=iso.group(0))
            return LLMResponse(
                text="To reschedule I need the booking uid and the new start time in ISO format, "
                "e.g. 'reschedule booking abc12345 to 2026-06-12T10:00:00Z'."
            )

        if re.search(r"\bbook\b", lowered) or "schedule a" in lowered or "set up" in lowered:
            if event_type_id and iso and email:
                return _call(
                    "create_booking",
                    event_type_id=int(event_type_id.group(0)),
                    start=iso.group(0),
                    attendee_name=email.group(0).split("@")[0],
                    attendee_email=email.group(0),
                    time_zone="UTC",
                )
            return LLMResponse(
                text="To book I need the event type ID, a start time in ISO format, and the attendee's "
                "email, e.g. 'book event type 123 at 2026-06-12T10:00:00Z for ada@example.com'."
            )

        if "event type" in lowered:
            return _call("list_event_types")

        if any(
            k in lowered
            for k in (
                "calendar",
                "upcoming",
                "what's on",
                "whats on",
                "meetings",
                "bookings",
                "schedule",
            )
        ):
            return _call("list_bookings", status="upcoming")

        return LLMResponse(text=HELP_TEXT)

    def _find_uid(self, text: str) -> str | None:
        # Pick the first token that looks like a booking uid (mixed alphanumeric,
        # which excludes plain words and plain numbers)
        for match in BOOKING_UID.finditer(text):
            token = match.group(0)
            if any(c.isdigit() for c in token) and any(c.isalpha() for c in token):
                return token
        return None

    def _summarize(self, messages: list[Message]) -> LLMResponse:
        tool_name = ""
        for message in reversed(messages):
            if message.role == "assistant" and message.tool_calls:
                tool_name = message.tool_calls[0].name
                break

        if not messages[-1].tool_results:
            return LLMResponse(text=HELP_TEXT)
        result = messages[-1].tool_results[0]
        if result.content.startswith("CONFIRMATION_REQUIRED"):
            return LLMResponse(
                text="This needs your sign-off — review the card below and hit Confirm."
            )
        if result.is_error:
            return LLMResponse(text=f"That didn't work: {result.content}")

        try:
            data = json.loads(result.content)
        except ValueError:
            data = result.content

        return LLMResponse(text=self._format(tool_name, data))

    def _format(self, tool_name: str, data) -> str:
        if tool_name == "list_bookings":
            if not data:
                return "Your calendar is clear — no upcoming bookings."
            lines = [
                f"- {b.get('title', 'Untitled')} — {b.get('start', '?')} (uid: {b.get('uid', '?')})"
                for b in data
            ]
            return "Here's what's on your calendar:\n" + "\n".join(lines)

        if tool_name == "get_available_slots":
            if not data:
                return "No open slots in that window."
            lines = [
                f"- {day}: " + ", ".join(s.get("start", "?") for s in slots)
                for day, slots in data.items()
            ]
            return "Available slots:\n" + "\n".join(lines)

        if tool_name == "list_event_types":
            if not data:
                return "You have no event types set up."
            lines = [
                f"- {e.get('title', '?')} (id: {e.get('id', '?')}, {e.get('lengthInMinutes', '?')} min)"
                for e in data
            ]
            return "Your event types:\n" + "\n".join(lines)

        if tool_name == "create_booking":
            return (
                f"Booked! {data.get('title', 'Your event')} at {data.get('start', '?')} "
                f"(uid: {data.get('uid', '?')})."
            )

        if tool_name == "cancel_booking":
            return "Done — the booking has been cancelled."

        if tool_name == "reschedule_booking":
            return (
                f"Rescheduled — new time is {data.get('start', '?')} (uid: {data.get('uid', '?')})."
            )

        return f"Result from {tool_name}: {json.dumps(data, indent=2)[:1500]}"
