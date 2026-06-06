import pytest

from app.agent import Agent, build_dispatch, build_system_prompt
from app.calcom import CalComError
from app.llm import LLMResponse, MockProvider, ToolCall


class FakeCalCom:
    """In-memory stand-in for CalComClient used as the dispatch backend."""

    def __init__(self):
        self.bookings = [
            {"uid": "abc123def", "title": "Intro call", "start": "2026-06-11T14:00:00Z"},
        ]
        self.cancelled: list[str] = []

    async def list_bookings(self, **kwargs):
        return self.bookings

    async def list_event_types(self, username):
        return [{"id": 123, "title": "30-min intro", "lengthInMinutes": 30}]

    async def get_slots(self, event_type_id, start, end, time_zone=None):
        return {"2026-06-11": [{"start": "2026-06-11T15:00:00Z"}]}

    async def create_booking(
        self, event_type_id, start, attendee_name, attendee_email, time_zone, **kwargs
    ):
        booking = {"uid": "new42uid", "title": "30-min intro", "start": start}
        self.bookings.append(booking)
        return booking

    async def cancel_booking(self, booking_uid, reason=None):
        self.cancelled.append(booking_uid)
        return {"uid": booking_uid, "status": "cancelled"}

    async def reschedule_booking(self, booking_uid, new_start, reason=None):
        return {"uid": booking_uid + "x", "start": new_start}


@pytest.fixture
def fake():
    return FakeCalCom()


@pytest.fixture
def agent(fake):
    dispatch = build_dispatch(fake, "testuser")
    return Agent(MockProvider(), dispatch, lambda: build_system_prompt("UTC"))


async def test_view_calendar(agent):
    reply = await agent.chat("s1", "What's on my calendar?")

    assert "Intro call" in reply.text
    assert "abc123def" in reply.text
    assert [a.name for a in reply.tool_activity] == ["list_bookings"]
    assert reply.tool_activity[0].ok


async def test_book_event(agent, fake):
    reply = await agent.chat(
        "s1", "Book event type 123 at 2026-06-12T10:00:00Z for ada@example.com"
    )

    assert "Booked" in reply.text
    assert "new42uid" in reply.text
    assert len(fake.bookings) == 2


async def test_cancel_booking(agent, fake):
    reply = await agent.chat("s1", "Cancel booking abc123def")

    assert "cancelled" in reply.text.lower()
    assert fake.cancelled == ["abc123def"]


async def test_reschedule_booking(agent):
    reply = await agent.chat("s1", "Reschedule booking abc123def to 2026-06-12T10:00:00Z")

    assert "Rescheduled" in reply.text
    assert "2026-06-12T10:00:00Z" in reply.text


async def test_list_event_types(agent):
    reply = await agent.chat("s1", "What event types do I have?")

    assert "30-min intro" in reply.text
    assert "123" in reply.text


async def test_tool_error_is_relayed_conversationally(agent):
    async def failing_list_bookings(**kwargs):
        raise CalComError(401, "Invalid API key")

    agent._dispatch["list_bookings"] = failing_list_bookings

    reply = await agent.chat("s1", "What's on my calendar?")

    assert "didn't work" in reply.text
    assert "Invalid API key" in reply.text
    assert not reply.tool_activity[0].ok


async def test_sessions_are_isolated(agent):
    await agent.chat("session-a", "What's on my calendar?")
    await agent.chat("session-b", "What event types do I have?")

    # user, assistant w/ tool_call, tool results, final assistant
    assert len(agent._sessions["session-a"]) == 4
    assert agent._sessions["session-a"] is not agent._sessions["session-b"]


async def test_missing_details_asks_instead_of_calling_tools(agent):
    reply = await agent.chat("s1", "Book a meeting")

    assert reply.tool_activity == []
    assert "event type" in reply.text.lower()


async def test_show_my_bookings_lists_instead_of_booking(agent):
    reply = await agent.chat("s1", "Show my bookings")

    assert [a.name for a in reply.tool_activity] == ["list_bookings"]


async def test_datetime_year_is_not_mistaken_for_event_type_id(agent):
    # No explicit event type id: the 2026 in the timestamp must not be used as one
    reply = await agent.chat("s1", "Book a meeting at 2026-06-12T10:00:00Z for ada@example.com")

    assert reply.tool_activity == []
    assert "event type" in reply.text.lower()


async def test_email_local_part_is_not_mistaken_for_uid(agent, fake):
    reply = await agent.chat("s1", "Cancel the booking with ada12345@example.com")

    # Must not cancel using "ada12345"; with no uid found it lists bookings instead
    assert fake.cancelled == []
    assert [a.name for a in reply.tool_activity] == ["list_bookings"]


async def test_concurrent_same_session_turns_do_not_interleave(agent):
    import asyncio

    await asyncio.gather(
        agent.chat("s1", "What's on my calendar?"),
        agent.chat("s1", "What event types do I have?"),
    )

    history = agent._sessions["s1"]
    # Two complete turns of 4 messages each, in order: each user message is
    # followed by its own assistant/tool/assistant block
    assert len(history) == 8
    assert [m.role for m in history] == ["user", "assistant", "tool", "assistant"] * 2


def test_trim_caps_history_and_realigns_to_user_turn():
    from app.agent.loop import MAX_HISTORY_MESSAGES
    from app.llm import Message

    # Build an over-long history of repeating 4-message turns
    turn = [
        Message(role="user", content="u"),
        Message(
            role="assistant", content="", tool_calls=[ToolCall(id="x", name="t", arguments={})]
        ),
        Message(role="tool"),
        Message(role="assistant", content="a"),
    ]
    history = [m for _ in range(30) for m in turn]  # 120 messages

    Agent._trim(history)

    assert len(history) <= MAX_HISTORY_MESSAGES
    assert history[0].role == "user"


async def test_lru_session_eviction_keeps_active_sessions(agent, monkeypatch):
    import app.agent.loop as loop_module

    monkeypatch.setattr(loop_module, "MAX_SESSIONS", 2)

    await agent.chat("a", "What's on my calendar?")
    await agent.chat("b", "What's on my calendar?")
    await agent.chat("a", "What event types do I have?")  # refresh "a"
    await agent.chat("c", "What's on my calendar?")  # evicts "b" (least recent), not "a"

    assert set(agent._sessions) == {"a", "c"}


async def test_provider_raw_payload_rides_along_on_assistant_messages(fake):
    """Vendor blocks (e.g. Anthropic thinking) must survive into history verbatim."""
    sentinel = [{"type": "thinking", "signature": "s"}]

    class RawProvider:
        def __init__(self):
            self.turn = 0

        async def complete(self, system, messages, tools):
            self.turn += 1
            if self.turn == 1:
                return LLMResponse(
                    tool_calls=[ToolCall(id="x", name="list_bookings", arguments={})],
                    raw=sentinel,
                )
            return LLMResponse(text="done")

    agent = Agent(RawProvider(), {"list_bookings": fake.list_bookings}, lambda: "system")

    await agent.chat("s1", "calendar?")

    tool_call_message = agent._sessions["s1"][1]
    assert tool_call_message.role == "assistant"
    assert tool_call_message.raw is sentinel


async def test_runaway_tool_loop_is_capped(fake):
    class AlwaysToolProvider:
        async def complete(self, system, messages, tools):
            return LLMResponse(tool_calls=[ToolCall(id="x", name="list_bookings", arguments={})])

    dispatch = {"list_bookings": fake.list_bookings}
    agent = Agent(AlwaysToolProvider(), dispatch, lambda: "system")

    reply = await agent.chat("s1", "loop forever")

    assert len(reply.tool_activity) == 8
    assert "rephrase" in reply.text
