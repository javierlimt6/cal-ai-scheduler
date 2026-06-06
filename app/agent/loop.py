"""The agent: per-session conversation state and the tool-use loop."""

import asyncio
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from uuid import uuid4

from app.agent.tools import TOOLS, ToolFunc, execute_tool
from app.calcom import CalComError
from app.llm import LLMProvider, Message, ToolCall, ToolResult

logger = logging.getLogger(__name__)

MAX_TOOL_ITERATIONS = 8
# Cap per-session history so long-lived sessions don't grow without bound
# (in tokens sent to the LLM or in memory).
MAX_HISTORY_MESSAGES = 60
# _trim realigns the window onto a user message; that only stays safe if one
# full turn (user + iterations*(assistant+tool) + assistant) fits in the cap.
assert MAX_TOOL_ITERATIONS * 2 + 2 <= MAX_HISTORY_MESSAGES
# Cap the number of tracked sessions; least-recently-used are evicted first.
MAX_SESSIONS = 500

# Destructive tools are never executed off the back of an LLM decision alone.
# The call is frozen server-side and only runs when the user explicitly
# approves it (see Agent.resolve_pending) — so a prompt-injected booking title
# can't talk the model into cancelling anything by itself.
DESTRUCTIVE_TOOLS = frozenset({"cancel_booking", "reschedule_booking"})

GATE_MESSAGE = (
    "CONFIRMATION_REQUIRED: {summary}. The action is on hold — the user has been shown a "
    "Confirm/Decline card in the chat UI and nothing happens until they click Confirm. "
    "Tell the user in one sentence what is awaiting their confirmation and point them at "
    "the card. Do not call the tool again."
)


class PendingActionError(Exception):
    """The referenced pending action no longer exists (stale or already resolved)."""


@dataclass
class PendingAction:
    """A destructive tool call frozen until the user approves it."""

    id: str
    call: ToolCall
    summary: str


@dataclass
class ToolActivity:
    """A record of one tool invocation, surfaced to the UI (name + outcome only)."""

    name: str
    ok: bool


@dataclass
class AgentReply:
    text: str
    tool_activity: list[ToolActivity] = field(default_factory=list)
    pending_action: PendingAction | None = None


class Agent:
    def __init__(
        self,
        provider: LLMProvider,
        dispatch: dict[str, ToolFunc],
        system_prompt: Callable[[], str],
    ):
        self._provider = provider
        self._dispatch = dispatch
        # Built per turn, not per process: the prompt embeds "now", which must
        # stay fresh so relative dates ("tomorrow") resolve correctly.
        self._system_prompt = system_prompt
        self._sessions: dict[str, list[Message]] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._pending: dict[str, PendingAction] = {}

    async def chat(self, session_id: str, user_message: str) -> AgentReply:
        # Serialize turns within a session so concurrent requests can't
        # interleave their messages into the shared history.
        async with self._locks.setdefault(session_id, asyncio.Lock()):
            return await self._chat(session_id, user_message)

    async def _chat(self, session_id: str, user_message: str) -> AgentReply:
        if session_id not in self._sessions and len(self._sessions) >= MAX_SESSIONS:
            # Evict the least-recently-used session that is NOT mid-turn: an
            # in-flight turn holds its lock, and popping that lock would let a
            # concurrent request fork the same history. If every session is
            # mid-turn (pathological), briefly exceed the cap instead.
            evicted = next(
                (
                    sid
                    for sid in self._sessions
                    if (lock := self._locks.get(sid)) is None or not lock.locked()
                ),
                None,
            )
            if evicted is not None:
                self._sessions.pop(evicted)
                self._locks.pop(evicted, None)
                self._pending.pop(evicted, None)
        # A new message moves the conversation on: any unconfirmed action is
        # stale and must never fire later off an old card.
        self._pending.pop(session_id, None)
        history = self._sessions.setdefault(session_id, [])
        # Re-insert to refresh recency: dict order doubles as the LRU order,
        # so active sessions aren't the ones evicted.
        self._sessions[session_id] = self._sessions.pop(session_id)
        self._trim(history)
        history.append(Message(role="user", content=user_message))
        system_prompt = self._system_prompt()
        activity: list[ToolActivity] = []

        for _ in range(MAX_TOOL_ITERATIONS):
            response = await self._provider.complete(system_prompt, history, TOOLS)

            if not response.tool_calls:
                history.append(Message(role="assistant", content=response.text))
                self._trim(history)
                return AgentReply(
                    text=response.text,
                    tool_activity=activity,
                    pending_action=self._pending.get(session_id),
                )

            history.append(
                Message(
                    role="assistant",
                    content=response.text,
                    tool_calls=response.tool_calls,
                    raw=response.raw,
                )
            )

            outcomes = await asyncio.gather(
                *(self._run_tool(session_id, call) for call in response.tool_calls)
            )
            results = []
            for call, outcome in zip(response.tool_calls, outcomes, strict=True):
                if isinstance(outcome, PendingAction):
                    # Held, not executed: tell the LLM why, add no activity.
                    results.append(
                        ToolResult(
                            tool_call_id=call.id,
                            content=GATE_MESSAGE.format(summary=outcome.summary),
                        )
                    )
                    continue
                content, ok = outcome
                results.append(ToolResult(tool_call_id=call.id, content=content, is_error=not ok))
                activity.append(ToolActivity(name=call.name, ok=ok))

            history.append(Message(role="tool", tool_results=results))

        text = "I wasn't able to finish that in a reasonable number of steps — could you rephrase or break it down?"
        history.append(Message(role="assistant", content=text))
        self._trim(history)
        return AgentReply(
            text=text, tool_activity=activity, pending_action=self._pending.get(session_id)
        )

    async def resolve_pending(self, session_id: str, action_id: str, approved: bool) -> AgentReply:
        """Execute (or drop) a held destructive action on the user's explicit say-so.

        This is the only path that runs a destructive tool, and it runs exactly
        the arguments frozen when the action was proposed — the LLM is not in
        the decision loop.
        """
        async with self._locks.setdefault(session_id, asyncio.Lock()):
            pending = self._pending.get(session_id)
            if pending is None or pending.id != action_id:
                raise PendingActionError(
                    "That confirmation has expired — tell me again what you'd like to do."
                )
            del self._pending[session_id]
            history = self._sessions.setdefault(session_id, [])

            if not approved:
                history.append(Message(role="user", content="(I declined — don't do that.)"))
                text = "Okay, I've left everything as it was."
                history.append(Message(role="assistant", content=text))
                self._trim(history)
                return AgentReply(text=text)

            content, ok = await self._execute(pending.call)
            history.append(Message(role="user", content=f"(Confirmed: {pending.summary}.)"))
            text = _describe_outcome(pending.call, content, ok)
            history.append(Message(role="assistant", content=text))
            self._trim(history)
            return AgentReply(
                text=text,
                tool_activity=[ToolActivity(name=pending.call.name, ok=ok)],
            )

    async def _run_tool(self, session_id: str, call: ToolCall) -> tuple[str, bool] | PendingAction:
        if call.name in DESTRUCTIVE_TOOLS:
            if session_id in self._pending:
                return (
                    "Another action is already awaiting the user's confirmation — "
                    "wait for their decision before proposing this one.",
                    False,
                )
            pending = PendingAction(id=uuid4().hex, call=call, summary=_describe_action(call))
            self._pending[session_id] = pending
            return pending
        return await self._execute(call)

    async def _execute(self, call: ToolCall) -> tuple[str, bool]:
        try:
            return await execute_tool(self._dispatch, call.name, call.arguments), True
        except CalComError as exc:
            return str(exc), False
        except Exception:
            logger.exception("Tool %s failed", call.name)
            return f"Internal error running tool {call.name}", False

    @staticmethod
    def _trim(history: list[Message]) -> None:
        """Drop oldest turns past the cap, keeping the window aligned on a user turn."""
        if len(history) <= MAX_HISTORY_MESSAGES:
            return
        del history[: len(history) - MAX_HISTORY_MESSAGES]
        while history and history[0].role != "user":
            del history[0]


def _describe_action(call: ToolCall) -> str:
    """One human-readable line for the confirmation card."""
    args = call.arguments
    if call.name == "cancel_booking":
        summary = f"Cancel booking {args.get('booking_uid', '?')}"
        if args.get("reason"):
            summary += f" — {args['reason']}"
        return summary
    if call.name == "reschedule_booking":
        return f"Reschedule booking {args.get('booking_uid', '?')} to {args.get('new_start', '?')}"
    return f"Run {call.name}"


def _describe_outcome(call: ToolCall, content: str, ok: bool) -> str:
    """Server-authored result text for a confirmed action (no LLM in this path)."""
    if not ok:
        return f"That didn't work: {content}"
    if call.name == "cancel_booking":
        return "Done — the booking has been cancelled."
    if call.name == "reschedule_booking":
        try:
            data = json.loads(content)
        except ValueError:
            data = None
        if isinstance(data, dict):
            return f"Done — rescheduled to {data.get('start', 'the new time')}."
        return "Done — the booking has been rescheduled."
    return f"Done — {call.name} completed."
