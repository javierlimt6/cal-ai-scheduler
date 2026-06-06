"""The agent: per-session conversation state and the tool-use loop."""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from app.agent.tools import TOOLS, ToolFunc, execute_tool
from app.calcom import CalComError
from app.llm import LLMProvider, Message, ToolCall, ToolResult

logger = logging.getLogger(__name__)

MAX_TOOL_ITERATIONS = 8
# Cap per-session history so long-lived sessions don't grow without bound
# (in tokens sent to the LLM or in memory).
MAX_HISTORY_MESSAGES = 60
# Cap the number of tracked sessions; least-recently-used are evicted first.
MAX_SESSIONS = 500


@dataclass
class ToolActivity:
    """A record of one tool invocation, surfaced to the UI."""

    name: str
    arguments: dict
    ok: bool


@dataclass
class AgentReply:
    text: str
    tool_activity: list[ToolActivity] = field(default_factory=list)


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

    async def chat(self, session_id: str, user_message: str) -> AgentReply:
        # Serialize turns within a session so concurrent requests can't
        # interleave their messages into the shared history.
        async with self._locks.setdefault(session_id, asyncio.Lock()):
            return await self._chat(session_id, user_message)

    async def _chat(self, session_id: str, user_message: str) -> AgentReply:
        if session_id not in self._sessions and len(self._sessions) >= MAX_SESSIONS:
            evicted = next(iter(self._sessions))
            self._sessions.pop(evicted)
            self._locks.pop(evicted, None)
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
                return AgentReply(text=response.text, tool_activity=activity)

            history.append(
                Message(role="assistant", content=response.text, tool_calls=response.tool_calls)
            )

            outcomes = await asyncio.gather(
                *(self._run_tool(call) for call in response.tool_calls)
            )
            results = []
            for call, (content, ok) in zip(response.tool_calls, outcomes):
                results.append(ToolResult(tool_call_id=call.id, content=content, is_error=not ok))
                activity.append(ToolActivity(name=call.name, arguments=call.arguments, ok=ok))

            history.append(Message(role="tool", tool_results=results))

        text = "I wasn't able to finish that in a reasonable number of steps — could you rephrase or break it down?"
        history.append(Message(role="assistant", content=text))
        self._trim(history)
        return AgentReply(text=text, tool_activity=activity)

    async def _run_tool(self, call: ToolCall) -> tuple[str, bool]:
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
