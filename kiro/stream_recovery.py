# -*- coding: utf-8 -*-

# Kiro Gateway
# https://github.com/jwadow/kiro-gateway
# Copyright (C) 2025 Jwadow
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
Mid-stream drop recovery (Kiro IDE parity), pure building blocks.

When Kiro closes a response stream in the middle of a turn (httpx.RemoteProtocolError,
httpx.ReadError), the official Kiro IDE client recovers as follows:

- drop before any output: resend the request;
- drop after text with no tool call: resend at most once, with the partial text kept in
  the history plus a "continue" prompt;
- drop while a tool call is streaming: keep the partial assistant message, answer every
  interrupted tool use with a synthetic failed tool result and call the model again
  (at most 3 consecutive recoveries);
- drop with reasoning only: no recovery, the turn fails;
- tool call fully received (stop frame) then a drop: no resend, the tool call is delivered.

This module holds the parts that need no I/O: the constants, the decision table, the
backoff formula and the builder of the continuation request. It never imports
kiro.streaming_core (the streaming layer imports this module).
"""

import asyncio
import copy
import json
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional

import httpx
from loguru import logger

from kiro.config import AUTO_TRIM_PAYLOAD, KIRO_MAX_PAYLOAD_BYTES
from kiro.payload_guards import check_payload_size, trim_payload_to_limit


# ==================================================================================================
# Constants
# ==================================================================================================

# Synthetic tool result sent for every tool use whose input stream was interrupted
INTERRUPTED_TOOL_RESULT_MESSAGE: str = (
    "The tool was NOT executed because its input stream was interrupted. "
    "Review the completed tool results and decide the next step."
)

# User prompt appended after partial assistant text when no tool call was involved
CONTINUE_PROMPT: str = (
    "The previous response was interrupted. "
    "Continue from the recorded response; do not repeat completed work."
)

# Message shown to the client when recovery is not possible or exhausted
CONNECTION_INTERRUPTED_MESSAGE: str = (
    "Your connection was interrupted. Please try again in a moment."
)

# Key under which the raw partial JSON text of an interrupted tool input is stored
INVALID_ARGS_KEY: str = "invalidArgs"

# Maximum number of consecutive tool recoveries per HTTP response (Kiro: pHp = 3)
TOOL_RECOVERY_MAX: int = 3

# Maximum number of text resends per HTTP response (Kiro: duplication cap = 1)
DUPLICATION_RETRY_MAX: int = 1

# Backoff before a retry: base * 2^(index - 1), +-jitter (Kiro: base 500 ms, jitter 0.2)
BACKOFF_BASE_S: float = 0.5
BACKOFF_JITTER: float = 0.2

# Tool status written into a synthetic tool result
TOOL_RESULT_STATUS_ERROR: str = "error"


# ==================================================================================================
# Data classes
# ==================================================================================================

class RecoveryAction(str, Enum):
    """
    What to do after a mid-stream drop.

    Attributes:
        FINISH: The response is complete enough: deliver what was received and end normally
        FAIL: Give up: the client gets an in-stream error
        RESEND: Resend the unchanged request (nothing was produced)
        RESEND_WITH_TEXT: Resend with the partial text in the history plus CONTINUE_PROMPT
        CONTINUE_TOOLS: Resend with the partial assistant message and failed tool results
    """
    FINISH = "finish"
    FAIL = "fail"
    RESEND = "resend"
    RESEND_WITH_TEXT = "resend_with_text"
    CONTINUE_TOOLS = "continue_tools"


@dataclass
class AttemptEvidence:
    """
    What one (dropped) upstream attempt produced before it ended.

    Attributes:
        text: Text content produced by the attempt (assistant message text)
        reasoning_seen: Whether any reasoning/thinking output was produced
        completed_tools: Tool calls whose stop frame arrived (OpenAI-style dicts with
            "id", "type" and "function": {"name", "arguments"})
        interrupted_tool: The tool call that was still streaming (raw partial arguments), or None
        saw_metering: Whether a meteringEvent frame arrived (the response was complete)
    """
    text: str = ""
    reasoning_seen: bool = False
    completed_tools: List[Dict[str, Any]] = field(default_factory=list)
    interrupted_tool: Optional[Dict[str, Any]] = None
    saw_metering: bool = False

    def record_event(self, event: Any) -> None:
        """
        Record the text / reasoning of a parsed event (only type and text fields are read).

        Args:
            event: KiroEvent produced by the stream parser
        """
        if event.type == "content" and event.content:
            self.text += event.content
        elif event.type == "thinking":
            self.reasoning_seen = True

    @property
    def has_tools(self) -> bool:
        """True if the attempt produced at least one tool call (completed or interrupted)."""
        return bool(self.completed_tools) or self.interrupted_tool is not None


@dataclass
class RecoveryBudget:
    """
    Counters shared by all attempts of one HTTP response.

    Attributes:
        stream_retries_used: Number of RESEND / RESEND_WITH_TEXT actions taken
        duplication_retries_used: Number of RESEND_WITH_TEXT actions taken
        tool_recoveries_used: Number of CONTINUE_TOOLS actions taken
        ever_text: An earlier attempt produced text
        ever_tool: An earlier attempt produced a tool call
    """
    stream_retries_used: int = 0
    duplication_retries_used: int = 0
    tool_recoveries_used: int = 0
    ever_text: bool = False
    ever_tool: bool = False

    def backoff_index(self, action: RecoveryAction) -> int:
        """
        Backoff index (1-based) for the given action, before the budget is advanced.

        Args:
            action: The chosen recovery action

        Returns:
            1 for a tool recovery, stream_retries_used + 1 for a resend
        """
        if action == RecoveryAction.CONTINUE_TOOLS:
            return 1
        return self.stream_retries_used + 1

    def advance(self, action: RecoveryAction, evidence: AttemptEvidence) -> None:
        """
        Account for one taken action and the evidence of the attempt that led to it.

        Args:
            action: The action that is about to be executed
            evidence: Evidence of the dropped attempt
        """
        if action == RecoveryAction.CONTINUE_TOOLS:
            self.tool_recoveries_used += 1
        elif action in (RecoveryAction.RESEND, RecoveryAction.RESEND_WITH_TEXT):
            self.stream_retries_used += 1
            if action == RecoveryAction.RESEND_WITH_TEXT:
                self.duplication_retries_used += 1
        if evidence.text:
            self.ever_text = True
        if evidence.has_tools:
            self.ever_tool = True


@dataclass(frozen=True)
class RecoveryDecision:
    """
    Result of decide_recovery.

    Attributes:
        action: What to do next
        reason: Short machine-readable reason (used in logs and tests)
    """
    action: RecoveryAction
    reason: str


class UpstreamStreamInterrupted(Exception):
    """
    Raised when a mid-stream drop cannot be recovered (or recovery is exhausted).

    Formatters turn it into an in-stream error, non-streaming routes into HTTP 502.

    Attributes:
        reason: Machine-readable reason (e.g. "reasoning_only", "streak_exhausted")
    """

    def __init__(self, reason: str, message: str = CONNECTION_INTERRUPTED_MESSAGE) -> None:
        self.reason = reason
        super().__init__(message)


@dataclass
class RecoverySettings:
    """
    Everything a streaming session needs to resend a request.

    Attributes:
        payload: The base Kiro request payload of the first attempt
        make_request: Coroutine function that sends a payload and returns the streaming response
        max_attempts: Total number of upstream attempts allowed for plain resends
    """
    payload: Dict[str, Any]
    make_request: Callable[[Dict[str, Any]], Awaitable[httpx.Response]]
    max_attempts: int = 3


# ==================================================================================================
# Decision table
# ==================================================================================================

def decide_recovery(
    evidence: AttemptEvidence,
    budget: RecoveryBudget,
    max_attempts: int,
) -> RecoveryDecision:
    """
    Decide what to do after a drop. The first matching rule wins.

    1. meteringEvent seen: FINISH (the answer was complete before the drop);
    2. interrupted tool: CONTINUE_TOOLS while tool_recoveries_used < TOOL_RECOVERY_MAX,
       else FAIL (streak_exhausted);
    3. completed tools only: FINISH (tool_complete);
    4. an earlier attempt had a tool: FAIL (tool_context);
    5. reasoning without text: FAIL (reasoning_only);
    6. attempt budget used up: FAIL (attempts_exhausted);
    7. text produced (now or earlier): RESEND_WITH_TEXT once, then FAIL (duplication_cap);
    8. nothing produced: RESEND.

    Args:
        evidence: What the dropped attempt produced
        budget: Counters of the attempts so far (not modified)
        max_attempts: Total number of upstream attempts allowed for plain resends

    Returns:
        The decision with its reason
    """
    if evidence.saw_metering:
        return RecoveryDecision(RecoveryAction.FINISH, "complete_before_drop")

    if evidence.interrupted_tool is not None:
        if budget.tool_recoveries_used < TOOL_RECOVERY_MAX:
            return RecoveryDecision(RecoveryAction.CONTINUE_TOOLS, "interrupted_tool")
        return RecoveryDecision(RecoveryAction.FAIL, "streak_exhausted")

    if evidence.completed_tools:
        return RecoveryDecision(RecoveryAction.FINISH, "tool_complete")

    if budget.ever_tool:
        return RecoveryDecision(RecoveryAction.FAIL, "tool_context")

    if evidence.reasoning_seen and not evidence.text:
        return RecoveryDecision(RecoveryAction.FAIL, "reasoning_only")

    if budget.stream_retries_used + 1 >= max_attempts:
        return RecoveryDecision(RecoveryAction.FAIL, "attempts_exhausted")

    if budget.ever_text or evidence.text:
        if budget.duplication_retries_used < DUPLICATION_RETRY_MAX:
            return RecoveryDecision(RecoveryAction.RESEND_WITH_TEXT, "text_produced")
        return RecoveryDecision(RecoveryAction.FAIL, "duplication_cap")

    return RecoveryDecision(RecoveryAction.RESEND, "no_output")


# ==================================================================================================
# Backoff
# ==================================================================================================

def compute_backoff_s(index: int, rng: Optional[random.Random] = None) -> float:
    """
    Compute the wait before a retry: base * 2^(index - 1) with +-jitter.

    Args:
        index: 1-based retry index
        rng: Optional random generator (tests pass a seeded one)

    Returns:
        Seconds to wait (never negative)
    """
    generator = rng if rng is not None else random
    base = BACKOFF_BASE_S * (2 ** max(index - 1, 0))
    jitter = generator.uniform(-BACKOFF_JITTER, BACKOFF_JITTER)
    return max(base * (1.0 + jitter), 0.0)


async def backoff_sleep(seconds: float) -> None:
    """
    Sleep before a retry. A module-level coroutine so tests can patch it.

    Args:
        seconds: Seconds to wait
    """
    await asyncio.sleep(seconds)


# ==================================================================================================
# Continuation payload
# ==================================================================================================

def _tool_use_entry(tool: Dict[str, Any], interrupted: bool) -> Dict[str, Any]:
    """
    Convert a parser tool call into a Kiro history toolUse entry.

    Args:
        tool: OpenAI-style tool call dict ("id", "function": {"name", "arguments"})
        interrupted: True if the tool input was still streaming (raw partial JSON text)

    Returns:
        {"toolUseId", "name", "input"}; input is {"invalidArgs": raw} for an interrupted tool,
        the parsed object (or {} if it is not an object) for a completed one
    """
    function = tool.get("function", {})
    raw_args = function.get("arguments", "")
    if interrupted:
        tool_input: Dict[str, Any] = {INVALID_ARGS_KEY: raw_args if isinstance(raw_args, str) else ""}
    else:
        try:
            parsed = json.loads(raw_args) if isinstance(raw_args, str) and raw_args.strip() else {}
        except json.JSONDecodeError:
            parsed = {}
        tool_input = parsed if isinstance(parsed, dict) else {}
    return {
        "toolUseId": tool.get("id", ""),
        "name": function.get("name", ""),
        "input": tool_input,
    }


def build_continuation_payload(
    payload: Dict[str, Any],
    action: RecoveryAction,
    evidence: AttemptEvidence,
) -> Dict[str, Any]:
    """
    Build the upstream request of the next attempt (the base payload is never mutated).

    CONTINUE_TOOLS: the current user message moves into the history (without modelId and
    tools), followed by an assistant message with the partial text and every tool use of the
    attempt (interrupted ones carry {"invalidArgs": raw}). The new current message is empty
    and answers every tool use with a failed tool result.

    RESEND_WITH_TEXT: same, without tool uses, and the new current message is CONTINUE_PROMPT.

    RESEND: an unchanged deep copy of the payload.

    Args:
        payload: The payload of the attempt that dropped
        action: RESEND, RESEND_WITH_TEXT or CONTINUE_TOOLS
        evidence: What the dropped attempt produced

    Returns:
        A new payload dict

    Raises:
        ValueError: If the action is not a resend action or the payload has no current message
    """
    if action not in (
        RecoveryAction.RESEND,
        RecoveryAction.RESEND_WITH_TEXT,
        RecoveryAction.CONTINUE_TOOLS,
    ):
        raise ValueError(f"Cannot build a continuation payload for action {action}")

    new_payload = copy.deepcopy(payload)
    if action == RecoveryAction.RESEND:
        return new_payload

    conversation_state = new_payload.get("conversationState")
    current_message = conversation_state.get("currentMessage") if conversation_state else None
    user_input = current_message.get("userInputMessage") if current_message else None
    if user_input is None:
        raise ValueError("Payload has no conversationState.currentMessage.userInputMessage")

    model_id = user_input.get("modelId")
    context = user_input.get("userInputMessageContext") or {}
    tools = context.get("tools")

    # The current user message becomes a history entry (history user messages carry neither
    # modelId nor tools)
    history_user = copy.deepcopy(user_input)
    history_user.pop("modelId", None)
    history_context = history_user.get("userInputMessageContext")
    if isinstance(history_context, dict):
        history_context.pop("tools", None)
        if not history_context:
            history_user.pop("userInputMessageContext", None)

    assistant: Dict[str, Any] = {"content": evidence.text}
    new_context: Dict[str, Any] = {}
    if tools:
        new_context["tools"] = tools

    if action == RecoveryAction.CONTINUE_TOOLS:
        tool_uses = [_tool_use_entry(tool, interrupted=False) for tool in evidence.completed_tools]
        if evidence.interrupted_tool is not None:
            tool_uses.append(_tool_use_entry(evidence.interrupted_tool, interrupted=True))
        assistant["toolUses"] = tool_uses
        new_context["toolResults"] = [
            {
                "toolUseId": tool_use["toolUseId"],
                "status": TOOL_RESULT_STATUS_ERROR,
                "content": [{"text": INTERRUPTED_TOOL_RESULT_MESSAGE}],
            }
            for tool_use in tool_uses
        ]
        new_content = ""
    else:
        new_content = CONTINUE_PROMPT

    new_user_input: Dict[str, Any] = {
        "content": new_content,
        "modelId": model_id,
        "origin": user_input.get("origin", "AI_EDITOR"),
    }
    if new_context:
        new_user_input["userInputMessageContext"] = new_context

    history = conversation_state.setdefault("history", [])
    history.append({"userInputMessage": history_user})
    history.append({"assistantResponseMessage": assistant})
    current_message["userInputMessage"] = new_user_input

    if AUTO_TRIM_PAYLOAD:
        size = check_payload_size(new_payload)
        if size > KIRO_MAX_PAYLOAD_BYTES:
            stats = trim_payload_to_limit(new_payload, KIRO_MAX_PAYLOAD_BYTES)
            logger.info(
                f"[StreamRecovery] continuation payload trimmed: "
                f"{stats.original_bytes} -> {stats.final_bytes} bytes"
            )

    return new_payload
