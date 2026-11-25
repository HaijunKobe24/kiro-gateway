# -*- coding: utf-8 -*-

"""
Unit tests for kiro.stream_recovery (mid-stream drop recovery, Kiro IDE parity).

Covers the decision table (every row and boundary), the backoff formula, the continuation
payload builder (the shape that the real Kiro upstream accepted with HTTP 200) and the
small data classes. No I/O happens here: everything is pure.
"""

import copy
import json
import random
from unittest.mock import AsyncMock, patch

import pytest

from kiro import stream_recovery
from kiro.stream_recovery import (
    BACKOFF_BASE_S,
    BACKOFF_JITTER,
    CONNECTION_INTERRUPTED_MESSAGE,
    CONTINUE_PROMPT,
    DUPLICATION_RETRY_MAX,
    INTERRUPTED_TOOL_RESULT_MESSAGE,
    INVALID_ARGS_KEY,
    TOOL_RECOVERY_MAX,
    AttemptEvidence,
    RecoveryAction,
    RecoveryBudget,
    RecoveryDecision,
    RecoverySettings,
    UpstreamStreamInterrupted,
    backoff_sleep,
    build_continuation_payload,
    compute_backoff_s,
    decide_recovery,
)

MAX_ATTEMPTS = 3


def tool_call(tool_id: str, args: str, name: str = "get_weather"):
    """Builds a parser-style tool call dict."""
    return {"id": tool_id, "type": "function", "function": {"name": name, "arguments": args}}


def base_payload():
    """A realistic base payload: one history pair, current message with tools."""
    return {
        "profileArn": "arn:aws:codewhisperer:us-east-1:123:profile/X",
        "conversationState": {
            "chatTriggerType": "MANUAL",
            "conversationId": "conv-1",
            "history": [
                {"userInputMessage": {"content": "hi", "origin": "AI_EDITOR"}},
                {"assistantResponseMessage": {"content": "hello"}},
            ],
            "currentMessage": {
                "userInputMessage": {
                    "content": "What is the weather in Paris?",
                    "modelId": "claude-haiku-5.5",
                    "origin": "AI_EDITOR",
                    "userInputMessageContext": {
                        "tools": [{"toolSpecification": {"name": "get_weather"}}],
                    },
                }
            },
        },
        "additionalModelRequestFields": {"reasoning_effort": "low"},
    }


# =============================================================================
# Constants
# =============================================================================

class TestConstants:
    """The Kiro-parity constants must keep their exact values."""

    def test_interrupted_tool_message_is_exact(self):
        """
        What it does: Compares the synthetic tool result text with the Kiro bundle text.
        Goal: The text is part of the model-visible contract and must not drift.
        """
        assert INTERRUPTED_TOOL_RESULT_MESSAGE == (
            "The tool was NOT executed because its input stream was interrupted. "
            "Review the completed tool results and decide the next step."
        )

    def test_user_facing_message_is_exact(self):
        """
        What it does: Compares the client-visible error message.
        Goal: Same wording as Kiro.
        """
        assert CONNECTION_INTERRUPTED_MESSAGE == "Your connection was interrupted. Please try again in a moment."

    def test_caps_and_backoff_match_kiro(self):
        """
        What it does: Reads the numeric constants.
        Goal: 3 tool recoveries, 1 duplication retry, 500 ms base, 20 percent jitter.
        """
        assert TOOL_RECOVERY_MAX == 3
        assert DUPLICATION_RETRY_MAX == 1
        assert BACKOFF_BASE_S == 0.5
        assert BACKOFF_JITTER == 0.2
        assert INVALID_ARGS_KEY == "invalidArgs"


# =============================================================================
# Decision table
# =============================================================================

class TestDecideRecovery:
    """decide_recovery: one test per row of the table, plus boundaries."""

    def test_metering_seen_finishes(self):
        """
        What it does: The attempt saw a meteringEvent, even with an interrupted tool.
        Goal: Row 1 wins over everything: FINISH complete_before_drop.
        """
        evidence = AttemptEvidence(
            text="x", interrupted_tool=tool_call("t1", '{"a'), saw_metering=True
        )
        decision = decide_recovery(evidence, RecoveryBudget(), MAX_ATTEMPTS)
        assert decision == RecoveryDecision(RecoveryAction.FINISH, "complete_before_drop")

    def test_interrupted_tool_continues(self):
        """
        What it does: Interrupted tool, no tool recoveries used.
        Goal: Row 2: CONTINUE_TOOLS.
        """
        evidence = AttemptEvidence(text="x", interrupted_tool=tool_call("t1", '{"a'))
        decision = decide_recovery(evidence, RecoveryBudget(), MAX_ATTEMPTS)
        assert decision.action == RecoveryAction.CONTINUE_TOOLS
        assert decision.reason == "interrupted_tool"

    @pytest.mark.parametrize("used,expected", [
        (0, RecoveryAction.CONTINUE_TOOLS),
        (1, RecoveryAction.CONTINUE_TOOLS),
        (2, RecoveryAction.CONTINUE_TOOLS),
        (3, RecoveryAction.FAIL),
        (4, RecoveryAction.FAIL),
    ])
    def test_tool_recovery_cap_is_three(self, used, expected):
        """
        What it does: Varies tool_recoveries_used around the cap.
        Goal: Boundary: the 3rd recovery is allowed, the 4th is refused (streak_exhausted).
        """
        evidence = AttemptEvidence(interrupted_tool=tool_call("t1", '{"a'))
        decision = decide_recovery(evidence, RecoveryBudget(tool_recoveries_used=used), MAX_ATTEMPTS)
        assert decision.action == expected
        if expected == RecoveryAction.FAIL:
            assert decision.reason == "streak_exhausted"

    def test_mixed_completed_and_interrupted_continues(self):
        """
        What it does: One completed and one interrupted tool.
        Goal: The interrupted tool decides: CONTINUE_TOOLS (not FINISH).
        """
        evidence = AttemptEvidence(
            completed_tools=[tool_call("t1", '{"a": 1}')],
            interrupted_tool=tool_call("t2", '{"b'),
        )
        decision = decide_recovery(evidence, RecoveryBudget(), MAX_ATTEMPTS)
        assert decision.action == RecoveryAction.CONTINUE_TOOLS

    def test_completed_tool_only_finishes(self):
        """
        What it does: A complete tool call (stop frame) and then a drop.
        Goal: Row 3: FINISH tool_complete, never a resend.
        """
        evidence = AttemptEvidence(text="t", completed_tools=[tool_call("t1", '{"a": 1}')])
        decision = decide_recovery(evidence, RecoveryBudget(), MAX_ATTEMPTS)
        assert decision == RecoveryDecision(RecoveryAction.FINISH, "tool_complete")

    def test_completed_tool_finishes_even_when_budget_is_used_up(self):
        """
        What it does: Completed tool with every counter at its cap.
        Goal: FINISH does not depend on the budget.
        """
        budget = RecoveryBudget(stream_retries_used=5, duplication_retries_used=5, tool_recoveries_used=5)
        evidence = AttemptEvidence(completed_tools=[tool_call("t1", "{}")])
        assert decide_recovery(evidence, budget, MAX_ATTEMPTS).action == RecoveryAction.FINISH

    def test_ever_tool_blocks_resend(self):
        """
        What it does: Nothing in this attempt, but an earlier attempt had a tool.
        Goal: Row 4: FAIL tool_context (Kiro: w needs !toolEver).
        """
        decision = decide_recovery(AttemptEvidence(), RecoveryBudget(ever_tool=True), MAX_ATTEMPTS)
        assert decision == RecoveryDecision(RecoveryAction.FAIL, "tool_context")

    def test_ever_tool_with_text_still_fails(self):
        """
        What it does: Text in this attempt after an earlier tool recovery.
        Goal: Still FAIL tool_context (no text resend after a tool recovery).
        """
        decision = decide_recovery(AttemptEvidence(text="abc"), RecoveryBudget(ever_tool=True), MAX_ATTEMPTS)
        assert decision.reason == "tool_context"

    def test_ever_tool_still_allows_a_new_tool_recovery(self):
        """
        What it does: Interrupted tool in a continuation attempt.
        Goal: Tool recovery is still possible (row 2 comes before row 4).
        """
        evidence = AttemptEvidence(interrupted_tool=tool_call("t2", '{"b'))
        budget = RecoveryBudget(ever_tool=True, tool_recoveries_used=1)
        assert decide_recovery(evidence, budget, MAX_ATTEMPTS).action == RecoveryAction.CONTINUE_TOOLS

    def test_reasoning_only_fails(self):
        """
        What it does: Reasoning output, no text, no tool.
        Goal: Row 5: FAIL reasoning_only.
        """
        decision = decide_recovery(AttemptEvidence(reasoning_seen=True), RecoveryBudget(), MAX_ATTEMPTS)
        assert decision == RecoveryDecision(RecoveryAction.FAIL, "reasoning_only")

    def test_reasoning_plus_text_is_recoverable(self):
        """
        What it does: Reasoning followed by text.
        Goal: Text exists, so row 7 applies: RESEND_WITH_TEXT.
        """
        evidence = AttemptEvidence(reasoning_seen=True, text="Answer")
        decision = decide_recovery(evidence, RecoveryBudget(), MAX_ATTEMPTS)
        assert decision.action == RecoveryAction.RESEND_WITH_TEXT

    def test_attempts_exhausted(self):
        """
        What it does: stream_retries_used + 1 reaches max_attempts.
        Goal: Row 6: FAIL attempts_exhausted.
        """
        decision = decide_recovery(AttemptEvidence(), RecoveryBudget(stream_retries_used=2), MAX_ATTEMPTS)
        assert decision == RecoveryDecision(RecoveryAction.FAIL, "attempts_exhausted")

    def test_attempts_boundary_last_allowed_resend(self):
        """
        What it does: One attempt left.
        Goal: stream_retries_used=1 with max_attempts=3 still resends.
        """
        decision = decide_recovery(AttemptEvidence(), RecoveryBudget(stream_retries_used=1), MAX_ATTEMPTS)
        assert decision.action == RecoveryAction.RESEND

    def test_max_attempts_one_never_resends(self):
        """
        What it does: max_attempts=1 (a single attempt is allowed).
        Goal: FAIL attempts_exhausted on the first drop.
        """
        decision = decide_recovery(AttemptEvidence(), RecoveryBudget(), 1)
        assert decision.reason == "attempts_exhausted"

    def test_attempt_budget_does_not_cap_tool_recovery(self):
        """
        What it does: Tool recovery with the plain resend budget used up.
        Goal: Tool recovery has its own cap (3), not max_attempts.
        """
        evidence = AttemptEvidence(interrupted_tool=tool_call("t1", '{"a'))
        budget = RecoveryBudget(stream_retries_used=9)
        assert decide_recovery(evidence, budget, MAX_ATTEMPTS).action == RecoveryAction.CONTINUE_TOOLS

    def test_text_resends_once(self):
        """
        What it does: Text, no tool, no duplication retry used.
        Goal: Row 7: RESEND_WITH_TEXT.
        """
        decision = decide_recovery(AttemptEvidence(text="Hello"), RecoveryBudget(), MAX_ATTEMPTS)
        assert decision == RecoveryDecision(RecoveryAction.RESEND_WITH_TEXT, "text_produced")

    def test_duplication_cap_is_one(self):
        """
        What it does: Text again after the one allowed duplication retry.
        Goal: FAIL duplication_cap.
        """
        budget = RecoveryBudget(stream_retries_used=1, duplication_retries_used=1, ever_text=True)
        decision = decide_recovery(AttemptEvidence(text="more"), budget, MAX_ATTEMPTS)
        assert decision == RecoveryDecision(RecoveryAction.FAIL, "duplication_cap")

    def test_earlier_text_counts_when_this_attempt_is_empty(self):
        """
        What it does: Earlier attempt had text, this attempt produced nothing.
        Goal: ever_text keeps the text rule (RESEND_WITH_TEXT while the cap allows).
        """
        decision = decide_recovery(AttemptEvidence(), RecoveryBudget(ever_text=True), MAX_ATTEMPTS)
        assert decision.action == RecoveryAction.RESEND_WITH_TEXT

    def test_nothing_produced_resends(self):
        """
        What it does: Drop before any output.
        Goal: Row 8: RESEND.
        """
        decision = decide_recovery(AttemptEvidence(), RecoveryBudget(), MAX_ATTEMPTS)
        assert decision == RecoveryDecision(RecoveryAction.RESEND, "no_output")

    def test_budget_is_not_modified_by_decide(self):
        """
        What it does: Calls decide_recovery and compares the budget before and after.
        Goal: decide_recovery is pure.
        """
        budget = RecoveryBudget(stream_retries_used=1)
        before = copy.deepcopy(budget)
        decide_recovery(AttemptEvidence(text="x"), budget, MAX_ATTEMPTS)
        assert budget == before


# =============================================================================
# Budget bookkeeping
# =============================================================================

class TestRecoveryBudget:
    """RecoveryBudget.advance and backoff_index."""

    def test_advance_resend(self):
        """
        What it does: advance(RESEND) with empty evidence.
        Goal: Only stream_retries_used grows.
        """
        budget = RecoveryBudget()
        budget.advance(RecoveryAction.RESEND, AttemptEvidence())
        assert (budget.stream_retries_used, budget.duplication_retries_used, budget.tool_recoveries_used) == (1, 0, 0)

    def test_advance_resend_with_text(self):
        """
        What it does: advance(RESEND_WITH_TEXT) with text evidence.
        Goal: stream and duplication counters grow, ever_text is set.
        """
        budget = RecoveryBudget()
        budget.advance(RecoveryAction.RESEND_WITH_TEXT, AttemptEvidence(text="a"))
        assert budget.stream_retries_used == 1
        assert budget.duplication_retries_used == 1
        assert budget.ever_text is True
        assert budget.ever_tool is False

    def test_advance_continue_tools(self):
        """
        What it does: advance(CONTINUE_TOOLS) with an interrupted tool.
        Goal: tool counter grows, stream counter does not, ever_tool is set.
        """
        budget = RecoveryBudget()
        budget.advance(RecoveryAction.CONTINUE_TOOLS, AttemptEvidence(interrupted_tool=tool_call("t", "{")))
        assert budget.tool_recoveries_used == 1
        assert budget.stream_retries_used == 0
        assert budget.ever_tool is True

    def test_backoff_index(self):
        """
        What it does: backoff_index for both kinds of action.
        Goal: Tool recovery always 1; resend is stream_retries_used + 1.
        """
        budget = RecoveryBudget(stream_retries_used=2, tool_recoveries_used=2)
        assert budget.backoff_index(RecoveryAction.CONTINUE_TOOLS) == 1
        assert budget.backoff_index(RecoveryAction.RESEND) == 3
        assert budget.backoff_index(RecoveryAction.RESEND_WITH_TEXT) == 3

    def test_three_consecutive_tool_recoveries_then_refusal(self):
        """
        What it does: Drives decide + advance through four interrupted-tool drops.
        Goal: Three CONTINUE_TOOLS then FAIL streak_exhausted.
        """
        budget = RecoveryBudget()
        actions = []
        for _ in range(4):
            evidence = AttemptEvidence(interrupted_tool=tool_call("t", '{"a'))
            decision = decide_recovery(evidence, budget, MAX_ATTEMPTS)
            actions.append(decision.action)
            if decision.action == RecoveryAction.FAIL:
                break
            budget.advance(decision.action, evidence)
        assert actions == [RecoveryAction.CONTINUE_TOOLS] * 3 + [RecoveryAction.FAIL]


class TestAttemptEvidence:
    """AttemptEvidence helpers."""

    def test_has_tools(self):
        """
        What it does: has_tools for empty, completed and interrupted evidence.
        Goal: True when any tool exists.
        """
        assert AttemptEvidence().has_tools is False
        assert AttemptEvidence(completed_tools=[tool_call("a", "{}")]).has_tools is True
        assert AttemptEvidence(interrupted_tool=tool_call("a", "{")).has_tools is True

    def test_record_event_collects_text_and_reasoning(self):
        """
        What it does: Feeds content, thinking, usage and empty-content events.
        Goal: text accumulates, reasoning flag is set, other events are ignored.
        """
        from kiro.streaming_core import KiroEvent
        evidence = AttemptEvidence()
        evidence.record_event(KiroEvent(type="content", content="ab"))
        evidence.record_event(KiroEvent(type="content", content=""))
        evidence.record_event(KiroEvent(type="content", content="cd"))
        evidence.record_event(KiroEvent(type="usage", usage=1.0))
        assert evidence.text == "abcd"
        assert evidence.reasoning_seen is False
        evidence.record_event(KiroEvent(type="thinking", thinking_content="..."))
        assert evidence.reasoning_seen is True


# =============================================================================
# Backoff
# =============================================================================

class TestBackoff:
    """compute_backoff_s and backoff_sleep."""

    @pytest.mark.parametrize("index,base", [(1, 0.5), (2, 1.0), (3, 2.0), (4, 4.0)])
    def test_bounds(self, index, base):
        """
        What it does: Samples many values per index.
        Goal: Always within base * (1 +- 0.2).
        """
        rng = random.Random(1234)
        values = [compute_backoff_s(index, rng) for _ in range(500)]
        assert min(values) >= base * 0.8 - 1e-9
        assert max(values) <= base * 1.2 + 1e-9
        assert min(values) < base < max(values)

    def test_index_zero_or_negative_is_clamped(self):
        """
        What it does: Index 0 and -3.
        Goal: Same as index 1, never a fraction of the base.
        """
        rng = random.Random(1)
        for index in (0, -3):
            value = compute_backoff_s(index, rng)
            assert 0.4 - 1e-9 <= value <= 0.6 + 1e-9

    def test_default_rng_works(self):
        """
        What it does: Calls without an rng.
        Goal: Result is within the index-1 bounds.
        """
        assert 0.4 - 1e-9 <= compute_backoff_s(1) <= 0.6 + 1e-9

    @pytest.mark.asyncio
    async def test_backoff_sleep_delegates_to_asyncio_sleep(self):
        """
        What it does: Patches asyncio.sleep.
        Goal: backoff_sleep waits exactly the given seconds.
        """
        with patch("kiro.stream_recovery.asyncio.sleep", new=AsyncMock()) as fake_sleep:
            await backoff_sleep(0.25)
        fake_sleep.assert_awaited_once_with(0.25)


# =============================================================================
# Continuation payload
# =============================================================================

class TestBuildContinuationPayload:
    """build_continuation_payload: the shape accepted by the real upstream (HTTP 200)."""

    def test_resend_is_an_unchanged_copy(self):
        """
        What it does: RESEND.
        Goal: Equal to the base but a different object (deep copy).
        """
        base = base_payload()
        result = build_continuation_payload(base, RecoveryAction.RESEND, AttemptEvidence())
        assert result == base
        assert result is not base
        assert result["conversationState"] is not base["conversationState"]

    def test_interrupted_tool_shape(self):
        """
        What it does: CONTINUE_TOOLS with partial text and one interrupted tool.
        Goal: History gets user + assistant (text, toolUse with invalidArgs); the current message
        is empty with tools kept and one error tool result carrying the generic message.
        """
        base = base_payload()
        evidence = AttemptEvidence(text="Let me check.", interrupted_tool=tool_call("tooluse_1", '{"city": "Par'))
        result = build_continuation_payload(base, RecoveryAction.CONTINUE_TOOLS, evidence)
        state = result["conversationState"]
        assert len(state["history"]) == 4
        moved_user, assistant = state["history"][2], state["history"][3]
        assert moved_user == {"userInputMessage": {"content": "What is the weather in Paris?", "origin": "AI_EDITOR"}}
        assert assistant == {
            "assistantResponseMessage": {
                "content": "Let me check.",
                "toolUses": [{
                    "toolUseId": "tooluse_1",
                    "name": "get_weather",
                    "input": {INVALID_ARGS_KEY: '{"city": "Par'},
                }],
            }
        }
        current = state["currentMessage"]["userInputMessage"]
        assert current["content"] == ""
        assert current["modelId"] == "claude-haiku-5.5"
        assert current["origin"] == "AI_EDITOR"
        assert current["userInputMessageContext"]["tools"] == base["conversationState"]["currentMessage"][
            "userInputMessage"]["userInputMessageContext"]["tools"]
        assert current["userInputMessageContext"]["toolResults"] == [{
            "toolUseId": "tooluse_1",
            "status": "error",
            "content": [{"text": INTERRUPTED_TOOL_RESULT_MESSAGE}],
        }]

    def test_moved_user_message_has_no_model_id_and_no_tools(self):
        """
        What it does: Checks the user message moved into the history.
        Goal: modelId and tools are removed; an emptied userInputMessageContext is dropped.
        """
        result = build_continuation_payload(
            base_payload(), RecoveryAction.CONTINUE_TOOLS,
            AttemptEvidence(interrupted_tool=tool_call("t1", "{")),
        )
        moved = result["conversationState"]["history"][-2]["userInputMessage"]
        assert "modelId" not in moved
        assert "userInputMessageContext" not in moved

    def test_moved_user_message_keeps_its_tool_results(self):
        """
        What it does: Base current message already carries toolResults (agent loop turn).
        Goal: The results stay on the moved history message, only tools are removed.
        """
        base = base_payload()
        context = base["conversationState"]["currentMessage"]["userInputMessage"]["userInputMessageContext"]
        context["toolResults"] = [{"toolUseId": "old", "status": "success", "content": [{"text": "ok"}]}]
        result = build_continuation_payload(
            base, RecoveryAction.CONTINUE_TOOLS,
            AttemptEvidence(interrupted_tool=tool_call("t1", "{")),
        )
        moved = result["conversationState"]["history"][-2]["userInputMessage"]
        assert moved["userInputMessageContext"] == {"toolResults": context["toolResults"]}

    def test_empty_assistant_text_is_kept_as_empty_string(self):
        """
        What it does: Tool-only attempt (no text).
        Goal: assistant content is "" (variant V2, accepted by the upstream).
        """
        result = build_continuation_payload(
            base_payload(), RecoveryAction.CONTINUE_TOOLS,
            AttemptEvidence(interrupted_tool=tool_call("t1", '{"a')),
        )
        assert result["conversationState"]["history"][-1]["assistantResponseMessage"]["content"] == ""

    def test_empty_partial_args(self):
        """
        What it does: Interrupted tool whose arguments are still empty.
        Goal: input is {"invalidArgs": ""} (variant V6, accepted by the upstream).
        """
        result = build_continuation_payload(
            base_payload(), RecoveryAction.CONTINUE_TOOLS,
            AttemptEvidence(interrupted_tool=tool_call("t1", "")),
        )
        tool_use = result["conversationState"]["history"][-1]["assistantResponseMessage"]["toolUses"][0]
        assert tool_use["input"] == {INVALID_ARGS_KEY: ""}

    def test_mixed_tools_all_get_error_results(self):
        """
        What it does: One completed and one interrupted tool.
        Goal: Completed input is the parsed object, interrupted uses invalidArgs, and BOTH get an
        error result in the same order (variant V3, accepted by the upstream).
        """
        evidence = AttemptEvidence(
            text="Two cities",
            completed_tools=[tool_call("a", '{"city": "Paris"}')],
            interrupted_tool=tool_call("b", '{"city": "Lon'),
        )
        result = build_continuation_payload(base_payload(), RecoveryAction.CONTINUE_TOOLS, evidence)
        tool_uses = result["conversationState"]["history"][-1]["assistantResponseMessage"]["toolUses"]
        assert [t["toolUseId"] for t in tool_uses] == ["a", "b"]
        assert tool_uses[0]["input"] == {"city": "Paris"}
        assert tool_uses[1]["input"] == {INVALID_ARGS_KEY: '{"city": "Lon'}
        results = result["conversationState"]["currentMessage"]["userInputMessage"][
            "userInputMessageContext"]["toolResults"]
        assert [r["toolUseId"] for r in results] == ["a", "b"]
        assert {r["status"] for r in results} == {"error"}

    @pytest.mark.parametrize("raw", ["", "   ", "not json", "[1, 2]", '"str"'])
    def test_completed_tool_with_odd_args_becomes_empty_object(self, raw):
        """
        What it does: Completed tool whose arguments are blank, invalid or not an object.
        Goal: input falls back to {} (the API wants an object).
        """
        evidence = AttemptEvidence(completed_tools=[tool_call("a", raw)])
        result = build_continuation_payload(base_payload(), RecoveryAction.CONTINUE_TOOLS, evidence)
        tool_use = result["conversationState"]["history"][-1]["assistantResponseMessage"]["toolUses"][0]
        assert tool_use["input"] == {}

    def test_base_payload_is_not_mutated(self):
        """
        What it does: Builds both continuation kinds from one base.
        Goal: The base payload is byte-for-byte unchanged.
        """
        base = base_payload()
        snapshot = json.dumps(base, sort_keys=True)
        build_continuation_payload(base, RecoveryAction.CONTINUE_TOOLS,
                                   AttemptEvidence(interrupted_tool=tool_call("t", "{")))
        build_continuation_payload(base, RecoveryAction.RESEND_WITH_TEXT, AttemptEvidence(text="x"))
        assert json.dumps(base, sort_keys=True) == snapshot

    def test_other_keys_are_preserved(self):
        """
        What it does: Checks profileArn, conversationId, chatTriggerType, additionalModelRequestFields.
        Goal: All stay exactly as in the base payload.
        """
        base = base_payload()
        result = build_continuation_payload(
            base, RecoveryAction.CONTINUE_TOOLS,
            AttemptEvidence(interrupted_tool=tool_call("t", "{")),
        )
        assert result["profileArn"] == base["profileArn"]
        assert result["additionalModelRequestFields"] == base["additionalModelRequestFields"]
        assert result["conversationState"]["conversationId"] == "conv-1"
        assert result["conversationState"]["chatTriggerType"] == "MANUAL"

    def test_second_continuation_builds_on_the_first(self):
        """
        What it does: Applies CONTINUE_TOOLS twice (the second input is the first output).
        Goal: History alternates user/assistant and grows by two entries per recovery; the
        first synthetic result message moves into the history.
        """
        first = build_continuation_payload(
            base_payload(), RecoveryAction.CONTINUE_TOOLS,
            AttemptEvidence(text="a", interrupted_tool=tool_call("t1", '{"x')),
        )
        second = build_continuation_payload(
            first, RecoveryAction.CONTINUE_TOOLS,
            AttemptEvidence(text="b", interrupted_tool=tool_call("t2", '{"y')),
        )
        history = second["conversationState"]["history"]
        assert len(history) == 6
        roles = ["userInputMessage" if "userInputMessage" in h else "assistantResponseMessage" for h in history]
        assert roles == ["userInputMessage", "assistantResponseMessage"] * 3
        moved = history[4]["userInputMessage"]
        assert moved["content"] == ""
        assert moved["userInputMessageContext"]["toolResults"][0]["toolUseId"] == "t1"
        assert "tools" not in moved["userInputMessageContext"]
        assert history[5]["assistantResponseMessage"]["toolUses"][0]["toolUseId"] == "t2"
        current = second["conversationState"]["currentMessage"]["userInputMessage"]
        assert current["userInputMessageContext"]["toolResults"][0]["toolUseId"] == "t2"
        assert current["modelId"] == "claude-haiku-5.5"

    def test_resend_with_text_shape(self):
        """
        What it does: RESEND_WITH_TEXT with partial text (rule 4, variant V4).
        Goal: Assistant message has the text and no toolUses; the current message is the
        continue prompt with tools and no toolResults.
        """
        result = build_continuation_payload(
            base_payload(), RecoveryAction.RESEND_WITH_TEXT, AttemptEvidence(text="The weather in"),
        )
        history = result["conversationState"]["history"]
        assert history[-1] == {"assistantResponseMessage": {"content": "The weather in"}}
        current = result["conversationState"]["currentMessage"]["userInputMessage"]
        assert current["content"] == CONTINUE_PROMPT
        assert "toolResults" not in current["userInputMessageContext"]
        assert "tools" in current["userInputMessageContext"]

    def test_payload_without_tools_has_no_context_key(self):
        """
        What it does: Base payload without tools, CONTINUE_PROMPT continuation.
        Goal: No empty userInputMessageContext is sent.
        """
        base = base_payload()
        del base["conversationState"]["currentMessage"]["userInputMessage"]["userInputMessageContext"]
        result = build_continuation_payload(base, RecoveryAction.RESEND_WITH_TEXT, AttemptEvidence(text="x"))
        assert "userInputMessageContext" not in result["conversationState"]["currentMessage"]["userInputMessage"]

    def test_payload_without_history_key(self):
        """
        What it does: Base payload with no history list.
        Goal: The history is created.
        """
        base = base_payload()
        del base["conversationState"]["history"]
        result = build_continuation_payload(
            base, RecoveryAction.CONTINUE_TOOLS, AttemptEvidence(interrupted_tool=tool_call("t", "{")),
        )
        assert len(result["conversationState"]["history"]) == 2

    def test_missing_current_message_raises(self):
        """
        What it does: Payload without currentMessage.
        Goal: ValueError (never a KeyError deep in the stack).
        """
        with pytest.raises(ValueError):
            build_continuation_payload({"conversationState": {}}, RecoveryAction.CONTINUE_TOOLS, AttemptEvidence())
        with pytest.raises(ValueError):
            build_continuation_payload({}, RecoveryAction.RESEND_WITH_TEXT, AttemptEvidence())

    @pytest.mark.parametrize("action", [RecoveryAction.FINISH, RecoveryAction.FAIL])
    def test_non_resend_actions_raise(self, action):
        """
        What it does: Asks for a payload for FINISH / FAIL.
        Goal: ValueError, those actions send nothing.
        """
        with pytest.raises(ValueError):
            build_continuation_payload(base_payload(), action, AttemptEvidence())

    def test_trim_guard_runs_when_enabled_and_over_limit(self):
        """
        What it does: AUTO_TRIM_PAYLOAD on, limit tiny.
        Goal: trim_payload_to_limit is called with the new payload and the limit.
        """
        with patch.object(stream_recovery, "AUTO_TRIM_PAYLOAD", True), \
                patch.object(stream_recovery, "KIRO_MAX_PAYLOAD_BYTES", 10), \
                patch.object(stream_recovery, "trim_payload_to_limit") as trim:
            result = build_continuation_payload(
                base_payload(), RecoveryAction.CONTINUE_TOOLS,
                AttemptEvidence(interrupted_tool=tool_call("t", "{")),
            )
        trim.assert_called_once()
        assert trim.call_args.args[0] is result
        assert trim.call_args.args[1] == 10

    def test_trim_guard_skipped_when_disabled_or_under_limit(self):
        """
        What it does: AUTO_TRIM_PAYLOAD off with a tiny limit, then on with a huge limit.
        Goal: trim_payload_to_limit is never called.
        """
        evidence = AttemptEvidence(interrupted_tool=tool_call("t", "{"))
        with patch.object(stream_recovery, "AUTO_TRIM_PAYLOAD", False), \
                patch.object(stream_recovery, "KIRO_MAX_PAYLOAD_BYTES", 10), \
                patch.object(stream_recovery, "trim_payload_to_limit") as trim:
            build_continuation_payload(base_payload(), RecoveryAction.CONTINUE_TOOLS, evidence)
        trim.assert_not_called()
        with patch.object(stream_recovery, "AUTO_TRIM_PAYLOAD", True), \
                patch.object(stream_recovery, "KIRO_MAX_PAYLOAD_BYTES", 10_000_000), \
                patch.object(stream_recovery, "trim_payload_to_limit") as trim:
            build_continuation_payload(base_payload(), RecoveryAction.CONTINUE_TOOLS, evidence)
        trim.assert_not_called()


# =============================================================================
# Exception and settings
# =============================================================================

class TestUpstreamStreamInterruptedAndSettings:
    """UpstreamStreamInterrupted and RecoverySettings."""

    def test_exception_carries_reason_and_default_message(self):
        """
        What it does: Raises with only a reason.
        Goal: str(exception) is the Kiro message and reason is kept.
        """
        error = UpstreamStreamInterrupted("reasoning_only")
        assert error.reason == "reasoning_only"
        assert str(error) == CONNECTION_INTERRUPTED_MESSAGE
        assert isinstance(error, Exception)

    def test_exception_custom_message(self):
        """
        What it does: Raises with a custom message.
        Goal: The message is used verbatim.
        """
        assert str(UpstreamStreamInterrupted("x", "custom")) == "custom"

    def test_exception_cause_is_preserved(self):
        """
        What it does: raise ... from drop.
        Goal: __cause__ points at the original drop.
        """
        drop = RuntimeError("drop")
        try:
            raise UpstreamStreamInterrupted("streak_exhausted") from drop
        except UpstreamStreamInterrupted as error:
            assert error.__cause__ is drop

    def test_settings_defaults(self):
        """
        What it does: Builds RecoverySettings with the required fields only.
        Goal: max_attempts defaults to 3.
        """
        settings = RecoverySettings(payload={}, make_request=AsyncMock())
        assert settings.max_attempts == 3
