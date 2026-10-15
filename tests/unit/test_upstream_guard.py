# -*- coding: utf-8 -*-

"""
Unit tests for kiro/upstream_guard.py.

Covers UpstreamFrameScanner (frame header scanning across arbitrary chunk boundaries),
the empty-response decision, GuardedResponse (replay of peeked bytes) and
EmptyUpstreamResponseError. All bytes are real AWS event-stream frames built by the
aws_frames fixture; nothing touches the network.
"""

from typing import AsyncGenerator, List
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from kiro.upstream_guard import (
    CARRY_BYTES,
    EMPTY_SIGNATURE_MAX_BYTES,
    EXCEPTION_HEADER_MARKER,
    HEADER_MARKER,
    KIND_EMPTY,
    KIND_INCOMPLETE,
    EmptyUpstreamResponseError,
    GuardedResponse,
    StreamOutcome,
    StreamVerdict,
    UpstreamFrameScanner,
)
from kiro.streaming_core import KiroEvent


def scan(chunks: List[bytes]) -> UpstreamFrameScanner:
    """Feeds all chunks into a fresh scanner and returns it."""
    scanner = UpstreamFrameScanner()
    for chunk in chunks:
        scanner.feed(chunk)
    return scanner


# =============================================================================
# Empty-response decision
# =============================================================================

class TestLooksEmptyDecision:
    """Decision table for UpstreamFrameScanner.looks_empty()."""

    def test_only_context_usage_frame_is_empty(self, aws_frames):
        """
        What it does: Feeds the real empty-response shape (one contextUsageEvent frame).
        Purpose: This is the exact upstream failure the guard exists for.
        """
        frame = aws_frames.context_usage(60.73619842529297)

        scanner = scan([frame])

        print(f"bytes={scanner.bytes_received}, counts={dict(scanner.counts)}")
        assert len(frame) == 147
        assert scanner.looks_empty() is True
        assert dict(scanner.counts) == {"contextUsageEvent": 1}
        assert scanner.bytes_received == 147

    def test_context_usage_plus_metering_without_output_is_not_empty(self, aws_frames):
        """
        What it does: contextUsageEvent + meteringEvent, no output frames.
        Purpose: The model really answered with nothing: pass through, never an error.
        """
        data = aws_frames.context_usage(1.0) + aws_frames.metering(0.5)

        scanner = scan([data])

        assert scanner.looks_empty() is False
        assert dict(scanner.counts) == {"contextUsageEvent": 1, "meteringEvent": 1}

    def test_metering_alone_is_not_empty(self, aws_frames):
        """
        What it does: Only a meteringEvent frame.
        Purpose: Metering is proof that the upstream processed the request.
        """
        assert scan([aws_frames.metering(1.5)]).looks_empty() is False

    def test_reasoning_without_metering_is_not_empty(self, aws_frames):
        """
        What it does: reasoning frame + contextUsageEvent, no metering (e.g. truncated stream).
        Purpose: Output exists, so it must be delivered, not reported as empty.
        """
        data = aws_frames.reasoning("thinking...") + aws_frames.context_usage(2.0)

        assert scan([data]).looks_empty() is False

    def test_content_frame_is_not_empty(self, aws_frames):
        """
        What it does: assistantResponseEvent + contextUsageEvent.
        Purpose: Regular text output.
        """
        data = aws_frames.content("Hello") + aws_frames.context_usage(2.0)

        assert scan([data]).looks_empty() is False

    def test_tool_use_frame_is_not_empty(self, aws_frames):
        """
        What it does: Only toolUseEvent frames.
        Purpose: A pure tool call response is output.
        """
        data = aws_frames.tool_start("get_weather", "tool_1") + aws_frames.context_usage(2.0)

        assert scan([data]).looks_empty() is False

    def test_unknown_frame_type_is_not_empty(self, aws_frames):
        """
        What it does: A frame type the gateway does not know.
        Purpose: Conservative: only the known-neutral frame type may mean "empty".
        """
        data = aws_frames.build("fooBarEvent", {}) + aws_frames.context_usage(2.0)

        scanner = scan([data])

        assert scanner.looks_empty() is False
        assert scanner.counts["fooBarEvent"] == 1

    def test_zero_bytes_is_empty(self):
        """
        What it does: Nothing fed at all.
        Purpose: A 200 with an empty body has neither output nor metering.
        """
        assert UpstreamFrameScanner().looks_empty() is True

    def test_empty_chunks_do_not_change_decision(self, aws_frames):
        """
        What it does: Interleaves empty chunks with the empty-response frame.
        Purpose: Empty chunks carry no information.
        """
        frame = aws_frames.context_usage(1.0)

        scanner = scan([b"", frame, b"", b""])

        assert scanner.looks_empty() is True
        assert scanner.bytes_received == len(frame)

    def test_bytes_without_any_frame_header_are_not_empty(self):
        """
        What it does: Bare JSON without AWS frame headers.
        Purpose: Unknown framing must never be classified as empty (guard skipped).
        """
        scanner = scan([b'{"content":"x"}'])

        assert scanner.looks_empty() is False
        assert scanner.frames_seen == 0

    def test_large_neutral_only_stream_is_not_empty(self, aws_frames):
        """
        What it does: Many contextUsageEvent frames adding up to more than the size cap.
        Purpose: Real empty responses are 143-148 bytes; the cap prevents misclassification.
        """
        frame = aws_frames.context_usage(1.0)
        count = EMPTY_SIGNATURE_MAX_BYTES // len(frame) + 2

        scanner = scan([frame * count])

        assert scanner.bytes_received > EMPTY_SIGNATURE_MAX_BYTES
        assert scanner.looks_empty() is False

    def test_several_context_usage_frames_under_cap_still_empty(self, aws_frames):
        """
        What it does: Two contextUsageEvent frames (still small).
        Purpose: Only the frame type matters, not the exact frame count.
        """
        scanner = scan([aws_frames.context_usage(1.0) * 2])

        assert scanner.looks_empty() is True
        assert scanner.counts["contextUsageEvent"] == 2


# =============================================================================
# Chunk boundary handling
# =============================================================================

class TestChunkBoundaries:
    """Frame headers split across chunks must be counted exactly once."""

    def test_empty_response_frame_split_at_every_offset(self, aws_frames):
        """
        What it does: Splits the empty-response frame in two at EVERY byte offset.
        Purpose: The decision and the counters must not depend on where the network splits.
        """
        frame = aws_frames.context_usage(60.73619842529297)

        for cut in range(1, len(frame)):
            scanner = scan([frame[:cut], frame[cut:]])
            assert scanner.looks_empty() is True, f"cut={cut}"
            assert dict(scanner.counts) == {"contextUsageEvent": 1}, f"cut={cut}"
            assert scanner.bytes_received == len(frame)

    def test_metering_frame_split_at_every_offset(self, aws_frames):
        """
        What it does: Context usage frame followed by a metering frame split at every offset.
        Purpose: A metering frame must be recognised however it is split, otherwise a
        genuine (metered) empty answer would be turned into an error.
        """
        data = aws_frames.context_usage(1.0) + aws_frames.metering(4.93)

        for cut in range(1, len(data)):
            scanner = scan([data[:cut], data[cut:]])
            assert scanner.looks_empty() is False, f"cut={cut}"
            assert scanner.counts["meteringEvent"] == 1, f"cut={cut}"
            assert scanner.counts["contextUsageEvent"] == 1, f"cut={cut}"

    def test_one_byte_per_chunk(self, aws_frames):
        """
        What it does: Feeds a multi-frame stream one byte at a time.
        Purpose: Extreme fragmentation must give exactly the same counters.
        """
        data = (
            aws_frames.reasoning("a") + aws_frames.content("b")
            + aws_frames.context_usage(1.0) + aws_frames.metering(1.0)
        )

        scanner = scan([data[i:i + 1] for i in range(len(data))])

        assert dict(scanner.counts) == {
            "reasoningContentEvent": 1, "assistantResponseEvent": 1,
            "contextUsageEvent": 1, "meteringEvent": 1,
        }

    def test_event_type_name_truncated_at_chunk_end(self, aws_frames):
        """
        What it does: First chunk ends in the middle of the event type name ("meteri" | "ngEvent").
        Purpose: A truncated name must neither be counted nor lost.
        """
        frame = aws_frames.metering(1.0)
        name_pos = frame.index(b"meteringEvent")
        cut = name_pos + len(b"meteri")

        scanner = UpstreamFrameScanner()
        scanner.feed(frame[:cut])
        assert scanner.counts == {}
        scanner.feed(frame[cut:])

        assert dict(scanner.counts) == {"meteringEvent": 1}

    def test_marker_split_inside_marker_bytes(self, aws_frames):
        """
        What it does: Splits inside the ":event-type" marker itself.
        Purpose: The carried tail must keep partial markers.
        """
        frame = aws_frames.context_usage(1.0)
        marker_pos = frame.index(HEADER_MARKER)

        scanner = scan([frame[:marker_pos + 5], frame[marker_pos + 5:]])

        assert dict(scanner.counts) == {"contextUsageEvent": 1}

    def test_length_field_split(self, aws_frames):
        """
        What it does: Splits between the two bytes of the 2-byte value length.
        Purpose: The prefix is incomplete until both length bytes arrived.
        """
        frame = aws_frames.context_usage(1.0)
        cut = frame.index(HEADER_MARKER) + len(HEADER_MARKER) + 1

        scanner = scan([frame[:cut], frame[cut:]])

        assert dict(scanner.counts) == {"contextUsageEvent": 1}

    def test_header_not_double_counted_when_next_chunk_is_tiny(self, aws_frames):
        """
        What it does: Complete header inside the carried tail, followed by tiny chunks.
        Purpose: A header that lies inside CARRY_BYTES must be counted once, not on every call.
        """
        frame = aws_frames.context_usage(1.0)
        scanner = UpstreamFrameScanner()
        scanner.feed(frame)
        for _ in range(5):
            scanner.feed(b"\x00")

        assert dict(scanner.counts) == {"contextUsageEvent": 1}

    def test_multiple_frames_in_one_chunk(self, aws_frames):
        """
        What it does: Many frames inside one chunk.
        Purpose: All headers in the chunk are counted.
        """
        data = aws_frames.content("a") * 3 + aws_frames.context_usage(1.0)

        scanner = scan([data])

        assert dict(scanner.counts) == {"assistantResponseEvent": 3, "contextUsageEvent": 1}

    @settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(sizes=st.lists(st.integers(min_value=1, max_value=200), min_size=1, max_size=12))
    def test_any_chunking_gives_same_counts(self, aws_frames, sizes):
        """
        What it does: Property test: random chunk sizes over a realistic stream.
        Purpose: Counters never depend on the chunking.
        """
        data = (
            aws_frames.reasoning("thinking") * 3 + aws_frames.content("hello world") * 2
            + aws_frames.tool_start("t", "id1") + aws_frames.context_usage(3.5)
            + aws_frames.metering(1.25)
        )
        expected = dict(scan([data]).counts)

        scanner = scan(aws_frames.split(data, sizes))

        assert dict(scanner.counts) == expected
        assert scanner.bytes_received == len(data)


# =============================================================================
# Malformed input
# =============================================================================

class TestMalformedHeaders:
    """Noise that resembles a header must not crash or be miscounted."""

    def test_zero_length_name_is_ignored(self):
        """
        What it does: Marker followed by length 0.
        Purpose: Zero-length names are noise.
        """
        scanner = scan([HEADER_MARKER + b"\x00\x00" + b"contextUsageEvent"])

        assert scanner.counts == {}

    def test_oversized_length_is_ignored(self):
        """
        What it does: Marker followed by length 255 with fewer bytes available.
        Purpose: A lying length field must not stall the scanner or count anything.
        """
        scanner = scan([HEADER_MARKER + b"\x00\xff" + b"contextUsageEvent"])

        assert scanner.counts == {}

    def test_non_alphabetic_name_is_ignored(self):
        """
        What it does: Name containing digits/control bytes.
        Purpose: Real event types are ASCII letters only.
        """
        scanner = scan([HEADER_MARKER + b"\x00\x05ab1de", HEADER_MARKER + b"\x00\x03a\x00b"])

        assert scanner.counts == {}

    def test_marker_at_very_end_of_stream(self):
        """
        What it does: Stream ends right after the marker.
        Purpose: A truncated header at end of stream must not raise or count.
        """
        scanner = scan([b"\x00\x00" + HEADER_MARKER])

        assert scanner.counts == {}
        assert scanner.looks_empty() is False

    def test_noise_marker_does_not_hide_following_real_header(self, aws_frames):
        """
        What it does: A bogus marker (length 0) directly before a real frame.
        Purpose: Scanning must continue after noise.
        """
        scanner = scan([HEADER_MARKER + b"\x00\x00" + aws_frames.metering(1.0)])

        assert dict(scanner.counts) == {"meteringEvent": 1}

    def test_binary_garbage_does_not_crash(self):
        """
        What it does: Random-looking bytes without markers.
        Purpose: Robustness.
        """
        scanner = scan([bytes(range(256)) * 20])

        assert scanner.counts == {}
        assert scanner.looks_empty() is False

    def test_carry_bytes_covers_longest_incomplete_header(self):
        """
        What it does: Checks the CARRY_BYTES constant against the header layout.
        Purpose: An incomplete header start is always within the last CARRY_BYTES bytes.
        """
        assert CARRY_BYTES == max(len(HEADER_MARKER), len(EXCEPTION_HEADER_MARKER)) + 2 + 64 - 1


# =============================================================================
# GuardedResponse
# =============================================================================

def make_httpx_like_response(chunks: List[bytes]) -> MagicMock:
    """Response double whose aiter_bytes() can be called ONCE (like httpx)."""
    response = MagicMock(spec=httpx.Response)
    response.status_code = 200
    response.headers = {"content-type": "application/vnd.amazon.eventstream"}
    response.aclose = AsyncMock()
    response.aread = AsyncMock(return_value=b"body")
    calls = {"count": 0}

    async def gen() -> AsyncGenerator[bytes, None]:
        for chunk in chunks:
            yield chunk

    def aiter_bytes(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] > 1:
            raise httpx.StreamConsumed()
        return gen()

    response.aiter_bytes = aiter_bytes
    response.calls = calls
    return response


class TestGuardedResponse:
    """Replay of peeked bytes."""

    @pytest.mark.asyncio
    async def test_replays_buffered_then_rest_in_order_with_boundaries(self):
        """
        What it does: Peeks two chunks, leaves two unread.
        Purpose: Replay yields exactly the original chunk sequence.
        """
        chunks = [b"aa", b"bbb", b"c", b"dddd"]
        response = make_httpx_like_response(chunks)
        iterator = response.aiter_bytes()
        peeked = [await iterator.__anext__(), await iterator.__anext__()]

        guarded = GuardedResponse(response, iterator, peeked, first_chunk_ms=12.0)
        replayed = [chunk async for chunk in guarded.aiter_bytes()]

        assert replayed == chunks
        assert response.calls["count"] == 1
        assert guarded.peeked_bytes == 5
        assert guarded.first_chunk_ms == 12.0

    @pytest.mark.asyncio
    async def test_replay_after_stream_end_during_peek(self):
        """
        What it does: Whole stream consumed while peeking.
        Purpose: The replay must still deliver everything and then end normally.
        """
        chunks = [b"x", b"y"]
        response = make_httpx_like_response(chunks)
        iterator = response.aiter_bytes()
        peeked = [chunk async for chunk in iterator]

        guarded = GuardedResponse(response, iterator, peeked)

        assert [chunk async for chunk in guarded.aiter_bytes()] == chunks

    @pytest.mark.asyncio
    async def test_delegates_attributes(self):
        """
        What it does: Reads status_code, headers and calls aread through the wrapper.
        Purpose: Everything except aiter_bytes/aclose behaves like the original response.
        """
        response = make_httpx_like_response([b"x"])
        guarded = GuardedResponse(response, response.aiter_bytes(), [])

        assert guarded.status_code == 200
        assert guarded.headers["content-type"] == "application/vnd.amazon.eventstream"
        assert await guarded.aread() == b"body"

    @pytest.mark.asyncio
    async def test_aclose_closes_iterator_and_response(self):
        """
        What it does: Closes the wrapper.
        Purpose: Both the byte iterator and the real response must be closed.
        """
        response = make_httpx_like_response([b"x"])
        iterator = MagicMock()
        iterator.aclose = AsyncMock()
        guarded = GuardedResponse(response, iterator, [])

        await guarded.aclose()

        iterator.aclose.assert_awaited_once()
        response.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_aclose_closes_response_even_if_iterator_close_fails(self):
        """
        What it does: Iterator close raises.
        Purpose: The real response must still be closed (no leaked connection).
        """
        response = make_httpx_like_response([b"x"])
        iterator = MagicMock()
        iterator.aclose = AsyncMock(side_effect=RuntimeError("boom"))
        guarded = GuardedResponse(response, iterator, [])

        with pytest.raises(RuntimeError):
            await guarded.aclose()

        response.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_aclose_without_iterator_aclose(self):
        """
        What it does: Iterator object without aclose().
        Purpose: Tolerate plain async iterators.
        """
        response = make_httpx_like_response([b"x"])

        class PlainIterator:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

        guarded = GuardedResponse(response, PlainIterator(), [])
        await guarded.aclose()

        response.aclose.assert_awaited_once()


class TestEmptyUpstreamResponseError:
    """Error attributes and message."""

    def test_attributes_and_message(self):
        """
        What it does: Builds the exception.
        Purpose: Routes log bytes_received and frame_counts from it.
        """
        err = EmptyUpstreamResponseError(147, {"contextUsageEvent": 1})

        assert err.bytes_received == 147
        assert err.frame_counts == {"contextUsageEvent": 1}
        assert "147" in str(err)
        assert "contextUsageEvent" in str(err)

    def test_frame_counts_are_copied(self):
        """
        What it does: Mutates the dict after creating the error.
        Purpose: The error must hold its own copy.
        """
        counts = {"contextUsageEvent": 1}
        err = EmptyUpstreamResponseError(1, counts)
        counts["x"] = 1

        assert err.frame_counts == {"contextUsageEvent": 1}


# =============================================================================
# StreamOutcome: incomplete response criterion (A, J)
# =============================================================================

REASONING_SECRET = "SECRET_REASONING_FOR_VERDICT"


def outcome_for(chunks: List[bytes], text: bool = False, tool: bool = False,
                finished: bool = True) -> StreamOutcome:
    """Builds a StreamOutcome by feeding chunks into its scanner and recording output flags."""
    outcome = StreamOutcome()
    for chunk in chunks:
        outcome.scanner.feed(chunk)
    if text:
        outcome.record_event(KiroEvent(type="content", content="hello"))
    if tool:
        outcome.record_event(KiroEvent(type="tool_use", tool_use={"id": "t1"}))
    outcome.finished = finished
    return outcome


class TestStreamVerdict:
    """Decision table of StreamOutcome.verdict() (the single behavior contract)."""

    def test_reasoning_and_context_usage_without_metering_is_incomplete(self, aws_frames):
        """
        What it does: 18 reasoning frames + contextUsageEvent, no metering (17:08:59 shape).
        Purpose: This is the reported failure: must be INCOMPLETE.
        """
        data = b"".join(aws_frames.reasoning(f"r{i}") for i in range(18)) + aws_frames.context_usage(3.0)

        outcome = outcome_for([data])

        assert outcome.verdict() == StreamVerdict.INCOMPLETE

    def test_reasoning_with_metering_and_no_body_is_complete(self, aws_frames):
        """
        What it does: reasoning + contextUsageEvent + meteringEvent, no body.
        Purpose: Any response with a meteringEvent is complete and passed through.
        """
        data = aws_frames.reasoning("r") + aws_frames.context_usage(1.0) + aws_frames.metering(0.2)

        assert outcome_for([data]).verdict() == StreamVerdict.COMPLETE

    def test_reasoning_and_text_without_metering_is_missing_metering_with_output(self, aws_frames):
        """
        What it does: reasoning + text, no metering (possibly truncated text).
        Purpose: Behavior must NOT change: only a WARNING is logged by the consumer.
        """
        data = aws_frames.reasoning("r") + aws_frames.content("partial")

        outcome = outcome_for([data], text=True)

        assert outcome.verdict() == StreamVerdict.MISSING_METERING_WITH_OUTPUT

    def test_reasoning_and_tool_use_without_metering_is_missing_metering_with_output(self, aws_frames):
        """
        What it does: reasoning + tool_use, no metering.
        Purpose: A tool call is real output: pass through.
        """
        data = aws_frames.reasoning("r") + aws_frames.tool_start("Bash", "t1")

        assert outcome_for([data], tool=True).verdict() == StreamVerdict.MISSING_METERING_WITH_OUTPUT

    def test_only_reasoning_without_context_usage_is_incomplete(self, aws_frames):
        """
        What it does: 49 reasoning frames (about 6 KB), no contextUsageEvent, no metering
        (the 17:15:07 shape).
        Purpose: Must be INCOMPLETE even though the stream carries no other frame type.
        """
        data = b"".join(aws_frames.reasoning(f"thought number {i} " * 3) for i in range(49))

        outcome = outcome_for([data])

        print(f"bytes={len(data)}")
        assert len(data) > 4096
        assert outcome.verdict() == StreamVerdict.INCOMPLETE

    def test_metering_frame_split_at_every_offset_is_still_complete(self, aws_frames):
        """
        What it does: Splits the stream into two chunks at EVERY byte offset.
        Purpose: Metering detection is exact across chunk boundaries.
        """
        data = aws_frames.reasoning("r") + aws_frames.context_usage(1.0) + aws_frames.metering(0.2)
        reference = outcome_for([data])

        for cut in range(1, len(data)):
            outcome = outcome_for([data[:cut], data[cut:]])
            assert outcome.verdict() == StreamVerdict.COMPLETE, f"cut at {cut}"
            assert dict(outcome.scanner.counts) == dict(reference.scanner.counts), f"cut at {cut}"

    def test_byte_by_byte_feeding_is_complete(self, aws_frames):
        """
        What it does: Feeds the stream one byte per chunk.
        Purpose: The worst case chunking gives the same verdict and counts.
        """
        data = aws_frames.reasoning("r") + aws_frames.metering(0.2)

        outcome = outcome_for(aws_frames.split(data, [1]))

        assert outcome.verdict() == StreamVerdict.COMPLETE
        assert dict(outcome.scanner.counts) == {"reasoningContentEvent": 1, "meteringEvent": 1}

    @pytest.mark.parametrize("chunks_factory", [
        lambda f: [b""],
        lambda f: [f.context_usage(60.7)],
        lambda f: [b"\x00garbage without any frame header" * 4],
    ])
    def test_empty_context_usage_only_and_unrecognised_are_no_output_frames(self, aws_frames, chunks_factory):
        """
        What it does: 0 bytes / only contextUsageEvent / bytes without a frame header.
        Purpose: Not INCOMPLETE: these are handled (or deliberately released) by the existing guard.
        """
        outcome = outcome_for(chunks_factory(aws_frames))

        assert outcome.verdict() == StreamVerdict.NO_OUTPUT_FRAMES

    def test_unfinished_stream_has_no_verdict(self, aws_frames):
        """
        What it does: Reasoning-only bytes but finished=False (mid-stream exception / cancel).
        Purpose: Never classify a stream that did not end normally.
        """
        outcome = outcome_for([aws_frames.reasoning("r")], finished=False)

        assert outcome.verdict() == StreamVerdict.NOT_FINISHED

    def test_signature_only_frame_is_incomplete(self, aws_frames):
        """
        What it does: Only a reasoningContentEvent that carries a signature.
        Purpose: Non-neutral frame, no output, no metering: INCOMPLETE.
        """
        assert outcome_for([aws_frames.signature("QUJD" * 10)]).verdict() == StreamVerdict.INCOMPLETE

    def test_metering_before_reasoning_is_complete(self, aws_frames):
        """
        What it does: meteringEvent first, then reasoning.
        Purpose: Order does not matter, only presence.
        """
        data = aws_frames.metering(0.1) + aws_frames.reasoning("r")

        assert outcome_for([data]).verdict() == StreamVerdict.COMPLETE

    def test_metering_alone_and_with_context_usage_are_complete(self, aws_frames):
        """
        What it does: meteringEvent alone / contextUsageEvent + meteringEvent.
        Purpose: Genuine empty answers are complete.
        """
        assert outcome_for([aws_frames.metering(0.1)]).verdict() == StreamVerdict.COMPLETE
        both = aws_frames.context_usage(1.0) + aws_frames.metering(0.1)
        assert outcome_for([both]).verdict() == StreamVerdict.COMPLETE

    def test_unknown_frame_type_without_output_is_incomplete(self, aws_frames):
        """
        What it does: A frame type the gateway does not know, no metering, no output.
        Purpose: Non-neutral frame without output or metering: INCOMPLETE.
        """
        assert outcome_for([aws_frames.build("mysteryEvent", {"x": 1})]).verdict() == StreamVerdict.INCOMPLETE

    @settings(max_examples=60, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(sizes=st.lists(st.integers(min_value=1, max_value=97), min_size=1, max_size=6))
    def test_verdict_is_independent_of_chunking(self, aws_frames, sizes):
        """
        What it does: Random chunk sizes over an incomplete and a complete stream.
        Purpose: The verdict never depends on how the network split the bytes.
        """
        incomplete = aws_frames.reasoning("abc") * 5 + aws_frames.context_usage(2.0)
        complete = incomplete + aws_frames.metering(0.3)

        assert outcome_for(aws_frames.split(incomplete, sizes)).verdict() == StreamVerdict.INCOMPLETE
        assert outcome_for(aws_frames.split(complete, sizes)).verdict() == StreamVerdict.COMPLETE


class TestStreamOutcomeRecording:
    """record_event(): what counts as output."""

    def test_empty_content_is_not_text(self):
        """
        What it does: A content event with an empty string.
        Purpose: produced_text only turns true for non-empty content.
        """
        outcome = StreamOutcome()
        outcome.record_event(KiroEvent(type="content", content=""))
        assert outcome.produced_text is False
        outcome.record_event(KiroEvent(type="content", content=" "))
        assert outcome.produced_text is True

    def test_thinking_never_counts_as_output(self):
        """
        What it does: Native and tag thinking events.
        Purpose: Reasoning is not output, whatever FAKE_REASONING_HANDLING is.
        """
        outcome = StreamOutcome()
        outcome.record_event(KiroEvent(type="thinking", thinking_content="x", is_native_reasoning=True))
        outcome.record_event(KiroEvent(type="thinking", thinking_content="y"))
        assert outcome.produced_text is False
        assert outcome.produced_tool_use is False

    def test_tool_use_counts(self):
        """
        What it does: A tool_use event.
        Purpose: Tool calls are real output.
        """
        outcome = StreamOutcome()
        outcome.record_event(KiroEvent(type="tool_use", tool_use={"id": "x"}))
        assert outcome.produced_tool_use is True

    def test_to_error_carries_stream_facts(self, aws_frames):
        """
        What it does: Builds the exception from an incomplete outcome.
        Purpose: kind, byte count, frame counts and output flags reach the route logs.
        """
        data = aws_frames.reasoning("r") + aws_frames.context_usage(1.0)
        outcome = outcome_for([data])

        err = outcome.to_error()

        assert err.kind == KIND_INCOMPLETE
        assert err.bytes_received == len(data)
        assert err.frame_counts == {"reasoningContentEvent": 1, "contextUsageEvent": 1}
        assert err.produced_text is False and err.produced_tool_use is False
        assert "incomplete" in str(err)

    def test_default_error_kind_is_empty(self):
        """
        What it does: Old style construction EmptyUpstreamResponseError(0, {}).
        Purpose: Existing callers keep working and stay kind "empty".
        """
        err = EmptyUpstreamResponseError(0, {})
        assert err.kind == KIND_EMPTY
        assert err.produced_text is False and err.produced_tool_use is False
        assert "empty" in str(err)


# =============================================================================
# AWS exception frames (I)
# =============================================================================

class TestExceptionFrames:
    """:exception-type headers are recognised (SPECULATION: layout derived from the AWS spec)."""

    def test_exception_frame_is_counted_and_non_neutral(self, aws_frames):
        """
        What it does: One ThrottlingException frame.
        Purpose: Counted as "exception:<Name>" and treated as a non-neutral frame.
        """
        scanner = scan([aws_frames.exception("ThrottlingException", {"message": "slow down"})])

        assert dict(scanner.counts) == {"exception:ThrottlingException": 1}
        assert scanner.saw_non_neutral_frame is True

    def test_exception_plus_context_usage_is_not_empty(self, aws_frames):
        """
        What it does: Exception frame + contextUsageEvent.
        Purpose: No longer mistaken for an empty response.
        """
        data = aws_frames.exception("ThrottlingException", {}) + aws_frames.context_usage(1.0)

        assert scan([data]).looks_empty() is False

    def test_exception_without_metering_or_output_is_incomplete(self, aws_frames):
        """
        What it does: Exception + contextUsageEvent, no metering, no output.
        Purpose: Falls into INCOMPLETE with the exception name visible in the counts.
        """
        data = aws_frames.exception("InternalServerException", {}) + aws_frames.context_usage(1.0)

        outcome = outcome_for([data])

        assert outcome.verdict() == StreamVerdict.INCOMPLETE
        assert outcome.scanner.counts["exception:InternalServerException"] == 1

    def test_exception_header_split_at_every_offset(self, aws_frames):
        """
        What it does: Splits an exception frame into two chunks at every offset.
        Purpose: Exact counting across chunk boundaries: never lost, never doubled.
        """
        data = aws_frames.exception("ThrottlingException", {"message": "x"}) + aws_frames.metering(0.1)
        for cut in range(1, len(data)):
            scanner = scan([data[:cut], data[cut:]])
            assert dict(scanner.counts) == {"exception:ThrottlingException": 1, "meteringEvent": 1}, f"cut {cut}"

    def test_byte_by_byte_exception_frame(self, aws_frames):
        """
        What it does: One byte per chunk.
        Purpose: Worst case chunking still counts exactly once.
        """
        data = aws_frames.exception("ThrottlingException", {})

        assert dict(scan(aws_frames.split(data, [1])).counts) == {"exception:ThrottlingException": 1}

    @pytest.mark.parametrize("noise", [
        EXCEPTION_HEADER_MARKER + b"\x00\x00abc",
        EXCEPTION_HEADER_MARKER + b"\x00\x41" + b"x" * 65,
        EXCEPTION_HEADER_MARKER + b"\x00\x03a1b",
    ])
    def test_noise_with_the_marker_is_not_counted(self, noise):
        """
        What it does: Marker followed by a zero length, a too long length, or a non-alphabetic name.
        Purpose: Payload noise must not create phantom frames.
        """
        assert scan([b"junk" + noise + b"junk"]).counts == {}

    def test_saw_metering_property(self, aws_frames):
        """
        What it does: Scanner with and without a meteringEvent.
        Purpose: saw_metering is the read-only signal the formatters use.
        """
        assert scan([aws_frames.reasoning("r")]).saw_metering is False
        assert scan([aws_frames.reasoning("r"), aws_frames.metering(1.0)]).saw_metering is True
