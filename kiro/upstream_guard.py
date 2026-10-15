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
Detection of empty upstream responses (pure logic, no network).

Kiro occasionally answers HTTP 200 with a stream that only carries a single
contextUsageEvent frame (143-148 bytes): no assistantResponseEvent, no
reasoningContentEvent, no toolUseEvent and no meteringEvent. Every genuine
response, including a genuinely empty model answer, ends with a meteringEvent frame.

Detection works on the AWS event-stream frame headers (the ":event-type" header
value), not on JSON payloads and not on parsed events:
- the frame header layout is part of the AWS protocol and independent of payload key order;
- header names can be located exactly across chunk boundaries (see UpstreamFrameScanner).

A second failure class is the "incomplete" response: Kiro ends the stream early while the
model is still reasoning. Only reasoningContentEvent frames (and maybe a contextUsageEvent)
arrive, with no meteringEvent and no text or tool call. StreamOutcome classifies that after
the stream ended (see StreamVerdict).

This module contains:
- EmptyUpstreamResponseError: raised for an empty or incomplete upstream response
- UpstreamFrameScanner: incremental, chunk-boundary-safe frame header scanner
- StreamVerdict / StreamOutcome: end-of-stream classification (complete / incomplete)
- GuardedResponse: response wrapper that replays the bytes consumed while peeking
"""

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, AsyncIterator, Dict, List, Optional

# Frame types that do not prove the model produced (or even tried to produce) anything.
NEUTRAL_FRAME_TYPES = frozenset({"contextUsageEvent"})

# Real empty upstream responses are 143-148 bytes. A stream larger than this is never
# classified as empty, which also bounds how many bytes the guard buffers.
EMPTY_SIGNATURE_MAX_BYTES = 4096

# AWS event-stream header: name length byte (0x0b) + ":event-type" is followed by
# value type 7 (string). The marker below is ":event-type" + the value type byte.
HEADER_MARKER = b":event-type\x07"

# Marker, then a 2-byte big-endian value length, then the value (the event type name).
HEADER_PREFIX_LEN = len(HEADER_MARKER) + 2

# Same layout for AWS exception frames (":exception-type" instead of ":event-type").
# SPECULATION: derived from the AWS event-stream specification, not from a real sample;
# the header layout of an exception frame has never been observed on this gateway.
EXCEPTION_HEADER_MARKER = b":exception-type\x07"

# Counter key prefix for exception frames, e.g. "exception:ThrottlingException"
EXCEPTION_KEY_PREFIX = "exception:"

# Frame type that the upstream sends once at the end of every complete response
METERING_FRAME_TYPE = "meteringEvent"

# Longest event type name accepted. Real names are at most 22 characters.
MAX_EVENT_TYPE_LEN = 64

# (marker, counter key prefix) pairs recognised by the scanner
_SCAN_MARKERS = (
    (HEADER_MARKER, ""),
    (EXCEPTION_HEADER_MARKER, EXCEPTION_KEY_PREFIX),
)

# Bytes kept between chunks. Any header that is not completely inside the scanned data
# must start in the last (longest marker + 2 + MAX_EVENT_TYPE_LEN - 1) bytes.
CARRY_BYTES = max(len(marker) for marker, _ in _SCAN_MARKERS) + 2 + MAX_EVENT_TYPE_LEN - 1

# Values of EmptyUpstreamResponseError.kind
KIND_EMPTY = "empty"
KIND_INCOMPLETE = "incomplete"


class EmptyUpstreamResponseError(Exception):
    """
    Raised when the upstream answered HTTP 200 with an empty or incomplete response.

    - kind "empty": a stream that ended normally and contained nothing but
      contextUsageEvent frames (or no bytes at all);
    - kind "incomplete": a stream that ended normally with reasoning (or other non-neutral)
      frames only: no meteringEvent, no text and no tool call.

    Attributes:
        bytes_received: Number of upstream bytes consumed before the stream ended
        frame_counts: Frame type name -> number of frames seen
        kind: KIND_EMPTY or KIND_INCOMPLETE
        produced_text: Whether any non-empty text was produced
        produced_tool_use: Whether any tool_use was produced
    """

    def __init__(
        self,
        bytes_received: int,
        frame_counts: Dict[str, int],
        kind: str = KIND_EMPTY,
        produced_text: bool = False,
        produced_tool_use: bool = False,
    ) -> None:
        self.bytes_received = bytes_received
        self.frame_counts = dict(frame_counts)
        self.kind = kind
        self.produced_text = produced_text
        self.produced_tool_use = produced_tool_use
        label = "an incomplete" if kind == KIND_INCOMPLETE else "an empty"
        super().__init__(
            f"Upstream returned {label} response "
            f"(bytes={bytes_received}, frames={self.frame_counts})"
        )


class UpstreamFrameScanner:
    """
    Incrementally counts AWS event-stream frames by their ":event-type" header.

    Frame headers can be split across arbitrary chunk boundaries. The scanner keeps the
    last CARRY_BYTES bytes between calls and never counts a header twice, so the counters
    are exact regardless of how the stream was chunked.

    Attributes:
        counts: Frame type name -> number of frames seen
        bytes_received: Total number of bytes fed
    """

    def __init__(self) -> None:
        self.counts: Counter = Counter()
        self.bytes_received: int = 0
        self._tail: bytes = b""

    def feed(self, chunk: bytes) -> None:
        """
        Scan one more chunk of the upstream stream.

        Args:
            chunk: Raw bytes exactly as received from the network
        """
        if not chunk:
            return
        self.bytes_received += len(chunk)

        data = self._tail + chunk
        tail_len = len(self._tail)
        for marker, key_prefix in _SCAN_MARKERS:
            self._scan_marker(data, tail_len, marker, key_prefix)

        self._tail = data[-CARRY_BYTES:]

    def _scan_marker(self, data: bytes, tail_len: int, marker: bytes, key_prefix: str) -> None:
        """
        Count the headers that start with one marker inside the scanned data.

        Args:
            data: Carried bytes followed by the new chunk
            tail_len: Number of carried bytes at the start of data
            marker: Header name + value type bytes to look for
            key_prefix: Prefix of the counter key (empty for event frames)
        """
        prefix_len = len(marker) + 2
        pos = 0
        while True:
            idx = data.find(marker, pos)
            if idx < 0:
                return
            pos = idx + 1

            prefix_end = idx + prefix_len
            if prefix_end > len(data):
                # Length field not complete yet: wait for the next chunk
                return

            name_len = int.from_bytes(data[idx + len(marker):prefix_end], "big")
            if name_len == 0 or name_len > MAX_EVENT_TYPE_LEN:
                # Not a real header (noise that happens to contain the marker)
                continue

            name_end = prefix_end + name_len
            if name_end > len(data):
                # Name truncated at the end of the data: wait for the next chunk
                return

            name = data[prefix_end:name_end]
            if not name.isalpha():
                continue

            if name_end <= tail_len:
                # Header lies completely inside the carried bytes: counted by an earlier call
                continue

            self.counts[key_prefix + name.decode("ascii")] += 1

    @property
    def saw_metering(self) -> bool:
        """True once at least one meteringEvent frame was recognised."""
        return self.counts.get(METERING_FRAME_TYPE, 0) > 0

    @property
    def frames_seen(self) -> int:
        """Total number of frames recognised so far."""
        return sum(self.counts.values())

    @property
    def saw_non_neutral_frame(self) -> bool:
        """True once any frame other than a NEUTRAL_FRAME_TYPES frame was recognised."""
        return any(name not in NEUTRAL_FRAME_TYPES for name in self.counts)

    def looks_empty(self) -> bool:
        """
        Decide whether a stream that ended normally was an empty upstream response.

        Only call this after the stream has ended. A stream is empty when it carried no
        bytes, or when every recognised frame is a neutral frame (contextUsageEvent) and
        the stream is small. Any other frame type (content, reasoning, tool use,
        meteringEvent, or an unknown type) means the response is not empty. Bytes without
        any recognisable frame header are never classified as empty (unknown framing).

        Returns:
            True if the stream is an empty upstream response
        """
        if self.bytes_received == 0:
            return True
        if self.saw_non_neutral_frame:
            return False
        if self.frames_seen == 0:
            return False
        return self.bytes_received <= EMPTY_SIGNATURE_MAX_BYTES


class StreamVerdict(str, Enum):
    """
    End-of-stream classification of one upstream response (see StreamOutcome.verdict).

    Attributes:
        NOT_FINISHED: The stream did not end normally (exception, cancel, consumer closed)
        COMPLETE: At least one meteringEvent was seen
        MISSING_METERING_WITH_OUTPUT: No meteringEvent, but text or tool_use was produced
        NO_OUTPUT_FRAMES: No meteringEvent, no output and no non-neutral frame at all
        INCOMPLETE: No meteringEvent, no text, no tool_use, but non-neutral frames were seen
    """
    NOT_FINISHED = "not_finished"
    COMPLETE = "complete"
    MISSING_METERING_WITH_OUTPUT = "missing_metering_with_output"
    NO_OUTPUT_FRAMES = "no_output_frames"
    INCOMPLETE = "incomplete"


@dataclass
class StreamOutcome:
    """
    Side channel filled by parse_kiro_stream and read by the stream consumers at stream end.

    Holds the ONE UpstreamFrameScanner of the stream (so nothing is scanned twice) and
    whether text or tool_use was produced. Thinking/reasoning events never count as output.

    Attributes:
        scanner: Frame scanner fed with every upstream byte of the stream
        produced_text: A non-empty content event was produced
        produced_tool_use: A tool_use event was produced
        finished: parse_kiro_stream reached the normal end of the stream
    """
    scanner: UpstreamFrameScanner = field(default_factory=UpstreamFrameScanner)
    produced_text: bool = False
    produced_tool_use: bool = False
    finished: bool = False

    def record_event(self, event: Any) -> None:
        """
        Record whether a parsed event counts as real output.

        Args:
            event: KiroEvent (only its type and content are read)
        """
        if event.type == "content" and event.content:
            self.produced_text = True
        elif event.type == "tool_use" and event.tool_use:
            self.produced_tool_use = True

    def verdict(self) -> StreamVerdict:
        """
        Classify the stream. The first matching rule wins.

        Returns:
            The StreamVerdict of the (finished) stream
        """
        if not self.finished:
            return StreamVerdict.NOT_FINISHED
        if self.scanner.saw_metering:
            return StreamVerdict.COMPLETE
        if self.produced_text or self.produced_tool_use:
            return StreamVerdict.MISSING_METERING_WITH_OUTPUT
        if not self.scanner.saw_non_neutral_frame:
            return StreamVerdict.NO_OUTPUT_FRAMES
        return StreamVerdict.INCOMPLETE

    def to_error(self) -> EmptyUpstreamResponseError:
        """
        Build the exception that reports this stream as an incomplete response.

        Returns:
            EmptyUpstreamResponseError with kind "incomplete" and the stream facts
        """
        return EmptyUpstreamResponseError(
            self.scanner.bytes_received,
            dict(self.scanner.counts),
            kind=KIND_INCOMPLETE,
            produced_text=self.produced_text,
            produced_tool_use=self.produced_tool_use,
        )


class GuardedResponse:
    """
    Response wrapper that replays the chunks consumed while peeking at the stream.

    httpx marks a streamed response as consumed as soon as aiter_bytes() starts, so the
    peeked chunks cannot be pushed back. This wrapper holds the already created byte
    iterator and the chunks read from it: aiter_bytes() yields the buffered chunks (with
    their original boundaries) first and then continues with the same iterator.
    Every other attribute (status_code, headers, aread, ...) is delegated to the response.

    Attributes:
        first_chunk_ms: Time between the start of peeking and the first upstream chunk
        peeked_bytes: Number of bytes consumed while peeking
    """

    def __init__(
        self,
        response: Any,
        iterator: AsyncIterator[bytes],
        buffered: List[bytes],
        first_chunk_ms: Optional[float] = None,
    ) -> None:
        self._response = response
        self._iterator = iterator
        self._buffered = list(buffered)
        self.first_chunk_ms = first_chunk_ms
        self.peeked_bytes = sum(len(chunk) for chunk in self._buffered)

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the wrapped response."""
        return getattr(self._response, name)

    async def aiter_bytes(self, chunk_size: Optional[int] = None) -> AsyncGenerator[bytes, None]:
        """
        Yield the peeked chunks, then the rest of the upstream stream.

        Args:
            chunk_size: Ignored (accepted for httpx.Response compatibility)

        Yields:
            Raw bytes chunks in their original order and with their original boundaries
        """
        for chunk in self._buffered:
            yield chunk
        async for chunk in self._iterator:
            yield chunk

    async def aclose(self) -> None:
        """Close the byte iterator, then the wrapped response."""
        iterator_close = getattr(self._iterator, "aclose", None)
        try:
            if iterator_close is not None:
                await iterator_close()
        finally:
            await self._response.aclose()
