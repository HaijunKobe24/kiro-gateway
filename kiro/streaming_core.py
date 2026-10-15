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
Core streaming logic for parsing Kiro API responses.

This module contains shared logic used by both OpenAI and Anthropic streaming:
- KiroEvent dataclass for unified events
- Kiro SSE stream parsing
- Full response collection
- First token timeout handling
- Empty upstream response detection (converted to EmptyUpstreamResponseError)
- Incomplete upstream response detection (reasoning only, no meteringEvent, no text/tool_use)
- Native reasoning (Kiro reasoningContentEvent frames) converted to thinking events

The core layer provides a unified interface that API-specific formatters use
to convert Kiro events to their respective SSE formats.
"""

import asyncio
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncGenerator, Callable, Awaitable, Dict, List, Optional, Tuple, Union

import httpx
from loguru import logger

from kiro.parsers import AwsEventStreamParser, parse_bracket_tool_calls, deduplicate_tool_calls
from kiro.config import (
    FIRST_TOKEN_TIMEOUT,
    FIRST_TOKEN_MAX_RETRIES,
    FAKE_REASONING_ENABLED,
    FAKE_REASONING_HANDLING,
    EMPTY_RESPONSE_AS_ERROR,
    INCOMPLETE_RESPONSE_AS_ERROR,
)
from kiro.thinking_parser import ThinkingParser, format_native_reasoning
from kiro.upstream_guard import (
    EMPTY_SIGNATURE_MAX_BYTES,
    EmptyUpstreamResponseError,
    GuardedResponse,
    StreamOutcome,
    StreamVerdict,
    UpstreamFrameScanner,
)

if TYPE_CHECKING:
    from kiro.cache import ModelInfoCache

# Import debug_logger for logging
try:
    from kiro.debug_logger import debug_logger
except ImportError:
    debug_logger = None


# ==================================================================================================
# Data Classes
# ==================================================================================================

@dataclass
class KiroEvent:
    """
    Unified event from Kiro API stream.
    
    This format is API-agnostic and can be converted to both OpenAI and Anthropic formats.
    
    Attributes:
        type: Event type (content, thinking, tool_use, usage, context_usage, error)
        content: Text content (for content events)
        thinking_content: Thinking/reasoning content (for thinking events)
        tool_use: Tool use data (for tool_use events)
        usage: Usage/metering data (for usage events)
        context_usage_percentage: Context usage percentage (for context_usage events)
        is_first_thinking_chunk: Whether this is the first thinking chunk
        is_last_thinking_chunk: Whether this is the last thinking chunk
        is_native_reasoning: True if the thinking text came from a native Kiro
            reasoningContentEvent frame (as opposed to <thinking> tag parsing)
    """
    type: str
    content: Optional[str] = None
    thinking_content: Optional[str] = None
    tool_use: Optional[Dict[str, Any]] = None
    usage: Optional[Dict[str, Any]] = None
    context_usage_percentage: Optional[float] = None
    is_first_thinking_chunk: bool = False
    is_last_thinking_chunk: bool = False
    is_native_reasoning: bool = False


@dataclass
class StreamResult:
    """
    Result of collecting a complete stream response.
    
    Attributes:
        content: Full text content
        thinking_content: Full thinking/reasoning content
        tool_calls: List of tool calls
        usage: Usage information
        context_usage_percentage: Context usage percentage from Kiro API
    """
    content: str = ""
    thinking_content: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    usage: Optional[Dict[str, Any]] = None
    context_usage_percentage: Optional[float] = None


@dataclass
class NativeReasoningStats:
    """
    Counters for native reasoning frames seen in one stream (never holds reasoning text).
    
    Attributes:
        frames: Number of native reasoning text frames received
        chars: Total number of characters in those frames
        signature_frames: Number of native reasoning signature frames received
        discarded_frames: Number of text frames dropped by FAKE_REASONING_HANDLING=remove
    """
    frames: int = 0
    chars: int = 0
    signature_frames: int = 0
    discarded_frames: int = 0


class FirstTokenTimeoutError(Exception):
    """Exception raised when first token timeout occurs."""
    pass


def count_frame_event_types(chunk: bytes, counts: Counter) -> None:
    """
    Count AWS event stream frame types found in a single raw chunk (diagnostics only).

    The chunk is scanned on its own, so a frame header split across two chunks is not
    counted here. parse_kiro_stream keeps one UpstreamFrameScanner for the whole stream
    instead, which is exact across chunk boundaries.

    Args:
        chunk: Raw bytes chunk from the Kiro stream
        counts: Counter updated in place with event type names (e.g. "assistantResponseEvent")
    """
    scanner = UpstreamFrameScanner()
    scanner.feed(chunk)
    counts.update(scanner.counts)


# ==================================================================================================
# Kiro Stream Parsing
# ==================================================================================================

async def parse_kiro_stream(
    response: httpx.Response,
    first_token_timeout: float = FIRST_TOKEN_TIMEOUT,
    enable_thinking_parser: bool = True,
    outcome: Optional[StreamOutcome] = None,
) -> AsyncGenerator[KiroEvent, None]:
    """
    Parses Kiro SSE stream and yields unified events.
    
    This is the core parsing function that converts Kiro's AWS SSE format
    into unified KiroEvent objects that can be formatted for any API.
    
    Native reasoning frames (reasoningContentEvent) are converted to
    KiroEvent(type="thinking", is_native_reasoning=True) so they flow through the same
    downstream channel as <thinking>-tag reasoning. They are emitted incrementally,
    independent of FAKE_REASONING / enable_thinking_parser, and honor
    FAKE_REASONING_HANDLING ("remove" discards them).
    
    Args:
        response: HTTP response with data stream
        first_token_timeout: First token wait timeout (seconds)
        enable_thinking_parser: Whether to enable thinking block parsing
        outcome: Optional side channel. Its scanner is fed with every upstream byte, text
            and tool_use output is recorded, and finished is set to True only when the
            stream ended normally (never on an exception, cancellation or early close).
            Consumers read outcome.verdict() after their loop (see check_stream_outcome).
    
    Yields:
        KiroEvent objects representing stream events
    
    Raises:
        FirstTokenTimeoutError: If first token not received within timeout
    """
    parser = AwsEventStreamParser()
    first_token_received = False
    reasoning_stats = NativeReasoningStats()
    frame_scanner = outcome.scanner if outcome is not None else UpstreamFrameScanner()
    
    # Initialize thinking parser if fake reasoning is enabled
    thinking_parser: Optional[ThinkingParser] = None
    if FAKE_REASONING_ENABLED and enable_thinking_parser:
        thinking_parser = ThinkingParser(handling_mode=FAKE_REASONING_HANDLING)
        logger.debug(f"Thinking parser initialized with mode: {FAKE_REASONING_HANDLING}")
    
    try:
        # Create iterator for reading bytes
        byte_iterator = response.aiter_bytes()
        
        # Wait for first chunk with timeout
        try:
            _t0 = asyncio.get_event_loop().time()
            logger.debug(f"Waiting for first token (timeout={first_token_timeout}s)...")
            first_byte_chunk = await asyncio.wait_for(
                byte_iterator.__anext__(),
                timeout=first_token_timeout
            )
            _first_token_ms = (asyncio.get_event_loop().time() - _t0) * 1000
            if isinstance(response, GuardedResponse) and response.first_chunk_ms is not None:
                # The guard already waited for the first chunk: report the real wait
                _first_token_ms = response.first_chunk_ms
            logger.info(f"[FirstToken] received in {_first_token_ms:.0f}ms")
        except asyncio.TimeoutError:
            logger.warning(f"[FirstTokenTimeout] Model did not respond within {first_token_timeout}s")
            raise FirstTokenTimeoutError(f"No response within {first_token_timeout} seconds")
        except StopAsyncIteration:
            # Empty response - this is normal, just finish
            logger.debug("Empty response from Kiro API")
            if outcome is not None:
                outcome.finished = True
            return
        
        # Process first chunk
        if debug_logger:
            debug_logger.log_raw_chunk(first_byte_chunk)
        frame_scanner.feed(first_byte_chunk)
        
        async for event in _process_chunk(parser, first_byte_chunk, thinking_parser, reasoning_stats):
            if event.type == "content" or event.type == "thinking":
                first_token_received = True
            if outcome is not None:
                outcome.record_event(event)
            yield event
        
        # Continue reading remaining chunks
        async for chunk in byte_iterator:
            if debug_logger:
                debug_logger.log_raw_chunk(chunk)
            frame_scanner.feed(chunk)
            
            async for event in _process_chunk(parser, chunk, thinking_parser, reasoning_stats):
                if outcome is not None:
                    outcome.record_event(event)
                yield event
        
        # Finalize thinking parser and yield any remaining content
        if thinking_parser:
            final_result = thinking_parser.finalize()
            
            if final_result.thinking_content:
                processed_thinking = thinking_parser.process_for_output(
                    final_result.thinking_content,
                    final_result.is_first_thinking_chunk,
                    final_result.is_last_thinking_chunk,
                )
                if processed_thinking:
                    yield KiroEvent(
                        type="thinking",
                        thinking_content=processed_thinking,
                        is_first_thinking_chunk=final_result.is_first_thinking_chunk,
                        is_last_thinking_chunk=final_result.is_last_thinking_chunk,
                    )
            
            if final_result.regular_content:
                final_content_event = KiroEvent(type="content", content=final_result.regular_content)
                if outcome is not None:
                    outcome.record_event(final_content_event)
                yield final_content_event
            
            if thinking_parser.found_thinking_block:
                logger.debug("Thinking block processing completed")
        
        # Check bracket-style tool calls in accumulated content
        all_tool_calls = parser.get_tool_calls()
        # Note: bracket tool calls are checked by the caller using full content
        
        # Yield tool calls if any
        for tc in all_tool_calls:
            tool_event = KiroEvent(type="tool_use", tool_use=tc)
            if outcome is not None:
                outcome.record_event(tool_event)
            yield tool_event
        
        # Normal end of the stream: only now the outcome may be classified
        if outcome is not None:
            outcome.finished = True
            
    except FirstTokenTimeoutError:
        raise
    except GeneratorExit:
        logger.debug("Client disconnected (GeneratorExit)")
        raise
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e) if str(e) else "(empty message)"
        logger.error(f"Error during stream parsing: [{error_type}] {error_msg}", exc_info=True)
        raise
    finally:
        # Per-stream frame summary (names and counts only, never payloads)
        logger.debug(
            f"[StreamFrames] bytes={frame_scanner.bytes_received}, "
            f"frames={dict(frame_scanner.counts)}"
        )
        # Counts and lengths only - reasoning text must never be logged
        if reasoning_stats.frames or reasoning_stats.signature_frames:
            logger.debug(
                f"[NativeReasoning] frames={reasoning_stats.frames}, "
                f"chars={reasoning_stats.chars}, "
                f"signature_frames={reasoning_stats.signature_frames}, "
                f"discarded_frames={reasoning_stats.discarded_frames}, "
                f"mode={FAKE_REASONING_HANDLING}"
            )


async def _process_chunk(
    parser: AwsEventStreamParser,
    chunk: bytes,
    thinking_parser: Optional[ThinkingParser],
    reasoning_stats: Optional[NativeReasoningStats] = None
) -> AsyncGenerator[KiroEvent, None]:
    """
    Process a single chunk from Kiro stream.
    
    Args:
        parser: AWS event stream parser
        chunk: Raw bytes chunk
        thinking_parser: Optional thinking parser for fake reasoning
        reasoning_stats: Optional counters updated for native reasoning frames
    
    Yields:
        KiroEvent objects
    """
    events = parser.feed(chunk)
    
    for event in events:
        if event["type"] == "content":
            content = event["data"]
            
            # Process through thinking parser if enabled
            if thinking_parser:
                parse_result = thinking_parser.feed(content)
                
                # Yield thinking content if any
                if parse_result.thinking_content:
                    processed_thinking = thinking_parser.process_for_output(
                        parse_result.thinking_content,
                        parse_result.is_first_thinking_chunk,
                        parse_result.is_last_thinking_chunk,
                    )
                    if processed_thinking:
                        yield KiroEvent(
                            type="thinking",
                            thinking_content=processed_thinking,
                            is_first_thinking_chunk=parse_result.is_first_thinking_chunk,
                            is_last_thinking_chunk=parse_result.is_last_thinking_chunk,
                        )
                
                # Yield regular content if any
                if parse_result.regular_content:
                    yield KiroEvent(type="content", content=parse_result.regular_content)
            else:
                # No thinking parser - pass through as-is
                yield KiroEvent(type="content", content=content)
        
        elif event["type"] == "reasoning":
            reasoning_text = event["data"]
            if reasoning_stats is not None:
                reasoning_stats.frames += 1
                reasoning_stats.chars += len(reasoning_text)
            
            # Native reasoning has no tags: only FAKE_REASONING_HANDLING applies
            processed_reasoning = format_native_reasoning(reasoning_text, FAKE_REASONING_HANDLING)
            if processed_reasoning is None:
                if reasoning_stats is not None:
                    reasoning_stats.discarded_frames += 1
                continue
            
            yield KiroEvent(
                type="thinking",
                thinking_content=processed_reasoning,
                is_native_reasoning=True,
            )
        
        elif event["type"] == "reasoning_signature":
            # The upstream signature is not forwarded (clients get a placeholder signature)
            if reasoning_stats is not None:
                reasoning_stats.signature_frames += 1
        
        elif event["type"] == "usage":
            yield KiroEvent(type="usage", usage=event["data"])
        
        elif event["type"] == "context_usage":
            yield KiroEvent(type="context_usage", context_usage_percentage=event["data"])


# ==================================================================================================
# Empty Upstream Response Guard
# ==================================================================================================

async def aclose_shielded(close: Callable[[], Awaitable[None]]) -> None:
    """
    Run a close coroutine function so that a pending cancellation cannot interrupt it.

    Used on cancellation paths: the cancelled task may be cancelled again at its next
    await (level-triggered cancellation of anyio based middleware), which would skip the
    close and leak the upstream connection. The close runs as a shielded task.
    Expected close errors are logged at DEBUG and swallowed; anything else propagates.

    Args:
        close: Zero-argument coroutine function, e.g. response.aclose or http_client.close
    """
    try:
        await asyncio.shield(close())
    except (httpx.HTTPError, RuntimeError, OSError) as close_error:
        logger.debug(f"Error closing upstream response after cancellation: {close_error}")


class StreamReadyMarker:
    """
    Type of the STREAM_READY sentinel (never a valid SSE chunk).

    stream_with_first_token_retry can yield it once, after the empty-response guard has
    accepted the upstream response and before the first SSE chunk. prime_sse_stream
    consumes it; it must never reach the client.
    """

    def __repr__(self) -> str:
        return "STREAM_READY"


STREAM_READY = StreamReadyMarker()


async def guard_upstream_response(
    response: httpx.Response,
    first_token_timeout: float = FIRST_TOKEN_TIMEOUT,
    enabled: Optional[bool] = None,
) -> Union[httpx.Response, GuardedResponse]:
    """
    Peek at the upstream stream and reject an empty upstream response.

    Reads raw bytes (without interpreting them) until one of:
    - a frame other than contextUsageEvent is seen (content, reasoning, tool use,
      meteringEvent, unknown types): the response is accepted immediately;
    - more than EMPTY_SIGNATURE_MAX_BYTES were read: accepted (never an empty response);
    - the stream ends: rejected with EmptyUpstreamResponseError when UpstreamFrameScanner
      says it is empty, otherwise accepted.

    The first chunk is awaited with first_token_timeout, exactly like parse_kiro_stream
    does, so FIRST_TOKEN_TIMEOUT keeps its meaning. Later chunks have no extra timeout.
    Any read error propagates unchanged (a failed read is not an empty response).

    Accepted responses are returned wrapped in GuardedResponse, which replays the peeked
    bytes with their original chunk boundaries, so downstream parsing and the SSE
    sequence are identical to a stream that was never peeked.

    Args:
        response: HTTP 200 response with the Kiro event stream
        first_token_timeout: Timeout for the first chunk (seconds)
        enabled: Override for EMPTY_RESPONSE_AS_ERROR (None reads the config value).
            When disabled, the response is returned untouched and no bytes are read.

    Returns:
        The original response (guard disabled) or a GuardedResponse replaying peeked bytes

    Raises:
        EmptyUpstreamResponseError: The stream ended and carried no output and no metering
        FirstTokenTimeoutError: No first chunk within first_token_timeout
    """
    if enabled is None:
        enabled = EMPTY_RESPONSE_AS_ERROR
    if not enabled:
        return response

    iterator = response.aiter_bytes()
    scanner = UpstreamFrameScanner()
    buffered: List[bytes] = []
    stream_ended = False
    started = asyncio.get_event_loop().time()

    try:
        try:
            logger.debug(f"[EmptyResponseGuard] Waiting for first chunk (timeout={first_token_timeout}s)...")
            chunk = await asyncio.wait_for(iterator.__anext__(), timeout=first_token_timeout)
        except asyncio.TimeoutError:
            logger.warning(f"[FirstTokenTimeout] Model did not respond within {first_token_timeout}s")
            raise FirstTokenTimeoutError(f"No response within {first_token_timeout} seconds")
        except StopAsyncIteration:
            stream_ended = True
            chunk = b""
        first_chunk_ms = (asyncio.get_event_loop().time() - started) * 1000

        release_reason = "stream_end"
        while not stream_ended:
            buffered.append(chunk)
            scanner.feed(chunk)
            if scanner.saw_non_neutral_frame:
                release_reason = "output_frame"
                break
            if scanner.bytes_received > EMPTY_SIGNATURE_MAX_BYTES:
                release_reason = "size_cap"
                break
            try:
                chunk = await iterator.__anext__()
            except StopAsyncIteration:
                stream_ended = True
    except asyncio.CancelledError:
        # Cancelled while peeking (client went away): the caller never got the response
        # object back, so close the upstream connection here instead of leaking it.
        await aclose_shielded(response.aclose)
        raise

    if stream_ended:
        if scanner.looks_empty():
            try:
                await response.aclose()
            except (httpx.HTTPError, RuntimeError, OSError) as close_error:
                logger.debug(f"Error closing empty response: {close_error}")
            raise EmptyUpstreamResponseError(scanner.bytes_received, dict(scanner.counts))
        if scanner.frames_seen == 0:
            release_reason = "unrecognised_framing"
            logger.warning(
                f"[EmptyResponseGuard] no AWS event-stream frame header recognised in "
                f"{scanner.bytes_received} bytes; empty-response check skipped"
            )

    logger.debug(
        f"[EmptyResponseGuard] released: reason={release_reason}, "
        f"first_chunk_ms={first_chunk_ms:.0f}, chunks={len(buffered)}, "
        f"bytes={scanner.bytes_received}"
    )
    return GuardedResponse(response, iterator, buffered, first_chunk_ms=first_chunk_ms)


async def prime_sse_stream(
    stream: AsyncGenerator[Union[str, StreamReadyMarker], None],
) -> AsyncGenerator[str, None]:
    """
    Advance an SSE generator until the response can safely start, then hand it back.

    Routes call this before building a StreamingResponse so an empty upstream response
    can still be answered with a proper HTTP error status. It reads the first item of
    the stream:
    - STREAM_READY (from stream_with_first_token_retry(emit_ready_marker=True)): the
      upstream response was accepted; the marker is dropped and the rest is replayed;
    - an SSE string: kept and replayed in front of the rest of the stream;
    - end of stream: an empty generator is returned.

    Errors are handled like this:
    - EmptyUpstreamResponseError is raised immediately (nothing was sent to the client);
    - any other Exception is re-raised by the returned generator on first iteration, so
      the route's existing stream_wrapper keeps turning it into an in-stream error event;
    - BaseException (e.g. CancelledError) closes the stream first and then propagates.

    The returned generator closes the underlying stream when it finishes or is closed,
    so a client disconnect reaches the stream's cleanup code.

    Args:
        stream: SSE generator, typically stream_with_first_token_retry*

    Returns:
        Generator yielding the SSE strings of the stream (never STREAM_READY)

    Raises:
        EmptyUpstreamResponseError: The upstream response was empty
    """
    head: List[str] = []
    deferred_error: Optional[Exception] = None
    try:
        first = await stream.__anext__()
        if not isinstance(first, StreamReadyMarker):
            head.append(first)
    except StopAsyncIteration:
        pass
    except EmptyUpstreamResponseError:
        raise
    except Exception as e:
        # Deliberately broad: the stream callbacks raise arbitrary types (HTTPException,
        # plain Exception from on_http_error / on_all_retries_failed). The error is not
        # swallowed: replay() re-raises the very same object on first iteration.
        deferred_error = e
    except BaseException:
        await stream.aclose()
        raise

    async def replay() -> AsyncGenerator[str, None]:
        """Yield the items read while priming, then the rest of the stream."""
        try:
            if deferred_error is not None:
                raise deferred_error
            for item in head:
                yield item
            async for item in stream:
                yield item
        finally:
            await stream.aclose()

    return replay()


def check_stream_outcome(outcome: StreamOutcome, enabled: Optional[bool] = None) -> bool:
    """
    Decide at the end of a stream whether it must be reported as an incomplete response.

    An incomplete response is a stream that ended normally with NO meteringEvent and NO
    text and NO tool_use, but with reasoning (non-neutral) frames. Reasoning never counts
    as output. See StreamVerdict for the full table.

    Logging (counts, byte size and booleans only, never content):
    - MISSING_METERING_WITH_OUTPUT (possibly truncated text): WARNING
      [IncompleteUpstreamResponse], behavior unchanged, whatever the switch says;
    - INCOMPLETE with the switch off: WARNING [IncompleteUpstreamResponse], pass-through;
    - INCOMPLETE with the switch on: not logged here, the caller reports the error.

    Args:
        outcome: StreamOutcome filled by parse_kiro_stream
        enabled: Override for INCOMPLETE_RESPONSE_AS_ERROR (None reads the config value)

    Returns:
        True if the caller must turn the stream into an error, False to finish it normally
    """
    if enabled is None:
        enabled = INCOMPLETE_RESPONSE_AS_ERROR

    verdict = outcome.verdict()
    if verdict == StreamVerdict.INCOMPLETE and enabled:
        return True

    if verdict in (StreamVerdict.INCOMPLETE, StreamVerdict.MISSING_METERING_WITH_OUTPUT):
        logger.warning(
            "[IncompleteUpstreamResponse] verdict={}, upstream_bytes={}, frames={}, "
            "text={}, tool_use={}, action=passthrough",
            verdict.value,
            outcome.scanner.bytes_received,
            dict(outcome.scanner.counts),
            outcome.produced_text,
            outcome.produced_tool_use,
        )
    return False


# ==================================================================================================
# Full Response Collection
# ==================================================================================================

async def collect_stream_to_result(
    response: httpx.Response,
    first_token_timeout: float = FIRST_TOKEN_TIMEOUT,
    enable_thinking_parser: bool = True
) -> StreamResult:
    """
    Collects full response from Kiro stream.
    
    This function consumes the entire stream and returns a StreamResult
    with all accumulated data.
    
    Before parsing, the empty-response guard rejects an empty upstream response
    (see guard_upstream_response); a response with a meteringEvent but no output is
    a genuine empty answer and yields an empty StreamResult.
    After aggregation, an incomplete response (reasoning only, no meteringEvent, no text,
    no tool_use) is rejected too (see check_stream_outcome).
    
    Args:
        response: HTTP response with stream
        first_token_timeout: First token wait timeout
        enable_thinking_parser: Whether to enable thinking block parsing
    
    Returns:
        StreamResult with full content, thinking, tool calls, and usage
    
    Raises:
        EmptyUpstreamResponseError: The upstream response was empty or incomplete
        FirstTokenTimeoutError: No first chunk within first_token_timeout
    """
    response = await guard_upstream_response(response, first_token_timeout)
    result = StreamResult()
    full_content_for_bracket_tools = ""
    outcome = StreamOutcome()
    
    async for event in parse_kiro_stream(
        response, first_token_timeout, enable_thinking_parser, outcome=outcome
    ):
        if event.type == "content" and event.content:
            result.content += event.content
            full_content_for_bracket_tools += event.content
        elif event.type == "thinking" and event.thinking_content:
            result.thinking_content += event.thinking_content
            # Native reasoning is free-form model thought and may quote tool-call syntax;
            # only tag-based thinking keeps its historical bracket tool call scanning.
            if not event.is_native_reasoning:
                full_content_for_bracket_tools += event.thinking_content
        elif event.type == "tool_use" and event.tool_use:
            result.tool_calls.append(event.tool_use)
        elif event.type == "usage" and event.usage:
            result.usage = event.usage
        elif event.type == "context_usage" and event.context_usage_percentage is not None:
            result.context_usage_percentage = event.context_usage_percentage
    
    if check_stream_outcome(outcome):
        raise outcome.to_error()
    
    # Check for bracket-style tool calls in full content
    bracket_tool_calls = parse_bracket_tool_calls(full_content_for_bracket_tools)
    if bracket_tool_calls:
        result.tool_calls = deduplicate_tool_calls(result.tool_calls + bracket_tool_calls)
    
    return result


# ==================================================================================================
# Token Counting Utilities
# ==================================================================================================

def calculate_tokens_from_context_usage(
    context_usage_percentage: Optional[float],
    completion_tokens: int,
    model_cache: "ModelInfoCache",
    model: str
) -> Tuple[int, int, str, str]:
    """
    Calculate token counts from Kiro's context usage percentage.
    
    Args:
        context_usage_percentage: Context usage percentage from Kiro API
        completion_tokens: Number of completion tokens (counted via tiktoken)
        model_cache: Model cache for getting max input tokens
        model: Model name
    
    Returns:
        Tuple of (prompt_tokens, total_tokens, prompt_source, total_source)
    """
    if context_usage_percentage is not None and context_usage_percentage > 0:
        max_input_tokens = model_cache.get_max_input_tokens(model)
        total_tokens = int((context_usage_percentage / 100) * max_input_tokens)
        prompt_tokens = max(0, total_tokens - completion_tokens)
        return prompt_tokens, total_tokens, "subtraction", "API Kiro"
    
    # Fallback: no context usage data
    return 0, completion_tokens, "unknown", "tiktoken"


# ==================================================================================================
# First Token Retry Logic
# ==================================================================================================

async def stream_with_first_token_retry(
    make_request: Callable[[], Awaitable[httpx.Response]],
    stream_processor: Callable[[httpx.Response], AsyncGenerator[str, None]],
    initial_response: Optional[httpx.Response] = None,
    max_retries: int = FIRST_TOKEN_MAX_RETRIES,
    first_token_timeout: float = FIRST_TOKEN_TIMEOUT,
    on_http_error: Optional[Callable[[int, str], Exception]] = None,
    on_all_retries_failed: Optional[Callable[[int, float], Exception]] = None,
    emit_ready_marker: bool = False,
    empty_response_as_error: Optional[bool] = None,
) -> AsyncGenerator[Union[str, StreamReadyMarker], None]:
    """
    Generic streaming with automatic retry on first token timeout.
    
    If model doesn't respond within first_token_timeout seconds,
    request is cancelled and a new one is made. Maximum max_retries attempts.
    
    This is seamless for user - they just see a delay,
    but eventually get a response (or error after all attempts).
    
    Every attempt first goes through guard_upstream_response (unless disabled): the first
    upstream chunk is awaited under first_token_timeout (a timeout is retried exactly like
    before), and an empty upstream response raises EmptyUpstreamResponseError WITHOUT any
    retry or backoff - the client is expected to retry.
    
    Args:
        make_request: Function to create new HTTP request (returns httpx.Response)
        stream_processor: Function that processes response and yields SSE strings.
                         Must use parse_kiro_stream internally for timeout handling.
        initial_response: Optional pre-validated response to use on first attempt.
                         If provided, make_request is only called on retries.
                         This allows reusing an already-opened HTTP 200 response.
        max_retries: Maximum number of attempts
        first_token_timeout: First token wait timeout (seconds)
        on_http_error: Optional callback to create exception for HTTP errors.
                      Receives (status_code, error_text), returns Exception.
                      If None, raises generic Exception.
        on_all_retries_failed: Optional callback to create exception when all retries fail.
                              Receives (max_retries, timeout), returns Exception.
                              If None, raises generic Exception.
        emit_ready_marker: If True, yield STREAM_READY once, after the upstream response was
                          accepted and before the first SSE chunk. Routes use it (via
                          prime_sse_stream) to decide between an HTTP error and a stream.
        empty_response_as_error: Override for EMPTY_RESPONSE_AS_ERROR (None reads the config).
    
    Yields:
        Strings in SSE format (format depends on stream_processor), preceded by
        STREAM_READY when emit_ready_marker is True
    
    Raises:
        EmptyUpstreamResponseError: The upstream response was empty (no retry is made)
        Exception from on_http_error or on_all_retries_failed callbacks
    
    Example:
        >>> async def make_req():
        ...     return await http_client.request_with_retry("POST", url, payload, stream=True)
        >>> async def process(response):
        ...     async for chunk in stream_kiro_to_openai(response, ...):
        ...         yield chunk
        >>> # With initial response (reuse already-validated 200 response)
        >>> response = await make_req()
        >>> async for chunk in stream_with_first_token_retry(make_req, process, initial_response=response):
        ...     print(chunk)
    """
    last_error: Optional[Exception] = None
    ready_marker_sent = False
    
    for attempt in range(max_retries):
        response: Optional[Union[httpx.Response, GuardedResponse]] = None
        try:
            # Make request
            if attempt > 0:
                logger.warning(f"Retry attempt {attempt + 1}/{max_retries} after first token timeout")
            
            # On first attempt, reuse initial_response if provided
            if attempt == 0 and initial_response is not None:
                response = initial_response
                logger.debug("Reusing initial response for first attempt")
            else:
                response = await make_request()
            
            if response.status_code != 200:
                # Error from API - close response and raise exception
                try:
                    error_content = await response.aread()
                    error_text = error_content.decode('utf-8', errors='replace')
                except Exception:
                    error_text = "Unknown error"
                
                try:
                    await response.aclose()
                except Exception:
                    pass
                
                logger.error(f"Error from Kiro API: {response.status_code} - {error_text}")
                
                if on_http_error:
                    raise on_http_error(response.status_code, error_text)
                else:
                    raise Exception(f"Upstream API error ({response.status_code}): {error_text}")
            
            # Peek at the upstream stream: first chunk under first_token_timeout,
            # empty upstream responses are rejected here (before anything is sent)
            response = await guard_upstream_response(
                response, first_token_timeout, empty_response_as_error
            )
            
            if emit_ready_marker and not ready_marker_sent:
                ready_marker_sent = True
                try:
                    yield STREAM_READY
                except GeneratorExit:
                    # Consumer went away before the stream processor took ownership of the response
                    try:
                        await response.aclose()
                    except (httpx.HTTPError, RuntimeError, OSError) as close_error:
                        logger.debug(f"Error closing response: {close_error}")
                    raise
            
            # Try to stream with first token timeout
            async for chunk in stream_processor(response):
                yield chunk
            
            # Successfully completed - exit
            return
            
        except EmptyUpstreamResponseError:
            # Not retried: the client decides whether to retry. Not an unexpected error.
            if response:
                try:
                    await response.aclose()
                except (httpx.HTTPError, RuntimeError, OSError) as close_error:
                    logger.debug(f"Error closing response: {close_error}")
            raise
            
        except asyncio.CancelledError:
            # Cancelled while waiting for the upstream (client went away): close the
            # response that is still open, then let the cancellation propagate.
            if response:
                await aclose_shielded(response.aclose)
            raise

        except FirstTokenTimeoutError as e:
            last_error = e
            logger.warning(
                f"[FirstTokenTimeout] Attempt {attempt + 1}/{max_retries} failed - "
                f"model did not respond within {first_token_timeout}s"
            )
            
            # Close current response if open
            if response:
                try:
                    await response.aclose()
                except Exception:
                    pass
            
            # Continue to next attempt
            continue
            
        except Exception as e:
            # Other errors - no retry, propagate
            # Use positional argument to avoid loguru interpreting curly braces in error message as format placeholders
            # f-string with repr() doesn't work because loguru still sees {type} inside the string
            error_msg = str(e) if str(e) else "(empty message)"
            logger.error("Unexpected error during streaming: {}", error_msg, exc_info=True)
            if response:
                try:
                    await response.aclose()
                except Exception:
                    pass
            raise
    
    # All attempts exhausted - raise error
    logger.error(
        f"[FirstTokenTimeout] All {max_retries} attempts exhausted - "
        f"model never responded within {first_token_timeout}s per attempt"
    )
    
    if on_all_retries_failed:
        raise on_all_retries_failed(max_retries, first_token_timeout)
    else:
        raise Exception(
            f"Model did not respond within {first_token_timeout}s after {max_retries} attempts. "
            "Please try again."
        )