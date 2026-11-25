
# -*- coding: utf-8 -*-

"""
Unit tests for streaming_core module.

Tests for:
- KiroEvent dataclass
- StreamResult dataclass
- FirstTokenTimeoutError exception
- parse_kiro_stream() function
- collect_stream_to_result() function
- calculate_tokens_from_context_usage() function
- upstream connection drop handling (is_upstream_drop, decide_drop_action,
  describe_stream_drop, retry of a dropped stream before any output was sent)
"""

import pytest
import asyncio
import time
import httpx
from loguru import logger
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import asdict

from kiro.streaming_core import (
    DROP_ACTION_DISABLED,
    DROP_ACTION_EXHAUSTED,
    DROP_ACTION_OUTPUT_SENT,
    DROP_ACTION_RETRY,
    STREAM_READY,
    decide_drop_action,
    describe_stream_drop,
    format_response_summary,
    is_upstream_drop,
    log_response_summary,
    KiroEvent,
    NativeReasoningStats,
    StreamResult,
    FirstTokenTimeoutError,
    parse_kiro_stream,
    collect_stream_to_result,
    calculate_tokens_from_context_usage,
    stream_with_first_token_retry,
    _process_chunk,
    _parse_single_stream,
)
from kiro.stream_recovery import AttemptEvidence


# ==================================================================================================
# Fixtures
# ==================================================================================================

@pytest.fixture
def mock_model_cache():
    """Mock for ModelInfoCache."""
    cache = MagicMock()
    cache.get_max_input_tokens.return_value = 200000
    return cache


@pytest.fixture
def mock_response():
    """Mock for httpx.Response."""
    response = AsyncMock()
    response.status_code = 200
    response.aclose = AsyncMock()
    return response


@pytest.fixture
def mock_parser():
    """Mock for AwsEventStreamParser."""
    parser = MagicMock()
    parser.feed.return_value = []
    parser.get_tool_calls.return_value = []
    return parser


# ==================================================================================================
# Tests for KiroEvent dataclass
# ==================================================================================================

class TestKiroEvent:
    """Tests for KiroEvent dataclass."""
    
    def test_creates_content_event(self):
        """
        What it does: Creates a content event with text.
        Goal: Verify KiroEvent can represent content events.
        """
        print("Action: Creating content event...")
        event = KiroEvent(type="content", content="Hello, world!")
        
        print(f"Comparing type: Expected 'content', Got '{event.type}'")
        assert event.type == "content"
        print(f"Comparing content: Expected 'Hello, world!', Got '{event.content}'")
        assert event.content == "Hello, world!"
        assert event.thinking_content is None
        assert event.tool_use is None
        print("✓ Content event created correctly")
    
    def test_creates_thinking_event(self):
        """
        What it does: Creates a thinking event with reasoning content.
        Goal: Verify KiroEvent can represent thinking events.
        """
        print("Action: Creating thinking event...")
        event = KiroEvent(
            type="thinking",
            thinking_content="Let me think...",
            is_first_thinking_chunk=True,
            is_last_thinking_chunk=False
        )
        
        print(f"Comparing type: Expected 'thinking', Got '{event.type}'")
        assert event.type == "thinking"
        print(f"Comparing thinking_content: Expected 'Let me think...', Got '{event.thinking_content}'")
        assert event.thinking_content == "Let me think..."
        assert event.is_first_thinking_chunk is True
        assert event.is_last_thinking_chunk is False
        print("✓ Thinking event created correctly")
    
    def test_creates_tool_use_event(self):
        """
        What it does: Creates a tool_use event with tool data.
        Goal: Verify KiroEvent can represent tool use events.
        """
        print("Action: Creating tool_use event...")
        tool_data = {
            "id": "call_123",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "Moscow"}'}
        }
        event = KiroEvent(type="tool_use", tool_use=tool_data)
        
        print(f"Comparing type: Expected 'tool_use', Got '{event.type}'")
        assert event.type == "tool_use"
        print(f"Comparing tool_use: Expected {tool_data}, Got {event.tool_use}")
        assert event.tool_use == tool_data
        print("✓ Tool use event created correctly")
    
    def test_creates_usage_event(self):
        """
        What it does: Creates a usage event with metering data.
        Goal: Verify KiroEvent can represent usage events.
        """
        print("Action: Creating usage event...")
        usage_data = {"credits": 0.001}
        event = KiroEvent(type="usage", usage=usage_data)
        
        print(f"Comparing type: Expected 'usage', Got '{event.type}'")
        assert event.type == "usage"
        print(f"Comparing usage: Expected {usage_data}, Got {event.usage}")
        assert event.usage == usage_data
        print("✓ Usage event created correctly")
    
    def test_creates_context_usage_event(self):
        """
        What it does: Creates a context_usage event with percentage.
        Goal: Verify KiroEvent can represent context usage events.
        """
        print("Action: Creating context_usage event...")
        event = KiroEvent(type="context_usage", context_usage_percentage=5.5)
        
        print(f"Comparing type: Expected 'context_usage', Got '{event.type}'")
        assert event.type == "context_usage"
        print(f"Comparing context_usage_percentage: Expected 5.5, Got {event.context_usage_percentage}")
        assert event.context_usage_percentage == 5.5
        print("✓ Context usage event created correctly")
    
    def test_default_values(self):
        """
        What it does: Verifies default values for optional fields.
        Goal: Ensure all optional fields default to None/False.
        """
        print("Action: Creating minimal event...")
        event = KiroEvent(type="content")
        
        print("Checking default values...")
        assert event.content is None
        assert event.thinking_content is None
        assert event.tool_use is None
        assert event.usage is None
        assert event.context_usage_percentage is None
        assert event.is_first_thinking_chunk is False
        assert event.is_last_thinking_chunk is False
        assert event.is_native_reasoning is False
        print("✓ All default values are correct")


# ==================================================================================================
# Tests for StreamResult dataclass
# ==================================================================================================

class TestStreamResult:
    """Tests for StreamResult dataclass."""
    
    def test_creates_empty_result(self):
        """
        What it does: Creates an empty StreamResult.
        Goal: Verify default values are correct.
        """
        print("Action: Creating empty StreamResult...")
        result = StreamResult()
        
        print("Checking default values...")
        assert result.content == ""
        assert result.thinking_content == ""
        assert result.tool_calls == []
        assert result.usage is None
        assert result.context_usage_percentage is None
        print("✓ Empty StreamResult created correctly")
    
    def test_creates_result_with_content(self):
        """
        What it does: Creates StreamResult with content.
        Goal: Verify content is stored correctly.
        """
        print("Action: Creating StreamResult with content...")
        result = StreamResult(content="Hello, world!")
        
        print(f"Comparing content: Expected 'Hello, world!', Got '{result.content}'")
        assert result.content == "Hello, world!"
        print("✓ StreamResult with content created correctly")
    
    def test_creates_result_with_tool_calls(self):
        """
        What it does: Creates StreamResult with tool calls.
        Goal: Verify tool calls are stored correctly.
        """
        print("Action: Creating StreamResult with tool calls...")
        tool_calls = [
            {"id": "call_1", "function": {"name": "func1"}},
            {"id": "call_2", "function": {"name": "func2"}}
        ]
        result = StreamResult(tool_calls=tool_calls)
        
        print(f"Comparing tool_calls count: Expected 2, Got {len(result.tool_calls)}")
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0]["id"] == "call_1"
        print("✓ StreamResult with tool calls created correctly")
    
    def test_creates_result_with_usage(self):
        """
        What it does: Creates StreamResult with usage data.
        Goal: Verify usage is stored correctly.
        """
        print("Action: Creating StreamResult with usage...")
        usage = {"credits": 0.002}
        result = StreamResult(usage=usage)
        
        print(f"Comparing usage: Expected {usage}, Got {result.usage}")
        assert result.usage == usage
        print("✓ StreamResult with usage created correctly")
    
    def test_creates_full_result(self):
        """
        What it does: Creates StreamResult with all fields.
        Goal: Verify all fields work together.
        """
        print("Action: Creating full StreamResult...")
        result = StreamResult(
            content="Response text",
            thinking_content="Thinking...",
            tool_calls=[{"id": "call_1"}],
            usage={"credits": 0.001},
            context_usage_percentage=3.5
        )
        
        print("Checking all fields...")
        assert result.content == "Response text"
        assert result.thinking_content == "Thinking..."
        assert len(result.tool_calls) == 1
        assert result.usage == {"credits": 0.001}
        assert result.context_usage_percentage == 3.5
        print("✓ Full StreamResult created correctly")


# ==================================================================================================
# Tests for FirstTokenTimeoutError
# ==================================================================================================

class TestFirstTokenTimeoutError:
    """Tests for FirstTokenTimeoutError exception."""
    
    def test_creates_exception_with_message(self):
        """
        What it does: Creates exception with custom message.
        Goal: Verify exception message is stored correctly.
        """
        print("Action: Creating FirstTokenTimeoutError...")
        error = FirstTokenTimeoutError("No response within 30 seconds")
        
        print(f"Comparing message: Expected 'No response within 30 seconds', Got '{str(error)}'")
        assert str(error) == "No response within 30 seconds"
        print("✓ Exception created correctly")
    
    def test_exception_is_catchable(self):
        """
        What it does: Verifies exception can be caught.
        Goal: Ensure exception inherits from Exception.
        """
        print("Action: Raising and catching FirstTokenTimeoutError...")
        
        with pytest.raises(FirstTokenTimeoutError) as exc_info:
            raise FirstTokenTimeoutError("Timeout!")
        
        print(f"Caught exception: {exc_info.value}")
        assert "Timeout!" in str(exc_info.value)
        print("✓ Exception is catchable")
    
    def test_exception_inherits_from_exception(self):
        """
        What it does: Verifies inheritance chain.
        Goal: Ensure proper exception hierarchy.
        """
        print("Action: Checking inheritance...")
        error = FirstTokenTimeoutError("Test")
        
        assert isinstance(error, Exception)
        print("✓ FirstTokenTimeoutError inherits from Exception")


# ==================================================================================================
# Tests for parse_kiro_stream()
# ==================================================================================================

class TestParseKiroStream:
    """Tests for parse_kiro_stream() function."""
    
    @pytest.mark.asyncio
    async def test_parses_content_events(self, mock_response, mock_parser):
        """
        What it does: Parses content events from Kiro stream.
        Goal: Verify content events are yielded correctly.
        """
        print("Setup: Mock parser to return content events...")
        mock_parser.feed.return_value = [
            {"type": "content", "data": "Hello"},
            {"type": "content", "data": " World"}
        ]
        
        async def mock_aiter_bytes():
            yield b'chunk1'
        
        mock_response.aiter_bytes = mock_aiter_bytes
        
        print("Action: Parsing stream...")
        events = []
        
        with patch('kiro.streaming_core.AwsEventStreamParser', return_value=mock_parser):
            with patch('kiro.streaming_core.FAKE_REASONING_ENABLED', False):
                async for event in parse_kiro_stream(mock_response, first_token_timeout=30):
                    events.append(event)
        
        print(f"Received {len(events)} events")
        content_events = [e for e in events if e.type == "content"]
        print(f"Content events: {len(content_events)}")
        
        assert len(content_events) == 2
        assert content_events[0].content == "Hello"
        assert content_events[1].content == " World"
        print("✓ Content events parsed correctly")
    
    @pytest.mark.asyncio
    async def test_parses_usage_events(self, mock_response, mock_parser):
        """
        What it does: Parses usage events from Kiro stream.
        Goal: Verify usage events are yielded correctly.
        """
        print("Setup: Mock parser to return usage event...")
        mock_parser.feed.return_value = [
            {"type": "usage", "data": {"credits": 0.001}}
        ]
        
        async def mock_aiter_bytes():
            yield b'chunk1'
        
        mock_response.aiter_bytes = mock_aiter_bytes
        
        print("Action: Parsing stream...")
        events = []
        
        with patch('kiro.streaming_core.AwsEventStreamParser', return_value=mock_parser):
            with patch('kiro.streaming_core.FAKE_REASONING_ENABLED', False):
                async for event in parse_kiro_stream(mock_response, first_token_timeout=30):
                    events.append(event)
        
        print(f"Received {len(events)} events")
        usage_events = [e for e in events if e.type == "usage"]
        
        assert len(usage_events) == 1
        assert usage_events[0].usage == {"credits": 0.001}
        print("✓ Usage events parsed correctly")
    
    @pytest.mark.asyncio
    async def test_parses_context_usage_events(self, mock_response, mock_parser):
        """
        What it does: Parses context_usage events from Kiro stream.
        Goal: Verify context usage percentage is yielded correctly.
        """
        print("Setup: Mock parser to return context_usage event...")
        mock_parser.feed.return_value = [
            {"type": "context_usage", "data": 5.5}
        ]
        
        async def mock_aiter_bytes():
            yield b'chunk1'
        
        mock_response.aiter_bytes = mock_aiter_bytes
        
        print("Action: Parsing stream...")
        events = []
        
        with patch('kiro.streaming_core.AwsEventStreamParser', return_value=mock_parser):
            with patch('kiro.streaming_core.FAKE_REASONING_ENABLED', False):
                async for event in parse_kiro_stream(mock_response, first_token_timeout=30):
                    events.append(event)
        
        print(f"Received {len(events)} events")
        context_events = [e for e in events if e.type == "context_usage"]
        
        assert len(context_events) == 1
        assert context_events[0].context_usage_percentage == 5.5
        print("✓ Context usage events parsed correctly")
    
    @pytest.mark.asyncio
    async def test_yields_tool_calls_at_end(self, mock_response, mock_parser):
        """
        What it does: Yields tool calls collected during parsing.
        Goal: Verify tool calls are yielded as events.
        """
        print("Setup: Mock parser with tool calls...")
        mock_parser.feed.return_value = [{"type": "content", "data": "text"}]
        mock_parser.get_tool_calls.return_value = [
            {"id": "call_1", "function": {"name": "func1", "arguments": "{}"}}
        ]
        
        async def mock_aiter_bytes():
            yield b'chunk1'
        
        mock_response.aiter_bytes = mock_aiter_bytes
        
        print("Action: Parsing stream...")
        events = []
        
        with patch('kiro.streaming_core.AwsEventStreamParser', return_value=mock_parser):
            with patch('kiro.streaming_core.FAKE_REASONING_ENABLED', False):
                async for event in parse_kiro_stream(mock_response, first_token_timeout=30):
                    events.append(event)
        
        print(f"Received {len(events)} events")
        tool_events = [e for e in events if e.type == "tool_use"]
        
        assert len(tool_events) == 1
        assert tool_events[0].tool_use["id"] == "call_1"
        print("✓ Tool calls yielded correctly")
    
    @pytest.mark.asyncio
    async def test_raises_timeout_on_first_token(self, mock_response):
        """
        What it does: Raises FirstTokenTimeoutError on timeout.
        Goal: Verify timeout handling for first token.
        """
        print("Setup: Mock response that times out...")
        
        async def mock_aiter_bytes():
            yield b'chunk'
        
        mock_response.aiter_bytes = mock_aiter_bytes
        
        async def mock_wait_for_timeout(*args, **kwargs):
            raise asyncio.TimeoutError()
        
        print("Action: Parsing stream with timeout...")
        
        with patch('kiro.streaming_core.asyncio.wait_for', side_effect=mock_wait_for_timeout):
            with pytest.raises(FirstTokenTimeoutError) as exc_info:
                async for event in parse_kiro_stream(mock_response, first_token_timeout=30):
                    pass
        
        print(f"Caught exception: {exc_info.value}")
        assert "30" in str(exc_info.value)
        print("✓ FirstTokenTimeoutError raised on timeout")
    
    @pytest.mark.asyncio
    async def test_handles_empty_response(self, mock_response):
        """
        What it does: Handles empty response gracefully.
        Goal: Verify no events yielded for empty response.
        """
        print("Setup: Mock empty response...")
        
        async def mock_aiter_bytes():
            return
            yield  # Make it a generator
        
        mock_response.aiter_bytes = mock_aiter_bytes
        
        # Mock wait_for to raise StopAsyncIteration (empty response)
        async def mock_wait_for_empty(*args, **kwargs):
            raise StopAsyncIteration()
        
        print("Action: Parsing empty stream...")
        events = []
        
        with patch('kiro.streaming_core.asyncio.wait_for', side_effect=mock_wait_for_empty):
            async for event in parse_kiro_stream(mock_response, first_token_timeout=30):
                events.append(event)
        
        print(f"Received {len(events)} events")
        assert len(events) == 0
        print("✓ Empty response handled correctly")
    
    @pytest.mark.asyncio
    async def test_handles_generator_exit(self, mock_response, mock_parser):
        """
        What it does: Handles GeneratorExit gracefully.
        Goal: Verify client disconnect is handled.
        """
        print("Setup: Mock response that raises GeneratorExit...")
        
        async def mock_aiter_bytes():
            yield b'chunk1'
            raise GeneratorExit()
        
        mock_response.aiter_bytes = mock_aiter_bytes
        mock_parser.feed.return_value = [{"type": "content", "data": "Hello"}]
        
        print("Action: Parsing stream with GeneratorExit...")
        events = []
        generator_exit_raised = False
        
        with patch('kiro.streaming_core.AwsEventStreamParser', return_value=mock_parser):
            with patch('kiro.streaming_core.FAKE_REASONING_ENABLED', False):
                try:
                    async for event in parse_kiro_stream(mock_response, first_token_timeout=30):
                        events.append(event)
                except GeneratorExit:
                    generator_exit_raised = True
        
        print(f"GeneratorExit raised: {generator_exit_raised}")
        assert generator_exit_raised
        print("✓ GeneratorExit handled correctly")


# ==================================================================================================
# Tests for _process_chunk()
# ==================================================================================================

class TestProcessChunk:
    """Tests for _process_chunk() helper function."""
    
    @pytest.mark.asyncio
    async def test_processes_content_event(self, mock_parser):
        """
        What it does: Processes content event from chunk.
        Goal: Verify content is converted to KiroEvent.
        """
        print("Setup: Mock parser with content event...")
        mock_parser.feed.return_value = [{"type": "content", "data": "Hello"}]
        
        print("Action: Processing chunk...")
        events = []
        async for event in _process_chunk(mock_parser, b'chunk', None):
            events.append(event)
        
        print(f"Received {len(events)} events")
        assert len(events) == 1
        assert events[0].type == "content"
        assert events[0].content == "Hello"
        print("✓ Content event processed correctly")
    
    @pytest.mark.asyncio
    async def test_processes_usage_event(self, mock_parser):
        """
        What it does: Processes usage event from chunk.
        Goal: Verify usage is converted to KiroEvent.
        """
        print("Setup: Mock parser with usage event...")
        mock_parser.feed.return_value = [{"type": "usage", "data": {"credits": 0.001}}]
        
        print("Action: Processing chunk...")
        events = []
        async for event in _process_chunk(mock_parser, b'chunk', None):
            events.append(event)
        
        print(f"Received {len(events)} events")
        assert len(events) == 1
        assert events[0].type == "usage"
        assert events[0].usage == {"credits": 0.001}
        print("✓ Usage event processed correctly")
    
    @pytest.mark.asyncio
    async def test_processes_context_usage_event(self, mock_parser):
        """
        What it does: Processes context_usage event from chunk.
        Goal: Verify context usage is converted to KiroEvent.
        """
        print("Setup: Mock parser with context_usage event...")
        mock_parser.feed.return_value = [{"type": "context_usage", "data": 7.5}]
        
        print("Action: Processing chunk...")
        events = []
        async for event in _process_chunk(mock_parser, b'chunk', None):
            events.append(event)
        
        print(f"Received {len(events)} events")
        assert len(events) == 1
        assert events[0].type == "context_usage"
        assert events[0].context_usage_percentage == 7.5
        print("✓ Context usage event processed correctly")
    
    @pytest.mark.asyncio
    async def test_processes_multiple_events(self, mock_parser):
        """
        What it does: Processes multiple events from single chunk.
        Goal: Verify all events are yielded.
        """
        print("Setup: Mock parser with multiple events...")
        mock_parser.feed.return_value = [
            {"type": "content", "data": "Hello"},
            {"type": "content", "data": " World"},
            {"type": "usage", "data": {"credits": 0.001}}
        ]
        
        print("Action: Processing chunk...")
        events = []
        async for event in _process_chunk(mock_parser, b'chunk', None):
            events.append(event)
        
        print(f"Received {len(events)} events")
        assert len(events) == 3
        assert events[0].type == "content"
        assert events[1].type == "content"
        assert events[2].type == "usage"
        print("✓ Multiple events processed correctly")
    
    @pytest.mark.asyncio
    async def test_processes_with_thinking_parser(self, mock_parser):
        """
        What it does: Processes content through thinking parser.
        Goal: Verify thinking parser integration.
        """
        print("Setup: Mock parser and thinking parser...")
        mock_parser.feed.return_value = [{"type": "content", "data": "Hello"}]
        
        mock_thinking_parser = MagicMock()
        mock_thinking_parser.feed.return_value = MagicMock(
            thinking_content=None,
            regular_content="Hello",
            is_first_thinking_chunk=False,
            is_last_thinking_chunk=False
        )
        
        print("Action: Processing chunk with thinking parser...")
        events = []
        async for event in _process_chunk(mock_parser, b'chunk', mock_thinking_parser):
            events.append(event)
        
        print(f"Received {len(events)} events")
        assert len(events) == 1
        assert events[0].type == "content"
        assert events[0].content == "Hello"
        print("✓ Thinking parser integration works correctly")
    
    @pytest.mark.asyncio
    async def test_yields_thinking_content(self, mock_parser):
        """
        What it does: Yields thinking content from thinking parser.
        Goal: Verify thinking events are created.
        """
        print("Setup: Mock parser and thinking parser with thinking content...")
        mock_parser.feed.return_value = [{"type": "content", "data": "<thinking>Let me think</thinking>"}]
        
        mock_thinking_parser = MagicMock()
        mock_thinking_parser.feed.return_value = MagicMock(
            thinking_content="Let me think",
            regular_content=None,
            is_first_thinking_chunk=True,
            is_last_thinking_chunk=True
        )
        mock_thinking_parser.process_for_output.return_value = "Let me think"
        
        print("Action: Processing chunk with thinking content...")
        events = []
        async for event in _process_chunk(mock_parser, b'chunk', mock_thinking_parser):
            events.append(event)
        
        print(f"Received {len(events)} events")
        thinking_events = [e for e in events if e.type == "thinking"]
        assert len(thinking_events) == 1
        assert thinking_events[0].thinking_content == "Let me think"
        print("✓ Thinking content yielded correctly")


# ==================================================================================================
# Tests for collect_stream_to_result()
# ==================================================================================================

class TestCollectStreamToResult:
    """Tests for collect_stream_to_result() function."""
    
    @pytest.mark.asyncio
    async def test_collects_content(self, mock_response, mock_parser):
        """
        What it does: Collects content from stream.
        Goal: Verify content is accumulated correctly.
        """
        print("Setup: Mock parser with content events...")
        mock_parser.feed.return_value = [
            {"type": "content", "data": "Hello"},
            {"type": "content", "data": " World"}
        ]
        mock_parser.get_tool_calls.return_value = []
        
        async def mock_aiter_bytes():
            yield b'chunk1'
        
        mock_response.aiter_bytes = mock_aiter_bytes
        
        print("Action: Collecting stream...")
        
        with patch('kiro.streaming_core.AwsEventStreamParser', return_value=mock_parser):
            with patch('kiro.streaming_core.FAKE_REASONING_ENABLED', False):
                with patch('kiro.streaming_core.parse_bracket_tool_calls', return_value=[]):
                    result = await collect_stream_to_result(mock_response, first_token_timeout=30)
        
        print(f"Collected content: '{result.content}'")
        assert result.content == "Hello World"
        print("✓ Content collected correctly")
    
    @pytest.mark.asyncio
    async def test_collects_tool_calls(self, mock_response, mock_parser):
        """
        What it does: Collects tool calls from stream.
        Goal: Verify tool calls are accumulated correctly.
        """
        print("Setup: Mock parser with tool calls...")
        mock_parser.feed.return_value = [{"type": "content", "data": "text"}]
        mock_parser.get_tool_calls.return_value = [
            {"id": "call_1", "function": {"name": "func1", "arguments": "{}"}}
        ]
        
        async def mock_aiter_bytes():
            yield b'chunk1'
        
        mock_response.aiter_bytes = mock_aiter_bytes
        
        print("Action: Collecting stream...")
        
        with patch('kiro.streaming_core.AwsEventStreamParser', return_value=mock_parser):
            with patch('kiro.streaming_core.FAKE_REASONING_ENABLED', False):
                with patch('kiro.streaming_core.parse_bracket_tool_calls', return_value=[]):
                    result = await collect_stream_to_result(mock_response, first_token_timeout=30)
        
        print(f"Collected tool calls: {len(result.tool_calls)}")
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["id"] == "call_1"
        print("✓ Tool calls collected correctly")
    
    @pytest.mark.asyncio
    async def test_collects_usage(self, mock_response, mock_parser):
        """
        What it does: Collects usage from stream.
        Goal: Verify usage is stored correctly.
        """
        print("Setup: Mock parser with usage event...")
        mock_parser.feed.return_value = [
            {"type": "content", "data": "text"},
            {"type": "usage", "data": {"credits": 0.002}}
        ]
        mock_parser.get_tool_calls.return_value = []
        
        async def mock_aiter_bytes():
            yield b'chunk1'
        
        mock_response.aiter_bytes = mock_aiter_bytes
        
        print("Action: Collecting stream...")
        
        with patch('kiro.streaming_core.AwsEventStreamParser', return_value=mock_parser):
            with patch('kiro.streaming_core.FAKE_REASONING_ENABLED', False):
                with patch('kiro.streaming_core.parse_bracket_tool_calls', return_value=[]):
                    result = await collect_stream_to_result(mock_response, first_token_timeout=30)
        
        print(f"Collected usage: {result.usage}")
        assert result.usage == {"credits": 0.002}
        print("✓ Usage collected correctly")
    
    @pytest.mark.asyncio
    async def test_collects_context_usage_percentage(self, mock_response, mock_parser):
        """
        What it does: Collects context usage percentage from stream.
        Goal: Verify context usage is stored correctly.
        """
        print("Setup: Mock parser with context_usage event...")
        mock_parser.feed.return_value = [
            {"type": "content", "data": "text"},
            {"type": "context_usage", "data": 8.5}
        ]
        mock_parser.get_tool_calls.return_value = []
        
        async def mock_aiter_bytes():
            yield b'chunk1'
        
        mock_response.aiter_bytes = mock_aiter_bytes
        
        print("Action: Collecting stream...")
        
        with patch('kiro.streaming_core.AwsEventStreamParser', return_value=mock_parser):
            with patch('kiro.streaming_core.FAKE_REASONING_ENABLED', False):
                with patch('kiro.streaming_core.parse_bracket_tool_calls', return_value=[]):
                    result = await collect_stream_to_result(mock_response, first_token_timeout=30)
        
        print(f"Collected context_usage_percentage: {result.context_usage_percentage}")
        assert result.context_usage_percentage == 8.5
        print("✓ Context usage percentage collected correctly")
    
    @pytest.mark.asyncio
    async def test_collects_thinking_content(self, mock_response, mock_parser):
        """
        What it does: Collects thinking content from stream.
        Goal: Verify thinking content is accumulated correctly.
        """
        print("Setup: Mock parser with thinking content...")
        # We need to mock the thinking parser behavior
        mock_parser.feed.return_value = [{"type": "content", "data": "thinking text"}]
        mock_parser.get_tool_calls.return_value = []
        
        async def mock_aiter_bytes():
            yield b'chunk1'
        
        mock_response.aiter_bytes = mock_aiter_bytes
        
        # Create mock events that include thinking
        mock_events = [
            KiroEvent(type="thinking", thinking_content="Let me think..."),
            KiroEvent(type="content", content="Here is my answer")
        ]
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            for event in mock_events:
                yield event
        
        print("Action: Collecting stream with thinking...")
        
        with patch('kiro.streaming_core.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_core.parse_bracket_tool_calls', return_value=[]):
                result = await collect_stream_to_result(mock_response, first_token_timeout=30)
        
        print(f"Collected thinking_content: '{result.thinking_content}'")
        print(f"Collected content: '{result.content}'")
        assert result.thinking_content == "Let me think..."
        assert result.content == "Here is my answer"
        print("✓ Thinking content collected correctly")
    
    @pytest.mark.asyncio
    async def test_deduplicates_bracket_tool_calls(self, mock_response, mock_parser):
        """
        What it does: Deduplicates bracket-style tool calls.
        Goal: Verify duplicate tool calls are removed.
        """
        print("Setup: Mock parser with tool calls and bracket tool calls...")
        mock_parser.feed.return_value = [{"type": "content", "data": "text"}]
        mock_parser.get_tool_calls.return_value = [
            {"id": "call_1", "function": {"name": "func1", "arguments": "{}"}}
        ]
        
        bracket_tool_calls = [
            {"id": "call_1", "function": {"name": "func1", "arguments": "{}"}},  # Duplicate
            {"id": "call_2", "function": {"name": "func2", "arguments": "{}"}}   # New
        ]
        
        async def mock_aiter_bytes():
            yield b'chunk1'
        
        mock_response.aiter_bytes = mock_aiter_bytes
        
        print("Action: Collecting stream with duplicates...")
        
        with patch('kiro.streaming_core.AwsEventStreamParser', return_value=mock_parser):
            with patch('kiro.streaming_core.FAKE_REASONING_ENABLED', False):
                with patch('kiro.streaming_core.parse_bracket_tool_calls', return_value=bracket_tool_calls):
                    with patch('kiro.streaming_core.deduplicate_tool_calls') as mock_dedup:
                        mock_dedup.return_value = [
                            {"id": "call_1", "function": {"name": "func1", "arguments": "{}"}},
                            {"id": "call_2", "function": {"name": "func2", "arguments": "{}"}}
                        ]
                        result = await collect_stream_to_result(mock_response, first_token_timeout=30)
        
        print(f"Collected tool calls: {len(result.tool_calls)}")
        assert len(result.tool_calls) == 2
        print("✓ Tool calls deduplicated correctly")


# ==================================================================================================
# Tests for calculate_tokens_from_context_usage()
# ==================================================================================================

class TestCalculateTokensFromContextUsage:
    """Tests for calculate_tokens_from_context_usage() function."""
    
    def test_calculates_tokens_from_percentage(self, mock_model_cache):
        """
        What it does: Calculates tokens from context usage percentage.
        Goal: Verify token calculation is correct.
        """
        print("Setup: Context usage 10% with 200000 max tokens...")
        context_usage_percentage = 10.0
        completion_tokens = 100
        
        print("Action: Calculating tokens...")
        prompt_tokens, total_tokens, prompt_source, total_source = calculate_tokens_from_context_usage(
            context_usage_percentage, completion_tokens, mock_model_cache, "claude-sonnet-4"
        )
        
        # 10% of 200000 = 20000 total tokens
        # prompt_tokens = 20000 - 100 = 19900
        print(f"Comparing total_tokens: Expected 20000, Got {total_tokens}")
        assert total_tokens == 20000
        print(f"Comparing prompt_tokens: Expected 19900, Got {prompt_tokens}")
        assert prompt_tokens == 19900
        assert prompt_source == "subtraction"
        assert total_source == "API Kiro"
        print("✓ Tokens calculated correctly")
    
    def test_handles_zero_percentage(self, mock_model_cache):
        """
        What it does: Handles zero context usage percentage.
        Goal: Verify fallback behavior for zero percentage.
        """
        print("Setup: Context usage 0%...")
        context_usage_percentage = 0.0
        completion_tokens = 100
        
        print("Action: Calculating tokens...")
        prompt_tokens, total_tokens, prompt_source, total_source = calculate_tokens_from_context_usage(
            context_usage_percentage, completion_tokens, mock_model_cache, "claude-sonnet-4"
        )
        
        print(f"Comparing prompt_tokens: Expected 0, Got {prompt_tokens}")
        assert prompt_tokens == 0
        print(f"Comparing total_tokens: Expected 100, Got {total_tokens}")
        assert total_tokens == 100
        assert prompt_source == "unknown"
        assert total_source == "tiktoken"
        print("✓ Zero percentage handled correctly")
    
    def test_handles_none_percentage(self, mock_model_cache):
        """
        What it does: Handles None context usage percentage.
        Goal: Verify fallback behavior for None percentage.
        """
        print("Setup: Context usage None...")
        context_usage_percentage = None
        completion_tokens = 100
        
        print("Action: Calculating tokens...")
        prompt_tokens, total_tokens, prompt_source, total_source = calculate_tokens_from_context_usage(
            context_usage_percentage, completion_tokens, mock_model_cache, "claude-sonnet-4"
        )
        
        print(f"Comparing prompt_tokens: Expected 0, Got {prompt_tokens}")
        assert prompt_tokens == 0
        print(f"Comparing total_tokens: Expected 100, Got {total_tokens}")
        assert total_tokens == 100
        assert prompt_source == "unknown"
        assert total_source == "tiktoken"
        print("✓ None percentage handled correctly")
    
    def test_prevents_negative_prompt_tokens(self, mock_model_cache):
        """
        What it does: Prevents negative prompt tokens.
        Goal: Verify prompt_tokens is never negative.
        """
        print("Setup: Very small context usage with large completion...")
        context_usage_percentage = 0.01  # 0.01% of 200000 = 20 total tokens
        completion_tokens = 100  # More than total!
        
        print("Action: Calculating tokens...")
        prompt_tokens, total_tokens, prompt_source, total_source = calculate_tokens_from_context_usage(
            context_usage_percentage, completion_tokens, mock_model_cache, "claude-sonnet-4"
        )
        
        print(f"Comparing prompt_tokens: Expected >= 0, Got {prompt_tokens}")
        assert prompt_tokens >= 0
        print("✓ Negative prompt tokens prevented")
    
    def test_uses_model_specific_max_tokens(self, mock_model_cache):
        """
        What it does: Uses model-specific max input tokens.
        Goal: Verify model cache is queried correctly.
        """
        print("Setup: Different max tokens for model...")
        mock_model_cache.get_max_input_tokens.return_value = 100000  # Different from default
        context_usage_percentage = 10.0
        completion_tokens = 100
        
        print("Action: Calculating tokens...")
        prompt_tokens, total_tokens, prompt_source, total_source = calculate_tokens_from_context_usage(
            context_usage_percentage, completion_tokens, mock_model_cache, "claude-haiku-3"
        )
        
        # 10% of 100000 = 10000 total tokens
        print(f"Comparing total_tokens: Expected 10000, Got {total_tokens}")
        assert total_tokens == 10000
        
        # Verify model cache was called with correct model
        mock_model_cache.get_max_input_tokens.assert_called_with("claude-haiku-3")
        print("✓ Model-specific max tokens used correctly")
    
    def test_small_percentage_calculation(self, mock_model_cache):
        """
        What it does: Calculates tokens for small percentage.
        Goal: Verify precision for small percentages.
        """
        print("Setup: Context usage 0.5%...")
        context_usage_percentage = 0.5
        completion_tokens = 50
        
        print("Action: Calculating tokens...")
        prompt_tokens, total_tokens, prompt_source, total_source = calculate_tokens_from_context_usage(
            context_usage_percentage, completion_tokens, mock_model_cache, "claude-sonnet-4"
        )
        
        # 0.5% of 200000 = 1000 total tokens
        # prompt_tokens = 1000 - 50 = 950
        print(f"Comparing total_tokens: Expected 1000, Got {total_tokens}")
        assert total_tokens == 1000
        print(f"Comparing prompt_tokens: Expected 950, Got {prompt_tokens}")
        assert prompt_tokens == 950
        print("✓ Small percentage calculated correctly")
    
    def test_large_percentage_calculation(self, mock_model_cache):
        """
        What it does: Calculates tokens for large percentage.
        Goal: Verify calculation for high context usage.
        """
        print("Setup: Context usage 95%...")
        context_usage_percentage = 95.0
        completion_tokens = 1000
        
        print("Action: Calculating tokens...")
        prompt_tokens, total_tokens, prompt_source, total_source = calculate_tokens_from_context_usage(
            context_usage_percentage, completion_tokens, mock_model_cache, "claude-sonnet-4"
        )
        
        # 95% of 200000 = 190000 total tokens
        # prompt_tokens = 190000 - 1000 = 189000
        print(f"Comparing total_tokens: Expected 190000, Got {total_tokens}")
        assert total_tokens == 190000
        print(f"Comparing prompt_tokens: Expected 189000, Got {prompt_tokens}")
        assert prompt_tokens == 189000
        print("✓ Large percentage calculated correctly")


# ==================================================================================================
# Tests for thinking parser integration
# ==================================================================================================

class TestThinkingParserIntegration:
    """Tests for thinking parser integration in streaming."""
    
    @pytest.mark.asyncio
    async def test_thinking_parser_enabled_when_fake_reasoning_on(self, mock_response, mock_parser):
        """
        What it does: Enables thinking parser when FAKE_REASONING_ENABLED is True.
        Goal: Verify thinking parser is created.
        """
        print("Setup: Enable fake reasoning...")
        mock_parser.feed.return_value = [{"type": "content", "data": "Hello"}]
        mock_parser.get_tool_calls.return_value = []
        
        async def mock_aiter_bytes():
            yield b'chunk1'
        
        mock_response.aiter_bytes = mock_aiter_bytes
        
        print("Action: Parsing stream with fake reasoning enabled...")
        events = []
        
        with patch('kiro.streaming_core.AwsEventStreamParser', return_value=mock_parser):
            with patch('kiro.streaming_core.FAKE_REASONING_ENABLED', True):
                with patch('kiro.streaming_core.ThinkingParser') as mock_thinking_parser_class:
                    mock_thinking_parser = MagicMock()
                    mock_thinking_parser.feed.return_value = MagicMock(
                        thinking_content=None,
                        regular_content="Hello",
                        is_first_thinking_chunk=False,
                        is_last_thinking_chunk=False
                    )
                    mock_thinking_parser.finalize.return_value = MagicMock(
                        thinking_content=None,
                        regular_content=None,
                        is_first_thinking_chunk=False,
                        is_last_thinking_chunk=False
                    )
                    mock_thinking_parser.found_thinking_block = False
                    mock_thinking_parser_class.return_value = mock_thinking_parser
                    
                    async for event in parse_kiro_stream(mock_response, first_token_timeout=30):
                        events.append(event)
                    
                    # Verify ThinkingParser was instantiated
                    mock_thinking_parser_class.assert_called_once()
        
        print("✓ Thinking parser enabled when fake reasoning is on")
    
    @pytest.mark.asyncio
    async def test_thinking_parser_disabled_when_fake_reasoning_off(self, mock_response, mock_parser):
        """
        What it does: Disables thinking parser when FAKE_REASONING_ENABLED is False.
        Goal: Verify thinking parser is not created.
        """
        print("Setup: Disable fake reasoning...")
        mock_parser.feed.return_value = [{"type": "content", "data": "Hello"}]
        mock_parser.get_tool_calls.return_value = []
        
        async def mock_aiter_bytes():
            yield b'chunk1'
        
        mock_response.aiter_bytes = mock_aiter_bytes
        
        print("Action: Parsing stream with fake reasoning disabled...")
        events = []
        
        with patch('kiro.streaming_core.AwsEventStreamParser', return_value=mock_parser):
            with patch('kiro.streaming_core.FAKE_REASONING_ENABLED', False):
                with patch('kiro.streaming_core.ThinkingParser') as mock_thinking_parser_class:
                    async for event in parse_kiro_stream(mock_response, first_token_timeout=30):
                        events.append(event)
                    
                    # Verify ThinkingParser was NOT instantiated
                    mock_thinking_parser_class.assert_not_called()
        
        print("✓ Thinking parser disabled when fake reasoning is off")
    
    @pytest.mark.asyncio
    async def test_thinking_parser_can_be_disabled_via_parameter(self, mock_response, mock_parser):
        """
        What it does: Disables thinking parser via enable_thinking_parser parameter.
        Goal: Verify parameter overrides config.
        """
        print("Setup: Enable fake reasoning but disable via parameter...")
        mock_parser.feed.return_value = [{"type": "content", "data": "Hello"}]
        mock_parser.get_tool_calls.return_value = []
        
        async def mock_aiter_bytes():
            yield b'chunk1'
        
        mock_response.aiter_bytes = mock_aiter_bytes
        
        print("Action: Parsing stream with thinking parser disabled via parameter...")
        events = []
        
        with patch('kiro.streaming_core.AwsEventStreamParser', return_value=mock_parser):
            with patch('kiro.streaming_core.FAKE_REASONING_ENABLED', True):
                with patch('kiro.streaming_core.ThinkingParser') as mock_thinking_parser_class:
                    async for event in parse_kiro_stream(
                        mock_response,
                        first_token_timeout=30,
                        enable_thinking_parser=False
                    ):
                        events.append(event)
                    
                    # Verify ThinkingParser was NOT instantiated
                    mock_thinking_parser_class.assert_not_called()
        
        print("✓ Thinking parser disabled via parameter")


# ==================================================================================================
# Tests for error handling
# ==================================================================================================

class TestStreamingCoreErrorHandling:
    """Tests for error handling in streaming_core."""
    
    @pytest.mark.asyncio
    async def test_propagates_first_token_timeout_error(self, mock_response):
        """
        What it does: Propagates FirstTokenTimeoutError.
        Goal: Verify timeout error is not caught internally.
        """
        print("Setup: Mock response that times out...")
        
        async def mock_aiter_bytes():
            yield b'chunk'
        
        mock_response.aiter_bytes = mock_aiter_bytes
        
        async def mock_wait_for_timeout(*args, **kwargs):
            raise asyncio.TimeoutError()
        
        print("Action: Parsing stream with timeout...")
        
        with patch('kiro.streaming_core.asyncio.wait_for', side_effect=mock_wait_for_timeout):
            with pytest.raises(FirstTokenTimeoutError):
                async for event in parse_kiro_stream(mock_response, first_token_timeout=30):
                    pass
        
        print("✓ FirstTokenTimeoutError propagated correctly")
    
    @pytest.mark.asyncio
    async def test_propagates_generator_exit(self, mock_response, mock_parser):
        """
        What it does: Propagates GeneratorExit.
        Goal: Verify client disconnect is handled.
        """
        print("Setup: Mock response that raises GeneratorExit...")
        
        async def mock_aiter_bytes():
            yield b'chunk1'
            raise GeneratorExit()
        
        mock_response.aiter_bytes = mock_aiter_bytes
        mock_parser.feed.return_value = [{"type": "content", "data": "Hello"}]
        
        print("Action: Parsing stream with GeneratorExit...")
        
        with patch('kiro.streaming_core.AwsEventStreamParser', return_value=mock_parser):
            with patch('kiro.streaming_core.FAKE_REASONING_ENABLED', False):
                with pytest.raises(GeneratorExit):
                    async for event in parse_kiro_stream(mock_response, first_token_timeout=30):
                        pass
        
        print("✓ GeneratorExit propagated correctly")
    
    @pytest.mark.asyncio
    async def test_propagates_other_exceptions(self, mock_response, mock_parser):
        """
        What it does: Propagates other exceptions.
        Goal: Verify errors are not swallowed.
        """
        print("Setup: Mock response that raises RuntimeError...")
        
        async def mock_aiter_bytes():
            yield b'chunk1'
            raise RuntimeError("Test error")
        
        mock_response.aiter_bytes = mock_aiter_bytes
        mock_parser.feed.return_value = [{"type": "content", "data": "Hello"}]
        
        print("Action: Parsing stream with RuntimeError...")
        
        with patch('kiro.streaming_core.AwsEventStreamParser', return_value=mock_parser):
            with patch('kiro.streaming_core.FAKE_REASONING_ENABLED', False):
                with pytest.raises(RuntimeError) as exc_info:
                    async for event in parse_kiro_stream(mock_response, first_token_timeout=30):
                        pass
        
        print(f"Caught exception: {exc_info.value}")
        assert "Test error" in str(exc_info.value)
        print("✓ RuntimeError propagated correctly")


# ==================================================================================================
# Tests for stream_with_first_token_retry()
# ==================================================================================================

@pytest.mark.usefixtures("no_empty_response_guard")
class TestStreamWithFirstTokenRetryCore:
    """
    Tests for stream_with_first_token_retry() generic function.
    
    This function provides automatic retry logic on first token timeout.
    It is used by both OpenAI and Anthropic streaming implementations.
    """
    
    @pytest.mark.asyncio
    async def test_yields_chunks_on_success(self):
        """
        What it does: Yields chunks on successful streaming.
        Goal: Verify normal operation without retries.
        """
        print("Setup: Mock successful request...")
        
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.aclose = AsyncMock()
        
        async def mock_make_request():
            return mock_response
        
        async def mock_stream_processor(response):
            yield "chunk1"
            yield "chunk2"
            yield "chunk3"
        
        print("Action: Streaming with retry wrapper...")
        chunks = []
        
        async for chunk in stream_with_first_token_retry(
            make_request=mock_make_request,
            stream_processor=mock_stream_processor,
            max_retries=3,
            first_token_timeout=30
        ):
            chunks.append(chunk)
        
        print(f"Received {len(chunks)} chunks")
        assert len(chunks) == 3
        assert chunks == ["chunk1", "chunk2", "chunk3"]
        print("✓ Chunks yielded on success")
    
    @pytest.mark.asyncio
    async def test_retries_on_first_token_timeout(self):
        """
        What it does: Retries on first token timeout.
        Goal: Verify retry logic is triggered.
        """
        print("Setup: Mock request that times out then succeeds...")
        
        call_count = 0
        
        async def mock_make_request():
            nonlocal call_count
            call_count += 1
            response = AsyncMock()
            response.status_code = 200
            response.aclose = AsyncMock()
            return response
        
        async def mock_stream_processor(response):
            nonlocal call_count
            if call_count == 1:
                raise FirstTokenTimeoutError("Timeout on first attempt")
            yield "success_chunk"
        
        print("Action: Streaming with retry on timeout...")
        chunks = []
        
        async for chunk in stream_with_first_token_retry(
            make_request=mock_make_request,
            stream_processor=mock_stream_processor,
            max_retries=3,
            first_token_timeout=30
        ):
            chunks.append(chunk)
        
        print(f"Call count: {call_count}")
        print(f"Received {len(chunks)} chunks")
        
        assert call_count == 2  # First timeout, second success
        assert len(chunks) == 1
        assert chunks[0] == "success_chunk"
        print("✓ Retry on timeout works correctly")
    
    @pytest.mark.asyncio
    async def test_raises_exception_after_all_retries(self):
        """
        What it does: Raises exception after all retries exhausted.
        Goal: Verify error handling when all retries fail.
        """
        print("Setup: Mock request that always times out...")
        
        call_count = 0
        
        async def mock_make_request():
            nonlocal call_count
            call_count += 1
            response = AsyncMock()
            response.status_code = 200
            response.aclose = AsyncMock()
            return response
        
        async def mock_stream_processor(response):
            raise FirstTokenTimeoutError("Timeout!")
            yield  # Make it a generator
        
        print("Action: Streaming with all retries failing...")
        
        with pytest.raises(Exception) as exc_info:
            async for chunk in stream_with_first_token_retry(
                make_request=mock_make_request,
                stream_processor=mock_stream_processor,
                max_retries=3,
                first_token_timeout=30
            ):
                pass
        
        print(f"Call count: {call_count}")
        print(f"Exception: {exc_info.value}")
        
        assert call_count == 3  # Should try exactly 3 times
        assert "30" in str(exc_info.value)  # Timeout value in message
        assert "3" in str(exc_info.value)  # Retry count in message
        print("✓ Exception raised after all retries")
    
    @pytest.mark.asyncio
    async def test_uses_custom_error_callbacks(self):
        """
        What it does: Uses custom error callbacks.
        Goal: Verify on_http_error and on_all_retries_failed callbacks.
        """
        print("Setup: Mock request that always times out with custom callbacks...")
        
        async def mock_make_request():
            response = AsyncMock()
            response.status_code = 200
            response.aclose = AsyncMock()
            return response
        
        async def mock_stream_processor(response):
            raise FirstTokenTimeoutError("Timeout!")
            yield  # Make it a generator
        
        def custom_all_retries_failed(max_retries, timeout):
            return ValueError(f"Custom error: {max_retries} retries, {timeout}s timeout")
        
        print("Action: Streaming with custom callback...")
        
        with pytest.raises(ValueError) as exc_info:
            async for chunk in stream_with_first_token_retry(
                make_request=mock_make_request,
                stream_processor=mock_stream_processor,
                max_retries=2,
                first_token_timeout=15,
                on_all_retries_failed=custom_all_retries_failed
            ):
                pass
        
        print(f"Exception: {exc_info.value}")
        assert "Custom error" in str(exc_info.value)
        assert "2 retries" in str(exc_info.value)
        assert "15" in str(exc_info.value)
        print("✓ Custom callback used correctly")
    
    @pytest.mark.asyncio
    async def test_handles_http_error(self):
        """
        What it does: Handles HTTP error from API.
        Goal: Verify HTTP errors are handled correctly.
        """
        print("Setup: Mock request that returns HTTP error...")
        
        async def mock_make_request():
            response = AsyncMock()
            response.status_code = 500
            response.aread = AsyncMock(return_value=b"Internal Server Error")
            response.aclose = AsyncMock()
            return response
        
        async def mock_stream_processor(response):
            yield "should not reach"
        
        print("Action: Streaming with HTTP error...")
        
        with pytest.raises(Exception) as exc_info:
            async for chunk in stream_with_first_token_retry(
                make_request=mock_make_request,
                stream_processor=mock_stream_processor,
                max_retries=3,
                first_token_timeout=30
            ):
                pass
        
        print(f"Exception: {exc_info.value}")
        assert "500" in str(exc_info.value)
        assert "Internal Server Error" in str(exc_info.value)
        print("✓ HTTP error handled correctly")
    
    @pytest.mark.asyncio
    async def test_uses_custom_http_error_callback(self):
        """
        What it does: Uses custom HTTP error callback.
        Goal: Verify on_http_error callback is used.
        """
        print("Setup: Mock request with custom HTTP error callback...")
        
        async def mock_make_request():
            response = AsyncMock()
            response.status_code = 429
            response.aread = AsyncMock(return_value=b"Rate limited")
            response.aclose = AsyncMock()
            return response
        
        async def mock_stream_processor(response):
            yield "should not reach"
        
        def custom_http_error(status_code, error_text):
            return RuntimeError(f"Custom HTTP error: {status_code} - {error_text}")
        
        print("Action: Streaming with custom HTTP error callback...")
        
        with pytest.raises(RuntimeError) as exc_info:
            async for chunk in stream_with_first_token_retry(
                make_request=mock_make_request,
                stream_processor=mock_stream_processor,
                max_retries=3,
                first_token_timeout=30,
                on_http_error=custom_http_error
            ):
                pass
        
        print(f"Exception: {exc_info.value}")
        assert "Custom HTTP error" in str(exc_info.value)
        assert "429" in str(exc_info.value)
        assert "Rate limited" in str(exc_info.value)
        print("✓ Custom HTTP error callback used correctly")
    
    @pytest.mark.asyncio
    async def test_closes_response_on_timeout(self):
        """
        What it does: Closes response on timeout.
        Goal: Verify response is properly closed after timeout.
        """
        print("Setup: Mock request that times out...")
        
        responses = []
        
        async def mock_make_request():
            response = AsyncMock()
            response.status_code = 200
            response.aclose = AsyncMock()
            responses.append(response)
            return response
        
        async def mock_stream_processor(response):
            raise FirstTokenTimeoutError("Timeout!")
            yield  # Make it a generator
        
        print("Action: Streaming with timeout...")
        
        try:
            async for chunk in stream_with_first_token_retry(
                make_request=mock_make_request,
                stream_processor=mock_stream_processor,
                max_retries=2,
                first_token_timeout=30
            ):
                pass
        except Exception:
            pass
        
        print(f"Created {len(responses)} responses")
        
        # All responses should have been closed
        for i, response in enumerate(responses):
            print(f"Response {i} aclose called: {response.aclose.called}")
            response.aclose.assert_called()
        
        print("✓ Responses closed on timeout")
    
    @pytest.mark.asyncio
    async def test_propagates_non_timeout_exceptions(self):
        """
        What it does: Propagates non-timeout exceptions without retry.
        Goal: Verify other exceptions are not retried.
        """
        print("Setup: Mock request that raises RuntimeError...")
        
        call_count = 0
        
        async def mock_make_request():
            nonlocal call_count
            call_count += 1
            response = AsyncMock()
            response.status_code = 200
            response.aclose = AsyncMock()
            return response
        
        async def mock_stream_processor(response):
            raise RuntimeError("Not a timeout error")
            yield  # Make it a generator
        
        print("Action: Streaming with non-timeout error...")
        
        with pytest.raises(RuntimeError) as exc_info:
            async for chunk in stream_with_first_token_retry(
                make_request=mock_make_request,
                stream_processor=mock_stream_processor,
                max_retries=3,
                first_token_timeout=30
            ):
                pass
        
        print(f"Call count: {call_count}")
        print(f"Exception: {exc_info.value}")
        
        assert call_count == 1  # Should NOT retry
        assert "Not a timeout error" in str(exc_info.value)
        print("✓ Non-timeout exceptions propagated without retry")
    
    @pytest.mark.asyncio
    async def test_uses_configured_max_retries(self):
        """
        What it does: Uses configured max_retries value.
        Goal: Verify max_retries parameter is respected.
        """
        print("Setup: Mock request that always times out...")
        
        call_count = 0
        
        async def mock_make_request():
            nonlocal call_count
            call_count += 1
            response = AsyncMock()
            response.status_code = 200
            response.aclose = AsyncMock()
            return response
        
        async def mock_stream_processor(response):
            raise FirstTokenTimeoutError("Timeout!")
            yield  # Make it a generator
        
        print("Action: Streaming with max_retries=5...")
        
        try:
            async for chunk in stream_with_first_token_retry(
                make_request=mock_make_request,
                stream_processor=mock_stream_processor,
                max_retries=5,
                first_token_timeout=30
            ):
                pass
        except Exception:
            pass
        
        print(f"Call count: {call_count}")
        assert call_count == 5  # Should try exactly 5 times
        print("✓ max_retries parameter respected")
    
    @pytest.mark.asyncio
    async def test_multiple_retries_then_success(self):
        """
        What it does: Succeeds after multiple retries.
        Goal: Verify recovery after multiple failures.
        """
        print("Setup: Mock request that fails twice then succeeds...")
        
        call_count = 0
        
        async def mock_make_request():
            nonlocal call_count
            call_count += 1
            response = AsyncMock()
            response.status_code = 200
            response.aclose = AsyncMock()
            return response
        
        async def mock_stream_processor(response):
            nonlocal call_count
            if call_count < 3:
                raise FirstTokenTimeoutError(f"Timeout on attempt {call_count}")
            yield "finally_success"
        
        print("Action: Streaming with multiple retries...")
        chunks = []
        
        async for chunk in stream_with_first_token_retry(
            make_request=mock_make_request,
            stream_processor=mock_stream_processor,
            max_retries=5,
            first_token_timeout=30
        ):
            chunks.append(chunk)
        
        print(f"Call count: {call_count}")
        print(f"Received {len(chunks)} chunks")
        
        assert call_count == 3  # Failed twice, succeeded on third
        assert len(chunks) == 1
        assert chunks[0] == "finally_success"
        print("✓ Multiple retries then success works correctly")
    
    @pytest.mark.asyncio
    async def test_closes_response_on_http_error(self):
        """
        What it does: Closes response on HTTP error.
        Goal: Verify response is properly closed after HTTP error.
        """
        print("Setup: Mock request that returns HTTP error...")
        
        response = AsyncMock()
        response.status_code = 503
        response.aread = AsyncMock(return_value=b"Service Unavailable")
        response.aclose = AsyncMock()
        
        async def mock_make_request():
            return response
        
        async def mock_stream_processor(resp):
            yield "should not reach"
        
        print("Action: Streaming with HTTP error...")
        
        try:
            async for chunk in stream_with_first_token_retry(
                make_request=mock_make_request,
                stream_processor=mock_stream_processor,
                max_retries=3,
                first_token_timeout=30
            ):
                pass
        except Exception:
            pass
        
        print(f"Response aclose called: {response.aclose.called}")
        response.aclose.assert_called()
        print("✓ Response closed on HTTP error")
    
    @pytest.mark.asyncio
    async def test_reuses_initial_response_on_first_attempt(self):
        """
        What it does: Reuses initial_response on first attempt.
        Goal: Verify initial_response is used instead of calling make_request.
        """
        print("Setup: Mock initial response...")
        
        initial_response = AsyncMock()
        initial_response.status_code = 200
        initial_response.aclose = AsyncMock()
        
        make_request_called = False
        
        async def mock_make_request():
            nonlocal make_request_called
            make_request_called = True
            response = AsyncMock()
            response.status_code = 200
            response.aclose = AsyncMock()
            return response
        
        async def mock_stream_processor(response):
            # Verify we got the initial response
            assert response is initial_response
            yield "chunk_from_initial"
        
        print("Action: Streaming with initial_response...")
        chunks = []
        
        async for chunk in stream_with_first_token_retry(
            make_request=mock_make_request,
            stream_processor=mock_stream_processor,
            initial_response=initial_response,
            max_retries=3,
            first_token_timeout=30
        ):
            chunks.append(chunk)
        
        print(f"make_request called: {make_request_called}")
        print(f"Received {len(chunks)} chunks")
        
        # make_request should NOT be called on first attempt
        assert not make_request_called
        assert len(chunks) == 1
        assert chunks[0] == "chunk_from_initial"
        print("✓ initial_response reused on first attempt")
    
    @pytest.mark.asyncio
    async def test_calls_make_request_on_retry(self):
        """
        What it does: Calls make_request on retry attempts.
        Goal: Verify make_request is called when retrying after timeout.
        """
        print("Setup: Mock initial response that times out...")
        
        initial_response = AsyncMock()
        initial_response.status_code = 200
        initial_response.aclose = AsyncMock()
        
        make_request_call_count = 0
        
        async def mock_make_request():
            nonlocal make_request_call_count
            make_request_call_count += 1
            print(f"make_request called (retry attempt {make_request_call_count})")
            response = AsyncMock()
            response.status_code = 200
            response.aclose = AsyncMock()
            return response
        
        attempt_count = 0
        
        async def mock_stream_processor(response):
            nonlocal attempt_count
            attempt_count += 1
            
            if attempt_count == 1:
                # First attempt with initial_response - timeout
                assert response is initial_response
                raise FirstTokenTimeoutError("Timeout on first attempt")
            else:
                # Retry attempts - should get response from make_request
                assert response is not initial_response
                yield "success_chunk"
        
        print("Action: Streaming with initial_response that times out...")
        chunks = []
        
        async for chunk in stream_with_first_token_retry(
            make_request=mock_make_request,
            stream_processor=mock_stream_processor,
            initial_response=initial_response,
            max_retries=3,
            first_token_timeout=30
        ):
            chunks.append(chunk)
        
        print(f"make_request call count: {make_request_call_count}")
        print(f"Total attempts: {attempt_count}")
        print(f"Received {len(chunks)} chunks")
        
        # make_request should be called once (on retry)
        assert make_request_call_count == 1
        assert attempt_count == 2  # First with initial_response, second with make_request
        assert len(chunks) == 1
        print("✓ make_request called on retry")
    
    @pytest.mark.asyncio
    async def test_initial_response_none_calls_make_request_immediately(self):
        """
        What it does: Calls make_request immediately when initial_response is None.
        Goal: Verify backward compatibility (old behavior).
        """
        print("Setup: No initial_response (None)...")
        
        make_request_call_count = 0
        
        async def mock_make_request():
            nonlocal make_request_call_count
            make_request_call_count += 1
            print(f"make_request called (attempt {make_request_call_count})")
            response = AsyncMock()
            response.status_code = 200
            response.aclose = AsyncMock()
            return response
        
        async def mock_stream_processor(response):
            yield "chunk"
        
        print("Action: Streaming without initial_response...")
        chunks = []
        
        async for chunk in stream_with_first_token_retry(
            make_request=mock_make_request,
            stream_processor=mock_stream_processor,
            initial_response=None,  # Explicitly None
            max_retries=3,
            first_token_timeout=30
        ):
            chunks.append(chunk)
        
        print(f"make_request call count: {make_request_call_count}")
        print(f"Received {len(chunks)} chunks")
        
        # make_request should be called immediately (old behavior)
        assert make_request_call_count == 1
        assert len(chunks) == 1
        print("✓ make_request called immediately when initial_response is None")


# ==================================================================================================
# Native reasoning (reasoningContentEvent) tests
# ==================================================================================================

SECRET_MARKER = "SECRET_REASONING_MARKER_7f3a"


def _bytes_response(mock_response, chunks, pulled=None):
    """Attaches an aiter_bytes yielding chunks to mock_response (optionally counting pulls)."""
    async def aiter_bytes():
        for chunk in chunks:
            if pulled is not None:
                pulled.append(1)
            yield chunk

    mock_response.aiter_bytes = aiter_bytes
    return mock_response


async def _collect_events(response, **kwargs):
    """Runs parse_kiro_stream and returns all KiroEvents."""
    return [e async for e in parse_kiro_stream(response, first_token_timeout=30, **kwargs)]


class TestNativeReasoningStats:
    """Tests for the NativeReasoningStats counters dataclass."""

    def test_defaults_are_zero(self):
        """
        What it does: Creates default stats.
        Goal: All counters start at zero.
        """
        stats = NativeReasoningStats()
        assert (stats.frames, stats.chars, stats.signature_frames, stats.discarded_frames) == (0, 0, 0, 0)

    def test_instances_do_not_share_state(self):
        """
        What it does: Mutates one instance.
        Goal: Another instance is untouched.
        """
        a, b = NativeReasoningStats(), NativeReasoningStats()
        a.frames += 5
        assert b.frames == 0


class TestKiroEventNativeReasoningFlag:
    """Tests for KiroEvent.is_native_reasoning."""

    def test_can_be_set_true(self):
        """
        What it does: Creates a thinking event flagged as native.
        Goal: Flag is stored.
        """
        event = KiroEvent(type="thinking", thinking_content="x", is_native_reasoning=True)
        assert event.is_native_reasoning is True


class TestProcessChunkNativeReasoning:
    """Tests for native reasoning handling inside _process_chunk()."""

    @staticmethod
    async def _run(mock_parser, events, thinking_parser=None, stats=None):
        mock_parser.feed.return_value = events
        if stats is None:
            return [e async for e in _process_chunk(mock_parser, b"x", thinking_parser)]
        return [e async for e in _process_chunk(mock_parser, b"x", thinking_parser, stats)]

    @pytest.mark.asyncio
    async def test_reasoning_event_becomes_native_thinking_event(self, mock_parser):
        """
        What it does: Feeds a reasoning parser event.
        Goal: Yields KiroEvent(thinking) flagged native, with first/last flags False.
        """
        with patch("kiro.streaming_core.FAKE_REASONING_HANDLING", "as_reasoning_content"):
            events = await self._run(mock_parser, [{"type": "reasoning", "data": "think"}])

        assert len(events) == 1
        assert events[0].type == "thinking"
        assert events[0].thinking_content == "think"
        assert events[0].is_native_reasoning is True
        assert events[0].is_first_thinking_chunk is False
        assert events[0].is_last_thinking_chunk is False
        assert events[0].content is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mode", ["as_reasoning_content", "strip_tags", "pass"])
    async def test_non_remove_modes_deliver_original_text(self, mock_parser, mode):
        """
        What it does: Applies each non-remove FAKE_REASONING_HANDLING mode.
        Goal: Text delivered unchanged (native reasoning has no tags to add or strip).
        """
        with patch("kiro.streaming_core.FAKE_REASONING_HANDLING", mode):
            events = await self._run(mock_parser, [{"type": "reasoning", "data": " spaced \n"}])

        assert [e.thinking_content for e in events] == [" spaced \n"]

    @pytest.mark.asyncio
    async def test_remove_mode_discards_and_counts(self, mock_parser):
        """
        What it does: Applies FAKE_REASONING_HANDLING=remove.
        Goal: No event is yielded; frames/chars/discarded counters are updated.
        """
        stats = NativeReasoningStats()
        with patch("kiro.streaming_core.FAKE_REASONING_HANDLING", "remove"):
            events = await self._run(
                mock_parser,
                [{"type": "reasoning", "data": "abc"}, {"type": "reasoning", "data": "de"}],
                stats=stats,
            )

        assert events == []
        assert (stats.frames, stats.chars, stats.discarded_frames) == (2, 5, 2)

    @pytest.mark.asyncio
    async def test_signature_event_is_counted_not_yielded(self, mock_parser):
        """
        What it does: Feeds a reasoning_signature event.
        Goal: No KiroEvent; signature_frames incremented.
        """
        stats = NativeReasoningStats()
        events = await self._run(mock_parser, [{"type": "reasoning_signature", "data": 40}], stats=stats)
        assert events == []
        assert stats.signature_frames == 1
        assert stats.frames == 0

    @pytest.mark.asyncio
    async def test_works_without_stats_argument(self, mock_parser):
        """
        What it does: Calls _process_chunk with the legacy three arguments.
        Goal: Backward compatible - no stats needed.
        """
        with patch("kiro.streaming_core.FAKE_REASONING_HANDLING", "remove"):
            assert await self._run(mock_parser, [{"type": "reasoning", "data": "a"}]) == []
        events = await self._run(
            mock_parser,
            [{"type": "reasoning", "data": "a"}, {"type": "reasoning_signature", "data": 3}],
        )
        assert [e.thinking_content for e in events] == ["a"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("parser_kind", ["none", "mock"])
    async def test_independent_of_thinking_parser(self, mock_parser, parser_kind):
        """
        What it does: Runs with thinking_parser=None and with a MagicMock parser.
        Goal: Native reasoning never touches the tag parser.
        """
        thinking_parser = None if parser_kind == "none" else MagicMock()
        events = await self._run(
            mock_parser, [{"type": "reasoning", "data": "r"}], thinking_parser=thinking_parser
        )

        assert [e.thinking_content for e in events] == ["r"]
        if thinking_parser is not None:
            thinking_parser.feed.assert_not_called()
            thinking_parser.process_for_output.assert_not_called()

    @pytest.mark.asyncio
    async def test_mixed_chunk_preserves_upstream_order(self, mock_parser):
        """
        What it does: reasoning, content, reasoning, usage, context_usage in one chunk.
        Goal: Output order equals upstream order.
        """
        events = await self._run(mock_parser, [
            {"type": "reasoning", "data": "r1"},
            {"type": "content", "data": "c1"},
            {"type": "reasoning", "data": "r2"},
            {"type": "usage", "data": 0.5},
            {"type": "context_usage", "data": 4.0},
        ])

        assert [e.type for e in events] == ["thinking", "content", "thinking", "usage", "context_usage"]
        assert events[0].thinking_content == "r1"
        assert events[2].thinking_content == "r2"

    @pytest.mark.asyncio
    async def test_unknown_event_type_is_ignored(self, mock_parser):
        """
        What it does: Feeds an unknown event type.
        Goal: Ignored, neighbours unaffected.
        """
        events = await self._run(mock_parser, [
            {"type": "mystery", "data": 1}, {"type": "reasoning", "data": "ok"},
        ])
        assert [e.thinking_content for e in events] == ["ok"]


class TestParseKiroStreamNativeReasoning:
    """parse_kiro_stream() with the REAL AwsEventStreamParser and real AWS frames."""

    TEXTS = ["I", "'ll", " check", ".\n\n", "中文推理", " 😀", " — done"]

    @pytest.fixture(autouse=True)
    def _config(self):
        """Pins reasoning config so a user .env cannot influence the tests."""
        with patch("kiro.streaming_core.FAKE_REASONING_ENABLED", True), \
                patch("kiro.streaming_core.FAKE_REASONING_HANDLING", "as_reasoning_content"):
            yield

    @pytest.mark.asyncio
    async def test_every_text_frame_becomes_exactly_one_thinking_event(self, mock_response, aws_frames):
        """
        What it does: N reasoning frames + signature + content.
        Goal: Exactly N thinking events (equal to the frame count), all flagged native, text intact.
        """
        data = b"".join(aws_frames.reasoning(t) for t in self.TEXTS)
        data += aws_frames.signature("QUJD") + aws_frames.content("answer")
        _bytes_response(mock_response, [data])

        events = await _collect_events(mock_response)

        thinking = [e for e in events if e.type == "thinking"]
        print(f"Thinking events: {len(thinking)} (frames: {len(self.TEXTS)})")
        assert len(thinking) == len(self.TEXTS)
        assert all(e.is_native_reasoning for e in thinking)
        assert [e.thinking_content for e in thinking] == self.TEXTS
        assert [e.content for e in events if e.type == "content"] == ["answer"]

    @pytest.mark.asyncio
    async def test_thinking_is_delivered_incrementally(self, mock_response, aws_frames):
        """
        What it does: One frame per network chunk; records how many chunks were pulled when
        each thinking event is yielded.
        Goal: The k-th thinking event is available before chunk k+1 is pulled - this is the
        bug fix (previously the client got zero bytes while reasoning).
        """
        pulled = []
        chunks = [aws_frames.reasoning(t) for t in self.TEXTS] + [aws_frames.content("end")]
        _bytes_response(mock_response, chunks, pulled)

        pulled_at_event = []
        async for event in parse_kiro_stream(mock_response, first_token_timeout=30):
            if event.type == "thinking":
                pulled_at_event.append(len(pulled))

        assert pulled_at_event == list(range(1, len(self.TEXTS) + 1))

    @pytest.mark.asyncio
    async def test_arbitrary_chunking_gives_identical_events(self, mock_response, aws_frames):
        """
        What it does: Same stream chunked 1 byte at a time and in odd sizes.
        Goal: Identical thinking text sequence.
        """
        data = b"".join(aws_frames.reasoning(t) for t in self.TEXTS) + aws_frames.content("tail")
        _bytes_response(mock_response, [data])
        baseline = [e.thinking_content for e in await _collect_events(mock_response) if e.type == "thinking"]

        for sizes in ([1], [2, 3], [7], [1000]):
            _bytes_response(mock_response, aws_frames.split(data, sizes))
            got = [e.thinking_content for e in await _collect_events(mock_response) if e.type == "thinking"]
            assert got == baseline, f"sizes {sizes}"

    @pytest.mark.asyncio
    async def test_reasoning_then_content_then_tool_use_order(self, mock_response, aws_frames):
        """
        What it does: reasoning -> content -> tool call frames.
        Goal: thinking before content before tool_use.
        """
        data = b"".join([
            aws_frames.reasoning("plan"),
            aws_frames.content("say"),
            aws_frames.tool_start("Bash", "toolu_1"),
            aws_frames.tool_input('{"command": "ls"}', "Bash", "toolu_1"),
            aws_frames.tool_stop("Bash", "toolu_1"),
        ])
        _bytes_response(mock_response, [data])

        events = await _collect_events(mock_response)

        assert [e.type for e in events] == ["thinking", "content", "tool_use"]
        assert events[2].tool_use["function"]["name"] == "Bash"

    @pytest.mark.asyncio
    async def test_reasoning_only_then_pure_tool_use(self, mock_response, aws_frames):
        """
        What it does: Reasoning frames followed only by a tool call (no text).
        Goal: thinking events first, then the tool_use at the end.
        """
        data = b"".join([
            aws_frames.reasoning("a"), aws_frames.reasoning("b"),
            aws_frames.tool_start("Read", "toolu_2"),
            aws_frames.tool_input('{"path": "x"}', "Read", "toolu_2"),
        ])
        _bytes_response(mock_response, [data])

        events = await _collect_events(mock_response)

        assert [e.type for e in events] == ["thinking", "thinking", "tool_use"]

    @pytest.mark.asyncio
    async def test_reasoning_only_stream(self, mock_response, aws_frames):
        """
        What it does: Only reasoning frames.
        Goal: Only thinking events; nothing else, no crash.
        """
        _bytes_response(mock_response, [aws_frames.reasoning("solo") + aws_frames.signature("QQ==")])
        events = await _collect_events(mock_response)
        assert [(e.type, e.thinking_content) for e in events] == [("thinking", "solo")]

    @pytest.mark.asyncio
    async def test_unknown_frames_only_yield_nothing(self, mock_response, aws_frames):
        """
        What it does: Unrecognized frame next to a meteringEvent (metering keeps the
        empty-response guard satisfied).
        Goal: Unknown frames yield no events; only the known metering event comes through.
        """
        _bytes_response(mock_response, [aws_frames.build("weirdEvent", {"foo": 1}) + aws_frames.metering(0.1)])
        events = await _collect_events(mock_response)
        assert [(e.type, e.usage) for e in events] == [("usage", 0.1)]

    @pytest.mark.asyncio
    async def test_empty_text_frames_yield_no_events(self, mock_response, aws_frames):
        """
        What it does: Empty-string reasoning frames.
        Goal: Skipped.
        """
        _bytes_response(mock_response, [aws_frames.reasoning("") + aws_frames.reasoning("")])
        assert await _collect_events(mock_response) == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("enabled,thinking_flag", [(False, True), (True, False), (False, False)])
    async def test_native_reasoning_ignores_fake_reasoning_switches(
        self, mock_response, aws_frames, enabled, thinking_flag
    ):
        """
        What it does: Disables FAKE_REASONING and/or enable_thinking_parser.
        Goal: Native reasoning is still surfaced (those switches only control tag handling).
        """
        _bytes_response(mock_response, [aws_frames.reasoning("r1") + aws_frames.reasoning("r2")])
        with patch("kiro.streaming_core.FAKE_REASONING_ENABLED", enabled):
            events = await _collect_events(mock_response, enable_thinking_parser=thinking_flag)

        assert [e.thinking_content for e in events] == ["r1", "r2"]

    @pytest.mark.asyncio
    async def test_remove_mode_drops_thinking_but_keeps_content(self, mock_response, aws_frames):
        """
        What it does: FAKE_REASONING_HANDLING=remove.
        Goal: Zero thinking events; content untouched.
        """
        _bytes_response(mock_response, [aws_frames.reasoning("hidden") + aws_frames.content("shown")])
        with patch("kiro.streaming_core.FAKE_REASONING_HANDLING", "remove"):
            events = await _collect_events(mock_response)

        assert [e.type for e in events] == ["content"]
        assert events[0].content == "shown"

    @pytest.mark.asyncio
    async def test_native_and_tag_reasoning_coexist(self, mock_response, aws_frames):
        """
        What it does: Native reasoning frames, then content that itself opens with <thinking>.
        Goal: Both become thinking events (native flagged, tag-based not); answer is content.
        """
        data = (
            aws_frames.reasoning("native")
            + aws_frames.content("<thinking>tagged</thinking>answer")
        )
        _bytes_response(mock_response, [data])

        events = await _collect_events(mock_response)

        thinking = [e for e in events if e.type == "thinking"]
        assert [(e.thinking_content, e.is_native_reasoning) for e in thinking] == [
            ("native", True), ("tagged", False),
        ]
        assert "".join(e.content for e in events if e.type == "content") == "answer"

    @pytest.mark.asyncio
    async def test_debug_summary_has_counts_but_never_reasoning_text(
        self, mock_response, aws_frames, loguru_messages
    ):
        """
        What it does: Streams reasoning containing a secret marker and checks all log output.
        Goal: One summary line with counts; the marker never appears in any log.
        """
        marker_text = f"{SECRET_MARKER} and more"
        data = aws_frames.reasoning(marker_text) + aws_frames.reasoning("x") + aws_frames.signature("QQ==")
        _bytes_response(mock_response, [data])

        await _collect_events(mock_response)

        joined = "\n".join(loguru_messages)
        summaries = [m for m in loguru_messages if "[NativeReasoning]" in m]
        print(f"Summaries: {summaries}")
        assert len(summaries) == 1
        assert "frames=2" in summaries[0]
        assert f"chars={len(marker_text) + 1}" in summaries[0]
        assert "signature_frames=1" in summaries[0]
        assert SECRET_MARKER not in joined

    @pytest.mark.asyncio
    async def test_no_summary_without_native_reasoning(self, mock_response, aws_frames, loguru_messages):
        """
        What it does: Stream without reasoning frames.
        Goal: No [NativeReasoning] log noise.
        """
        _bytes_response(mock_response, [aws_frames.content("plain")])
        await _collect_events(mock_response)
        assert not [m for m in loguru_messages if "[NativeReasoning]" in m]

    @pytest.mark.asyncio
    async def test_summary_still_logged_when_consumer_closes_early(
        self, mock_response, aws_frames, loguru_messages
    ):
        """
        What it does: Consumer closes the generator after the first event (client disconnect).
        Goal: The finally block still logs the summary and the close completes cleanly.
        """
        _bytes_response(mock_response, [aws_frames.reasoning("a"), aws_frames.reasoning("b")])

        agen = parse_kiro_stream(mock_response, first_token_timeout=30)
        first = await agen.__anext__()
        await agen.aclose()

        assert first.thinking_content == "a"
        assert any("[NativeReasoning]" in m and "frames=1" in m for m in loguru_messages)

    @pytest.mark.asyncio
    async def test_first_token_timeout_still_raised(self, mock_response):
        """
        What it does: First chunk never arrives.
        Goal: FirstTokenTimeoutError still propagates (the new finally must not swallow it).
        """
        async def aiter_bytes():
            yield b"x"

        mock_response.aiter_bytes = aiter_bytes

        async def timeout(*args, **kwargs):
            raise asyncio.TimeoutError()

        with patch("kiro.streaming_core.asyncio.wait_for", side_effect=timeout):
            with pytest.raises(FirstTokenTimeoutError):
                await _collect_events(mock_response)

    @pytest.mark.asyncio
    async def test_upstream_error_mid_stream_propagates_after_partial_reasoning(
        self, mock_response, aws_frames
    ):
        """
        What it does: Upstream raises after delivering some reasoning.
        Goal: Earlier thinking events were already delivered and the error propagates.
        """
        async def aiter_bytes():
            yield aws_frames.reasoning("before")
            raise RuntimeError("boom")

        mock_response.aiter_bytes = aiter_bytes
        seen = []

        with pytest.raises(RuntimeError, match="boom"):
            async for event in parse_kiro_stream(mock_response, first_token_timeout=30):
                seen.append(event.thinking_content)

        assert seen == ["before"]


class TestCollectStreamToResultNativeReasoning:
    """collect_stream_to_result() aggregation of native reasoning."""

    @pytest.fixture(autouse=True)
    def _config(self):
        with patch("kiro.streaming_core.FAKE_REASONING_ENABLED", True), \
                patch("kiro.streaming_core.FAKE_REASONING_HANDLING", "as_reasoning_content"):
            yield

    @pytest.mark.asyncio
    async def test_thinking_content_is_all_native_reasoning_concatenated(self, mock_response, aws_frames):
        """
        What it does: Collects a stream with reasoning, content, context usage.
        Goal: thinking_content == concatenation of reasoning frames; usage fields preserved.
        """
        data = b"".join([
            aws_frames.reasoning("one "), aws_frames.reasoning("two"),
            aws_frames.content("answer"), aws_frames.context_usage(5.0),
            aws_frames.build("usageEvent", {"usage": 0.25}),
        ])
        _bytes_response(mock_response, [data])

        result = await collect_stream_to_result(mock_response, first_token_timeout=30)

        assert result.thinking_content == "one two"
        assert result.content == "answer"
        assert result.context_usage_percentage == 5.0
        assert result.usage == 0.25

    @pytest.mark.asyncio
    async def test_bracket_syntax_inside_native_reasoning_is_not_a_tool_call(self, mock_response, aws_frames):
        """
        What it does: Native reasoning quotes "[Called x with args: {...}]".
        Goal: No tool call is fabricated from the model's inner monologue.
        """
        quote = '[Called evil_tool with args: {"a": 1}]'
        _bytes_response(mock_response, [aws_frames.reasoning(quote) + aws_frames.content("done")])

        result = await collect_stream_to_result(mock_response, first_token_timeout=30)

        assert result.thinking_content == quote
        assert result.tool_calls == []

    @pytest.mark.asyncio
    async def test_bracket_syntax_inside_tag_thinking_still_detected(self, mock_response, aws_frames):
        """
        What it does: Same quote inside a <thinking> tag block.
        Goal: Historical behavior is locked - tag-based thinking is still scanned.
        """
        quote = '[Called legacy_tool with args: {"a": 1}]'
        _bytes_response(mock_response, [aws_frames.content(f"<thinking>{quote}</thinking>ok")])

        result = await collect_stream_to_result(mock_response, first_token_timeout=30)

        assert [tc["function"]["name"] for tc in result.tool_calls] == ["legacy_tool"]

    @pytest.mark.asyncio
    async def test_remove_mode_yields_empty_thinking_content(self, mock_response, aws_frames):
        """
        What it does: FAKE_REASONING_HANDLING=remove.
        Goal: thinking_content is empty; content kept.
        """
        _bytes_response(mock_response, [aws_frames.reasoning("hidden") + aws_frames.content("shown")])
        with patch("kiro.streaming_core.FAKE_REASONING_HANDLING", "remove"):
            result = await collect_stream_to_result(mock_response, first_token_timeout=30)

        assert result.thinking_content == ""
        assert result.content == "shown"


# ==================================================================================================
# Empty upstream response guard
# ==================================================================================================

import httpx

from kiro.streaming_core import (
    STREAM_READY,
    StreamReadyMarker,
    guard_upstream_response,
    prime_sse_stream,
)
from kiro.upstream_guard import (
    EMPTY_SIGNATURE_MAX_BYTES,
    EmptyUpstreamResponseError,
    GuardedResponse,
)


def _stream_response(chunks):
    """
    Response double with a REAL one-shot byte stream (like httpx: aiter_bytes() works once).

    Args:
        chunks: List of bytes chunks, or exceptions to raise at that position

    Returns:
        AsyncMock response with status 200, aclose and a pull counter in `.pulled`
    """
    response = AsyncMock()
    response.status_code = 200
    response.aclose = AsyncMock()
    response.pulled = []
    state = {"calls": 0}

    async def gen():
        for chunk in chunks:
            if isinstance(chunk, BaseException):
                raise chunk
            response.pulled.append(len(chunk))
            yield chunk

    def aiter_bytes(*args, **kwargs):
        state["calls"] += 1
        if state["calls"] > 1:
            raise httpx.StreamConsumed()
        return gen()

    response.aiter_bytes = aiter_bytes
    return response


def _hanging_response():
    """Response double whose byte stream never produces a chunk (first token timeout)."""
    response = AsyncMock()
    response.status_code = 200
    response.aclose = AsyncMock()

    async def gen():
        await asyncio.sleep(3600)
        yield b""

    response.aiter_bytes = lambda *a, **k: gen()
    return response


def _success_bytes(aws_frames):
    """A realistic successful stream: reasoning, content, contextUsage, metering."""
    return (
        aws_frames.reasoning("Let me think") + aws_frames.reasoning(" more")
        + aws_frames.content("Hello") + aws_frames.content(" world")
        + aws_frames.context_usage(12.5) + aws_frames.metering(0.75)
    )


@pytest.fixture
def guard_on():
    """Enables the guard explicitly (it is also the default)."""
    with patch("kiro.streaming_core.EMPTY_RESPONSE_AS_ERROR", True):
        yield


@pytest.mark.usefixtures("guard_on")
class TestGuardUpstreamResponse:
    """Tests for guard_upstream_response()."""

    @pytest.mark.asyncio
    async def test_context_usage_only_raises_empty_error(self, aws_frames):
        """
        What it does: Upstream stream is a single contextUsageEvent frame.
        Purpose: The real failure shape must be rejected, with diagnostics attached.
        """
        frame = aws_frames.context_usage(60.73619842529297)
        response = _stream_response([frame])

        with pytest.raises(EmptyUpstreamResponseError) as exc_info:
            await guard_upstream_response(response, first_token_timeout=5)

        print(f"error: {exc_info.value}")
        assert exc_info.value.bytes_received == len(frame)
        assert exc_info.value.frame_counts == {"contextUsageEvent": 1}
        response.aclose.assert_awaited()

    @pytest.mark.asyncio
    async def test_zero_byte_stream_raises_empty_error(self):
        """
        What it does: Upstream stream ends without a single byte.
        Purpose: No bytes means neither output nor metering.
        """
        response = _stream_response([])

        with pytest.raises(EmptyUpstreamResponseError) as exc_info:
            await guard_upstream_response(response, first_token_timeout=5)

        assert exc_info.value.bytes_received == 0

    @pytest.mark.asyncio
    async def test_empty_frame_split_across_chunks_raises(self, aws_frames):
        """
        What it does: The empty-response frame arrives in 1-byte chunks.
        Purpose: Detection must not depend on network chunking.
        """
        frame = aws_frames.context_usage(60.7)
        response = _stream_response(aws_frames.split(frame, [1]))

        with pytest.raises(EmptyUpstreamResponseError) as exc_info:
            await guard_upstream_response(response, first_token_timeout=5)

        assert exc_info.value.frame_counts == {"contextUsageEvent": 1}

    @pytest.mark.asyncio
    async def test_metering_without_output_is_passed_through(self, aws_frames):
        """
        What it does: contextUsageEvent + meteringEvent, no output frame.
        Purpose: A metered empty answer is the model's real answer: never an error.
        The metering frame itself is passed through as a usage event.
        """
        data = aws_frames.context_usage(1.0) + aws_frames.metering(0.5)
        response = _stream_response([data])

        guarded = await guard_upstream_response(response, first_token_timeout=5)

        assert isinstance(guarded, GuardedResponse)
        events = await _collect_events(guarded)
        assert [(e.type, e.usage) for e in events] == [("context_usage", None), ("usage", 0.5)]

    @pytest.mark.asyncio
    async def test_metering_split_across_chunks_is_passed_through(self, aws_frames):
        """
        What it does: The metering frame is split in the middle of its event type name.
        Purpose: Boundary-split metering must still be recognised.
        """
        data = aws_frames.context_usage(1.0) + aws_frames.metering(0.5)
        cut = data.index(b"meteringEvent") + 6
        response = _stream_response([data[:cut], data[cut:]])

        guarded = await guard_upstream_response(response, first_token_timeout=5)

        assert isinstance(guarded, GuardedResponse)

    @pytest.mark.asyncio
    async def test_reasoning_without_metering_is_passed_through(self, aws_frames):
        """
        What it does: reasoning frame + contextUsageEvent, no metering.
        Purpose: Output exists; the response must be delivered.
        """
        data = aws_frames.reasoning("hmm") + aws_frames.context_usage(2.0)
        response = _stream_response([data])

        guarded = await guard_upstream_response(response, first_token_timeout=5)
        events = await _collect_events(guarded)

        assert [e.type for e in events if e.type == "thinking"] == ["thinking"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("sizes", [[1], [2], [7], [13], [50], [147], [165], [1000], [3, 1, 5, 200]])
    async def test_guarded_events_equal_unguarded_events(self, aws_frames, sizes):
        """
        What it does: Runs the same stream through parse_kiro_stream with and without the guard,
        for many chunk sizes.
        Purpose: The guard must not lose, reorder or alter a single event.
        """
        data = _success_bytes(aws_frames)
        chunks = aws_frames.split(data, sizes)
        baseline = await _collect_events(_stream_response(chunks))

        guarded = await guard_upstream_response(_stream_response(chunks), first_token_timeout=5)
        replayed = await _collect_events(guarded)

        assert [asdict(e) for e in replayed] == [asdict(e) for e in baseline]
        assert any(e.type == "content" for e in replayed)

    @pytest.mark.asyncio
    async def test_first_output_frame_releases_without_reading_more(self, aws_frames):
        """
        What it does: Counts upstream pulls after the first content frame arrived.
        Purpose: The guard must not delay a real response (no extra buffering/waiting).
        """
        chunks = [aws_frames.content("a"), aws_frames.content("b"), aws_frames.content("c")]
        response = _stream_response(chunks)

        guarded = await guard_upstream_response(response, first_token_timeout=5)

        assert isinstance(guarded, GuardedResponse)
        assert len(response.pulled) == 1

    @pytest.mark.asyncio
    async def test_neutral_first_chunk_keeps_reading_until_output(self, aws_frames):
        """
        What it does: contextUsageEvent arrives first, content later (hypothetical ordering).
        Purpose: Only output/metering/unknown frames release; neutral frames keep peeking.
        """
        response = _stream_response([aws_frames.context_usage(1.0), aws_frames.content("late")])

        guarded = await guard_upstream_response(response, first_token_timeout=5)
        events = await _collect_events(guarded)

        assert [e.content for e in events if e.type == "content"] == ["late"]

    @pytest.mark.asyncio
    async def test_size_cap_releases_neutral_only_stream(self, aws_frames):
        """
        What it does: Neutral frames beyond EMPTY_SIGNATURE_MAX_BYTES, then the stream goes on.
        Purpose: The buffer stays bounded and large streams are never "empty".
        """
        frame = aws_frames.context_usage(1.0)
        count = EMPTY_SIGNATURE_MAX_BYTES // len(frame) + 2
        response = _stream_response([frame] * count + [aws_frames.content("x")])

        guarded = await guard_upstream_response(response, first_token_timeout=5)

        assert isinstance(guarded, GuardedResponse)

    @pytest.mark.asyncio
    async def test_unrecognised_framing_is_passed_through_with_warning(self, loguru_messages):
        """
        What it does: Stream of bare JSON with no AWS frame headers.
        Purpose: When the protocol changes the guard must step aside, and say so.
        """
        response = _stream_response([b'{"content":"Hello"}'])

        guarded = await guard_upstream_response(response, first_token_timeout=5)

        assert isinstance(guarded, GuardedResponse)
        assert any("no AWS event-stream frame header recognised" in m for m in loguru_messages)

    @pytest.mark.asyncio
    async def test_first_chunk_timeout_raises_first_token_timeout(self):
        """
        What it does: The upstream never sends a first chunk.
        Purpose: FIRST_TOKEN_TIMEOUT keeps its meaning.
        """
        response = _hanging_response()

        with pytest.raises(FirstTokenTimeoutError):
            await guard_upstream_response(response, first_token_timeout=0.05)

    @pytest.mark.asyncio
    async def test_no_timeout_applies_after_first_chunk(self, aws_frames):
        """
        What it does: Second chunk is slower than first_token_timeout.
        Purpose: Only the FIRST chunk is subject to the timeout (as in parse_kiro_stream).
        """
        frame = aws_frames.context_usage(1.0)
        response = AsyncMock()
        response.aclose = AsyncMock()

        async def gen():
            yield frame[:10]
            await asyncio.sleep(0.2)
            yield frame[10:]

        response.aiter_bytes = lambda *a, **k: gen()

        with pytest.raises(EmptyUpstreamResponseError):
            await guard_upstream_response(response, first_token_timeout=0.05)

    @pytest.mark.asyncio
    async def test_read_error_before_stream_end_propagates_unchanged(self, aws_frames):
        """
        What it does: Network error after a partial empty-looking prefix.
        Purpose: A failed read is not an empty response; the original error must surface.
        """
        frame = aws_frames.context_usage(1.0)
        response = _stream_response([frame, httpx.ReadError("connection reset")])

        with pytest.raises(httpx.ReadError):
            await guard_upstream_response(response, first_token_timeout=5)

    @pytest.mark.asyncio
    async def test_read_error_on_first_chunk_propagates_unchanged(self):
        """
        What it does: The very first read fails.
        Purpose: Same as above for the first chunk.
        """
        response = _stream_response([httpx.ReadError("boom")])

        with pytest.raises(httpx.ReadError):
            await guard_upstream_response(response, first_token_timeout=5)

    @pytest.mark.asyncio
    async def test_disabled_returns_original_and_reads_nothing(self, aws_frames):
        """
        What it does: enabled=False with an empty-response stream.
        Purpose: With the switch off the response must be untouched (native behavior).
        """
        response = _stream_response([aws_frames.context_usage(1.0)])

        result = await guard_upstream_response(response, first_token_timeout=5, enabled=False)

        assert result is response
        assert response.pulled == []

    @pytest.mark.asyncio
    async def test_config_default_is_read_at_call_time(self, aws_frames):
        """
        What it does: Patches the module level EMPTY_RESPONSE_AS_ERROR to False.
        Purpose: The config flag controls the guard when `enabled` is not passed.
        """
        response = _stream_response([aws_frames.context_usage(1.0)])

        with patch("kiro.streaming_core.EMPTY_RESPONSE_AS_ERROR", False):
            result = await guard_upstream_response(response, first_token_timeout=5)

        assert result is response

    @pytest.mark.asyncio
    async def test_aiter_bytes_of_original_response_called_once(self, aws_frames):
        """
        What it does: Guard + full parse over a real one-shot stream (StreamConsumed on 2nd call).
        Purpose: The replay wrapper must never ask httpx for a second iterator.
        """
        response = _stream_response([_success_bytes(aws_frames)])

        guarded = await guard_upstream_response(response, first_token_timeout=5)
        events = await _collect_events(guarded)

        assert events


async def _sse_processor(response):
    """stream_processor double: yields one SSE-like string per content event."""
    async for event in parse_kiro_stream(response, first_token_timeout=30):
        if event.type == "content":
            yield f"data:{event.content}"


@pytest.mark.usefixtures("guard_on")
class TestFirstTokenRetryWithGuard:
    """The guard lives inside stream_with_first_token_retry attempts."""

    @pytest.mark.asyncio
    async def test_ready_marker_then_chunks(self, aws_frames):
        """
        What it does: emit_ready_marker=True with a normal response.
        Purpose: Exactly one STREAM_READY, before all SSE chunks, content complete.
        """
        response = _stream_response(aws_frames.split(_success_bytes(aws_frames), [11]))

        items = [
            item async for item in stream_with_first_token_retry(
                make_request=AsyncMock(), stream_processor=_sse_processor,
                initial_response=response, max_retries=3, first_token_timeout=5,
                emit_ready_marker=True,
            )
        ]

        print(items)
        assert items[0] is STREAM_READY
        assert items[1:] == ["data:Hello", "data: world"]
        assert sum(1 for i in items if isinstance(i, StreamReadyMarker)) == 1

    @pytest.mark.asyncio
    async def test_no_marker_by_default(self, aws_frames):
        """
        What it does: emit_ready_marker left at its default.
        Purpose: Existing callers must not see a sentinel.
        """
        response = _stream_response([_success_bytes(aws_frames)])

        items = [
            item async for item in stream_with_first_token_retry(
                make_request=AsyncMock(), stream_processor=_sse_processor,
                initial_response=response, max_retries=3, first_token_timeout=5,
            )
        ]

        assert all(isinstance(i, str) for i in items)
        assert items == ["data:Hello", "data: world"]

    @pytest.mark.asyncio
    async def test_empty_response_raises_before_any_item_and_is_not_retried(self, aws_frames):
        """
        What it does: initial response is a lone contextUsageEvent frame.
        Purpose: EmptyUpstreamResponseError surfaces with nothing yielded and make_request
        untouched (no gateway-side retry).
        """
        response = _stream_response([aws_frames.context_usage(1.0)])
        make_request = AsyncMock()
        items = []

        with pytest.raises(EmptyUpstreamResponseError):
            async for item in stream_with_first_token_retry(
                make_request=make_request, stream_processor=_sse_processor,
                initial_response=response, max_retries=3, first_token_timeout=5,
                emit_ready_marker=True,
            ):
                items.append(item)

        assert items == []
        make_request.assert_not_called()
        response.aclose.assert_awaited()

    @pytest.mark.asyncio
    async def test_empty_response_logs_no_error_level(self, aws_frames, loguru_messages):
        """
        What it does: Empty response through the retry loop.
        Purpose: It is an expected condition: no ERROR line, no stack trace.
        """
        response = _stream_response([aws_frames.context_usage(1.0)])

        with pytest.raises(EmptyUpstreamResponseError):
            async for _ in stream_with_first_token_retry(
                make_request=AsyncMock(), stream_processor=_sse_processor,
                initial_response=response, max_retries=3, first_token_timeout=5,
            ):
                pass

        assert not [m for m in loguru_messages if m.startswith("ERROR|")]

    @pytest.mark.asyncio
    async def test_timeout_in_guard_is_retried_then_succeeds(self, aws_frames):
        """
        What it does: Attempt 1 hangs (guard timeout), attempt 2 is a normal response.
        Purpose: FIRST_TOKEN_TIMEOUT retry works exactly as before, marker emitted once.
        """
        hanging = _hanging_response()
        good = _stream_response([_success_bytes(aws_frames)])
        make_request = AsyncMock(return_value=good)

        items = [
            item async for item in stream_with_first_token_retry(
                make_request=make_request, stream_processor=_sse_processor,
                initial_response=hanging, max_retries=3, first_token_timeout=0.05,
                emit_ready_marker=True,
            )
        ]

        assert items[0] is STREAM_READY
        assert items[1:] == ["data:Hello", "data: world"]
        assert make_request.await_count == 1
        hanging.aclose.assert_awaited()

    @pytest.mark.asyncio
    async def test_timeout_then_empty_response_raises_empty_error(self, aws_frames):
        """
        What it does: Attempt 1 times out, the retried response is empty.
        Purpose: The retried response is peeked too; empty error, nothing yielded, both closed.
        """
        hanging = _hanging_response()
        empty = _stream_response([aws_frames.context_usage(1.0)])
        make_request = AsyncMock(return_value=empty)
        items = []

        with pytest.raises(EmptyUpstreamResponseError):
            async for item in stream_with_first_token_retry(
                make_request=make_request, stream_processor=_sse_processor,
                initial_response=hanging, max_retries=3, first_token_timeout=0.05,
                emit_ready_marker=True,
            ):
                items.append(item)

        assert items == []
        hanging.aclose.assert_awaited()
        empty.aclose.assert_awaited()
        assert make_request.await_count == 1

    @pytest.mark.asyncio
    async def test_all_attempts_timing_out_keeps_old_error_semantics(self):
        """
        What it does: Every attempt hangs.
        Purpose: on_all_retries_failed is invoked after max_retries attempts, as before.
        """
        make_request = AsyncMock(side_effect=lambda: _hanging_response())
        calls = []

        def on_all_failed(retries, timeout):
            calls.append((retries, timeout))
            return RuntimeError("all failed")

        with pytest.raises(RuntimeError, match="all failed"):
            async for _ in stream_with_first_token_retry(
                make_request=make_request, stream_processor=_sse_processor,
                initial_response=_hanging_response(), max_retries=3, first_token_timeout=0.02,
                on_all_retries_failed=on_all_failed, emit_ready_marker=True,
            ):
                pass

        assert calls == [(3, 0.02)]
        assert make_request.await_count == 2

    @pytest.mark.asyncio
    async def test_guard_disabled_emits_marker_before_first_token_wait(self, aws_frames):
        """
        What it does: empty_response_as_error=False with an empty-looking stream.
        Purpose: Native behavior: marker immediately, empty stream passes through without error.
        """
        response = _stream_response([aws_frames.context_usage(1.0)])

        items = [
            item async for item in stream_with_first_token_retry(
                make_request=AsyncMock(), stream_processor=_sse_processor,
                initial_response=response, max_retries=3, first_token_timeout=5,
                emit_ready_marker=True, empty_response_as_error=False,
            )
        ]

        assert items == [STREAM_READY]

    @pytest.mark.asyncio
    async def test_closing_after_marker_closes_response(self, aws_frames):
        """
        What it does: Consumer closes the generator right after receiving STREAM_READY.
        Purpose: The accepted response must not leak when no stream processor ever ran.
        """
        response = _stream_response([_success_bytes(aws_frames)])
        stream = stream_with_first_token_retry(
            make_request=AsyncMock(), stream_processor=_sse_processor,
            initial_response=response, max_retries=3, first_token_timeout=5,
            emit_ready_marker=True,
        )

        assert await stream.__anext__() is STREAM_READY
        await stream.aclose()

        response.aclose.assert_awaited()


class TestPrimeSseStream:
    """Tests for prime_sse_stream()."""

    @staticmethod
    def _source(items, after_error=None):
        """Async generator yielding items; raises `after_error` at the end if given."""
        closed = {"value": False}

        async def gen():
            try:
                for item in items:
                    yield item
                if after_error is not None:
                    raise after_error
            finally:
                closed["value"] = True

        inner = gen()

        class Source:
            """Async generator facade exposing the `closed` flag."""

            def __init__(self):
                self.closed = closed

            def __aiter__(self):
                return self

            async def __anext__(self):
                return await inner.__anext__()

            async def aclose(self):
                await inner.aclose()

        return Source()

    @pytest.mark.asyncio
    async def test_drops_marker_and_replays_rest_in_order(self):
        """
        What it does: Stream starts with STREAM_READY.
        Purpose: The marker must never reach the client; everything else in order.
        """
        primed = await prime_sse_stream(self._source([STREAM_READY, "a", "b", "c"]))

        assert [item async for item in primed] == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_first_string_item_is_kept(self):
        """
        What it does: Stream without marker (first item is an SSE string).
        Purpose: The first chunk must be replayed, not dropped.
        """
        primed = await prime_sse_stream(self._source(["a", "b"]))

        assert [item async for item in primed] == ["a", "b"]

    @pytest.mark.asyncio
    async def test_empty_stream(self):
        """
        What it does: Stream ends immediately.
        Purpose: An empty generator is returned, no exception.
        """
        primed = await prime_sse_stream(self._source([]))

        assert [item async for item in primed] == []

    @pytest.mark.asyncio
    async def test_marker_only_stream(self):
        """
        What it does: Only the marker, then end.
        Purpose: Nothing is replayed.
        """
        primed = await prime_sse_stream(self._source([STREAM_READY]))

        assert [item async for item in primed] == []

    @pytest.mark.asyncio
    async def test_empty_error_is_raised_immediately(self):
        """
        What it does: Stream raises EmptyUpstreamResponseError at the first step.
        Purpose: Routes need it before the StreamingResponse exists.
        """
        source = self._source([], after_error=EmptyUpstreamResponseError(147, {"contextUsageEvent": 1}))

        with pytest.raises(EmptyUpstreamResponseError):
            await prime_sse_stream(source)

    @pytest.mark.asyncio
    async def test_other_exception_is_deferred_to_iteration(self):
        """
        What it does: Stream raises a generic error at the first step.
        Purpose: Existing route behavior (in-stream error event) must stay: raised on iteration.
        """
        source = self._source([], after_error=RuntimeError("upstream broke"))

        primed = await prime_sse_stream(source)

        with pytest.raises(RuntimeError, match="upstream broke"):
            async for _ in primed:
                pass

    @pytest.mark.asyncio
    async def test_deferred_exception_raised_only_once(self):
        """
        What it does: Iterates a primed stream twice after a deferred error.
        Purpose: The error surfaces once; a finished generator is simply exhausted.
        """
        primed = await prime_sse_stream(self._source([], after_error=RuntimeError("x")))

        with pytest.raises(RuntimeError):
            await primed.__anext__()
        with pytest.raises(StopAsyncIteration):
            await primed.__anext__()

    @pytest.mark.asyncio
    async def test_cancelled_error_closes_stream_and_propagates(self):
        """
        What it does: Stream raises CancelledError while priming.
        Purpose: Cancellation must propagate and the source must be closed.
        """
        source = self._source([], after_error=asyncio.CancelledError())

        with pytest.raises(asyncio.CancelledError):
            await prime_sse_stream(source)

        assert source.closed["value"] is True

    @pytest.mark.asyncio
    async def test_closing_primed_stream_closes_source(self):
        """
        What it does: Consumer stops after the first item and closes the primed generator.
        Purpose: Client disconnect must reach the source generator's cleanup.
        """
        source = self._source([STREAM_READY, "a", "b", "c"])
        primed = await prime_sse_stream(source)

        assert await primed.__anext__() == "a"
        await primed.aclose()

        assert source.closed["value"] is True

    @pytest.mark.asyncio
    async def test_mid_stream_error_propagates_after_items(self):
        """
        What it does: Source yields two items then fails.
        Purpose: Items first, then the error (route turns it into an SSE error event).
        """
        source = self._source([STREAM_READY, "a", "b"], after_error=RuntimeError("late"))
        primed = await prime_sse_stream(source)
        seen = []

        with pytest.raises(RuntimeError, match="late"):
            async for item in primed:
                seen.append(item)

        assert seen == ["a", "b"]


@pytest.mark.usefixtures("guard_on")
class TestCollectStreamToResultGuard:
    """collect_stream_to_result() with the empty-response guard."""

    @pytest.mark.asyncio
    async def test_empty_response_raises(self, aws_frames):
        """
        What it does: Non-streaming collection of a lone contextUsageEvent frame.
        Purpose: Non-streaming path reports the empty upstream response too.
        """
        response = _stream_response([aws_frames.context_usage(60.7)])

        with pytest.raises(EmptyUpstreamResponseError):
            await collect_stream_to_result(response, first_token_timeout=5)

    @pytest.mark.asyncio
    async def test_metered_empty_answer_returns_empty_result(self, aws_frames):
        """
        What it does: contextUsageEvent + meteringEvent, no output.
        Purpose: The model's empty answer is delivered as an empty result (no error).
        """
        response = _stream_response([aws_frames.context_usage(3.0) + aws_frames.metering(0.1)])

        result = await collect_stream_to_result(response, first_token_timeout=5)

        assert result.content == ""
        assert result.tool_calls == []
        assert result.context_usage_percentage == 3.0

    @pytest.mark.asyncio
    async def test_normal_response_unchanged(self, aws_frames):
        """
        What it does: Normal reasoning + content response in awkward chunks.
        Purpose: Collected result equals the one without the guard.
        """
        chunks = aws_frames.split(_success_bytes(aws_frames), [5, 9])
        guarded = await collect_stream_to_result(_stream_response(chunks), first_token_timeout=5)
        with patch("kiro.streaming_core.EMPTY_RESPONSE_AS_ERROR", False):
            unguarded = await collect_stream_to_result(_stream_response(chunks), first_token_timeout=5)

        assert asdict(guarded) == asdict(unguarded)
        assert guarded.content == "Hello world"

    @pytest.mark.asyncio
    async def test_tool_only_response(self, aws_frames):
        """
        What it does: A tool call response (no text).
        Purpose: Tool-only output is output.
        """
        data = (
            aws_frames.tool_start("get_weather", "tool_1")
            + aws_frames.tool_input('{"city":"Paris"}', "get_weather", "tool_1")
            + aws_frames.tool_stop("get_weather", "tool_1")
            + aws_frames.context_usage(2.0) + aws_frames.metering(0.2)
        )

        result = await collect_stream_to_result(_stream_response([data]), first_token_timeout=5)

        assert len(result.tool_calls) == 1

    @pytest.mark.asyncio
    async def test_switch_off_returns_empty_result(self, aws_frames):
        """
        What it does: EMPTY_RESPONSE_AS_ERROR=False with an empty upstream response.
        Purpose: Native behavior is restored: empty result, no error.
        """
        response = _stream_response([aws_frames.context_usage(60.7)])

        with patch("kiro.streaming_core.EMPTY_RESPONSE_AS_ERROR", False):
            result = await collect_stream_to_result(response, first_token_timeout=5)

        assert result.content == ""


class TestParseKiroStreamDiagnostics:
    """[StreamFrames] and [FirstToken] diagnostics."""

    @pytest.mark.asyncio
    async def test_stream_frames_log_is_exact_across_chunk_boundaries(self, aws_frames, loguru_messages):
        """
        What it does: Parses a stream cut into 3-byte chunks.
        Purpose: [StreamFrames] counts must be exact even when headers are split.
        """
        data = _success_bytes(aws_frames)
        response = _stream_response(aws_frames.split(data, [3]))

        await _collect_events(response)

        line = next(m for m in loguru_messages if "[StreamFrames]" in m)
        print(line)
        assert f"bytes={len(data)}" in line
        assert "'reasoningContentEvent': 2" in line
        assert "'assistantResponseEvent': 2" in line
        assert "'contextUsageEvent': 1" in line
        assert "'meteringEvent': 1" in line

    @pytest.mark.asyncio
    async def test_first_token_log_uses_guard_measured_time(self, aws_frames, loguru_messages):
        """
        What it does: Parses a GuardedResponse whose first chunk took a known time.
        Purpose: The replay is instant; the log must show the real upstream wait.
        """
        response = _stream_response([_success_bytes(aws_frames)])
        guarded = await guard_upstream_response(response, first_token_timeout=5, enabled=True)
        guarded.first_chunk_ms = 1234.0

        await _collect_events(guarded)

        assert any("[FirstToken] received in 1234ms" in m for m in loguru_messages)

    def test_count_frame_event_types_still_counts_single_chunk(self, aws_frames):
        """
        What it does: Calls the legacy diagnostics helper on one chunk.
        Purpose: The public helper keeps working.
        """
        from collections import Counter
        from kiro.streaming_core import count_frame_event_types

        counts = Counter()
        count_frame_event_types(aws_frames.content("a") + aws_frames.metering(1.0), counts)

        assert dict(counts) == {"assistantResponseEvent": 1, "meteringEvent": 1}


@pytest.mark.usefixtures("guard_on")
class TestGuardReleasesOnFirstToolFrame:
    """A pure tool-call response must not wait for the end of the stream."""

    @pytest.mark.asyncio
    async def test_first_tool_use_frame_releases_without_reading_more(self, aws_frames):
        """
        What it does: Upstream sends toolUseEvent frames in separate chunks; counts pulls.
        Purpose: Tool-only responses stream their first byte as soon as the first
        toolUseEvent header is seen (the guard does not wait for the stream end).
        """
        chunks = [
            aws_frames.tool_start("Bash", "t1"),
            aws_frames.tool_input('{"command": "ls"}', "Bash", "t1"),
            aws_frames.tool_stop("Bash", "t1"),
            aws_frames.context_usage(1.0),
        ]
        response = _stream_response(chunks)

        guarded = await guard_upstream_response(response, first_token_timeout=5)

        assert isinstance(guarded, GuardedResponse)
        assert len(response.pulled) == 1
        events = await _collect_events(guarded)
        assert [e.type for e in events if e.type == "tool_use"] == ["tool_use"]


# ==================================================================================================
# Incomplete upstream response (reasoning only, no meteringEvent, no text, no tool_use)
# ==================================================================================================

from kiro.streaming_core import (  # noqa: E402
    aclose_shielded,
    check_stream_outcome,
)
from kiro.upstream_guard import (  # noqa: E402
    KIND_INCOMPLETE,
    StreamOutcome,
    StreamVerdict,
)

BODY_SECRET = "SECRET_BODY_TEXT_91c2"
THOUGHT_SECRET = "SECRET_THOUGHT_TEXT_55d1"


@pytest.fixture
def pinned_reasoning():
    """Pins native reasoning handling so tests do not depend on the local .env."""
    with patch("kiro.streaming_core.FAKE_REASONING_ENABLED", True), \
            patch("kiro.streaming_core.FAKE_REASONING_HANDLING", "as_reasoning_content"):
        yield


def incomplete_bytes(aws_frames, with_context_usage=True):
    """Reasoning frames (+ contextUsageEvent), no metering, no body: the reported failure."""
    data = b"".join(aws_frames.reasoning(f"{THOUGHT_SECRET} {i}") for i in range(5))
    if with_context_usage:
        data += aws_frames.context_usage(3.5)
    return data


@pytest.mark.usefixtures("pinned_reasoning")
class TestParseKiroStreamOutcome:
    """parse_kiro_stream fills the StreamOutcome side channel."""

    @pytest.mark.asyncio
    async def test_reasoning_and_context_usage_without_metering_is_incomplete(self, aws_frames):
        """
        What it does: Parses the reported failure shape with an outcome.
        Purpose: verdict INCOMPLETE; no text, no tool_use; frame counts from the ONE scanner.
        """
        data = incomplete_bytes(aws_frames)
        outcome = StreamOutcome()

        events = await _collect_events(_stream_response([data]), outcome=outcome)

        assert [e.type for e in events] == ["thinking"] * 5 + ["context_usage"]
        assert outcome.finished is True
        assert outcome.verdict() == StreamVerdict.INCOMPLETE
        assert outcome.produced_text is False and outcome.produced_tool_use is False
        assert dict(outcome.scanner.counts) == {"reasoningContentEvent": 5, "contextUsageEvent": 1}
        assert outcome.scanner.bytes_received == len(data)

    @pytest.mark.asyncio
    async def test_metering_makes_it_complete(self, aws_frames):
        """
        What it does: Same stream plus a meteringEvent.
        Purpose: verdict COMPLETE (pass through verbatim).
        """
        outcome = StreamOutcome()

        await _collect_events(
            _stream_response([incomplete_bytes(aws_frames) + aws_frames.metering(0.4)]), outcome=outcome
        )

        assert outcome.verdict() == StreamVerdict.COMPLETE

    @pytest.mark.asyncio
    async def test_reasoning_and_body_without_metering(self, aws_frames):
        """
        What it does: Reasoning + text, no metering.
        Purpose: verdict MISSING_METERING_WITH_OUTPUT with produced_text.
        """
        outcome = StreamOutcome()

        await _collect_events(
            _stream_response([aws_frames.reasoning("r") + aws_frames.content("hi")]), outcome=outcome
        )

        assert outcome.verdict() == StreamVerdict.MISSING_METERING_WITH_OUTPUT
        assert outcome.produced_text is True

    @pytest.mark.asyncio
    async def test_reasoning_and_tool_use_without_metering(self, aws_frames):
        """
        What it does: Reasoning + a complete tool call, no metering.
        Purpose: produced_tool_use is recorded; verdict MISSING_METERING_WITH_OUTPUT.
        """
        data = (
            aws_frames.reasoning("r")
            + aws_frames.tool_start("Bash", "toolu_1")
            + aws_frames.tool_input('{"command": "ls"}', "Bash", "toolu_1")
            + aws_frames.tool_stop("Bash", "toolu_1")
        )
        outcome = StreamOutcome()

        await _collect_events(_stream_response([data]), outcome=outcome)

        assert outcome.produced_tool_use is True
        assert outcome.verdict() == StreamVerdict.MISSING_METERING_WITH_OUTPUT

    @pytest.mark.asyncio
    async def test_only_reasoning_frames_no_context_usage(self, aws_frames):
        """
        What it does: 49 reasoning frames only (the 17:15:07 shape).
        Purpose: INCOMPLETE even without a contextUsageEvent.
        """
        data = b"".join(aws_frames.reasoning(f"thought {i} " * 4) for i in range(49))
        outcome = StreamOutcome()

        await _collect_events(_stream_response([data]), outcome=outcome)

        assert outcome.scanner.counts["reasoningContentEvent"] == 49
        assert outcome.verdict() == StreamVerdict.INCOMPLETE

    @pytest.mark.asyncio
    async def test_empty_stream_is_finished_with_no_output_frames(self):
        """
        What it does: Upstream stream with no bytes at all (guard disabled).
        Purpose: Early return still marks the stream finished; verdict NO_OUTPUT_FRAMES.
        """
        outcome = StreamOutcome()

        events = await _collect_events(_stream_response([]), outcome=outcome)

        assert events == []
        assert outcome.finished is True
        assert outcome.verdict() == StreamVerdict.NO_OUTPUT_FRAMES

    @pytest.mark.asyncio
    @pytest.mark.parametrize("error", [
        httpx.ReadError("connection reset"),
        httpx.RemoteProtocolError("peer closed connection without sending complete message body"),
    ])
    async def test_mid_stream_exception_is_not_finished(self, aws_frames, error):
        """
        What it does: The upstream raises after some reasoning frames.
        Purpose: finished stays False -> NOT_FINISHED; the exception propagates unchanged.
        """
        outcome = StreamOutcome()
        response = _stream_response([aws_frames.reasoning("r"), error])

        with pytest.raises(type(error)):
            await _collect_events(response, outcome=outcome)

        assert outcome.finished is False
        assert outcome.verdict() == StreamVerdict.NOT_FINISHED

    @pytest.mark.asyncio
    async def test_consumer_close_is_not_finished(self, aws_frames):
        """
        What it does: The consumer closes the generator after the first event (GeneratorExit).
        Purpose: finished stays False: no verdict for an abandoned stream.
        """
        outcome = StreamOutcome()
        response = _stream_response([aws_frames.reasoning("a"), aws_frames.reasoning("b")])
        gen = parse_kiro_stream(response, first_token_timeout=30, outcome=outcome)

        await gen.__anext__()
        await gen.aclose()

        assert outcome.finished is False
        assert outcome.verdict() == StreamVerdict.NOT_FINISHED

    @pytest.mark.asyncio
    async def test_events_are_identical_with_and_without_outcome(self, aws_frames):
        """
        What it does: Parses the same bytes with and without an outcome.
        Purpose: The outcome is a pure side channel: the event sequence does not change.
        """
        data = _success_bytes(aws_frames)

        with_outcome = await _collect_events(_stream_response([data]), outcome=StreamOutcome())
        without = await _collect_events(_stream_response([data]))

        assert with_outcome == without

    @pytest.mark.asyncio
    async def test_guard_replay_counts_the_whole_stream_with_one_scanner(self, aws_frames):
        """
        What it does: Runs the guard first (it peeks), then parses the GuardedResponse.
        Purpose: outcome.scanner counts exactly the bytes/frames of the whole stream, equal to
        the unguarded result: peeked bytes are not lost and not counted twice.
        """
        data = incomplete_bytes(aws_frames)
        chunks = aws_frames.split(data, [13, 70])
        guarded_outcome = StreamOutcome()
        plain_outcome = StreamOutcome()
        guarded = await guard_upstream_response(_stream_response(chunks), 5, True)

        await _collect_events(guarded, outcome=guarded_outcome)
        await _collect_events(_stream_response(chunks), outcome=plain_outcome)

        assert isinstance(guarded, GuardedResponse)
        assert guarded_outcome.scanner.bytes_received == plain_outcome.scanner.bytes_received == len(data)
        assert dict(guarded_outcome.scanner.counts) == dict(plain_outcome.scanner.counts)
        assert guarded_outcome.verdict() == plain_outcome.verdict() == StreamVerdict.INCOMPLETE

    @pytest.mark.asyncio
    @pytest.mark.parametrize("handling", ["include_as_text", "strip_tags", "pass", "remove", "as_reasoning_content"])
    async def test_reasoning_never_counts_as_text_in_any_mode(self, aws_frames, handling):
        """
        What it does: Reasoning-only stream under every FAKE_REASONING_HANDLING mode.
        Purpose: Mode changes formatting, never the verdict: always INCOMPLETE, never text.
        """
        outcome = StreamOutcome()

        with patch("kiro.streaming_core.FAKE_REASONING_HANDLING", handling):
            await _collect_events(_stream_response([incomplete_bytes(aws_frames)]), outcome=outcome)

        assert outcome.produced_text is False
        assert outcome.verdict() == StreamVerdict.INCOMPLETE

    @pytest.mark.asyncio
    async def test_exception_frame_without_output_is_incomplete(self, aws_frames):
        """
        What it does: AWS exception frame + contextUsageEvent through the real parser.
        Purpose: The parser tolerates the frame and the verdict is INCOMPLETE.
        """
        data = aws_frames.exception("ThrottlingException", {"message": "x"}) + aws_frames.context_usage(1.0)
        outcome = StreamOutcome()

        await _collect_events(_stream_response([data]), outcome=outcome)

        assert outcome.verdict() == StreamVerdict.INCOMPLETE


class TestCheckStreamOutcome:
    """check_stream_outcome(): switch semantics and WARNING content."""

    @staticmethod
    def _outcome(aws_frames, kind):
        """Builds a finished StreamOutcome of the requested verdict."""
        outcome = StreamOutcome()
        if kind == "incomplete":
            outcome.scanner.feed(incomplete_bytes(aws_frames))
        elif kind == "complete":
            outcome.scanner.feed(incomplete_bytes(aws_frames) + aws_frames.metering(0.1))
        elif kind == "missing_metering":
            outcome.scanner.feed(aws_frames.reasoning(THOUGHT_SECRET) + aws_frames.content(BODY_SECRET))
            outcome.produced_text = True
        elif kind == "no_output":
            outcome.scanner.feed(aws_frames.context_usage(1.0))
        outcome.finished = kind != "not_finished"
        if kind == "not_finished":
            outcome.scanner.feed(incomplete_bytes(aws_frames))
        return outcome

    def test_incomplete_with_switch_on_requires_error(self, aws_frames, loguru_messages):
        """
        What it does: INCOMPLETE, switch on.
        Purpose: Returns True; logs nothing itself (the caller reports the error once).
        """
        assert check_stream_outcome(self._outcome(aws_frames, "incomplete"), enabled=True) is True
        assert not [m for m in loguru_messages if "IncompleteUpstreamResponse" in m]

    def test_incomplete_with_switch_off_passes_through_with_warning(self, aws_frames, loguru_messages):
        """
        What it does: INCOMPLETE, switch off.
        Purpose: Returns False (old behavior) but leaves one diagnostic WARNING.
        """
        assert check_stream_outcome(self._outcome(aws_frames, "incomplete"), enabled=False) is False

        lines = [m for m in loguru_messages if "[IncompleteUpstreamResponse]" in m]
        assert len(lines) == 1 and lines[0].startswith("WARNING|")
        assert "action=passthrough" in lines[0]

    @pytest.mark.parametrize("enabled", [True, False])
    def test_missing_metering_with_output_warns_whatever_the_switch(self, aws_frames, loguru_messages, enabled):
        """
        What it does: Text without metering (possibly truncated), both switch states.
        Purpose: Never an error; one WARNING with counts/booleans and NO content.
        """
        outcome = self._outcome(aws_frames, "missing_metering")

        assert check_stream_outcome(outcome, enabled=enabled) is False

        lines = [m for m in loguru_messages if "[IncompleteUpstreamResponse]" in m]
        assert len(lines) == 1 and lines[0].startswith("WARNING|")
        assert "text=True" in lines[0] and "tool_use=False" in lines[0]
        assert "upstream_bytes=" in lines[0] and "reasoningContentEvent" in lines[0]
        assert BODY_SECRET not in lines[0] and THOUGHT_SECRET not in lines[0]

    @pytest.mark.parametrize("kind", ["complete", "no_output", "not_finished"])
    def test_other_verdicts_are_silent_passthrough(self, aws_frames, loguru_messages, kind):
        """
        What it does: COMPLETE / NO_OUTPUT_FRAMES / NOT_FINISHED.
        Purpose: Returns False and writes no [IncompleteUpstreamResponse] line.
        """
        assert check_stream_outcome(self._outcome(aws_frames, kind), enabled=True) is False
        assert not [m for m in loguru_messages if "IncompleteUpstreamResponse" in m]

    def test_enabled_none_reads_the_config(self, aws_frames):
        """
        What it does: enabled=None with the module setting patched both ways.
        Purpose: The config value is the default.
        """
        outcome = self._outcome(aws_frames, "incomplete")
        with patch("kiro.streaming_core.INCOMPLETE_RESPONSE_AS_ERROR", True):
            assert check_stream_outcome(outcome) is True
        with patch("kiro.streaming_core.INCOMPLETE_RESPONSE_AS_ERROR", False):
            assert check_stream_outcome(outcome) is False


@pytest.mark.usefixtures("pinned_reasoning", "guard_on")
class TestCollectStreamToResultIncomplete:
    """collect_stream_to_result() with the incomplete-response criterion."""

    @pytest.mark.asyncio
    async def test_incomplete_raises_with_diagnostics(self, aws_frames):
        """
        What it does: Non-streaming collection of reasoning + contextUsageEvent, no metering.
        Purpose: Raises EmptyUpstreamResponseError(kind=incomplete) with the stream facts.
        """
        data = incomplete_bytes(aws_frames)

        with pytest.raises(EmptyUpstreamResponseError) as exc_info:
            await collect_stream_to_result(_stream_response([data]), first_token_timeout=5)

        err = exc_info.value
        assert err.kind == KIND_INCOMPLETE
        assert err.bytes_received == len(data)
        assert err.frame_counts == {"reasoningContentEvent": 5, "contextUsageEvent": 1}
        assert err.produced_text is False and err.produced_tool_use is False

    @pytest.mark.asyncio
    async def test_only_reasoning_no_context_usage_raises(self, aws_frames):
        """
        What it does: Only reasoning frames (17:15:07 shape), non-streaming.
        Purpose: Raises as well.
        """
        data = incomplete_bytes(aws_frames, with_context_usage=False)

        with pytest.raises(EmptyUpstreamResponseError) as exc_info:
            await collect_stream_to_result(_stream_response(aws_frames.split(data, [7])), first_token_timeout=5)

        assert exc_info.value.kind == KIND_INCOMPLETE

    @pytest.mark.asyncio
    async def test_metering_returns_result_with_thinking_only(self, aws_frames):
        """
        What it does: Reasoning + contextUsageEvent + meteringEvent, no body.
        Purpose: Complete: returns the thinking-only result unchanged.
        """
        data = incomplete_bytes(aws_frames) + aws_frames.metering(0.2)

        result = await collect_stream_to_result(_stream_response([data]), first_token_timeout=5)

        assert THOUGHT_SECRET in result.thinking_content
        assert result.content == ""

    @pytest.mark.asyncio
    async def test_reasoning_and_body_without_metering_passes_with_warning(self, aws_frames, loguru_messages):
        """
        What it does: Reasoning + text, no metering.
        Purpose: Result returned unchanged; one WARNING; no exception.
        """
        data = aws_frames.reasoning("r") + aws_frames.content(BODY_SECRET) + aws_frames.context_usage(1.0)

        result = await collect_stream_to_result(_stream_response([data]), first_token_timeout=5)

        assert result.content == BODY_SECRET
        lines = [m for m in loguru_messages if "[IncompleteUpstreamResponse]" in m]
        assert len(lines) == 1 and BODY_SECRET not in lines[0]

    @pytest.mark.asyncio
    async def test_reasoning_and_tool_use_without_metering_passes(self, aws_frames):
        """
        What it does: Reasoning + tool call, no metering.
        Purpose: Result with the tool call is returned.
        """
        data = (
            aws_frames.reasoning("r")
            + aws_frames.tool_start("Bash", "toolu_1")
            + aws_frames.tool_input('{"command": "ls"}', "Bash", "toolu_1")
            + aws_frames.tool_stop("Bash", "toolu_1")
        )

        result = await collect_stream_to_result(_stream_response([data]), first_token_timeout=5)

        assert len(result.tool_calls) == 1

    @pytest.mark.asyncio
    async def test_switch_off_restores_old_result(self, aws_frames):
        """
        What it does: INCOMPLETE_RESPONSE_AS_ERROR=False with an incomplete stream.
        Purpose: Old behavior: the thinking-only result is returned, no exception.
        """
        with patch("kiro.streaming_core.INCOMPLETE_RESPONSE_AS_ERROR", False):
            result = await collect_stream_to_result(
                _stream_response([incomplete_bytes(aws_frames)]), first_token_timeout=5
            )

        assert THOUGHT_SECRET in result.thinking_content
        assert result.content == ""

    @pytest.mark.asyncio
    async def test_empty_response_keeps_kind_empty(self, aws_frames):
        """
        What it does: A lone contextUsageEvent (the earlier empty-response criterion).
        Purpose: Still raised by the guard with kind "empty".
        """
        with pytest.raises(EmptyUpstreamResponseError) as exc_info:
            await collect_stream_to_result(_stream_response([aws_frames.context_usage(1.0)]), first_token_timeout=5)

        assert exc_info.value.kind == "empty"

    @pytest.mark.asyncio
    async def test_mid_stream_exception_is_not_reported_as_incomplete(self, aws_frames):
        """
        What it does: Reasoning frames, then the upstream connection breaks.
        Purpose: The original httpx error propagates (different failure class).
        """
        response = _stream_response([aws_frames.reasoning("r"), httpx.RemoteProtocolError("cut")])

        with pytest.raises(httpx.RemoteProtocolError):
            await collect_stream_to_result(response, first_token_timeout=5)


# ==================================================================================================
# Peek-phase cancellation cleanup
# ==================================================================================================

class TestCancellationCleanup:
    """The upstream response is closed when the task is cancelled while waiting for bytes."""

    @staticmethod
    def _blocking_response(first_chunk=None):
        """Response whose stream yields first_chunk (optional) and then blocks forever."""
        response = AsyncMock()
        response.status_code = 200
        response.aclose = AsyncMock()
        started = asyncio.Event()

        async def gen():
            if first_chunk is not None:
                yield first_chunk
            started.set()
            await asyncio.sleep(3600)
            yield b""

        response.aiter_bytes = lambda *a, **k: gen()
        response.started = started
        return response

    @pytest.mark.asyncio
    async def test_guard_closes_response_when_cancelled_waiting_for_first_chunk(self):
        """
        What it does: Cancels guard_upstream_response while it waits for the first chunk.
        Purpose: response.aclose is awaited and CancelledError propagates.
        """
        response = self._blocking_response()
        task = asyncio.ensure_future(guard_upstream_response(response, first_token_timeout=30, enabled=True))
        await asyncio.sleep(0.01)

        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task
        response.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_guard_closes_response_when_cancelled_while_peeking_further(self, aws_frames):
        """
        What it does: Cancels the guard after a neutral first chunk, while it waits for more.
        Purpose: Same cleanup in the later-chunk loop.
        """
        response = self._blocking_response(first_chunk=aws_frames.context_usage(1.0))
        task = asyncio.ensure_future(guard_upstream_response(response, first_token_timeout=30, enabled=True))
        await asyncio.wait_for(response.started.wait(), timeout=2)

        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task
        response.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_retry_loop_closes_response_when_cancelled(self):
        """
        What it does: Cancels stream_with_first_token_retry while the guard waits.
        Purpose: The open response is closed and the cancellation propagates.
        """
        response = self._blocking_response()

        async def make_request():
            return response

        async def processor(resp):
            yield "never"

        async def consume():
            async for _ in stream_with_first_token_retry(
                make_request, processor, first_token_timeout=30, empty_response_as_error=True
            ):
                pass

        task = asyncio.ensure_future(consume())
        await asyncio.sleep(0.01)

        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task
        assert response.aclose.await_count >= 1

    @pytest.mark.asyncio
    async def test_aclose_shielded_swallows_expected_close_errors(self):
        """
        What it does: The close coroutine raises httpx.HTTPError / RuntimeError / OSError.
        Purpose: Expected close errors are swallowed (logged at DEBUG).
        """
        for error in (httpx.ReadError("x"), RuntimeError("x"), OSError("x")):
            close = AsyncMock(side_effect=error)
            await aclose_shielded(close)
            close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_aclose_shielded_does_not_swallow_other_errors(self):
        """
        What it does: The close coroutine raises a ValueError.
        Purpose: Unexpected errors are not hidden.
        """
        with pytest.raises(ValueError):
            await aclose_shielded(AsyncMock(side_effect=ValueError("bug")))

    @pytest.mark.asyncio
    async def test_aclose_shielded_finishes_close_even_if_caller_is_cancelled(self):
        """
        What it does: Cancels the caller while the close coroutine is still running.
        Purpose: The shielded close runs to completion anyway.
        """
        finished = asyncio.Event()

        async def slow_close():
            await asyncio.sleep(0.05)
            finished.set()

        task = asyncio.ensure_future(aclose_shielded(slow_close))
        await asyncio.sleep(0.01)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(finished.wait(), timeout=2)


# ==================================================================================================
# Upstream connection drop (mid-stream) handling
# ==================================================================================================

DROP_MESSAGE = "peer closed connection without sending complete message body (incomplete chunked read)"


def _drop_error():
    """Builds the exception httpx raises when the upstream closes a chunked body early."""
    return httpx.RemoteProtocolError(DROP_MESSAGE)


def _ok_response():
    """Response double with status 200 and a recordable aclose()."""
    response = AsyncMock()
    response.status_code = 200
    response.aclose = AsyncMock()
    return response


def _capture_logs(level="WARNING"):
    """Starts collecting loguru messages; returns (messages, handler id)."""
    messages = []
    handler_id = logger.add(lambda m: messages.append(str(m)), level=level, format="{level}|{message}")
    return messages, handler_id


class TestIsUpstreamDrop:
    """Tests for is_upstream_drop()."""

    def test_remote_protocol_error_is_a_drop(self):
        """
        What it does: Passes the exception seen in production ("incomplete chunked read").
        Purpose: This is the failure the retry exists for.
        """
        assert is_upstream_drop(_drop_error()) is True

    def test_read_error_is_a_drop(self):
        """
        What it does: Passes httpx.ReadError (connection reset while reading).
        Purpose: A reset is the same failure class as an early close.
        """
        assert is_upstream_drop(httpx.ReadError("reset")) is True

    @pytest.mark.parametrize("error", [
        httpx.ReadTimeout("slow"),
        httpx.ConnectError("refused"),
        httpx.WriteError("broken pipe"),
        RuntimeError("boom"),
        ValueError("bad"),
        asyncio.TimeoutError(),
        FirstTokenTimeoutError("no token"),
        Exception(DROP_MESSAGE),
    ])
    def test_other_exceptions_are_not_drops(self, error):
        """
        What it does: Passes timeouts, connect/write errors, generic errors and a plain
        Exception that merely has the drop message text.
        Purpose: Only the two httpx classes are retried; matching on message text is not done.
        """
        assert is_upstream_drop(error) is False


class TestDecideDropAction:
    """Tests for decide_drop_action()."""

    def test_retries_when_nothing_was_sent(self):
        """
        What it does: No client output, switch on, attempts left.
        Purpose: The only case that retries.
        """
        assert decide_drop_action(0, True, 0, 3) == DROP_ACTION_RETRY

    def test_retries_on_second_attempt_too(self):
        """
        What it does: Second of three attempts failed.
        Purpose: A third attempt is still available.
        """
        assert decide_drop_action(0, True, 1, 3) == DROP_ACTION_RETRY

    def test_gives_up_on_last_attempt(self):
        """
        What it does: The last allowed attempt dropped.
        Purpose: No attempt left, the drop is propagated.
        """
        assert decide_drop_action(0, True, 2, 3) == DROP_ACTION_EXHAUSTED

    def test_single_attempt_budget_never_retries(self):
        """
        What it does: max_attempts=1.
        Purpose: Boundary: the first attempt is also the last one.
        """
        assert decide_drop_action(0, True, 0, 1) == DROP_ACTION_EXHAUSTED

    def test_output_already_sent_wins_over_everything(self):
        """
        What it does: One chunk with output was already yielded.
        Purpose: A retry would append a second answer to the first one.
        """
        assert decide_drop_action(1, True, 0, 3) == DROP_ACTION_OUTPUT_SENT
        assert decide_drop_action(500, False, 2, 3) == DROP_ACTION_OUTPUT_SENT

    def test_disabled_switch_never_retries(self):
        """
        What it does: Switch off, nothing sent.
        Purpose: Users can restore the old behavior.
        """
        assert decide_drop_action(0, False, 0, 3) == DROP_ACTION_DISABLED


class TestDescribeStreamDrop:
    """Tests for describe_stream_drop()."""

    def test_uses_counters_of_guarded_response(self):
        """
        What it does: Describes a drop of a GuardedResponse that delivered 3 chunks / 300 bytes.
        Purpose: Bytes, chunks and idle time are what tells idle timeouts from size limits.
        """
        from kiro.upstream_guard import GuardedResponse
        guarded = GuardedResponse(MagicMock(), None, [])
        guarded.bytes_read = 300
        guarded.chunks_read = 3
        guarded.last_chunk_at = time.monotonic() - 12.0
        text = describe_stream_drop(guarded, _drop_error(), time.monotonic() - 20.0, 4)
        print(text)
        assert "error=RemoteProtocolError" in text
        assert "bytes=300" in text
        assert "chunks=3" in text
        assert "client_chunks=4" in text
        assert "idle=12." in text
        assert "elapsed=20." in text

    def test_reports_na_without_counters(self):
        """
        What it does: Describes a drop of a plain response double (guard disabled).
        Purpose: Missing counters are shown as n/a, never as a crash or a bogus number.
        """
        text = describe_stream_drop(object(), httpx.ReadError("x"), time.monotonic(), 0)
        assert "bytes=n/a" in text
        assert "chunks=n/a" in text
        assert "idle=n/a" in text
        assert "error=ReadError" in text

    def test_ignores_mock_attributes(self):
        """
        What it does: Passes an AsyncMock whose attributes are auto-created mocks.
        Purpose: Non-numeric attributes must be treated as unknown.
        """
        text = describe_stream_drop(AsyncMock(), _drop_error(), time.monotonic(), 0)
        assert "bytes=n/a" in text

    def test_never_includes_the_error_message_text(self):
        """
        What it does: Describes an error whose message contains a secret.
        Purpose: The line may carry counts and timings only.
        """
        text = describe_stream_drop(object(), httpx.ReadError("SECRET_PAYLOAD"), time.monotonic(), 0)
        assert "SECRET_PAYLOAD" not in text


class TestStreamWithFirstTokenRetryUpstreamDrop:
    """stream_with_first_token_retry(): retry of a stream the upstream dropped."""

    @staticmethod
    async def _collect(**kwargs):
        """Runs the core wrapper, returns (items, raised exception or None)."""
        items = []
        raised = None
        kwargs.setdefault("empty_response_as_error", False)
        try:
            async for item in stream_with_first_token_retry(first_token_timeout=30, **kwargs):
                items.append(item)
        except Exception as e:
            raised = e
        return items, raised

    @pytest.mark.asyncio
    async def test_drop_before_any_output_is_retried(self):
        """
        What it does: Attempt 1 raises the drop error without yielding; attempt 2 works.
        Purpose: The client only sees the answer of attempt 2 (the production failure).
        """
        calls = []
        make_request = AsyncMock(side_effect=lambda: _ok_response())

        async def processor(response):
            calls.append(response)
            if len(calls) == 1:
                raise _drop_error()
            yield "chunk-a"
            yield "chunk-b"

        items, raised = await self._collect(
            make_request=make_request, stream_processor=processor,
            initial_response=_ok_response(), max_retries=3, retry_upstream_drop=True,
        )
        assert raised is None
        assert items == ["chunk-a", "chunk-b"]
        assert len(calls) == 2
        assert make_request.await_count == 1

    @pytest.mark.asyncio
    async def test_drop_after_output_is_not_retried(self):
        """
        What it does: Attempt 1 yields one chunk, then drops.
        Purpose: A retry would append a second answer; the error is propagated and the chunk
        already delivered is not repeated.
        """
        make_request = AsyncMock(side_effect=lambda: _ok_response())

        async def processor(response):
            yield "partial text"
            raise _drop_error()

        items, raised = await self._collect(
            make_request=make_request, stream_processor=processor,
            initial_response=_ok_response(), max_retries=3, retry_upstream_drop=True,
        )
        assert items == ["partial text"]
        assert isinstance(raised, httpx.RemoteProtocolError)
        assert make_request.await_count == 0

    @pytest.mark.asyncio
    async def test_preamble_chunk_does_not_block_retry(self):
        """
        What it does: Attempt 1 yields only a preamble chunk, then drops. Attempt 2 starts
        with the same kind of preamble.
        Purpose: is_preamble_chunk marks chunks without model output (Anthropic message_start).
        """
        make_request = AsyncMock(side_effect=lambda: _ok_response())
        attempts = []

        async def processor(response):
            attempts.append(1)
            if len(attempts) == 1:
                yield "PREAMBLE"
                raise _drop_error()
            yield "text"

        items, raised = await self._collect(
            make_request=make_request, stream_processor=processor,
            initial_response=_ok_response(), max_retries=3, retry_upstream_drop=True,
            is_preamble_chunk=lambda c: c == "PREAMBLE",
        )
        assert raised is None
        assert items == ["PREAMBLE", "text"]

    @pytest.mark.asyncio
    async def test_output_chunk_after_preamble_blocks_retry(self):
        """
        What it does: Preamble, then a content chunk, then a drop.
        Purpose: Only chunks that are not preamble count as output.
        """
        make_request = AsyncMock(side_effect=lambda: _ok_response())

        async def processor(response):
            yield "PREAMBLE"
            yield "content"
            raise _drop_error()

        items, raised = await self._collect(
            make_request=make_request, stream_processor=processor,
            initial_response=_ok_response(), max_retries=3, retry_upstream_drop=True,
            is_preamble_chunk=lambda c: c == "PREAMBLE",
        )
        assert items == ["PREAMBLE", "content"]
        assert isinstance(raised, httpx.RemoteProtocolError)
        assert make_request.await_count == 0

    @pytest.mark.asyncio
    async def test_switch_off_restores_old_behavior(self):
        """
        What it does: retry_upstream_drop=False and a drop before any output.
        Purpose: Opt-out: the error propagates after a single attempt.
        """
        make_request = AsyncMock(side_effect=lambda: _ok_response())

        async def processor(response):
            raise _drop_error()
            yield

        items, raised = await self._collect(
            make_request=make_request, stream_processor=processor,
            initial_response=_ok_response(), max_retries=3, retry_upstream_drop=False,
        )
        assert items == []
        assert isinstance(raised, httpx.RemoteProtocolError)
        assert make_request.await_count == 0

    @pytest.mark.asyncio
    async def test_reads_config_when_override_is_none(self):
        """
        What it does: No override; UPSTREAM_DROP_RETRY patched to False, then to True.
        Purpose: The default comes from the config switch.
        """
        async def processor(response):
            raise _drop_error()
            yield

        with patch("kiro.streaming_core.UPSTREAM_DROP_RETRY", False):
            make_request = AsyncMock(side_effect=lambda: _ok_response())
            _, raised = await self._collect(
                make_request=make_request, stream_processor=processor,
                initial_response=_ok_response(), max_retries=3,
            )
            assert make_request.await_count == 0
            assert isinstance(raised, httpx.RemoteProtocolError)
        with patch("kiro.streaming_core.UPSTREAM_DROP_RETRY", True):
            make_request = AsyncMock(side_effect=lambda: _ok_response())
            _, raised = await self._collect(
                make_request=make_request, stream_processor=processor,
                initial_response=_ok_response(), max_retries=3,
            )
            assert make_request.await_count == 2

    @pytest.mark.asyncio
    async def test_all_attempts_dropping_raises_the_drop_error(self):
        """
        What it does: Every attempt drops before output; budget is 3.
        Purpose: The real cause is raised (not a misleading first token timeout) and exactly
        3 requests are made.
        """
        make_request = AsyncMock(side_effect=lambda: _ok_response())

        async def processor(response):
            raise _drop_error()
            yield

        items, raised = await self._collect(
            make_request=make_request, stream_processor=processor,
            initial_response=_ok_response(), max_retries=3, retry_upstream_drop=True,
        )
        assert items == []
        assert isinstance(raised, httpx.RemoteProtocolError)
        assert make_request.await_count == 2  # initial response + 2 retries = 3 attempts

    @pytest.mark.asyncio
    async def test_timeout_and_drop_share_the_attempt_budget(self):
        """
        What it does: Attempt 1 first token timeout, attempt 2 drop, attempt 3 success.
        Purpose: Both failure kinds are retried and counted against the same budget.
        """
        responses = []
        make_request = AsyncMock(side_effect=lambda: _ok_response())

        async def processor(response):
            responses.append(response)
            if len(responses) == 1:
                raise FirstTokenTimeoutError("slow")
            if len(responses) == 2:
                raise _drop_error()
            yield "done"

        items, raised = await self._collect(
            make_request=make_request, stream_processor=processor,
            initial_response=_ok_response(), max_retries=3, retry_upstream_drop=True,
        )
        assert raised is None
        assert items == ["done"]
        assert make_request.await_count == 2

    @pytest.mark.asyncio
    async def test_failed_response_is_closed_before_retry(self):
        """
        What it does: Attempt 1 drops; its response must be closed.
        Purpose: No leaked upstream connection (CLOSE_WAIT) on retry.
        """
        first = _ok_response()
        second = _ok_response()
        make_request = AsyncMock(return_value=second)
        seen = []

        async def processor(response):
            seen.append(response)
            if len(seen) == 1:
                raise _drop_error()
            yield "x"

        await self._collect(
            make_request=make_request, stream_processor=processor,
            initial_response=first, max_retries=3, retry_upstream_drop=True,
        )
        first.aclose.assert_awaited()

    @pytest.mark.asyncio
    async def test_ready_marker_is_sent_once_across_a_drop_retry(self):
        """
        What it does: emit_ready_marker=True; attempt 1 drops after the marker, attempt 2 works.
        Purpose: The route consumes the marker once; a second one would reach the client.
        """
        make_request = AsyncMock(side_effect=lambda: _ok_response())
        seen = []

        async def processor(response):
            seen.append(1)
            if len(seen) == 1:
                raise _drop_error()
            yield "x"

        items, raised = await self._collect(
            make_request=make_request, stream_processor=processor,
            initial_response=_ok_response(), max_retries=3, retry_upstream_drop=True,
            emit_ready_marker=True, empty_response_as_error=False,
        )
        assert raised is None
        assert items.count(STREAM_READY) == 1
        assert items[-1] == "x"

    @pytest.mark.asyncio
    async def test_drop_inside_the_guard_peek_is_retried(self):
        """
        What it does: A real response double whose byte stream raises the drop error on the
        first read, so the failure happens in guard_upstream_response.
        Purpose: The production traceback shows drops during the peek phase as well.
        """
        def dropping():
            response = _ok_response()

            async def gen():
                raise _drop_error()
                yield b""

            response.aiter_bytes = lambda *a, **k: gen()
            return response

        good = _ok_response()

        async def good_gen():
            yield b"abc"

        good.aiter_bytes = lambda *a, **k: good_gen()
        make_request = AsyncMock(return_value=good)

        async def processor(response):
            data = b"".join([c async for c in response.aiter_bytes()])
            yield data.decode()

        items, raised = await self._collect(
            make_request=make_request, stream_processor=processor,
            initial_response=dropping(), max_retries=3, retry_upstream_drop=True,
            empty_response_as_error=True,
        )
        assert raised is None
        assert items == ["abc"]
        assert make_request.await_count == 1

    @pytest.mark.asyncio
    async def test_non_drop_errors_are_still_not_retried(self):
        """
        What it does: A RuntimeError and an httpx.ReadTimeout before any output.
        Purpose: Only the drop classes are retried; everything else propagates immediately.
        """
        for error in (RuntimeError("boom"), httpx.ReadTimeout("slow")):
            make_request = AsyncMock(side_effect=lambda: _ok_response())

            async def processor(response, error=error):
                raise error
                yield

            _, raised = await self._collect(
                make_request=make_request, stream_processor=processor,
                initial_response=_ok_response(), max_retries=3, retry_upstream_drop=True,
            )
            assert raised is error
            assert make_request.await_count == 0

    @pytest.mark.asyncio
    async def test_retry_logs_one_stream_drop_warning(self):
        """
        What it does: Drop before output, then success; captures the logs.
        Purpose: A [StreamDrop] WARNING with action=retry explains the recovery, and no
        "Unexpected error" ERROR is logged for a recovered drop.
        """
        messages, handler_id = _capture_logs("WARNING")
        try:
            make_request = AsyncMock(side_effect=lambda: _ok_response())
            seen = []

            async def processor(response):
                seen.append(1)
                if len(seen) == 1:
                    raise _drop_error()
                yield "x"

            await self._collect(
                make_request=make_request, stream_processor=processor,
                initial_response=_ok_response(), max_retries=3, retry_upstream_drop=True,
            )
        finally:
            logger.remove(handler_id)
        drop_lines = [m for m in messages if "[StreamDrop]" in m]
        print(drop_lines)
        assert len(drop_lines) == 1
        assert drop_lines[0].startswith("WARNING|")
        assert "action=retry" in drop_lines[0]
        assert "attempt=1/3" in drop_lines[0]
        assert not [m for m in messages if "Unexpected error during streaming" in m]

    @pytest.mark.asyncio
    async def test_give_up_after_output_logs_an_error_with_reason(self):
        """
        What it does: Drop after output; captures the logs.
        Purpose: The [StreamDrop] ERROR says why no retry was made (output_sent) and carries
        the counters, without leaking the message text of the exception.
        """
        messages, handler_id = _capture_logs("WARNING")
        try:
            async def processor(response):
                yield "partial"
                raise httpx.RemoteProtocolError("SECRET_PAYLOAD")

            await self._collect(
                make_request=AsyncMock(), stream_processor=processor,
                initial_response=_ok_response(), max_retries=3, retry_upstream_drop=True,
            )
        finally:
            logger.remove(handler_id)
        drop_lines = [m for m in messages if "[StreamDrop]" in m]
        assert len(drop_lines) == 1
        assert drop_lines[0].startswith("ERROR|")
        assert f"action={DROP_ACTION_OUTPUT_SENT}" in drop_lines[0]
        assert "client_chunks=1" in drop_lines[0]
        assert "SECRET_PAYLOAD" not in drop_lines[0]


class TestStreamStatsLogging:
    """[StreamStats] summary for streams that complete normally."""

    def test_describe_accepts_no_error(self):
        """
        What it does: describe_stream_drop with error=None and a max_gap_s counter.
        Purpose: The same formatter serves completed streams.
        """
        from kiro.upstream_guard import GuardedResponse
        guarded = GuardedResponse(MagicMock(), None, [])
        guarded.max_gap_s = 7.25
        text = describe_stream_drop(guarded, None, time.monotonic(), 3)
        assert "error=none" in text
        assert "max_gap=7.2s" in text or "max_gap=7.3s" in text

    def test_describe_reports_na_max_gap_without_counter(self):
        """
        What it does: Plain object without max_gap_s.
        Purpose: Unknown is shown as n/a.
        """
        assert "max_gap=n/a" in describe_stream_drop(object(), None, time.monotonic(), 0)

    @pytest.mark.asyncio
    async def test_completed_stream_logs_one_stats_line(self):
        """
        What it does: A normal stream through the retry wrapper.
        Purpose: Exactly one INFO [StreamStats] line with outcome=complete.
        """
        messages, handler_id = _capture_logs("INFO")
        try:
            async def processor(response):
                yield "a"
            async for _ in stream_with_first_token_retry(
                make_request=AsyncMock(), stream_processor=processor,
                initial_response=_ok_response(), max_retries=3, first_token_timeout=30,
                empty_response_as_error=False,
            ):
                pass
        finally:
            logger.remove(handler_id)
        stats = [m for m in messages if "[StreamStats]" in m]
        assert len(stats) == 1
        assert stats[0].startswith("INFO|")
        assert "outcome=complete" in stats[0]
        assert "client_chunks=1" in stats[0]



# ==================================================================================================
# Attempt evidence capture (stream-drop recovery, _parse_single_stream)
# ==================================================================================================

async def _drain_with_evidence(response, evidence, **kwargs):
    """Runs _parse_single_stream until it ends or raises; returns (events, exception or None)."""
    events = []
    error = None
    try:
        async for event in _parse_single_stream(
            response, first_token_timeout=30, evidence=evidence, **kwargs
        ):
            events.append(event)
    except Exception as exc:  # noqa: BLE001 - the test inspects any drop exception
        error = exc
    return events, error


@pytest.mark.usefixtures("pinned_reasoning")
class TestAttemptEvidenceCapture:
    """_parse_single_stream fills AttemptEvidence on every way out; parse_kiro_stream is unchanged."""

    @pytest.mark.asyncio
    async def test_drop_after_text(self, aws_frames):
        """
        What it does: Text frames, then a RemoteProtocolError.
        Goal: Evidence has the text, no tools, no reasoning, no metering; the drop propagates.
        """
        evidence = AttemptEvidence()
        response = _stream_response([
            aws_frames.content("Hello"), aws_frames.content(" world"), _drop_error(),
        ])
        events, error = await _drain_with_evidence(response, evidence)
        assert isinstance(error, httpx.RemoteProtocolError)
        assert [e.content for e in events] == ["Hello", " world"]
        assert evidence.text == "Hello world"
        assert evidence.reasoning_seen is False
        assert evidence.completed_tools == [] and evidence.interrupted_tool is None
        assert evidence.saw_metering is False

    @pytest.mark.asyncio
    async def test_drop_mid_tool_keeps_raw_partial_args_and_id(self, aws_frames):
        """
        What it does: Text, a tool start and two input fragments, then a drop.
        Goal: interrupted_tool has the id, the name and the RAW partial arguments (not "{}").
        """
        evidence = AttemptEvidence()
        response = _stream_response([
            aws_frames.content("Let me write it."),
            aws_frames.tool_start("Write", "tooluse_1"),
            aws_frames.tool_input('{"path": "/tmp/a", "content": "par', "Write", "tooluse_1"),
            aws_frames.tool_input("tial", "Write", "tooluse_1"),
            _drop_error(),
        ])
        events, error = await _drain_with_evidence(response, evidence)
        assert isinstance(error, httpx.RemoteProtocolError)
        assert [e.type for e in events] == ["content"]  # tool events are only yielded at the end
        assert evidence.text == "Let me write it."
        assert evidence.completed_tools == []
        assert evidence.interrupted_tool["id"] == "tooluse_1"
        assert evidence.interrupted_tool["function"]["name"] == "Write"
        assert evidence.interrupted_tool["function"]["arguments"] == '{"path": "/tmp/a", "content": "partial'

    @pytest.mark.asyncio
    async def test_drop_after_complete_tool(self, aws_frames):
        """
        What it does: A tool with its stop frame, then a drop.
        Goal: The tool is in completed_tools, nothing is interrupted.
        """
        evidence = AttemptEvidence()
        response = _stream_response([
            aws_frames.tool_start("Read", "t1"),
            aws_frames.tool_input('{"path": "/x"}', "Read", "t1"),
            aws_frames.tool_stop("Read", "t1"),
            _drop_error(),
        ])
        _, error = await _drain_with_evidence(response, evidence)
        assert isinstance(error, httpx.RemoteProtocolError)
        assert [t["id"] for t in evidence.completed_tools] == ["t1"]
        assert evidence.completed_tools[0]["function"]["arguments"] == '{"path": "/x"}'
        assert evidence.interrupted_tool is None

    @pytest.mark.asyncio
    async def test_drop_with_complete_and_interrupted_tool(self, aws_frames):
        """
        What it does: One finished tool and a second one still streaming, then a drop.
        Goal: Both are captured in their own field.
        """
        evidence = AttemptEvidence()
        response = _stream_response([
            aws_frames.tool_start("A", "t1"),
            aws_frames.tool_input('{"a": 1}', "A", "t1"),
            aws_frames.tool_stop("A", "t1"),
            aws_frames.tool_start("B", "t2"),
            aws_frames.tool_input('{"b": [', "B", "t2"),
            _drop_error(),
        ])
        await _drain_with_evidence(response, evidence)
        assert [t["id"] for t in evidence.completed_tools] == ["t1"]
        assert evidence.interrupted_tool["id"] == "t2"
        assert evidence.interrupted_tool["function"]["arguments"] == '{"b": ['

    @pytest.mark.asyncio
    async def test_drop_after_reasoning_only(self, aws_frames):
        """
        What it does: Native reasoning frames, then a drop.
        Goal: reasoning_seen is True, no text, no tools.
        """
        evidence = AttemptEvidence()
        response = _stream_response([
            aws_frames.reasoning("thinking"), aws_frames.reasoning(" more"), _drop_error(),
        ])
        events, error = await _drain_with_evidence(response, evidence)
        assert isinstance(error, httpx.RemoteProtocolError)
        assert [e.type for e in events] == ["thinking", "thinking"]
        assert evidence.reasoning_seen is True
        assert evidence.text == ""
        assert not evidence.has_tools

    @pytest.mark.asyncio
    async def test_discarded_reasoning_still_counts_as_reasoning(self, aws_frames):
        """
        What it does: FAKE_REASONING_HANDLING=remove discards the reasoning events.
        Goal: Nothing is yielded but the evidence still says reasoning was produced.
        """
        evidence = AttemptEvidence()
        response = _stream_response([aws_frames.reasoning("hidden"), _drop_error()])
        with patch("kiro.streaming_core.FAKE_REASONING_HANDLING", "remove"):
            events, _ = await _drain_with_evidence(response, evidence)
        assert events == []
        assert evidence.reasoning_seen is True

    @pytest.mark.asyncio
    async def test_drop_before_any_output(self, aws_frames):
        """
        What it does: Only a contextUsageEvent arrives, then a drop.
        Goal: Evidence stays empty (the resend row of the decision table).
        """
        evidence = AttemptEvidence()
        response = _stream_response([aws_frames.context_usage(1.0), _drop_error()])
        _, error = await _drain_with_evidence(response, evidence)
        assert isinstance(error, httpx.RemoteProtocolError)
        assert evidence == AttemptEvidence()

    @pytest.mark.asyncio
    async def test_metering_before_drop_is_recorded(self, aws_frames):
        """
        What it does: Text and a meteringEvent, then a drop.
        Goal: saw_metering is True (the response was complete).
        """
        evidence = AttemptEvidence()
        response = _stream_response([
            aws_frames.content("done"), aws_frames.metering(0.2), _drop_error(),
        ])
        await _drain_with_evidence(response, evidence)
        assert evidence.saw_metering is True
        assert evidence.text == "done"

    @pytest.mark.asyncio
    async def test_normal_end_leaves_consistent_evidence(self, aws_frames):
        """
        What it does: A complete stream with text, a finished tool and metering.
        Goal: No exception; evidence matches what was yielded.
        """
        evidence = AttemptEvidence()
        response = _stream_response([
            aws_frames.content("Hi"),
            aws_frames.tool_start("A", "t1"),
            aws_frames.tool_input('{"a": 1}', "A", "t1"),
            aws_frames.tool_stop("A", "t1"),
            aws_frames.metering(0.3),
        ])
        events, error = await _drain_with_evidence(response, evidence)
        assert error is None
        event_types = [e.type for e in events]
        assert event_types[0] == "content"
        assert event_types.count("tool_use") == 1
        assert evidence.text == "Hi"
        assert [t["id"] for t in evidence.completed_tools] == ["t1"]
        assert evidence.interrupted_tool is None
        assert evidence.saw_metering is True

    @pytest.mark.asyncio
    async def test_normal_end_with_unfinished_tool_is_finalized_not_interrupted(self, aws_frames):
        """
        What it does: Stream ends normally while a tool has no stop frame.
        Goal: The parser finalizes it on the normal path, so it is completed, not interrupted.
        """
        evidence = AttemptEvidence()
        response = _stream_response([
            aws_frames.tool_start("A", "t1"),
            aws_frames.tool_input('{"a": 1}', "A", "t1"),
        ])
        _, error = await _drain_with_evidence(response, evidence)
        assert error is None
        assert [t["id"] for t in evidence.completed_tools] == ["t1"]
        assert evidence.interrupted_tool is None

    @pytest.mark.asyncio
    async def test_first_token_timeout_leaves_empty_evidence(self):
        """
        What it does: The stream never produces a chunk.
        Goal: FirstTokenTimeoutError propagates and the evidence stays empty.
        """
        evidence = AttemptEvidence()
        events = []
        with pytest.raises(FirstTokenTimeoutError):
            async for event in _parse_single_stream(
                _hanging_response(), first_token_timeout=0.05, evidence=evidence
            ):
                events.append(event)
        assert evidence == AttemptEvidence()

    @pytest.mark.asyncio
    async def test_evidence_is_filled_when_the_consumer_closes_early(self, aws_frames):
        """
        What it does: The consumer stops after the first event and closes the generator.
        Goal: The evidence reflects what was read so far (finally block runs).
        """
        evidence = AttemptEvidence()
        response = _stream_response([
            aws_frames.content("first"),
            aws_frames.tool_start("A", "t1"),
            aws_frames.tool_input('{"a', "A", "t1"),
        ])
        generator = _parse_single_stream(response, first_token_timeout=30, evidence=evidence)
        first = await generator.__anext__()
        assert first.content == "first"
        await generator.aclose()
        assert evidence.text == "first"

    @pytest.mark.asyncio
    async def test_utf8_text_split_across_chunks_is_complete_in_evidence(self, aws_frames):
        """
        What it does: Multi-byte text frame split into two network chunks, then a drop.
        Goal: Evidence text is the full string.
        """
        frame = aws_frames.content("你好，世界")
        mid = len(frame) - 12
        evidence = AttemptEvidence()
        await _drain_with_evidence(
            _stream_response([frame[:mid], frame[mid:], _drop_error()]), evidence
        )
        assert evidence.text == "你好，世界"

    @pytest.mark.asyncio
    async def test_dispatcher_matches_single_stream_for_plain_responses(self, aws_frames):
        """
        What it does: Runs parse_kiro_stream and _parse_single_stream on identical streams.
        Goal: Same events: the dispatcher adds no behaviour for plain responses.
        """
        chunks = [aws_frames.content("A"), aws_frames.content("B"), aws_frames.metering(0.1)]
        via_dispatcher = [e async for e in parse_kiro_stream(_stream_response(chunks), first_token_timeout=30)]
        via_single = [e async for e in _parse_single_stream(_stream_response(chunks), first_token_timeout=30)]
        assert via_dispatcher == via_single

    @pytest.mark.asyncio
    async def test_dispatcher_closes_inner_generator_on_early_close(self, aws_frames):
        """
        What it does: Closes parse_kiro_stream after the first event.
        Goal: The inner generator is finalized at once (its outcome is not marked finished).
        """
        outcome = StreamOutcome()
        generator = parse_kiro_stream(
            _stream_response([aws_frames.content("a"), aws_frames.content("b")]),
            first_token_timeout=30,
            outcome=outcome,
        )
        await generator.__anext__()
        await generator.aclose()
        assert outcome.finished is False

    @pytest.mark.asyncio
    async def test_no_content_in_logs(self, aws_frames):
        """
        What it does: Drops a stream that carried secret text and secret tool arguments.
        Goal: Neither appears in WARNING+ or INFO logs emitted while capturing evidence.
        """
        messages, handler_id = _capture_logs(level="INFO")
        try:
            evidence = AttemptEvidence()
            await _drain_with_evidence(_stream_response([
                aws_frames.content(BODY_SECRET),
                aws_frames.tool_start("A", "t1"),
                aws_frames.tool_input('{"k": "' + THOUGHT_SECRET, "A", "t1"),
                _drop_error(),
            ]), evidence)
        finally:
            logger.remove(handler_id)
        joined = "\n".join(messages)
        assert BODY_SECRET not in joined
        assert THOUGHT_SECRET not in joined


# ==================================================================================================
# Recovery session (RecoverableResponse, Kiro IDE parity)
# ==================================================================================================

from fastapi import HTTPException
from kiro.stream_recovery import (
    CONTINUE_PROMPT,
    INTERRUPTED_TOOL_RESULT_MESSAGE,
    RecoverySettings,
    UpstreamStreamInterrupted,
)
from kiro.streaming_core import RecoverableResponse, wrap_recoverable

PARTIAL_SECRET = "PARTIAL_ARGS_SECRET_91c2"


def _base_payload():
    """Minimal Kiro payload with one tool, history and a current user message."""
    return {
        "profileArn": "arn:test",
        "conversationState": {
            "conversationId": "conv-1",
            "chatTriggerType": "MANUAL",
            "history": [],
            "currentMessage": {"userInputMessage": {
                "content": "write the file",
                "modelId": "claude-haiku-5.5",
                "origin": "AI_EDITOR",
                "userInputMessageContext": {"tools": [{"toolSpecification": {"name": "Write"}}]},
            }},
        },
    }


@pytest.fixture
def recovery_env():
    """Recovery switch on, empty guard on, fake reasoning off, backoff sleep patched (yielded)."""
    sleep = AsyncMock()
    with patch("kiro.streaming_core.UPSTREAM_DROP_RETRY", True), \
            patch("kiro.streaming_core.EMPTY_RESPONSE_AS_ERROR", True), \
            patch("kiro.streaming_core.FAKE_REASONING_ENABLED", False), \
            patch("kiro.streaming_core.FAKE_REASONING_HANDLING", "as_reasoning_content"), \
            patch("kiro.streaming_core.backoff_sleep", sleep):
        yield sleep


def _session(first, *later, max_attempts=3):
    """
    Builds a RecoverableResponse over scripted attempts.

    Args:
        first: Chunks (bytes or exceptions) of the first attempt
        later: Chunk lists of the following attempts (served by make_request, in order)
        max_attempts: Attempt budget for plain resends

    Returns:
        (RecoverableResponse, make_request AsyncMock, first response double, later response doubles)
    """
    first_response = _stream_response(first)
    later_responses = [_stream_response(chunks) for chunks in later]
    make_request = AsyncMock(side_effect=later_responses)
    settings = RecoverySettings(
        payload=_base_payload(), make_request=make_request, max_attempts=max_attempts
    )
    return RecoverableResponse(first_response, settings), make_request, first_response, later_responses


async def _run_session(session, outcome=None, first_token_timeout=5):
    """Drains parse_kiro_stream(session); returns (events, raised exception or None)."""
    events = []
    raised = None
    try:
        async for event in parse_kiro_stream(
            session, first_token_timeout=first_token_timeout, outcome=outcome
        ):
            events.append(event)
    except Exception as exc:  # noqa: BLE001 - the test inspects the exception
        raised = exc
    return events, raised


def _tool_frames(aws_frames, tool_id, name="Write", args='{"path": "/a", "content": "x"}', stop=True):
    """Frames of one complete tool call."""
    frames = [aws_frames.tool_start(name, tool_id), aws_frames.tool_input(args, name, tool_id)]
    if stop:
        frames.append(aws_frames.tool_stop(name, tool_id))
    return frames


class TestWrapRecoverable:
    """wrap_recoverable() decides whether a response gets a recovery session."""

    def test_no_settings_returns_response(self):
        """
        What it does: settings is None.
        Goal: The response is returned unchanged.
        """
        response = _stream_response([])
        assert wrap_recoverable(response, None) is response

    def test_switch_off_returns_response(self):
        """
        What it does: UPSTREAM_DROP_RETRY is False.
        Goal: The response is returned unchanged (old behavior).
        """
        response = _stream_response([])
        settings = RecoverySettings(payload={}, make_request=AsyncMock())
        with patch("kiro.streaming_core.UPSTREAM_DROP_RETRY", False):
            assert wrap_recoverable(response, settings) is response

    def test_switch_on_wraps(self):
        """
        What it does: Switch on and settings given.
        Goal: A RecoverableResponse with status 200 and delegated attributes.
        """
        response = _stream_response([])
        response.extra_attr = "x"
        settings = RecoverySettings(payload={}, make_request=AsyncMock())
        with patch("kiro.streaming_core.UPSTREAM_DROP_RETRY", True):
            wrapped = wrap_recoverable(response, settings)
        assert isinstance(wrapped, RecoverableResponse)
        assert wrapped.status_code == 200
        assert wrapped.extra_attr == "x"

    def test_explicit_enabled_overrides_config(self):
        """
        What it does: enabled=True while the config switch is off.
        Goal: The override wins.
        """
        settings = RecoverySettings(payload={}, make_request=AsyncMock())
        with patch("kiro.streaming_core.UPSTREAM_DROP_RETRY", False):
            assert isinstance(wrap_recoverable(_stream_response([]), settings, enabled=True), RecoverableResponse)

    def test_already_wrapped_is_not_wrapped_twice(self):
        """
        What it does: Wraps a RecoverableResponse again.
        Goal: Same object back.
        """
        settings = RecoverySettings(payload={}, make_request=AsyncMock())
        wrapped = RecoverableResponse(_stream_response([]), settings)
        assert wrap_recoverable(wrapped, settings, enabled=True) is wrapped

    @pytest.mark.asyncio
    async def test_guard_passes_a_recoverable_response_through(self):
        """
        What it does: guard_upstream_response() on a RecoverableResponse.
        Goal: Returned unchanged and nothing is read from it.
        """
        response = _stream_response([])
        settings = RecoverySettings(payload={}, make_request=AsyncMock())
        wrapped = RecoverableResponse(response, settings)
        assert await guard_upstream_response(wrapped, first_token_timeout=5) is wrapped
        assert response.pulled == []


class TestRecoverableResponseRules:
    """Decision rules 2-6 end to end through parse_kiro_stream()."""

    @pytest.mark.asyncio
    async def test_rule2_interrupted_tool_is_continued(self, aws_frames, recovery_env):
        """
        What it does: Text and a tool call still streaming, then a drop; the next attempt
        re-issues the tool and ends normally.
        Goal: One event stream: text, usage, the new tool call (tool calls come last). The second request carries the
        partial assistant message and the failed tool result; the backoff is about 500 ms.
        """
        session, make_request, first, later = _session(
            [aws_frames.content("Let me write it."),
             aws_frames.tool_start("Write", "t1"),
             aws_frames.tool_input('{"path": "/a", "content": "' + PARTIAL_SECRET, "Write", "t1"),
             _drop_error()],
            _tool_frames(aws_frames, "t2") + [aws_frames.metering(0.5)],
        )
        outcome = StreamOutcome()
        events, raised = await _run_session(session, outcome)

        assert raised is None
        assert [e.type for e in events] == ["content", "usage", "tool_use"]
        assert events[2].tool_use["id"] == "t2"
        assert outcome.finished is True and outcome.produced_tool_use is True
        make_request.assert_awaited_once()
        sent = make_request.await_args.args[0]
        history = sent["conversationState"]["history"]
        assert history[-2]["userInputMessage"]["content"] == "write the file"
        assistant = history[-1]["assistantResponseMessage"]
        assert assistant["content"] == "Let me write it."
        assert assistant["toolUses"][0]["toolUseId"] == "t1"
        assert assistant["toolUses"][0]["input"] == {
            "invalidArgs": '{"path": "/a", "content": "' + PARTIAL_SECRET
        }
        current = sent["conversationState"]["currentMessage"]["userInputMessage"]
        assert current["content"] == ""
        result = current["userInputMessageContext"]["toolResults"][0]
        assert result["toolUseId"] == "t1" and result["status"] == "error"
        assert result["content"] == [{"text": INTERRUPTED_TOOL_RESULT_MESSAGE}]
        recovery_env.assert_awaited_once()
        assert 0.4 <= recovery_env.await_args.args[0] <= 0.6
        first.aclose.assert_awaited()

    @pytest.mark.asyncio
    async def test_rule3_drop_before_output_resends_the_same_payload(self, aws_frames, recovery_env):
        """
        What it does: Only a neutral frame, then a drop; the second attempt succeeds.
        Goal: The request is the unchanged base payload object (identity kept).
        """
        session, make_request, _, _ = _session(
            [aws_frames.context_usage(1.0), _drop_error()],
            [aws_frames.content("Hello"), aws_frames.metering(0.1)],
        )
        events, raised = await _run_session(session)

        assert raised is None
        assert [e.content for e in events if e.type == "content"] == ["Hello"]
        assert make_request.await_args.args[0] is session._settings.payload

    @pytest.mark.asyncio
    async def test_rule4_drop_after_text_resends_once_with_text(self, aws_frames, recovery_env):
        """
        What it does: Text, a drop, the second attempt streams more text and ends.
        Goal: Both texts are forwarded as is; the request holds the partial text plus the
        continue prompt and no tool results.
        """
        session, make_request, _, _ = _session(
            [aws_frames.content("The answer is"), _drop_error()],
            [aws_frames.content("The answer is 42."), aws_frames.metering(0.1)],
        )
        events, raised = await _run_session(session)

        assert raised is None
        assert "".join(e.content for e in events if e.type == "content") == "The answer isThe answer is 42."
        sent = make_request.await_args.args[0]
        assert sent["conversationState"]["history"][-1]["assistantResponseMessage"] == {"content": "The answer is"}
        current = sent["conversationState"]["currentMessage"]["userInputMessage"]
        assert current["content"] == CONTINUE_PROMPT
        assert "toolResults" not in current["userInputMessageContext"]

    @pytest.mark.asyncio
    async def test_rule4_second_text_drop_hits_the_duplication_cap(self, aws_frames, recovery_env):
        """
        What it does: Text drops twice.
        Goal: One resend only, then UpstreamStreamInterrupted(duplication_cap).
        """
        session, make_request, _, _ = _session(
            [aws_frames.content("a"), _drop_error()],
            [aws_frames.content("b"), _drop_error()],
        )
        events, raised = await _run_session(session)

        assert isinstance(raised, UpstreamStreamInterrupted)
        assert raised.reason == "duplication_cap"
        assert isinstance(raised.__cause__, httpx.RemoteProtocolError)
        assert make_request.await_count == 1
        assert [e.content for e in events] == ["a", "b"]

    @pytest.mark.asyncio
    async def test_rule5_reasoning_only_fails_without_a_request(self, aws_frames, recovery_env):
        """
        What it does: Native reasoning, then a drop.
        Goal: UpstreamStreamInterrupted(reasoning_only), no new request, no sleep.
        """
        session, make_request, first, _ = _session(
            [aws_frames.reasoning("thinking"), _drop_error()]
        )
        events, raised = await _run_session(session)

        assert isinstance(raised, UpstreamStreamInterrupted)
        assert raised.reason == "reasoning_only"
        assert [e.type for e in events] == ["thinking"]
        make_request.assert_not_awaited()
        recovery_env.assert_not_awaited()
        first.aclose.assert_awaited()

    @pytest.mark.asyncio
    async def test_rule6_complete_tool_then_drop_is_delivered(self, aws_frames, recovery_env):
        """
        What it does: Text, a complete tool call (stop frame), then a drop.
        Goal: The tool call is yielded, the stream ends normally, no request, outcome finished.
        """
        session, make_request, _, _ = _session(
            [aws_frames.content("Reading.")] + _tool_frames(aws_frames, "t1", "Read", '{"path": "/x"}')
            + [_drop_error()]
        )
        outcome = StreamOutcome()
        events, raised = await _run_session(session, outcome)

        assert raised is None
        assert [e.type for e in events] == ["content", "tool_use"]
        assert events[1].tool_use["id"] == "t1"
        assert events[1].tool_use["function"]["arguments"] == '{"path": "/x"}'
        assert outcome.finished is True and outcome.produced_tool_use is True
        make_request.assert_not_awaited()
        recovery_env.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_metering_seen_then_drop_finishes_without_resend(self, aws_frames, recovery_env):
        """
        What it does: Text and the meteringEvent arrive, then the connection drops.
        Goal: The answer was complete: normal end, usage event kept, no request.
        """
        session, make_request, _, _ = _session(
            [aws_frames.content("Done."), aws_frames.metering(0.3), _drop_error()]
        )
        outcome = StreamOutcome()
        events, raised = await _run_session(session, outcome)

        assert raised is None
        assert [e.type for e in events] == ["content", "usage"]
        assert outcome.finished is True
        make_request.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_normal_stream_needs_no_recovery(self, aws_frames, recovery_env):
        """
        What it does: A stream that ends normally.
        Goal: Same events as a plain response, no request, no sleep.
        """
        session, make_request, _, _ = _session(
            [aws_frames.content("Hi"), aws_frames.metering(0.1)]
        )
        events, raised = await _run_session(session)

        assert raised is None
        assert [e.type for e in events] == ["content", "usage"]
        make_request.assert_not_awaited()
        recovery_env.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_non_drop_error_propagates(self, aws_frames, recovery_env):
        """
        What it does: The stream raises a ValueError.
        Goal: Not a drop: it propagates unchanged and nothing is requested.
        """
        session, make_request, _, _ = _session(
            [aws_frames.content("a"), ValueError("boom")]
        )
        events, raised = await _run_session(session)

        assert isinstance(raised, ValueError)
        make_request.assert_not_awaited()


class TestRecoverableResponseBoundaries:
    """Caps, drops during a continuation, failing continuation requests, cleanup."""

    @pytest.mark.asyncio
    async def test_tool_recovery_cap_of_three(self, aws_frames, recovery_env):
        """
        What it does: Four consecutive attempts drop mid tool call.
        Goal: Three continuations, then UpstreamStreamInterrupted(streak_exhausted); the third
        request builds on the second (history grows by two entries per recovery).
        """
        def dropping(tool_id):
            return [aws_frames.tool_start("Write", tool_id),
                    aws_frames.tool_input('{"p": "', "Write", tool_id), _drop_error()]

        session, make_request, _, _ = _session(
            dropping("t1"), dropping("t2"), dropping("t3"), dropping("t4")
        )
        events, raised = await _run_session(session)

        assert isinstance(raised, UpstreamStreamInterrupted)
        assert raised.reason == "streak_exhausted"
        assert make_request.await_count == 3
        assert recovery_env.await_count == 3
        third = make_request.await_args_list[2].args[0]
        assert len(third["conversationState"]["history"]) == 6

    @pytest.mark.asyncio
    async def test_drop_during_continuation_without_output_fails(self, aws_frames, recovery_env):
        """
        What it does: Attempt 1 drops mid tool; the continuation drops before any frame.
        Goal: A tool was involved earlier, so no plain resend: reason tool_context.
        """
        session, make_request, _, _ = _session(
            [aws_frames.tool_start("Write", "t1"), aws_frames.tool_input('{"a"', "Write", "t1"), _drop_error()],
            [_drop_error()],
        )
        events, raised = await _run_session(session)

        assert isinstance(raised, UpstreamStreamInterrupted)
        assert raised.reason == "tool_context"
        assert make_request.await_count == 1

    @pytest.mark.asyncio
    async def test_drop_during_continuation_after_text_fails(self, aws_frames, recovery_env):
        """
        What it does: The continuation streams text, then drops.
        Goal: tool_context again: the new text stays with the client, no further request.
        """
        session, make_request, _, _ = _session(
            [aws_frames.tool_start("Write", "t1"), aws_frames.tool_input('{"a"', "Write", "t1"), _drop_error()],
            [aws_frames.content("partial"), _drop_error()],
        )
        events, raised = await _run_session(session)

        assert isinstance(raised, UpstreamStreamInterrupted)
        assert raised.reason == "tool_context"
        assert [e.content for e in events] == ["partial"]
        assert make_request.await_count == 1

    @pytest.mark.asyncio
    async def test_drop_during_continuation_after_complete_tool_finishes(self, aws_frames, recovery_env):
        """
        What it does: The continuation emits a complete tool call, then drops.
        Goal: Rule 6 applies inside the continuation: the tool call is delivered.
        """
        session, make_request, _, _ = _session(
            [aws_frames.tool_start("Write", "t1"), aws_frames.tool_input('{"a"', "Write", "t1"), _drop_error()],
            _tool_frames(aws_frames, "t2") + [_drop_error()],
        )
        outcome = StreamOutcome()
        events, raised = await _run_session(session, outcome)

        assert raised is None
        assert [e.tool_use["id"] for e in events if e.type == "tool_use"] == ["t2"]
        assert outcome.finished is True

    @pytest.mark.asyncio
    async def test_plain_resends_stop_at_the_attempt_budget(self, aws_frames, recovery_env):
        """
        What it does: Every attempt drops before output, budget 3.
        Goal: Two resends, then attempts_exhausted (3 attempts in total).
        """
        session, make_request, _, _ = _session(
            [_drop_error()], [_drop_error()], [_drop_error()], max_attempts=3
        )
        events, raised = await _run_session(session)

        assert isinstance(raised, UpstreamStreamInterrupted)
        assert raised.reason == "attempts_exhausted"
        assert make_request.await_count == 2
        assert [call.args[0] for call in recovery_env.await_args_list] != []
        delays = [call.args[0] for call in recovery_env.await_args_list]
        assert 0.4 <= delays[0] <= 0.6 and 0.8 <= delays[1] <= 1.2

    @pytest.mark.asyncio
    async def test_budget_of_one_attempt_never_resends(self, aws_frames, recovery_env):
        """
        What it does: max_attempts=1 and a drop before output.
        Goal: attempts_exhausted at once, no request.
        """
        session, make_request, _, _ = _session([_drop_error()], max_attempts=1)
        _, raised = await _run_session(session)

        assert raised.reason == "attempts_exhausted"
        make_request.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_continuation_http_error_status_fails(self, aws_frames, recovery_env):
        """
        What it does: The continuation request answers HTTP 400.
        Goal: UpstreamStreamInterrupted(continuation_failed); the rejected response is closed.
        """
        session, make_request, _, _ = _session(
            [aws_frames.tool_start("Write", "t1"), _drop_error()]
        )
        rejected = _ok_response()
        rejected.status_code = 400
        make_request.side_effect = [rejected]
        _, raised = await _run_session(session)

        assert isinstance(raised, UpstreamStreamInterrupted)
        assert raised.reason == "continuation_failed"
        rejected.aclose.assert_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("error", [
        httpx.ConnectError("no route"),
        HTTPException(status_code=502, detail="bad gateway"),
        OSError("socket"),
    ])
    async def test_continuation_request_error_fails(self, aws_frames, recovery_env, error):
        """
        What it does: make_request itself raises.
        Goal: UpstreamStreamInterrupted(continuation_failed) with the error as the cause.
        """
        session, make_request, _, _ = _session(
            [aws_frames.tool_start("Write", "t1"), _drop_error()]
        )
        make_request.side_effect = error
        _, raised = await _run_session(session)

        assert isinstance(raised, UpstreamStreamInterrupted)
        assert raised.reason == "continuation_failed"
        assert raised.__cause__ is error

    @pytest.mark.asyncio
    async def test_continuation_first_token_timeout_fails(self, aws_frames, recovery_env):
        """
        What it does: The continuation never sends a first chunk.
        Goal: UpstreamStreamInterrupted(continuation_failed), the hanging response is closed.
        """
        session, make_request, _, _ = _session(
            [aws_frames.tool_start("Write", "t1"), _drop_error()]
        )
        hanging = _hanging_response()
        make_request.side_effect = [hanging]
        _, raised = await _run_session(session, first_token_timeout=0.05)

        assert isinstance(raised, UpstreamStreamInterrupted)
        assert raised.reason == "continuation_failed"
        hanging.aclose.assert_awaited()

    @pytest.mark.asyncio
    async def test_continuation_empty_response_fails(self, aws_frames, recovery_env):
        """
        What it does: The continuation returns an empty (neutral frames only) response.
        Goal: UpstreamStreamInterrupted(continuation_failed).
        """
        session, make_request, _, _ = _session(
            [aws_frames.tool_start("Write", "t1"), _drop_error()],
            [aws_frames.context_usage(1.0)],
        )
        _, raised = await _run_session(session)

        assert isinstance(raised, UpstreamStreamInterrupted)
        assert raised.reason == "continuation_failed"

    @pytest.mark.asyncio
    async def test_drop_while_peeking_the_continuation_is_decided(self, aws_frames, recovery_env):
        """
        What it does: After a plain resend, the new stream drops before its first frame.
        Goal: The drop is decided like any other (another resend), the third attempt succeeds.
        """
        session, make_request, _, later = _session(
            [_drop_error()], [_drop_error()], [aws_frames.content("ok"), aws_frames.metering(0.1)],
            max_attempts=4,
        )
        events, raised = await _run_session(session)

        assert raised is None
        assert [e.content for e in events if e.type == "content"] == ["ok"]
        assert make_request.await_count == 2
        later[0].aclose.assert_awaited()

    @pytest.mark.asyncio
    async def test_first_attempt_first_token_timeout_propagates(self, aws_frames, recovery_env):
        """
        What it does: The first attempt is not guarded and never answers.
        Goal: FirstTokenTimeoutError propagates unchanged (the core retry handles it).
        """
        settings = RecoverySettings(payload=_base_payload(), make_request=AsyncMock())
        session = RecoverableResponse(_hanging_response(), settings)
        _, raised = await _run_session(session, first_token_timeout=0.05)

        assert isinstance(raised, FirstTokenTimeoutError)
        settings.make_request.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_consumer_close_closes_the_open_response(self, aws_frames, recovery_env):
        """
        What it does: The consumer closes the generator after the first event of a continuation.
        Goal: The continuation's response is closed (no leaked upstream connection).
        """
        session, make_request, _, later = _session(
            [aws_frames.tool_start("Write", "t1"), _drop_error()],
            [aws_frames.content("a"), aws_frames.content("b"), aws_frames.metering(0.1)],
        )
        generator = parse_kiro_stream(session, first_token_timeout=5)
        first_event = await generator.__anext__()
        await generator.aclose()

        assert first_event.content == "a"
        later[0].aclose.assert_awaited()

    @pytest.mark.asyncio
    async def test_scanner_tail_is_reset_between_attempts(self, aws_frames, recovery_env):
        """
        What it does: Attempt 1 ends inside a frame header name; attempt 2 starts with the
        rest of that name (the worst case for joined bytes).
        Goal: The shared scanner does not count the joined header, and it keeps the counters
        of attempt 1 (without reset_tail the count would be 3).
        """
        frame = aws_frames.content("cut")
        cut = frame.index(b"assistantResponseEvent") + 5
        session, make_request, _, _ = _session(
            [aws_frames.content("a"), frame[:cut], _drop_error()],
            [frame[cut:], aws_frames.content("b"), aws_frames.metering(0.1)],
        )
        outcome = StreamOutcome()
        await _run_session(session, outcome)

        assert outcome.scanner.counts["assistantResponseEvent"] == 2
        assert outcome.scanner.counts["meteringEvent"] == 1

    @pytest.mark.asyncio
    async def test_logs_hold_counts_only(self, aws_frames, recovery_env):
        """
        What it does: Recovers a drop whose partial arguments and text are secrets.
        Goal: A [StreamRecovery] line with counts exists; no content in any log line.
        """
        messages, handler_id = _capture_logs(level="DEBUG")
        try:
            session, _, _, _ = _session(
                [aws_frames.content(BODY_SECRET),
                 aws_frames.tool_start("Write", "t1"),
                 aws_frames.tool_input('{"k": "' + PARTIAL_SECRET, "Write", "t1"),
                 _drop_error()],
                _tool_frames(aws_frames, "t2") + [aws_frames.metering(0.1)],
            )
            await _run_session(session)
        finally:
            logger.remove(handler_id)
        joined = "\n".join(messages)
        recovery_lines = [m for m in messages if "[StreamRecovery]" in m]
        assert len(recovery_lines) == 1
        assert "action=continue_tools" in recovery_lines[0]
        assert "tools_interrupted=1" in recovery_lines[0]
        assert f"text_chars={len(BODY_SECRET)}" in recovery_lines[0]
        assert BODY_SECRET not in joined
        assert PARTIAL_SECRET not in joined

    @pytest.mark.asyncio
    async def test_give_up_is_logged_as_error(self, aws_frames, recovery_env):
        """
        What it does: A reasoning-only drop.
        Goal: The [StreamRecovery] line is ERROR level and names action=fail and the reason.
        """
        messages, handler_id = _capture_logs(level="WARNING")
        try:
            session, _, _, _ = _session([aws_frames.reasoning("r"), _drop_error()])
            await _run_session(session)
        finally:
            logger.remove(handler_id)
        lines = [m for m in messages if "[StreamRecovery]" in m]
        assert len(lines) == 1
        assert lines[0].startswith("ERROR|")
        assert "action=fail" in lines[0] and "reason=reasoning_only" in lines[0]

    @pytest.mark.asyncio
    async def test_unbuildable_continuation_fails(self, aws_frames, recovery_env):
        """
        What it does: The base payload has no current message.
        Goal: UpstreamStreamInterrupted(payload_unsupported) instead of a crash.
        """
        settings = RecoverySettings(payload={"conversationState": {}}, make_request=AsyncMock())
        session = RecoverableResponse(
            _stream_response([aws_frames.tool_start("Write", "t1"), _drop_error()]), settings
        )
        _, raised = await _run_session(session)

        assert isinstance(raised, UpstreamStreamInterrupted)
        assert raised.reason == "payload_unsupported"


class TestCollectStreamToResultRecovery:
    """collect_stream_to_result(recovery=...) recovers invisibly (non-streaming)."""

    @pytest.mark.asyncio
    async def test_interrupted_tool_is_recovered_invisibly(self, aws_frames, recovery_env):
        """
        What it does: Text and a half tool call, a drop, then a continuation with the full tool.
        Goal: One StreamResult with the text, exactly one tool call (the re-issued one), usage.
        """
        make_request = AsyncMock(return_value=_stream_response(
            _tool_frames(aws_frames, "t2") + [aws_frames.metering(0.5)]
        ))
        settings = RecoverySettings(payload=_base_payload(), make_request=make_request)
        response = _stream_response([
            aws_frames.content("Writing."),
            aws_frames.tool_start("Write", "t1"),
            aws_frames.tool_input('{"p', "Write", "t1"),
            _drop_error(),
        ])
        result = await collect_stream_to_result(response, first_token_timeout=5, recovery=settings)

        assert result.content == "Writing."
        assert [t["id"] for t in result.tool_calls] == ["t2"]
        assert result.usage == 0.5

    @pytest.mark.asyncio
    async def test_give_up_raises(self, aws_frames, recovery_env):
        """
        What it does: Reasoning-only drop.
        Goal: UpstreamStreamInterrupted propagates to the caller.
        """
        settings = RecoverySettings(payload=_base_payload(), make_request=AsyncMock())
        response = _stream_response([aws_frames.reasoning("r"), _drop_error()])
        with pytest.raises(UpstreamStreamInterrupted) as info:
            await collect_stream_to_result(response, first_token_timeout=5, recovery=settings)

        assert info.value.reason == "reasoning_only"

    @pytest.mark.asyncio
    async def test_complete_tool_then_drop_returns_the_tool(self, aws_frames, recovery_env):
        """
        What it does: A complete tool call, then a drop.
        Goal: The result holds the tool call; no request was made.
        """
        make_request = AsyncMock()
        settings = RecoverySettings(payload=_base_payload(), make_request=make_request)
        response = _stream_response(_tool_frames(aws_frames, "t1") + [_drop_error()])
        result = await collect_stream_to_result(response, first_token_timeout=5, recovery=settings)

        assert [t["id"] for t in result.tool_calls] == ["t1"]
        make_request.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_switch_off_keeps_the_old_behavior(self, aws_frames, recovery_env):
        """
        What it does: UPSTREAM_DROP_RETRY False and a drop.
        Goal: The raw httpx error propagates, no request.
        """
        make_request = AsyncMock()
        settings = RecoverySettings(payload=_base_payload(), make_request=make_request)
        response = _stream_response([aws_frames.content("a"), _drop_error()])
        with patch("kiro.streaming_core.UPSTREAM_DROP_RETRY", False):
            with pytest.raises(httpx.RemoteProtocolError):
                await collect_stream_to_result(response, first_token_timeout=5, recovery=settings)

        make_request.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_recovery_argument_keeps_the_old_behavior(self, aws_frames, recovery_env):
        """
        What it does: No recovery settings at all.
        Goal: The raw httpx error propagates (existing callers are unchanged).
        """
        response = _stream_response([aws_frames.content("a"), _drop_error()])
        with pytest.raises(httpx.RemoteProtocolError):
            await collect_stream_to_result(response, first_token_timeout=5)


class TestFirstTokenTimeoutAfterOutput:
    """stream_with_first_token_retry() must not restart a request after client-visible output."""

    @staticmethod
    def _processor_raising_after(chunks):
        """Stream processor that yields chunks and then raises FirstTokenTimeoutError."""
        async def processor(response):
            for chunk in chunks:
                yield chunk
            raise FirstTokenTimeoutError("late")
        return processor

    @pytest.mark.asyncio
    async def test_timeout_after_output_is_not_retried(self, aws_frames):
        """
        What it does: One chunk reaches the client, then a FirstTokenTimeoutError.
        Goal: It propagates, make_request is never called, the response is closed.
        """
        make_request = AsyncMock()
        response = _ok_response()
        received = []
        with patch("kiro.streaming_core.EMPTY_RESPONSE_AS_ERROR", False):
            with pytest.raises(FirstTokenTimeoutError):
                async for chunk in stream_with_first_token_retry(
                    make_request, self._processor_raising_after(["data: a\n\n"]),
                    initial_response=response, max_retries=3, first_token_timeout=5,
                ):
                    received.append(chunk)

        assert received == ["data: a\n\n"]
        make_request.assert_not_awaited()
        response.aclose.assert_awaited()

    @pytest.mark.asyncio
    async def test_timeout_before_output_is_still_retried(self, aws_frames):
        """
        What it does: The first attempt times out without output, the second succeeds.
        Goal: Old behavior kept: one new request, the client gets the second answer.
        """
        make_request = AsyncMock(return_value=_ok_response())
        calls = {"n": 0}

        async def processor(response):
            calls["n"] += 1
            if calls["n"] == 1:
                raise FirstTokenTimeoutError("early")
            yield "data: ok\n\n"

        received = []
        with patch("kiro.streaming_core.EMPTY_RESPONSE_AS_ERROR", False):
            async for chunk in stream_with_first_token_retry(
                make_request, processor, initial_response=_ok_response(),
                max_retries=3, first_token_timeout=5,
            ):
                received.append(chunk)

        assert received == ["data: ok\n\n"]
        make_request.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_preamble_only_output_still_counts_as_no_output(self, aws_frames):
        """
        What it does: Only a preamble chunk (is_preamble_chunk) was sent before the timeout.
        Goal: A preamble carries no model output, so the retry is still allowed.
        """
        make_request = AsyncMock(return_value=_ok_response())
        calls = {"n": 0}

        async def processor(response):
            calls["n"] += 1
            yield "event: message_start\n\n"
            if calls["n"] == 1:
                raise FirstTokenTimeoutError("early")
            yield "data: ok\n\n"

        received = []
        with patch("kiro.streaming_core.EMPTY_RESPONSE_AS_ERROR", False):
            async for chunk in stream_with_first_token_retry(
                make_request, processor, initial_response=_ok_response(),
                max_retries=3, first_token_timeout=5,
                is_preamble_chunk=lambda c: c.startswith("event: message_start"),
            ):
                received.append(chunk)

        assert received[-1] == "data: ok\n\n"
        make_request.assert_awaited_once()


class TestResponseSummary:
    """format_response_summary() / log_response_summary(): one-line fingerprint of a response."""

    def test_normal_text_response(self):
        """
        What it does: Text and reasoning, no tools, metering seen.
        Purpose: All fields are present and the response is not empty.
        """
        text = format_response_summary("openai", "claude-sonnet-5.5", "stop", 120, 30, [], 0, True)
        assert text == (
            "api=openai model=claude-sonnet-5.5 finish=stop text_chars=120 reasoning_chars=30 "
            "tool_calls=0[] truncated_tools=0 metering=true empty=false"
        )

    def test_empty_response_is_flagged(self):
        """
        What it does: No text, no reasoning, no tool call.
        Purpose: This is the case the line exists to reveal: empty=true.
        """
        text = format_response_summary("anthropic", "m", "end_turn", 0, 0, [], 0, True)
        assert "empty=true" in text
        assert "tool_calls=0[]" in text

    def test_reasoning_only_is_not_empty(self):
        """
        What it does: Only reasoning text.
        Purpose: Reasoning counts as output for the empty flag.
        """
        assert "empty=false" in format_response_summary("openai", "m", "stop", 0, 5, [], 0, True)

    def test_tool_call_only_is_not_empty(self):
        """
        What it does: One tool call, no text.
        Purpose: A pure tool call is a normal agent step, not an empty answer.
        """
        text = format_response_summary("openai", "m", "tool_calls", 0, 0, ["Write"], 0, True)
        assert "empty=false" in text
        assert "tool_calls=1[Write]" in text

    def test_tool_names_are_shortened_cleaned_and_limited(self):
        """
        What it does: 7 tools, one with whitespace and braces, one very long.
        Purpose: The line stays short and single-line: max 5 names, 40 chars each, "+N" for the rest.
        """
        names = ["a b\nc", "{x}", "L" * 100, "d", "e", "f", "g"]
        text = format_response_summary("openai", "m", "tool_calls", 1, 0, names, 0, True)
        assert "\n" not in text
        assert "tool_calls=7[abc,{x}," + "L" * 40 + ",d,e,+2]" in text

    def test_missing_metering_and_truncation_are_reported(self):
        """
        What it does: Truncated tool and no metering signal.
        Purpose: Both facts are visible in the line.
        """
        text = format_response_summary("openai", "m", "length", 10, 0, ["Write"], 1, False)
        assert "truncated_tools=1" in text
        assert "metering=false" in text

    def test_log_writes_one_info_line_with_tag(self):
        """
        What it does: log_response_summary with braces in a tool name and model.
        Purpose: One INFO [ResponseSummary] line; braces do not break loguru formatting;
        no message content is possible because none is passed in.
        """
        messages, handler_id = _capture_logs("INFO")
        try:
            log_response_summary("openai", "m{0}", "tool_calls", 5, 0, ["t{1}"], 0, True)
        finally:
            logger.remove(handler_id)
        lines = [m for m in messages if "[ResponseSummary]" in m]
        assert len(lines) == 1
        assert lines[0].startswith("INFO|")
        assert "model=m{0}" in lines[0]
        assert "tool_calls=1[t{1}]" in lines[0]
