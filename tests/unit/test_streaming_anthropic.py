
# -*- coding: utf-8 -*-

"""
Unit tests for streaming_anthropic module.

Tests for:
- generate_message_id() function
- format_sse_event() function
- stream_kiro_to_anthropic() generator
- collect_anthropic_response() function
"""

import pytest
import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

from kiro.streaming_anthropic import (
    generate_message_id,
    generate_thinking_signature,
    format_sse_event,
    stream_kiro_to_anthropic,
    collect_anthropic_response,
    stream_with_first_token_retry_anthropic,
)
from kiro.streaming_core import KiroEvent, StreamResult


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
def mock_auth_manager():
    """Mock for KiroAuthManager."""
    manager = MagicMock()
    return manager


@pytest.fixture
def mock_response():
    """Mock for httpx.Response."""
    response = AsyncMock()
    response.status_code = 200
    response.aclose = AsyncMock()
    return response


# ==================================================================================================
# Tests for generate_message_id()
# ==================================================================================================

class TestGenerateMessageId:
    """Tests for generate_message_id() function."""
    
    def test_generates_message_id_with_prefix(self):
        """
        What it does: Generates message ID with 'msg_' prefix.
        Goal: Verify Anthropic message ID format.
        """
        print("Action: Generating message ID...")
        message_id = generate_message_id()
        
        print(f"Generated ID: {message_id}")
        assert message_id.startswith("msg_")
        print("✓ Message ID has correct prefix")
    
    def test_generates_unique_ids(self):
        """
        What it does: Generates unique message IDs.
        Goal: Verify IDs are unique.
        """
        print("Action: Generating multiple message IDs...")
        ids = [generate_message_id() for _ in range(100)]
        
        print(f"Generated {len(ids)} IDs")
        unique_ids = set(ids)
        print(f"Unique IDs: {len(unique_ids)}")
        
        assert len(unique_ids) == 100
        print("✓ All message IDs are unique")
    
    def test_message_id_has_correct_length(self):
        """
        What it does: Verifies message ID length.
        Goal: Ensure ID format matches Anthropic spec.
        """
        print("Action: Generating message ID...")
        message_id = generate_message_id()
        
        # Format: msg_ + 24 hex chars
        print(f"Generated ID: {message_id}, length: {len(message_id)}")
        assert len(message_id) == 4 + 24  # "msg_" + 24 chars
        print("✓ Message ID has correct length")


# ==================================================================================================
# Tests for format_sse_event()
# ==================================================================================================

class TestFormatSseEvent:
    """Tests for format_sse_event() function."""
    
    def test_formats_message_start_event(self):
        """
        What it does: Formats message_start event.
        Goal: Verify Anthropic SSE format.
        """
        print("Action: Formatting message_start event...")
        data = {
            "type": "message_start",
            "message": {
                "id": "msg_123",
                "type": "message",
                "role": "assistant"
            }
        }
        
        result = format_sse_event("message_start", data)
        
        print(f"Formatted event:\n{result}")
        assert result.startswith("event: message_start\n")
        assert "data: " in result
        assert result.endswith("\n\n")
        print("✓ Event formatted correctly")
    
    def test_formats_content_block_delta_event(self):
        """
        What it does: Formats content_block_delta event.
        Goal: Verify delta event format.
        """
        print("Action: Formatting content_block_delta event...")
        data = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "text_delta",
                "text": "Hello"
            }
        }
        
        result = format_sse_event("content_block_delta", data)
        
        print(f"Formatted event:\n{result}")
        assert "event: content_block_delta\n" in result
        assert '"text": "Hello"' in result
        print("✓ Delta event formatted correctly")
    
    def test_formats_message_stop_event(self):
        """
        What it does: Formats message_stop event.
        Goal: Verify stop event format.
        """
        print("Action: Formatting message_stop event...")
        data = {"type": "message_stop"}
        
        result = format_sse_event("message_stop", data)
        
        print(f"Formatted event:\n{result}")
        assert "event: message_stop\n" in result
        print("✓ Stop event formatted correctly")
    
    def test_handles_unicode_content(self):
        """
        What it does: Handles Unicode content in events.
        Goal: Verify non-ASCII characters are preserved.
        """
        print("Action: Formatting event with Unicode...")
        data = {
            "type": "content_block_delta",
            "delta": {"text": "Привет мир! 🌍"}
        }
        
        result = format_sse_event("content_block_delta", data)
        
        print(f"Formatted event:\n{result}")
        assert "Привет мир!" in result
        assert "🌍" in result
        print("✓ Unicode content preserved")
    
    def test_json_data_is_valid(self):
        """
        What it does: Verifies JSON data is valid.
        Goal: Ensure data can be parsed back.
        """
        print("Action: Formatting and parsing event...")
        data = {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 100}
        }
        
        result = format_sse_event("message_delta", data)
        
        # Extract JSON from result
        lines = result.strip().split("\n")
        data_line = [l for l in lines if l.startswith("data: ")][0]
        json_str = data_line[6:]  # Remove "data: " prefix
        
        print(f"JSON string: {json_str}")
        parsed = json.loads(json_str)
        
        assert parsed["type"] == "message_delta"
        assert parsed["delta"]["stop_reason"] == "end_turn"
        print("✓ JSON data is valid and parseable")


# ==================================================================================================
# Tests for stream_kiro_to_anthropic()
# ==================================================================================================

class TestStreamKiroToAnthropic:
    """Tests for stream_kiro_to_anthropic() generator."""
    
    @pytest.mark.asyncio
    async def test_yields_message_start_event(self, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Yields message_start event at beginning.
        Goal: Verify Anthropic streaming protocol.
        """
        print("Setup: Mock empty stream...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            return
            yield  # Make it a generator
        
        print("Action: Streaming to Anthropic format...")
        events = []
        
        with patch('kiro.streaming_anthropic.parse_kiro_stream', mock_parse_kiro_stream):
            async for event in stream_kiro_to_anthropic(
                mock_response, "claude-sonnet-4", mock_model_cache, mock_auth_manager
            ):
                events.append(event)
        
        print(f"Received {len(events)} events")
        
        # First event should be message_start
        assert len(events) > 0
        assert "event: message_start" in events[0]
        print("✓ message_start event yielded first")
    
    @pytest.mark.asyncio
    async def test_yields_content_block_start_on_first_content(self, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Yields content_block_start before first content.
        Goal: Verify content block lifecycle.
        """
        print("Setup: Mock stream with content...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Hello")
        
        print("Action: Streaming to Anthropic format...")
        events = []
        
        with patch('kiro.streaming_anthropic.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_anthropic.parse_bracket_tool_calls', return_value=[]):
                async for event in stream_kiro_to_anthropic(
                    mock_response, "claude-sonnet-4", mock_model_cache, mock_auth_manager
                ):
                    events.append(event)
        
        print(f"Received {len(events)} events")
        
        # Should have content_block_start
        content_block_start_found = any("content_block_start" in e for e in events)
        assert content_block_start_found
        print("✓ content_block_start event yielded")
    
    @pytest.mark.asyncio
    async def test_yields_content_block_delta_for_content(self, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Yields content_block_delta for content events.
        Goal: Verify content streaming.
        """
        print("Setup: Mock stream with content...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Hello")
            yield KiroEvent(type="content", content=" World")
        
        print("Action: Streaming to Anthropic format...")
        events = []
        
        with patch('kiro.streaming_anthropic.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_anthropic.parse_bracket_tool_calls', return_value=[]):
                async for event in stream_kiro_to_anthropic(
                    mock_response, "claude-sonnet-4", mock_model_cache, mock_auth_manager
                ):
                    events.append(event)
        
        print(f"Received {len(events)} events")
        
        # Should have content_block_delta events
        delta_events = [e for e in events if "content_block_delta" in e]
        print(f"Delta events: {len(delta_events)}")
        
        assert len(delta_events) >= 2
        assert "Hello" in delta_events[0]
        assert "World" in delta_events[1]
        print("✓ content_block_delta events yielded for content")
    
    @pytest.mark.asyncio
    async def test_yields_tool_use_block_for_tool_calls(self, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Yields tool_use block for tool calls.
        Goal: Verify tool use streaming.
        """
        print("Setup: Mock stream with tool call...")
        
        tool_use_data = {
            "id": "toolu_123",
            "function": {
                "name": "get_weather",
                "arguments": '{"city": "Moscow"}'
            }
        }
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Let me check")
            yield KiroEvent(type="tool_use", tool_use=tool_use_data)
        
        print("Action: Streaming to Anthropic format...")
        events = []
        
        with patch('kiro.streaming_anthropic.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_anthropic.parse_bracket_tool_calls', return_value=[]):
                async for event in stream_kiro_to_anthropic(
                    mock_response, "claude-sonnet-4", mock_model_cache, mock_auth_manager
                ):
                    events.append(event)
        
        print(f"Received {len(events)} events")
        
        # Should have tool_use content block
        tool_use_events = [e for e in events if "tool_use" in e and "content_block_start" in e]
        print(f"Tool use events: {len(tool_use_events)}")
        
        assert len(tool_use_events) >= 1
        assert "get_weather" in tool_use_events[0]
        print("✓ tool_use block yielded for tool calls")
    
    @pytest.mark.asyncio
    async def test_yields_message_delta_with_stop_reason(self, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Yields message_delta with stop_reason.
        Goal: Verify message completion.
        """
        print("Setup: Mock stream with content...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Hello")
            yield KiroEvent(type="context_usage", context_usage_percentage=5.0)
        
        print("Action: Streaming to Anthropic format...")
        events = []
        
        with patch('kiro.streaming_anthropic.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_anthropic.parse_bracket_tool_calls', return_value=[]):
                async for event in stream_kiro_to_anthropic(
                    mock_response, "claude-sonnet-4", mock_model_cache, mock_auth_manager
                ):
                    events.append(event)
        
        print(f"Received {len(events)} events")
        
        # Should have message_delta with stop_reason
        message_delta_events = [e for e in events if "message_delta" in e]
        assert len(message_delta_events) >= 1
        assert "end_turn" in message_delta_events[0]
        print("✓ message_delta with stop_reason yielded")
    
    @pytest.mark.asyncio
    async def test_yields_message_stop_at_end(self, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Yields message_stop at end.
        Goal: Verify stream termination.
        """
        print("Setup: Mock stream with content...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Hello")
        
        print("Action: Streaming to Anthropic format...")
        events = []
        
        with patch('kiro.streaming_anthropic.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_anthropic.parse_bracket_tool_calls', return_value=[]):
                async for event in stream_kiro_to_anthropic(
                    mock_response, "claude-sonnet-4", mock_model_cache, mock_auth_manager
                ):
                    events.append(event)
        
        print(f"Received {len(events)} events")
        
        # Last event should be message_stop
        assert "message_stop" in events[-1]
        print("✓ message_stop yielded at end")
    
    @pytest.mark.asyncio
    async def test_stop_reason_is_tool_use_when_tools_present(self, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Sets stop_reason to tool_use when tools are present.
        Goal: Verify correct stop reason for tool calls.
        """
        print("Setup: Mock stream with tool call...")
        
        tool_use_data = {
            "id": "toolu_123",
            "function": {"name": "func1", "arguments": "{}"}
        }
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="tool_use", tool_use=tool_use_data)
        
        print("Action: Streaming to Anthropic format...")
        events = []
        
        with patch('kiro.streaming_anthropic.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_anthropic.parse_bracket_tool_calls', return_value=[]):
                async for event in stream_kiro_to_anthropic(
                    mock_response, "claude-sonnet-4", mock_model_cache, mock_auth_manager
                ):
                    events.append(event)
        
        print(f"Received {len(events)} events")
        
        # message_delta should have stop_reason: tool_use
        message_delta_events = [e for e in events if "message_delta" in e]
        assert len(message_delta_events) >= 1
        assert "tool_use" in message_delta_events[0]
        print("✓ stop_reason is tool_use when tools present")
    
    @pytest.mark.asyncio
    async def test_handles_bracket_tool_calls(self, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Handles bracket-style tool calls in content.
        Goal: Verify bracket tool call detection.
        """
        print("Setup: Mock stream with bracket tool calls...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="[tool_call: func1]")
        
        bracket_tool_calls = [
            {"id": "call_1", "function": {"name": "func1", "arguments": "{}"}}
        ]
        
        print("Action: Streaming to Anthropic format...")
        events = []
        
        with patch('kiro.streaming_anthropic.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_anthropic.parse_bracket_tool_calls', return_value=bracket_tool_calls):
                async for event in stream_kiro_to_anthropic(
                    mock_response, "claude-sonnet-4", mock_model_cache, mock_auth_manager
                ):
                    events.append(event)
        
        print(f"Received {len(events)} events")
        
        # Should have tool_use block from bracket tool calls
        tool_use_events = [e for e in events if "tool_use" in e and "content_block_start" in e]
        assert len(tool_use_events) >= 1
        print("✓ Bracket tool calls handled correctly")
    
    @pytest.mark.asyncio
    async def test_closes_response_on_completion(self, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Closes response on completion.
        Goal: Verify resource cleanup.
        """
        print("Setup: Mock stream...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Hello")
        
        print("Action: Streaming to Anthropic format...")
        
        with patch('kiro.streaming_anthropic.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_anthropic.parse_bracket_tool_calls', return_value=[]):
                async for event in stream_kiro_to_anthropic(
                    mock_response, "claude-sonnet-4", mock_model_cache, mock_auth_manager
                ):
                    pass
        
        print("Check: response.aclose() should be called...")
        mock_response.aclose.assert_called()
        print("✓ Response closed on completion")
    
    @pytest.mark.asyncio
    async def test_closes_response_on_error(self, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Closes response on error.
        Goal: Verify resource cleanup on error.
        """
        print("Setup: Mock stream that raises error...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Hello")
            raise RuntimeError("Test error")
        
        print("Action: Streaming to Anthropic format with error...")
        
        with patch('kiro.streaming_anthropic.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_anthropic.parse_bracket_tool_calls', return_value=[]):
                try:
                    async for event in stream_kiro_to_anthropic(
                        mock_response, "claude-sonnet-4", mock_model_cache, mock_auth_manager
                    ):
                        pass
                except RuntimeError:
                    pass
        
        print("Check: response.aclose() should be called...")
        mock_response.aclose.assert_called()
        print("✓ Response closed on error")


# ==================================================================================================
# Tests for collect_anthropic_response()
# ==================================================================================================

class TestCollectAnthropicResponse:
    """Tests for collect_anthropic_response() function."""
    
    @pytest.mark.asyncio
    async def test_collects_text_content(self, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Collects text content into response.
        Goal: Verify content collection.
        """
        print("Setup: Mock stream result with content...")
        
        mock_result = StreamResult(
            content="Hello, world!",
            thinking_content="",
            tool_calls=[],
            usage=None,
            context_usage_percentage=None
        )
        
        print("Action: Collecting Anthropic response...")
        
        with patch('kiro.streaming_anthropic.collect_stream_to_result', return_value=mock_result):
            result = await collect_anthropic_response(
                mock_response, "claude-sonnet-4", mock_model_cache, mock_auth_manager
            )
        
        print(f"Result: {result}")
        
        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "Hello, world!"
        print("✓ Text content collected correctly")
    
    @pytest.mark.asyncio
    async def test_collects_tool_use_content(self, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Collects tool use into response.
        Goal: Verify tool use collection.
        """
        print("Setup: Mock stream result with tool calls...")
        
        mock_result = StreamResult(
            content="Let me check",
            thinking_content="",
            tool_calls=[
                {
                    "id": "toolu_123",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Moscow"}'
                    }
                }
            ],
            usage=None,
            context_usage_percentage=None
        )
        
        print("Action: Collecting Anthropic response...")
        
        with patch('kiro.streaming_anthropic.collect_stream_to_result', return_value=mock_result):
            result = await collect_anthropic_response(
                mock_response, "claude-sonnet-4", mock_model_cache, mock_auth_manager
            )
        
        print(f"Result: {result}")
        
        # Should have text and tool_use blocks
        assert len(result["content"]) == 2
        
        text_block = result["content"][0]
        assert text_block["type"] == "text"
        
        tool_block = result["content"][1]
        assert tool_block["type"] == "tool_use"
        assert tool_block["name"] == "get_weather"
        assert tool_block["input"] == {"city": "Moscow"}
        print("✓ Tool use content collected correctly")
    
    @pytest.mark.asyncio
    async def test_sets_stop_reason_end_turn(self, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Sets stop_reason to end_turn for normal completion.
        Goal: Verify stop reason.
        """
        print("Setup: Mock stream result without tool calls...")
        
        mock_result = StreamResult(
            content="Hello",
            thinking_content="",
            tool_calls=[],
            usage=None,
            context_usage_percentage=5.0
        )
        
        print("Action: Collecting Anthropic response...")
        
        with patch('kiro.streaming_anthropic.collect_stream_to_result', return_value=mock_result):
            result = await collect_anthropic_response(
                mock_response, "claude-sonnet-4", mock_model_cache, mock_auth_manager
            )
        
        print(f"stop_reason: {result['stop_reason']}")
        assert result["stop_reason"] == "end_turn"
        print("✓ stop_reason is end_turn")
    
    @pytest.mark.asyncio
    async def test_sets_stop_reason_tool_use(self, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Sets stop_reason to tool_use when tools present.
        Goal: Verify stop reason for tool calls.
        """
        print("Setup: Mock stream result with tool calls...")
        
        mock_result = StreamResult(
            content="",
            thinking_content="",
            tool_calls=[{"id": "call_1", "function": {"name": "func1", "arguments": "{}"}}],
            usage=None,
            context_usage_percentage=None
        )
        
        print("Action: Collecting Anthropic response...")
        
        with patch('kiro.streaming_anthropic.collect_stream_to_result', return_value=mock_result):
            result = await collect_anthropic_response(
                mock_response, "claude-sonnet-4", mock_model_cache, mock_auth_manager
            )
        
        print(f"stop_reason: {result['stop_reason']}")
        assert result["stop_reason"] == "tool_use"
        print("✓ stop_reason is tool_use")
    
    @pytest.mark.asyncio
    async def test_includes_usage_info(self, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Includes usage information in response.
        Goal: Verify usage is included.
        """
        print("Setup: Mock stream result...")
        
        mock_result = StreamResult(
            content="Hello, world!",
            thinking_content="",
            tool_calls=[],
            usage=None,
            context_usage_percentage=None
        )
        
        print("Action: Collecting Anthropic response...")
        
        with patch('kiro.streaming_anthropic.collect_stream_to_result', return_value=mock_result):
            with patch('kiro.streaming_anthropic.estimate_request_tokens', return_value={"total_tokens": 10}):
                with patch('kiro.streaming_anthropic.count_tokens', return_value=5):
                    result = await collect_anthropic_response(
                        mock_response, "claude-sonnet-4", mock_model_cache, mock_auth_manager,
                        request_messages=[{"role": "user", "content": "Hi"}]
                    )
        
        print(f"Usage: {result['usage']}")
        assert "input_tokens" in result["usage"]
        assert "output_tokens" in result["usage"]
        print("✓ Usage info included")
    
    @pytest.mark.asyncio
    async def test_generates_message_id(self, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Generates message ID for response.
        Goal: Verify message ID is present.
        """
        print("Setup: Mock stream result...")
        
        mock_result = StreamResult(
            content="Hello",
            thinking_content="",
            tool_calls=[],
            usage=None,
            context_usage_percentage=None
        )
        
        print("Action: Collecting Anthropic response...")
        
        with patch('kiro.streaming_anthropic.collect_stream_to_result', return_value=mock_result):
            result = await collect_anthropic_response(
                mock_response, "claude-sonnet-4", mock_model_cache, mock_auth_manager
            )
        
        print(f"Message ID: {result['id']}")
        assert result["id"].startswith("msg_")
        print("✓ Message ID generated")
    
    @pytest.mark.asyncio
    async def test_includes_model_name(self, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Includes model name in response.
        Goal: Verify model is included.
        """
        print("Setup: Mock stream result...")
        
        mock_result = StreamResult(
            content="Hello",
            thinking_content="",
            tool_calls=[],
            usage=None,
            context_usage_percentage=None
        )
        
        print("Action: Collecting Anthropic response...")
        
        with patch('kiro.streaming_anthropic.collect_stream_to_result', return_value=mock_result):
            result = await collect_anthropic_response(
                mock_response, "claude-sonnet-4", mock_model_cache, mock_auth_manager
            )
        
        print(f"Model: {result['model']}")
        assert result["model"] == "claude-sonnet-4"
        print("✓ Model name included")
    
    @pytest.mark.asyncio
    async def test_parses_tool_arguments_from_string(self, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Parses tool arguments from JSON string.
        Goal: Verify arguments are parsed to dict.
        """
        print("Setup: Mock stream result with string arguments...")
        
        mock_result = StreamResult(
            content="",
            thinking_content="",
            tool_calls=[
                {
                    "id": "call_1",
                    "function": {
                        "name": "func1",
                        "arguments": '{"key": "value"}'  # String, not dict
                    }
                }
            ],
            usage=None,
            context_usage_percentage=None
        )
        
        print("Action: Collecting Anthropic response...")
        
        with patch('kiro.streaming_anthropic.collect_stream_to_result', return_value=mock_result):
            result = await collect_anthropic_response(
                mock_response, "claude-sonnet-4", mock_model_cache, mock_auth_manager
            )
        
        print(f"Result: {result}")
        
        # Tool input should be parsed to dict
        tool_block = result["content"][0]  # Only tool_use since content is empty
        assert tool_block["type"] == "tool_use"
        assert tool_block["input"] == {"key": "value"}
        assert isinstance(tool_block["input"], dict)
        print("✓ Tool arguments parsed from string to dict")
    
    @pytest.mark.asyncio
    async def test_handles_invalid_json_arguments(self, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Handles invalid JSON in tool arguments.
        Goal: Verify graceful handling of invalid JSON.
        """
        print("Setup: Mock stream result with invalid JSON arguments...")
        
        mock_result = StreamResult(
            content="",
            thinking_content="",
            tool_calls=[
                {
                    "id": "call_1",
                    "function": {
                        "name": "func1",
                        "arguments": "not valid json"  # Invalid JSON
                    }
                }
            ],
            usage=None,
            context_usage_percentage=None
        )
        
        print("Action: Collecting Anthropic response...")
        
        with patch('kiro.streaming_anthropic.collect_stream_to_result', return_value=mock_result):
            result = await collect_anthropic_response(
                mock_response, "claude-sonnet-4", mock_model_cache, mock_auth_manager
            )
        
        print(f"Result: {result}")
        
        # Should handle gracefully with empty dict
        tool_block = result["content"][0]
        assert tool_block["type"] == "tool_use"
        assert tool_block["input"] == {}
        print("✓ Invalid JSON arguments handled gracefully")
    
    @pytest.mark.asyncio
    async def test_handles_empty_content(self, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Handles empty content in response.
        Goal: Verify empty content is handled.
        """
        print("Setup: Mock stream result with empty content...")
        
        mock_result = StreamResult(
            content="",
            thinking_content="",
            tool_calls=[],
            usage=None,
            context_usage_percentage=None
        )
        
        print("Action: Collecting Anthropic response...")
        
        with patch('kiro.streaming_anthropic.collect_stream_to_result', return_value=mock_result):
            result = await collect_anthropic_response(
                mock_response, "claude-sonnet-4", mock_model_cache, mock_auth_manager
            )
        
        print(f"Result: {result}")
        
        # Content should be empty list
        assert result["content"] == []
        print("✓ Empty content handled correctly")


# ==================================================================================================
# Tests for error handling
# ==================================================================================================

class TestStreamingAnthropicErrorHandling:
    """Tests for error handling in streaming_anthropic."""
    
    @pytest.mark.asyncio
    async def test_propagates_first_token_timeout_error(self, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Propagates FirstTokenTimeoutError.
        Goal: Verify timeout error is not caught internally.
        """
        from kiro.streaming_core import FirstTokenTimeoutError
        
        print("Setup: Mock stream that raises timeout...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            raise FirstTokenTimeoutError("Timeout!")
            yield  # Make it a generator
        
        print("Action: Streaming to Anthropic format with timeout...")
        
        with patch('kiro.streaming_anthropic.parse_kiro_stream', mock_parse_kiro_stream):
            with pytest.raises(FirstTokenTimeoutError):
                async for event in stream_kiro_to_anthropic(
                    mock_response, "claude-sonnet-4", mock_model_cache, mock_auth_manager
                ):
                    pass
        
        print("✓ FirstTokenTimeoutError propagated correctly")
    
    @pytest.mark.asyncio
    async def test_propagates_generator_exit(self, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Propagates GeneratorExit.
        Goal: Verify client disconnect is handled.
        """
        print("Setup: Mock stream that raises GeneratorExit...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Hello")
            raise GeneratorExit()
        
        print("Action: Streaming to Anthropic format with GeneratorExit...")
        
        with patch('kiro.streaming_anthropic.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_anthropic.parse_bracket_tool_calls', return_value=[]):
                with pytest.raises(GeneratorExit):
                    async for event in stream_kiro_to_anthropic(
                        mock_response, "claude-sonnet-4", mock_model_cache, mock_auth_manager
                    ):
                        pass
        
        print("✓ GeneratorExit propagated correctly")
    
    @pytest.mark.asyncio
    async def test_yields_error_event_on_exception(self, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Yields error event on exception.
        Goal: Verify error event is sent to client.
        """
        print("Setup: Mock stream that raises RuntimeError...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Hello")
            raise RuntimeError("Test error")
        
        print("Action: Streaming to Anthropic format with error...")
        events = []
        
        with patch('kiro.streaming_anthropic.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_anthropic.parse_bracket_tool_calls', return_value=[]):
                try:
                    async for event in stream_kiro_to_anthropic(
                        mock_response, "claude-sonnet-4", mock_model_cache, mock_auth_manager
                    ):
                        events.append(event)
                except RuntimeError:
                    pass
        
        print(f"Received {len(events)} events")
        
        # Should have error event
        error_events = [e for e in events if "event: error" in e]
        assert len(error_events) >= 1
        assert "Test error" in error_events[0]
        print("✓ Error event yielded on exception")
    
    @pytest.mark.asyncio
    async def test_closes_response_in_finally(self, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Closes response in finally block.
        Goal: Verify resource cleanup always happens.
        """
        print("Setup: Mock stream that raises error...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            raise ValueError("Test error")
            yield  # Make it a generator
        
        print("Action: Streaming to Anthropic format with error...")
        
        with patch('kiro.streaming_anthropic.parse_kiro_stream', mock_parse_kiro_stream):
            try:
                async for event in stream_kiro_to_anthropic(
                    mock_response, "claude-sonnet-4", mock_model_cache, mock_auth_manager
                ):
                    pass
            except ValueError:
                pass
        
        print("Check: response.aclose() should be called...")
        mock_response.aclose.assert_called()
        print("✓ Response closed in finally block")


# ==================================================================================================
# Tests for thinking content handling
# ==================================================================================================

class TestStreamingAnthropicThinkingContent:
    """Tests for thinking content handling in Anthropic streaming."""
    
    @pytest.mark.asyncio
    async def test_includes_thinking_as_text_when_configured(self, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Includes thinking content as text when configured.
        Goal: Verify thinking content handling.
        """
        print("Setup: Mock stream with thinking content...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="thinking", thinking_content="Let me think...")
            yield KiroEvent(type="content", content="Here is my answer")
        
        print("Action: Streaming to Anthropic format with thinking...")
        events = []
        
        with patch('kiro.streaming_anthropic.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_anthropic.parse_bracket_tool_calls', return_value=[]):
                with patch('kiro.streaming_anthropic.FAKE_REASONING_HANDLING', 'include_as_text'):
                    async for event in stream_kiro_to_anthropic(
                        mock_response, "claude-sonnet-4", mock_model_cache, mock_auth_manager
                    ):
                        events.append(event)
        
        print(f"Received {len(events)} events")
        
        # Should have thinking content as text delta
        delta_events = [e for e in events if "content_block_delta" in e]
        thinking_found = any("Let me think" in e for e in delta_events)
        assert thinking_found
        print("✓ Thinking content included as text")
    
    @pytest.mark.asyncio
    async def test_strips_thinking_when_configured(self, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Strips thinking content when configured.
        Goal: Verify thinking content is stripped.
        """
        print("Setup: Mock stream with thinking content...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="thinking", thinking_content="Let me think...")
            yield KiroEvent(type="content", content="Here is my answer")
        
        print("Action: Streaming to Anthropic format with strip mode...")
        events = []
        
        with patch('kiro.streaming_anthropic.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_anthropic.parse_bracket_tool_calls', return_value=[]):
                with patch('kiro.streaming_anthropic.FAKE_REASONING_HANDLING', 'strip'):
                    async for event in stream_kiro_to_anthropic(
                        mock_response, "claude-sonnet-4", mock_model_cache, mock_auth_manager
                    ):
                        events.append(event)
        
        print(f"Received {len(events)} events")
        
        # Should NOT have thinking content
        delta_events = [e for e in events if "content_block_delta" in e]
        thinking_found = any("Let me think" in e for e in delta_events)
        assert not thinking_found
        print("✓ Thinking content stripped")


# ==================================================================================================
# Tests for context usage calculation
# ==================================================================================================

class TestStreamingAnthropicContextUsage:
    """Tests for context usage calculation in Anthropic streaming."""
    
    @pytest.mark.asyncio
    async def test_calculates_tokens_from_context_usage(self, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Calculates tokens from context usage percentage.
        Goal: Verify token calculation.
        """
        print("Setup: Mock stream with context usage...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Hello")
            yield KiroEvent(type="context_usage", context_usage_percentage=5.0)
        
        print("Action: Streaming to Anthropic format...")
        events = []
        
        with patch('kiro.streaming_anthropic.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_anthropic.parse_bracket_tool_calls', return_value=[]):
                async for event in stream_kiro_to_anthropic(
                    mock_response, "claude-sonnet-4", mock_model_cache, mock_auth_manager
                ):
                    events.append(event)
        
        print(f"Received {len(events)} events")
        
        # message_delta should have usage with output_tokens
        message_delta_events = [e for e in events if "message_delta" in e]
        assert len(message_delta_events) >= 1
        assert "output_tokens" in message_delta_events[0]
        print("✓ Tokens calculated from context usage")
    
    @pytest.mark.asyncio
    async def test_uses_request_messages_for_input_tokens(self, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Uses request messages for input token count.
        Goal: Verify input tokens are counted from request.
        """
        print("Setup: Mock stream...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Hello")
        
        request_messages = [
            {"role": "user", "content": "Hi there!"}
        ]
        
        print("Action: Streaming to Anthropic format with request messages...")
        events = []
        
        with patch('kiro.streaming_anthropic.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_anthropic.parse_bracket_tool_calls', return_value=[]):
                with patch('kiro.streaming_anthropic.estimate_request_tokens', return_value={"total_tokens": 10}) as mock_estimate:
                    async for event in stream_kiro_to_anthropic(
                        mock_response, "claude-sonnet-4", mock_model_cache, mock_auth_manager,
                        request_messages=request_messages
                    ):
                        events.append(event)
                    
                    # Verify estimate_request_tokens was called
                    mock_estimate.assert_called_once_with(
                        messages=request_messages,
                        tools=None,
                        system_prompt=None,
                        apply_claude_correction=False
                    )
        
        print("✓ Request messages used for input token count")

    @pytest.mark.asyncio
    async def test_uses_tools_and_system_for_input_tokens(self, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Includes request tools and system in input token estimation.
        Goal: Verify Anthropic fallback token counting uses full request.
        """
        print("Setup: Mock stream...")

        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Hello")

        request_messages = [{"role": "user", "content": "Hi"}]
        request_tools = [{"name": "get_weather", "input_schema": {"type": "object"}}]
        request_system = [{"type": "text", "text": "你是助手"}]

        with patch('kiro.streaming_anthropic.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_anthropic.parse_bracket_tool_calls', return_value=[]):
                with patch('kiro.streaming_anthropic.estimate_request_tokens', return_value={"total_tokens": 12}) as mock_estimate:
                    events = []
                    async for event in stream_kiro_to_anthropic(
                        mock_response, "claude-sonnet-4", mock_model_cache, mock_auth_manager,
                        request_messages=request_messages,
                        request_tools=request_tools,
                        request_system=request_system,
                    ):
                        events.append(event)

                    assert events, "Should produce streaming events"
                    mock_estimate.assert_called_once_with(
                        messages=request_messages,
                        tools=request_tools,
                        system_prompt=request_system,
                        apply_claude_correction=False
                    )
        print("✓ Request tools and system included in token count")

    @pytest.mark.asyncio
    async def test_context_usage_zero_keeps_fallback_estimate(self, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Keeps fallback estimate when context usage is 0.
        Goal: Prevent overriding with zero prompt tokens.
        """
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Hello")
            yield KiroEvent(type="context_usage", context_usage_percentage=0.0)

        events = []
        with patch('kiro.streaming_anthropic.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_anthropic.parse_bracket_tool_calls', return_value=[]):
                with patch('kiro.streaming_anthropic.estimate_request_tokens', return_value={"total_tokens": 99}):
                    async for event in stream_kiro_to_anthropic(
                        mock_response, "claude-sonnet-4", mock_model_cache, mock_auth_manager,
                        request_messages=[{"role": "user", "content": "hi"}]
                    ):
                        events.append(event)

        message_start_event = next(e for e in events if "event: message_start" in e)
        assert '"input_tokens": 99' in message_start_event

    @pytest.mark.asyncio
    async def test_non_streaming_passes_upstream_cache_usage_fields(self, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Passes through upstream cache usage fields when available.
        Goal: Ensure no fake values, only real upstream usage keys.
        """
        mock_result = MagicMock(
            content="done",
            thinking_content="",
            tool_calls=[],
            context_usage_percentage=None,
            usage={"cacheReadInputTokens": 12, "cacheCreationInputTokens": 34},
        )

        with patch('kiro.streaming_anthropic.collect_stream_to_result', return_value=mock_result):
            with patch('kiro.streaming_anthropic.generate_message_id', return_value="msg_test"):
                response_data = await collect_anthropic_response(
                    mock_response, "claude-sonnet-4", mock_model_cache, mock_auth_manager
                )

        usage = response_data["usage"]
        assert usage["cache_read_input_tokens"] == 12
        assert usage["cache_creation_input_tokens"] == 34


# ==================================================================================================
# Tests for generate_thinking_signature()
# ==================================================================================================

class TestGenerateThinkingSignature:
    """
    Tests for generate_thinking_signature() function.
    
    This function generates placeholder signatures for thinking content blocks.
    In real Anthropic API, this is a cryptographic signature for verification.
    Since we use fake reasoning via tag injection, we generate a placeholder.
    """
    
    def test_generates_signature_with_prefix(self):
        """
        What it does: Generates signature with 'sig_' prefix.
        Goal: Verify signature format matches expected pattern.
        """
        print("Action: Generating thinking signature...")
        signature = generate_thinking_signature()
        
        print(f"Generated signature: {signature}")
        assert signature.startswith("sig_")
        print("✓ Signature has correct prefix")
    
    def test_generates_unique_signatures(self):
        """
        What it does: Generates unique signatures.
        Goal: Verify signatures are unique across multiple calls.
        """
        print("Action: Generating multiple signatures...")
        signatures = [generate_thinking_signature() for _ in range(100)]
        
        print(f"Generated {len(signatures)} signatures")
        unique_signatures = set(signatures)
        print(f"Unique signatures: {len(unique_signatures)}")
        
        assert len(unique_signatures) == 100
        print("✓ All signatures are unique")
    
    def test_signature_has_correct_length(self):
        """
        What it does: Verifies signature length.
        Goal: Ensure signature format is consistent.
        """
        print("Action: Generating signature...")
        signature = generate_thinking_signature()
        
        # Format: sig_ + 32 hex chars
        print(f"Generated signature: {signature}, length: {len(signature)}")
        assert len(signature) == 4 + 32  # "sig_" + 32 chars
        print("✓ Signature has correct length")
    
    def test_signature_contains_only_valid_characters(self):
        """
        What it does: Verifies signature contains only valid hex characters.
        Goal: Ensure signature is properly formatted.
        """
        print("Action: Generating signature...")
        signature = generate_thinking_signature()
        
        print(f"Generated signature: {signature}")
        # Remove prefix and check remaining chars are hex
        hex_part = signature[4:]  # Remove "sig_"
        assert all(c in '0123456789abcdef' for c in hex_part)
        print("✓ Signature contains only valid hex characters")


# ==================================================================================================
# Tests for stream_with_first_token_retry_anthropic()
# ==================================================================================================

@pytest.mark.usefixtures("no_empty_response_guard")
class TestStreamWithFirstTokenRetryAnthropic:
    """
    Tests for stream_with_first_token_retry_anthropic() function.
    
    This function wraps stream_kiro_to_anthropic with automatic retry
    on first token timeout. It uses the generic stream_with_first_token_retry
    from streaming_core.py with Anthropic-specific error formatting.
    """
    
    @pytest.mark.asyncio
    async def test_yields_chunks_on_success(self, mock_model_cache, mock_auth_manager):
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
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Hello")
        
        print("Action: Streaming with retry wrapper...")
        chunks = []
        
        with patch('kiro.streaming_anthropic.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_anthropic.parse_bracket_tool_calls', return_value=[]):
                async for chunk in stream_with_first_token_retry_anthropic(
                    make_request=mock_make_request,
                    model="claude-sonnet-4",
                    model_cache=mock_model_cache,
                    auth_manager=mock_auth_manager,
                    max_retries=3,
                    first_token_timeout=30
                ):
                    chunks.append(chunk)
        
        print(f"Received {len(chunks)} chunks")
        assert len(chunks) > 0
        assert any("message_start" in c for c in chunks)
        print("✓ Chunks yielded on success")
    
    @pytest.mark.asyncio
    async def test_retries_on_first_token_timeout(self, mock_model_cache, mock_auth_manager):
        """
        What it does: Retries on first token timeout.
        Goal: Verify retry logic is triggered.
        """
        from kiro.streaming_core import FirstTokenTimeoutError
        
        print("Setup: Mock request that times out then succeeds...")
        
        call_count = 0
        
        async def mock_make_request():
            nonlocal call_count
            call_count += 1
            response = AsyncMock()
            response.status_code = 200
            response.aclose = AsyncMock()
            return response
        
        async def mock_stream_kiro_to_anthropic(*args, **kwargs):
            nonlocal call_count
            if call_count == 1:
                raise FirstTokenTimeoutError("Timeout on first attempt")
            yield "event: message_start\ndata: {}\n\n"
            yield "event: message_stop\ndata: {}\n\n"
        
        print("Action: Streaming with retry on timeout...")
        chunks = []
        
        with patch('kiro.streaming_anthropic.stream_kiro_to_anthropic', mock_stream_kiro_to_anthropic):
            async for chunk in stream_with_first_token_retry_anthropic(
                make_request=mock_make_request,
                model="claude-sonnet-4",
                model_cache=mock_model_cache,
                auth_manager=mock_auth_manager,
                max_retries=3,
                first_token_timeout=30
            ):
                chunks.append(chunk)
        
        print(f"Call count: {call_count}")
        print(f"Received {len(chunks)} chunks")
        
        assert call_count == 2  # First timeout, second success
        assert len(chunks) > 0
        print("✓ Retry on timeout works correctly")
    
    @pytest.mark.asyncio
    async def test_raises_anthropic_error_after_all_retries(self, mock_model_cache, mock_auth_manager):
        """
        What it does: Raises Anthropic-formatted error after all retries exhausted.
        Goal: Verify error format matches Anthropic API.
        """
        from kiro.streaming_core import FirstTokenTimeoutError
        
        print("Setup: Mock request that always times out...")
        
        async def mock_make_request():
            response = AsyncMock()
            response.status_code = 200
            response.aclose = AsyncMock()
            return response
        
        async def mock_stream_kiro_to_anthropic(*args, **kwargs):
            raise FirstTokenTimeoutError("Timeout!")
            yield  # Make it a generator
        
        print("Action: Streaming with all retries failing...")
        
        with patch('kiro.streaming_anthropic.stream_kiro_to_anthropic', mock_stream_kiro_to_anthropic):
            with pytest.raises(Exception) as exc_info:
                async for chunk in stream_with_first_token_retry_anthropic(
                    make_request=mock_make_request,
                    model="claude-sonnet-4",
                    model_cache=mock_model_cache,
                    auth_manager=mock_auth_manager,
                    max_retries=2,
                    first_token_timeout=30
                ):
                    pass
        
        print(f"Exception: {exc_info.value}")
        
        # Error should be in Anthropic format (JSON)
        error_json = json.loads(str(exc_info.value))
        assert error_json["type"] == "error"
        assert error_json["error"]["type"] == "timeout_error"
        assert "30" in error_json["error"]["message"]
        print("✓ Anthropic-formatted error raised after all retries")
    
    @pytest.mark.asyncio
    async def test_raises_anthropic_error_on_http_error(self, mock_model_cache, mock_auth_manager):
        """
        What it does: Raises Anthropic-formatted error on HTTP error.
        Goal: Verify HTTP errors are formatted correctly.
        """
        print("Setup: Mock request that returns HTTP error...")
        
        async def mock_make_request():
            response = AsyncMock()
            response.status_code = 500
            response.aread = AsyncMock(return_value=b"Internal Server Error")
            response.aclose = AsyncMock()
            return response
        
        print("Action: Streaming with HTTP error...")
        
        with pytest.raises(Exception) as exc_info:
            async for chunk in stream_with_first_token_retry_anthropic(
                make_request=mock_make_request,
                model="claude-sonnet-4",
                model_cache=mock_model_cache,
                auth_manager=mock_auth_manager,
                max_retries=2,
                first_token_timeout=30
            ):
                pass
        
        print(f"Exception: {exc_info.value}")
        
        # Error should be in Anthropic format (JSON)
        error_json = json.loads(str(exc_info.value))
        assert error_json["type"] == "error"
        assert error_json["error"]["type"] == "api_error"
        assert "Upstream API error" in error_json["error"]["message"]
        print("✓ Anthropic-formatted error raised on HTTP error")
    
    @pytest.mark.asyncio
    async def test_passes_request_messages_to_stream(self, mock_model_cache, mock_auth_manager):
        """
        What it does: Passes request_messages to underlying stream function.
        Goal: Verify token counting parameters are forwarded.
        """
        print("Setup: Mock request with messages...")
        
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.aclose = AsyncMock()
        
        async def mock_make_request():
            return mock_response
        
        captured_kwargs = {}
        
        async def mock_stream_kiro_to_anthropic(*args, **kwargs):
            captured_kwargs.update(kwargs)
            yield "event: message_start\ndata: {}\n\n"
            yield "event: message_stop\ndata: {}\n\n"
        
        request_messages = [{"role": "user", "content": "Hello"}]
        
        print("Action: Streaming with request_messages...")
        
        with patch('kiro.streaming_anthropic.stream_kiro_to_anthropic', mock_stream_kiro_to_anthropic):
            async for chunk in stream_with_first_token_retry_anthropic(
                make_request=mock_make_request,
                model="claude-sonnet-4",
                model_cache=mock_model_cache,
                auth_manager=mock_auth_manager,
                request_messages=request_messages
            ):
                pass
        
        print(f"Captured kwargs: {captured_kwargs}")
        assert captured_kwargs.get("request_messages") == request_messages
        print("✓ request_messages passed to stream function")
    
    @pytest.mark.asyncio
    async def test_uses_configured_max_retries(self, mock_model_cache, mock_auth_manager):
        """
        What it does: Uses configured max_retries value.
        Goal: Verify max_retries parameter is respected.
        """
        from kiro.streaming_core import FirstTokenTimeoutError
        
        print("Setup: Mock request that always times out...")
        
        call_count = 0
        
        async def mock_make_request():
            nonlocal call_count
            call_count += 1
            response = AsyncMock()
            response.status_code = 200
            response.aclose = AsyncMock()
            return response
        
        async def mock_stream_kiro_to_anthropic(*args, **kwargs):
            raise FirstTokenTimeoutError("Timeout!")
            yield  # Make it a generator
        
        print("Action: Streaming with max_retries=5...")
        
        with patch('kiro.streaming_anthropic.stream_kiro_to_anthropic', mock_stream_kiro_to_anthropic):
            try:
                async for chunk in stream_with_first_token_retry_anthropic(
                    make_request=mock_make_request,
                    model="claude-sonnet-4",
                    model_cache=mock_model_cache,
                    auth_manager=mock_auth_manager,
                    max_retries=5,
                    first_token_timeout=30
                ):
                    pass
            except Exception:
                pass
        
        print(f"Call count: {call_count}")
        assert call_count == 5  # Should try exactly 5 times
        print("✓ max_retries parameter respected")


# ==================================================================================================
# Tests for truncation detection
# ==================================================================================================

class TestStreamingAnthropicTruncationDetection:
    """Tests for truncation detection in Anthropic streaming."""
    
    @pytest.mark.asyncio
    async def test_stop_reason_is_max_tokens_when_truncated(self, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Sets stop_reason to max_tokens when content is truncated.
        Goal: Verify truncation detection without completion signals.
        """
        print("Setup: Mock stream without completion signals (truncated)...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="This response was cut off mid-sentence because")
            # No context_usage event = truncation
        
        print("Action: Streaming to Anthropic format...")
        events = []
        
        with patch('kiro.streaming_anthropic.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_anthropic.parse_bracket_tool_calls', return_value=[]):
                async for event in stream_kiro_to_anthropic(
                    mock_response, "claude-sonnet-4", mock_model_cache, mock_auth_manager
                ):
                    events.append(event)
        
        print(f"Received {len(events)} events")
        
        # Should have message_delta with stop_reason: max_tokens
        message_delta_events = [e for e in events if "message_delta" in e]
        assert len(message_delta_events) >= 1
        print(f"Comparing stop_reason: Expected 'max_tokens', Got event: {message_delta_events[0]}")
        assert "max_tokens" in message_delta_events[0]
        print("✓ stop_reason is max_tokens when truncated")
    
    @pytest.mark.asyncio
    async def test_stop_reason_is_tool_use_even_without_completion_signals(self, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Sets stop_reason to tool_use when tool use present.
        Goal: Verify tool_use takes priority (not confused with content truncation).
        """
        print("Setup: Mock stream with tool use but no completion signals...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Let me call a tool")
            yield KiroEvent(type="tool_use", tool_use={
                "id": "toolu_1",
                "function": {"name": "get_weather", "arguments": "{}"}
            })
            # No context_usage event, but tool use present = tool_use stop_reason
        
        print("Action: Streaming to Anthropic format...")
        events = []
        
        with patch('kiro.streaming_anthropic.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_anthropic.parse_bracket_tool_calls', return_value=[]):
                async for event in stream_kiro_to_anthropic(
                    mock_response, "claude-sonnet-4", mock_model_cache, mock_auth_manager
                ):
                    events.append(event)
        
        print(f"Received {len(events)} events")
        
        # Tool use takes priority (not confused with content truncation)
        message_delta_events = [e for e in events if "message_delta" in e]
        assert len(message_delta_events) >= 1
        print(f"Comparing stop_reason: Expected 'tool_use', Got event: {message_delta_events[0]}")
        assert "tool_use" in message_delta_events[0]
        print("✓ stop_reason is tool_use (not confused with content truncation)")
    
    @pytest.mark.asyncio
    async def test_stop_reason_is_end_turn_with_completion_signals(self, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Sets stop_reason to end_turn when completion signals present.
        Goal: Verify normal completion is detected correctly.
        """
        print("Setup: Mock stream with completion signals (not truncated)...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Complete response")
            yield KiroEvent(type="context_usage", context_usage_percentage=5.0)
        
        print("Action: Streaming to Anthropic format...")
        events = []
        
        with patch('kiro.streaming_anthropic.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_anthropic.parse_bracket_tool_calls', return_value=[]):
                async for event in stream_kiro_to_anthropic(
                    mock_response, "claude-sonnet-4", mock_model_cache, mock_auth_manager
                ):
                    events.append(event)
        
        print(f"Received {len(events)} events")
        
        # With completion signals, should be end_turn
        message_delta_events = [e for e in events if "message_delta" in e]
        assert len(message_delta_events) >= 1
        print(f"Comparing stop_reason: Expected 'end_turn', Got event: {message_delta_events[0]}")
        assert "end_turn" in message_delta_events[0]
        print("✓ stop_reason is end_turn with completion signals")
    
    @pytest.mark.asyncio
    async def test_collect_detects_truncation_in_non_streaming(self, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Non-streaming detects truncation correctly.
        Goal: Verify collect_anthropic_response detects truncation.
        """
        print("Setup: Mock stream result without completion signals...")
        
        mock_result = StreamResult(
            content="Truncated response",
            thinking_content="",
            tool_calls=[],
            usage=None,
            context_usage_percentage=None  # No completion signal = truncation
        )
        
        print("Action: Collecting Anthropic response...")
        
        with patch('kiro.streaming_anthropic.collect_stream_to_result', return_value=mock_result):
            result = await collect_anthropic_response(
                mock_response, "claude-sonnet-4", mock_model_cache, mock_auth_manager
            )
        
        print(f"stop_reason: {result['stop_reason']}")
        
        # Should detect truncation and set max_tokens
        assert result["stop_reason"] == "max_tokens"
        print("✓ collect_anthropic_response detects truncation correctly")


# ==================================================================================================
# Native reasoning (reasoningContentEvent) tests
# ==================================================================================================

from kiro.tokenizer import count_tokens

NATIVE_SECRET = "SECRET_REASONING_MARKER_7f3a"


def _patch_handling(mode, enabled=True):
    """Pins FAKE_REASONING_* in BOTH modules that read them (core parsing + Anthropic formatting)."""
    from contextlib import ExitStack

    stack = ExitStack()
    stack.enter_context(patch("kiro.streaming_core.FAKE_REASONING_HANDLING", mode))
    stack.enter_context(patch("kiro.streaming_anthropic.FAKE_REASONING_HANDLING", mode))
    stack.enter_context(patch("kiro.streaming_core.FAKE_REASONING_ENABLED", enabled))
    return stack


def _events_source(events):
    """Returns a replacement for parse_kiro_stream that yields the given KiroEvents."""
    async def fake_parse(*args, **kwargs):
        for event in events:
            yield event
    return fake_parse


def _bytes_response(mock_response, chunks):
    """Attaches an aiter_bytes yielding chunks."""
    async def aiter_bytes():
        for chunk in chunks:
            yield chunk
    mock_response.aiter_bytes = aiter_bytes
    return mock_response


def _native(text):
    """Native reasoning KiroEvent."""
    return KiroEvent(type="thinking", thinking_content=text, is_native_reasoning=True)


def _tag(text):
    """Tag-based thinking KiroEvent."""
    return KiroEvent(type="thinking", thinking_content=text)


def _content(text):
    return KiroEvent(type="content", content=text)


def _tool(name="Bash", tool_id="toolu_1", args='{"command": "ls"}'):
    return KiroEvent(type="tool_use", tool_use={
        "id": tool_id, "type": "function", "function": {"name": name, "arguments": args},
    })


async def _run_events(events, mock_response, mock_model_cache, mock_auth_manager, mode="as_reasoning_content"):
    """Runs stream_kiro_to_anthropic over fake KiroEvents and returns parsed SSE tuples."""
    chunks = []
    # Completion signal so the stream is not treated as truncated (and no truncation state is saved)
    events = list(events) + [KiroEvent(type="context_usage", context_usage_percentage=1.0)]
    with _patch_handling(mode), \
            patch("kiro.streaming_anthropic.parse_kiro_stream", _events_source(events)):
        async for chunk in stream_kiro_to_anthropic(
            mock_response, "claude-sonnet-4", mock_model_cache, mock_auth_manager
        ):
            chunks.append(chunk)
    return chunks


def _shape(sse_events):
    """Compact description of an SSE sequence: (event, index, block/delta type)."""
    shape = []
    for name, data in sse_events:
        if name == "content_block_start":
            shape.append((name, data["index"], data["content_block"]["type"]))
        elif name == "content_block_delta":
            shape.append((name, data["index"], data["delta"]["type"]))
        elif name == "content_block_stop":
            shape.append((name, data["index"], None))
        else:
            shape.append((name, None, None))
    return shape


def _collapse(shape):
    """Collapses consecutive identical delta entries so sequences stay readable."""
    out = []
    for item in shape:
        if out and item == out[-1] and item[0] == "content_block_delta":
            continue
        out.append(item)
    return out


START = ("message_start", None, None)
DELTA = ("message_delta", None, None)
STOP = ("message_stop", None, None)


class TestStreamingAnthropicNativeReasoning:
    """Block ordering for native reasoning (KiroEvent level, FAKE_REASONING_HANDLING=as_reasoning_content)."""

    @pytest.mark.asyncio
    async def test_reasoning_then_content(self, mock_response, mock_model_cache, mock_auth_manager, anthropic_sse):
        """
        What it does: reasoning x3 -> content x2.
        Goal: thinking block 0 (3 deltas) is stopped BEFORE text block 1 opens; legal sequence.
        """
        chunks = await _run_events(
            [_native("a"), _native("b"), _native("c"), _content("X"), _content("Y")],
            mock_response, mock_model_cache, mock_auth_manager,
        )
        sse = anthropic_sse.parse(chunks)

        assert anthropic_sse.assert_legal(sse) == ["thinking", "text"]
        assert _collapse(_shape(sse)) == [
            START,
            ("content_block_start", 0, "thinking"),
            ("content_block_delta", 0, "thinking_delta"),
            ("content_block_stop", 0, None),
            ("content_block_start", 1, "text"),
            ("content_block_delta", 1, "text_delta"),
            ("content_block_stop", 1, None),
            DELTA, STOP,
        ]
        thinking = [d["delta"]["thinking"] for n, d in sse if n == "content_block_delta" and d["delta"]["type"] == "thinking_delta"]
        assert thinking == ["a", "b", "c"]
        assert [d["delta"]["text"] for n, d in sse if n == "content_block_delta" and d["delta"]["type"] == "text_delta"] == ["X", "Y"]
        assert sse[-2][1]["delta"]["stop_reason"] == "end_turn"

    @pytest.mark.asyncio
    async def test_reasoning_then_tool_use(self, mock_response, mock_model_cache, mock_auth_manager, anthropic_sse):
        """
        What it does: reasoning -> tool_use.
        Goal: thinking stopped before the tool_use block (index 1); stop_reason=tool_use.
        """
        chunks = await _run_events([_native("plan"), _tool()], mock_response, mock_model_cache, mock_auth_manager)
        sse = anthropic_sse.parse(chunks)

        assert anthropic_sse.assert_legal(sse) == ["thinking", "tool_use"]
        assert _collapse(_shape(sse))[:5] == [
            START,
            ("content_block_start", 0, "thinking"),
            ("content_block_delta", 0, "thinking_delta"),
            ("content_block_stop", 0, None),
            ("content_block_start", 1, "tool_use"),
        ]
        assert sse[-2][1]["delta"]["stop_reason"] == "tool_use"

    @pytest.mark.asyncio
    async def test_reasoning_then_two_tool_uses(self, mock_response, mock_model_cache, mock_auth_manager, anthropic_sse):
        """
        What it does: reasoning -> 2 tool calls.
        Goal: Tool blocks get indices 1 and 2.
        """
        chunks = await _run_events(
            [_native("plan"), _tool(tool_id="toolu_1"), _tool("Read", "toolu_2", '{"path": "x"}')],
            mock_response, mock_model_cache, mock_auth_manager,
        )
        sse = anthropic_sse.parse(chunks)
        assert anthropic_sse.assert_legal(sse) == ["thinking", "tool_use", "tool_use"]
        assert [d["index"] for n, d in sse if n == "content_block_start"] == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_reasoning_only_then_pure_tool_use(self, mock_response, mock_model_cache, mock_auth_manager, anthropic_sse):
        """
        What it does: Many reasoning chunks and a tool call that arrives only at the end (no text).
        Goal: Thinking block stays open across all chunks and is closed only when the
        tool_use arrives; no text block is ever created.
        """
        events = [_native(str(i)) for i in range(20)] + [_tool()]
        chunks = await _run_events(events, mock_response, mock_model_cache, mock_auth_manager)
        sse = anthropic_sse.parse(chunks)

        assert anthropic_sse.assert_legal(sse) == ["thinking", "tool_use"]
        assert sum(1 for n, _ in sse if n == "content_block_delta" and _["delta"]["type"] == "thinking_delta") == 20
        assert sse[-2][1]["delta"]["stop_reason"] == "tool_use"

    @pytest.mark.asyncio
    async def test_reasoning_only_closes_at_end(self, mock_response, mock_model_cache, mock_auth_manager, anthropic_sse):
        """
        What it does: Only reasoning, nothing else.
        Goal: Thinking block closed at stream end; stop_reason=end_turn.
        """
        chunks = await _run_events([_native("only")], mock_response, mock_model_cache, mock_auth_manager)
        sse = anthropic_sse.parse(chunks)
        assert anthropic_sse.assert_legal(sse) == ["thinking"]
        assert sse[-2][1]["delta"]["stop_reason"] == "end_turn"

    @pytest.mark.asyncio
    async def test_interleaved_reasoning_content_reasoning_content(self, mock_response, mock_model_cache, mock_auth_manager, anthropic_sse):
        """
        What it does: reasoning -> content -> reasoning -> content.
        Goal: Four blocks with indices 0..3; text block is closed before the second thinking
        block opens (previously produced a duplicate index while text was still open).
        """
        chunks = await _run_events(
            [_native("r1"), _content("c1"), _native("r2"), _content("c2")],
            mock_response, mock_model_cache, mock_auth_manager,
        )
        sse = anthropic_sse.parse(chunks)

        assert anthropic_sse.assert_legal(sse) == ["thinking", "text", "thinking", "text"]
        assert [d["index"] for n, d in sse if n == "content_block_start"] == [0, 1, 2, 3]

    @pytest.mark.asyncio
    async def test_text_before_reasoning(self, mock_response, mock_model_cache, mock_auth_manager, anthropic_sse):
        """
        What it does: content -> reasoning.
        Goal: Text block 0 is stopped before thinking block 1 starts.
        """
        chunks = await _run_events([_content("hello"), _native("then think")], mock_response, mock_model_cache, mock_auth_manager)
        sse = anthropic_sse.parse(chunks)
        assert anthropic_sse.assert_legal(sse) == ["text", "thinking"]

    @pytest.mark.asyncio
    async def test_text_reasoning_tool_use(self, mock_response, mock_model_cache, mock_auth_manager, anthropic_sse):
        """
        What it does: content -> reasoning -> tool_use.
        Goal: Legal three-block sequence.
        """
        chunks = await _run_events([_content("c"), _native("r"), _tool()], mock_response, mock_model_cache, mock_auth_manager)
        assert anthropic_sse.assert_legal(anthropic_sse.parse(chunks)) == ["text", "thinking", "tool_use"]

    @pytest.mark.asyncio
    async def test_native_and_tag_thinking_share_one_block(self, mock_response, mock_model_cache, mock_auth_manager, anthropic_sse):
        """
        What it does: native reasoning, tag thinking, native reasoning in a row.
        Goal: Exactly ONE thinking block is opened (never double-open).
        """
        chunks = await _run_events(
            [_native("n1"), _tag("t1"), _native("n2"), _content("done")],
            mock_response, mock_model_cache, mock_auth_manager,
        )
        sse = anthropic_sse.parse(chunks)
        assert anthropic_sse.assert_legal(sse) == ["thinking", "text"]
        assert sum(1 for n, d in sse if n == "content_block_start" and d["content_block"]["type"] == "thinking") == 1
        joined = "".join(d["delta"]["thinking"] for n, d in sse if n == "content_block_delta" and d["delta"]["type"] == "thinking_delta")
        assert joined == "n1t1n2"

    @pytest.mark.asyncio
    async def test_chinese_and_emoji_pass_through_unescaped(self, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Chinese / emoji / em dash reasoning.
        Goal: SSE payload carries the characters themselves (ensure_ascii=False).
        """
        chunks = await _run_events([_native("中文推理 😀 — ok")], mock_response, mock_model_cache, mock_auth_manager)
        assert any("中文推理 😀 — ok" in c for c in chunks)

    @pytest.mark.asyncio
    async def test_output_tokens_include_reasoning(self, mock_response, mock_model_cache, mock_auth_manager, anthropic_sse):
        """
        What it does: Long reasoning + short content.
        Goal: usage.output_tokens counts reasoning text as well.
        """
        reasoning = "reasoning words " * 200
        chunks = await _run_events([_native(reasoning), _content("hi")], mock_response, mock_model_cache, mock_auth_manager)
        sse = anthropic_sse.parse(chunks)
        out = sse[-2][1]["usage"]["output_tokens"]
        assert out >= count_tokens(reasoning)
        assert out > count_tokens("hi")

    @pytest.mark.asyncio
    async def test_message_start_is_first_and_unique(self, mock_response, mock_model_cache, mock_auth_manager, anthropic_sse):
        """
        What it does: Reasoning-first stream.
        Goal: message_start is the first event and appears once.
        """
        chunks = await _run_events([_native("a")], mock_response, mock_model_cache, mock_auth_manager)
        sse = anthropic_sse.parse(chunks)
        assert sse[0][0] == "message_start"
        assert [n for n, _ in sse].count("message_start") == 1

    @pytest.mark.asyncio
    async def test_empty_thinking_content_still_legal(self, mock_response, mock_model_cache, mock_auth_manager, anthropic_sse):
        """
        What it does: A thinking event whose text is None followed by content.
        Goal: Stream remains legal (empty thinking block, no empty delta).
        """
        chunks = await _run_events(
            [KiroEvent(type="thinking", thinking_content=None, is_native_reasoning=True), _content("x")],
            mock_response, mock_model_cache, mock_auth_manager,
        )
        sse = anthropic_sse.parse(chunks)
        assert anthropic_sse.assert_legal(sse) == ["thinking", "text"]
        assert not [1 for n, d in sse if n == "content_block_delta" and d["delta"]["type"] == "thinking_delta"]


class TestStreamingAnthropicNativeReasoningRealBytes:
    """End-to-end through the REAL parser: AWS frames -> parse_kiro_stream -> Anthropic SSE."""

    TEXTS = ["I", "'ll", " check", ".\n\n", "中文推理", " 😀", " — done"]

    async def _stream(self, mock_response, mock_model_cache, mock_auth_manager, chunks, mode="as_reasoning_content"):
        _bytes_response(mock_response, chunks)
        out = []
        with _patch_handling(mode):
            async for item in stream_kiro_to_anthropic(mock_response, "claude-sonnet-4", mock_model_cache, mock_auth_manager):
                out.append(item)
        return out

    def _data(self, aws_frames, tail=True):
        data = b"".join(aws_frames.reasoning(t) for t in self.TEXTS)
        data += aws_frames.signature("QUJD" * 40)
        if tail:
            data += b"".join(aws_frames.content(c) for c in ["Hello", " world"])
            data += aws_frames.context_usage(2.5)
        return data

    @pytest.mark.asyncio
    async def test_full_stream_is_legal_with_one_delta_per_frame(self, mock_response, mock_model_cache, mock_auth_manager, anthropic_sse, aws_frames):
        """
        What it does: reasoning x7 -> signature -> content x2 -> context usage.
        Goal: Legal SSE, thinking_delta count == reasoning frame count, text intact,
        upstream signature never forwarded, stop_reason=end_turn.
        """
        chunks = await self._stream(mock_response, mock_model_cache, mock_auth_manager, [self._data(aws_frames)])
        sse = anthropic_sse.parse(chunks)

        assert anthropic_sse.assert_legal(sse) == ["thinking", "text"]
        deltas = [d["delta"]["thinking"] for n, d in sse if n == "content_block_delta" and d["delta"]["type"] == "thinking_delta"]
        assert deltas == self.TEXTS
        assert "".join(chunks).count("QUJD" * 40) == 0
        assert sse[-2][1]["delta"]["stop_reason"] == "end_turn"
        placeholder = [d for n, d in sse if n == "content_block_start" and d["content_block"]["type"] == "thinking"][0]
        assert placeholder["content_block"]["signature"].startswith("sig_")

    @pytest.mark.asyncio
    async def test_chunking_does_not_change_sse_semantics(self, mock_response, mock_model_cache, mock_auth_manager, anthropic_sse, aws_frames):
        """
        What it does: Same bytes chunked 1 byte at a time and with fixed-seed random sizes.
        Goal: Same legal block types and same concatenated thinking text.
        """
        import random

        data = self._data(aws_frames)
        expected = "".join(self.TEXTS)
        for sizes in ([1], [3, 5, 11], [random.Random(7).randint(1, 50) for _ in range(8)]):
            chunks = await self._stream(mock_response, mock_model_cache, mock_auth_manager, aws_frames.split(data, sizes))
            sse = anthropic_sse.parse(chunks)
            assert anthropic_sse.assert_legal(sse) == ["thinking", "text"]
            joined = "".join(d["delta"]["thinking"] for n, d in sse if n == "content_block_delta" and d["delta"]["type"] == "thinking_delta")
            assert joined == expected

    @pytest.mark.asyncio
    async def test_reasoning_then_tool_call_frames(self, mock_response, mock_model_cache, mock_auth_manager, anthropic_sse, aws_frames):
        """
        What it does: reasoning frames then tool call frames, no text.
        Goal: thinking -> tool_use, stop_reason=tool_use.
        """
        data = b"".join([
            aws_frames.reasoning("need a tool"),
            aws_frames.tool_start("Bash", "toolu_9"),
            aws_frames.tool_input('{"command": "ls"}', "Bash", "toolu_9"),
            aws_frames.tool_stop("Bash", "toolu_9"),
            aws_frames.context_usage(1.0),
        ])
        chunks = await self._stream(mock_response, mock_model_cache, mock_auth_manager, [data])
        sse = anthropic_sse.parse(chunks)
        assert anthropic_sse.assert_legal(sse) == ["thinking", "tool_use"]
        assert sse[-2][1]["delta"]["stop_reason"] == "tool_use"

    @pytest.mark.asyncio
    async def test_reasoning_with_bracket_text_does_not_invent_tool_use(self, mock_response, mock_model_cache, mock_auth_manager, anthropic_sse, aws_frames):
        """
        What it does: Reasoning that quotes bracket tool-call syntax.
        Goal: No tool_use block is generated from reasoning text.
        """
        data = aws_frames.reasoning('[Called evil with args: {"a": 1}]') + aws_frames.content("fine") + aws_frames.context_usage(1.0)
        chunks = await self._stream(mock_response, mock_model_cache, mock_auth_manager, [data])
        assert anthropic_sse.assert_legal(anthropic_sse.parse(chunks)) == ["thinking", "text"]

    @pytest.mark.asyncio
    async def test_reasoning_only_stream(self, mock_response, mock_model_cache, mock_auth_manager, anthropic_sse, aws_frames):
        """
        What it does: Reasoning-only stream that ends with a meteringEvent (genuine answer
        without a body, so it is complete and passed through).
        Goal: Still legal and closed.
        """
        data = self._data(aws_frames, tail=False) + aws_frames.metering(0.1)
        chunks = await self._stream(mock_response, mock_model_cache, mock_auth_manager, [data])
        assert anthropic_sse.assert_legal(anthropic_sse.parse(chunks)) == ["thinking"]


class TestStreamingAnthropicReasoningModes:
    """FAKE_REASONING_HANDLING semantics for native reasoning, end to end."""

    async def _stream(self, mock_response, mock_model_cache, mock_auth_manager, aws_frames, mode):
        data = aws_frames.reasoning("R1") + aws_frames.reasoning("R2") + aws_frames.content("C") + aws_frames.context_usage(1.0)
        _bytes_response(mock_response, [data])
        out = []
        with _patch_handling(mode):
            async for item in stream_kiro_to_anthropic(mock_response, "claude-sonnet-4", mock_model_cache, mock_auth_manager):
                out.append(item)
        return out

    @pytest.mark.asyncio
    async def test_as_reasoning_content_uses_thinking_block(self, mock_response, mock_model_cache, mock_auth_manager, anthropic_sse, aws_frames):
        """
        What it does: as_reasoning_content.
        Goal: thinking block with thinking_delta.
        """
        sse = anthropic_sse.parse(await self._stream(mock_response, mock_model_cache, mock_auth_manager, aws_frames, "as_reasoning_content"))
        assert anthropic_sse.assert_legal(sse) == ["thinking", "text"]

    @pytest.mark.asyncio
    async def test_remove_discards_reasoning_but_keeps_content(self, mock_response, mock_model_cache, mock_auth_manager, anthropic_sse, aws_frames):
        """
        What it does: remove.
        Goal: No thinking block and no reasoning text anywhere; content normal; stream legal.
        """
        chunks = await self._stream(mock_response, mock_model_cache, mock_auth_manager, aws_frames, "remove")
        sse = anthropic_sse.parse(chunks)
        assert anthropic_sse.assert_legal(sse) == ["text"]
        assert "R1" not in "".join(chunks)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mode", ["strip_tags", "pass", "include_as_text"])
    async def test_text_modes_emit_reasoning_as_a_single_text_block(self, mock_response, mock_model_cache, mock_auth_manager, anthropic_sse, aws_frames, mode):
        """
        What it does: strip_tags / pass / legacy include_as_text.
        Goal: Reasoning appears as text_delta in ONE text block before the content; no thinking block.
        """
        sse = anthropic_sse.parse(await self._stream(mock_response, mock_model_cache, mock_auth_manager, aws_frames, mode))
        assert anthropic_sse.assert_legal(sse) == ["text"]
        text = "".join(d["delta"]["text"] for n, d in sse if n == "content_block_delta")
        assert text == "R1R2C"

    @pytest.mark.asyncio
    async def test_unknown_mode_drops_thinking(self, mock_response, mock_model_cache, mock_auth_manager, anthropic_sse, aws_frames):
        """
        What it does: An unrecognized handling value (legacy 'strip').
        Goal: Thinking is dropped as before; content survives.
        """
        chunks = await self._stream(mock_response, mock_model_cache, mock_auth_manager, aws_frames, "strip")
        sse = anthropic_sse.parse(chunks)
        assert anthropic_sse.assert_legal(sse) == ["text"]
        assert "R1" not in "".join(chunks)

    @pytest.mark.asyncio
    async def test_text_mode_with_text_then_reasoning_then_content_is_legal(self, mock_response, mock_model_cache, mock_auth_manager, anthropic_sse):
        """
        What it does: content -> native reasoning -> content in text mode.
        Goal: Reuses the open text block - one block, legal.
        """
        chunks = await _run_events([_content("a"), _native("b"), _content("c")], mock_response, mock_model_cache, mock_auth_manager, mode="strip_tags")
        sse = anthropic_sse.parse(chunks)
        assert anthropic_sse.assert_legal(sse) == ["text"]


class TestCollectAnthropicResponseNativeReasoning:
    """Non-streaming Anthropic response with native reasoning."""

    async def _collect(self, mock_response, mock_model_cache, mock_auth_manager, data, mode="as_reasoning_content"):
        _bytes_response(mock_response, [data])
        with _patch_handling(mode):
            return await collect_anthropic_response(mock_response, "claude-sonnet-4", mock_model_cache, mock_auth_manager)

    @pytest.mark.asyncio
    async def test_thinking_text_tool_use_in_order(self, mock_response, mock_model_cache, mock_auth_manager, aws_frames):
        """
        What it does: reasoning + content + tool call.
        Goal: content == [thinking, text, tool_use]; thinking has all reasoning and a sig_ placeholder.
        """
        data = b"".join([
            aws_frames.reasoning("step one, "), aws_frames.reasoning("step two"),
            aws_frames.content("answer"),
            aws_frames.tool_start("Bash", "toolu_1"),
            aws_frames.tool_input('{"command": "ls"}', "Bash", "toolu_1"),
            aws_frames.tool_stop("Bash", "toolu_1"),
            aws_frames.context_usage(2.0),
        ])
        result = await self._collect(mock_response, mock_model_cache, mock_auth_manager, data)

        blocks = result["content"]
        assert [b["type"] for b in blocks] == ["thinking", "text", "tool_use"]
        assert blocks[0]["thinking"] == "step one, step two"
        assert blocks[0]["signature"].startswith("sig_")
        assert blocks[1]["text"] == "answer"
        assert blocks[2]["input"] == {"command": "ls"}
        assert result["stop_reason"] == "tool_use"

    @pytest.mark.asyncio
    async def test_reasoning_only(self, mock_response, mock_model_cache, mock_auth_manager, aws_frames):
        """
        What it does: Only reasoning, with a meteringEvent (complete response without a body).
        Goal: content == [thinking]; end_turn.
        """
        data = aws_frames.reasoning("solo 中文") + aws_frames.metering(0.1)
        result = await self._collect(mock_response, mock_model_cache, mock_auth_manager, data)
        assert [b["type"] for b in result["content"]] == ["thinking"]
        assert result["content"][0]["thinking"] == "solo 中文"
        assert result["stop_reason"] == "end_turn"

    @pytest.mark.asyncio
    async def test_remove_mode_has_no_thinking_block(self, mock_response, mock_model_cache, mock_auth_manager, aws_frames):
        """
        What it does: remove.
        Goal: Only the text block; reasoning text absent from the whole response.
        """
        data = aws_frames.reasoning(NATIVE_SECRET) + aws_frames.content("shown")
        result = await self._collect(mock_response, mock_model_cache, mock_auth_manager, data, mode="remove")
        assert [b["type"] for b in result["content"]] == ["text"]
        assert NATIVE_SECRET not in json.dumps(result)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mode", ["strip_tags", "pass", "include_as_text"])
    async def test_text_modes_prepend_reasoning_to_text(self, mock_response, mock_model_cache, mock_auth_manager, aws_frames, mode):
        """
        What it does: strip_tags / pass / include_as_text.
        Goal: Reasoning is prepended to the text block; no thinking block.
        """
        data = aws_frames.reasoning("R.") + aws_frames.content("C")
        result = await self._collect(mock_response, mock_model_cache, mock_auth_manager, data, mode=mode)
        assert [b["type"] for b in result["content"]] == ["text"]
        assert result["content"][0]["text"] == "R.C"

    @pytest.mark.asyncio
    async def test_unknown_mode_drops_reasoning(self, mock_response, mock_model_cache, mock_auth_manager, aws_frames):
        """
        What it does: Unknown handling value.
        Goal: Reasoning dropped (legacy behavior), content kept.
        """
        data = aws_frames.reasoning("R.") + aws_frames.content("C")
        result = await self._collect(mock_response, mock_model_cache, mock_auth_manager, data, mode="strip")
        assert [b["type"] for b in result["content"]] == ["text"]
        assert result["content"][0]["text"] == "C"

    @pytest.mark.asyncio
    async def test_bracket_syntax_in_reasoning_does_not_create_tool_use(self, mock_response, mock_model_cache, mock_auth_manager, aws_frames):
        """
        What it does: Reasoning quoting [Called x with args: {...}].
        Goal: No tool_use block; stop_reason=end_turn.
        """
        data = aws_frames.reasoning('[Called evil with args: {"a": 1}]') + aws_frames.content("ok") + aws_frames.context_usage(1.0)
        result = await self._collect(mock_response, mock_model_cache, mock_auth_manager, data)
        assert [b["type"] for b in result["content"]] == ["thinking", "text"]
        assert result["stop_reason"] == "end_turn"

    @pytest.mark.asyncio
    async def test_output_tokens_include_reasoning(self, mock_response, mock_model_cache, mock_auth_manager, aws_frames):
        """
        What it does: Long reasoning, short answer.
        Goal: output_tokens covers the reasoning text.
        """
        reasoning = "reasoning words " * 200
        result = await self._collect(mock_response, mock_model_cache, mock_auth_manager, aws_frames.reasoning(reasoning) + aws_frames.content("hi"))
        assert result["usage"]["output_tokens"] >= count_tokens(reasoning)


# ==================================================================================================
# Empty upstream response (guard) and message_start de-duplication
# ==================================================================================================

import asyncio

import httpx

from kiro.streaming_core import STREAM_READY
from kiro.upstream_guard import EmptyUpstreamResponseError


def _real_stream_response(chunks):
    """Response double with a real one-shot byte stream (aiter_bytes() works once, like httpx)."""
    response = AsyncMock()
    response.status_code = 200
    response.aclose = AsyncMock()
    state = {"calls": 0}

    async def gen():
        for chunk in chunks:
            yield chunk

    def aiter_bytes(*args, **kwargs):
        state["calls"] += 1
        if state["calls"] > 1:
            raise httpx.StreamConsumed()
        return gen()

    response.aiter_bytes = aiter_bytes
    return response


def _hanging_stream_response():
    """Response double whose stream never yields (first token timeout)."""
    response = AsyncMock()
    response.status_code = 200
    response.aclose = AsyncMock()

    async def gen():
        await asyncio.sleep(3600)
        yield b""

    response.aiter_bytes = lambda *a, **k: gen()
    return response


def _normal_bytes(aws_frames):
    """Successful stream: reasoning, text, contextUsage, metering."""
    return (
        aws_frames.reasoning("think") + aws_frames.content("Hello")
        + aws_frames.content(" world") + aws_frames.context_usage(5.0) + aws_frames.metering(0.5)
    )


class TestCollectAnthropicResponseEmptyUpstream:
    """collect_anthropic_response() with the empty-response guard (non-streaming)."""

    @pytest.mark.asyncio
    async def test_empty_upstream_response_raises(self, aws_frames, mock_model_cache, mock_auth_manager):
        """
        What it does: Non-streaming collection of a lone contextUsageEvent frame.
        Purpose: The route gets EmptyUpstreamResponseError instead of an empty message.
        """
        response = _real_stream_response([aws_frames.context_usage(60.7)])

        with patch("kiro.streaming_core.EMPTY_RESPONSE_AS_ERROR", True):
            with pytest.raises(EmptyUpstreamResponseError):
                await collect_anthropic_response(response, "claude-sonnet-4.5", mock_model_cache, mock_auth_manager)

    @pytest.mark.asyncio
    async def test_metered_empty_answer_is_a_normal_empty_message(
        self, aws_frames, mock_model_cache, mock_auth_manager
    ):
        """
        What it does: contextUsageEvent + meteringEvent, no output.
        Purpose: The model really answered with nothing: HTTP 200 style empty message, no error.
        """
        response = _real_stream_response([aws_frames.context_usage(2.0) + aws_frames.metering(0.1)])

        with patch("kiro.streaming_core.EMPTY_RESPONSE_AS_ERROR", True):
            result = await collect_anthropic_response(
                response, "claude-sonnet-4.5", mock_model_cache, mock_auth_manager
            )

        assert result["content"] == []
        assert result["usage"]["output_tokens"] == 0
        assert result["stop_reason"] == "end_turn"

    @pytest.mark.asyncio
    async def test_switch_off_returns_empty_message(self, aws_frames, mock_model_cache, mock_auth_manager):
        """
        What it does: EMPTY_RESPONSE_AS_ERROR=False with an empty upstream response.
        Purpose: Native behavior: empty message, no error.
        """
        response = _real_stream_response([aws_frames.context_usage(60.7)])

        with patch("kiro.streaming_core.EMPTY_RESPONSE_AS_ERROR", False):
            result = await collect_anthropic_response(
                response, "claude-sonnet-4.5", mock_model_cache, mock_auth_manager
            )

        assert result["content"] == []
        assert result["stop_reason"] == "end_turn"

    @pytest.mark.asyncio
    async def test_normal_response_is_unchanged(self, aws_frames, mock_model_cache, mock_auth_manager):
        """
        What it does: Normal response, guard on.
        Purpose: Output blocks are delivered as before.
        """
        response = _real_stream_response(aws_frames.split(_normal_bytes(aws_frames), [17]))

        with patch("kiro.streaming_core.EMPTY_RESPONSE_AS_ERROR", True):
            result = await collect_anthropic_response(
                response, "claude-sonnet-4.5", mock_model_cache, mock_auth_manager
            )

        assert [b["type"] for b in result["content"]] == ["thinking", "text"]
        assert result["content"][1]["text"] == "Hello world"


class TestStreamingAnthropicEmptyUpstream:
    """stream_with_first_token_retry_anthropic() with the guard."""

    @staticmethod
    async def _run(response, make_request=None, timeout=5, ready=True, **kwargs):
        """Runs the retry wrapper and returns all yielded items."""
        cache = MagicMock()
        cache.get_max_input_tokens.return_value = 200000
        return [
            item async for item in stream_with_first_token_retry_anthropic(
                make_request=make_request or AsyncMock(),
                model="claude-sonnet-4.5",
                model_cache=cache,
                auth_manager=MagicMock(),
                initial_response=response,
                first_token_timeout=timeout,
                emit_ready_marker=ready,
                **kwargs,
            )
        ]

    @pytest.mark.asyncio
    async def test_empty_response_raises_before_yielding_anything(self, aws_frames):
        """
        What it does: Empty upstream response through the Anthropic streaming wrapper.
        Purpose: Raised before message_start (and before the marker): no SSE byte is produced.
        """
        response = _real_stream_response([aws_frames.context_usage(60.7)])
        items = []

        with patch("kiro.streaming_core.EMPTY_RESPONSE_AS_ERROR", True):
            with pytest.raises(EmptyUpstreamResponseError):
                cache = MagicMock()
                async for item in stream_with_first_token_retry_anthropic(
                    make_request=AsyncMock(), model="m", model_cache=cache, auth_manager=MagicMock(),
                    initial_response=response, first_token_timeout=5, emit_ready_marker=True,
                ):
                    items.append(item)

        assert items == []

    @pytest.mark.asyncio
    async def test_normal_stream_is_legal_with_single_message_start(self, aws_frames, anthropic_sse):
        """
        What it does: Normal response, marker enabled.
        Purpose: Marker first, then a legal Anthropic sequence message_start -> ... -> message_stop.
        """
        response = _real_stream_response(aws_frames.split(_normal_bytes(aws_frames), [9]))

        with patch("kiro.streaming_core.EMPTY_RESPONSE_AS_ERROR", True):
            items = await self._run(response)

        assert items[0] is STREAM_READY
        events = anthropic_sse.parse(items[1:])
        assert anthropic_sse.assert_legal(events) == ["thinking", "text"]

    @pytest.mark.asyncio
    async def test_timeout_inside_guard_gives_exactly_one_message_start(self, aws_frames, anthropic_sse):
        """
        What it does: Attempt 1 hangs inside the guard, attempt 2 is normal.
        Purpose: The retried stream contains one message_start and is a legal sequence.
        """
        good = _real_stream_response([_normal_bytes(aws_frames)])
        make_request = AsyncMock(return_value=good)

        with patch("kiro.streaming_core.EMPTY_RESPONSE_AS_ERROR", True):
            items = await self._run(_hanging_stream_response(), make_request=make_request, timeout=0.05)

        assert items[0] is STREAM_READY
        events = anthropic_sse.parse(items[1:])
        assert sum(1 for name, _ in events if name == "message_start") == 1
        anthropic_sse.assert_legal(events)
        assert make_request.await_count == 1

    @pytest.mark.asyncio
    async def test_timeout_after_message_start_does_not_repeat_it_when_guard_is_off(
        self, aws_frames, anthropic_sse
    ):
        """
        What it does: Guard disabled; attempt 1 emits message_start then times out waiting
        for the first token; attempt 2 is normal.
        Purpose: Fixes the duplicate message_start on first-token retry.
        """
        good = _real_stream_response([_normal_bytes(aws_frames)])
        make_request = AsyncMock(return_value=good)

        with patch("kiro.streaming_core.EMPTY_RESPONSE_AS_ERROR", False):
            items = await self._run(_hanging_stream_response(), make_request=make_request, timeout=0.05)

        assert items[0] is STREAM_READY
        strings = items[1:]
        events = anthropic_sse.parse(strings)
        print([name for name, _ in events])
        assert sum(1 for name, _ in events if name == "message_start") == 1
        anthropic_sse.assert_legal(events)
        assert make_request.await_count == 1

    @pytest.mark.asyncio
    async def test_two_timeouts_then_success_guard_off_single_message_start(self, aws_frames, anthropic_sse):
        """
        What it does: Guard off; attempts 1 and 2 time out after message_start, attempt 3 works.
        Purpose: message_start is sent once no matter how many retries happened.
        """
        good = _real_stream_response([_normal_bytes(aws_frames)])
        responses = [_hanging_stream_response(), good]
        make_request = AsyncMock(side_effect=lambda: responses.pop(0))

        with patch("kiro.streaming_core.EMPTY_RESPONSE_AS_ERROR", False):
            items = await self._run(_hanging_stream_response(), make_request=make_request, timeout=0.05)

        events = anthropic_sse.parse(items[1:])
        assert sum(1 for name, _ in events if name == "message_start") == 1
        anthropic_sse.assert_legal(events)

    @pytest.mark.asyncio
    async def test_sequence_identical_with_and_without_guard(self, aws_frames):
        """
        What it does: Same successful bytes with the guard on and off.
        Purpose: The guard must not change the SSE sequence (ids/signatures aside).
        """
        chunks = aws_frames.split(_normal_bytes(aws_frames), [23])

        def shape(items):
            names = []
            for item in items:
                if isinstance(item, str):
                    names.extend(line[len("event: "):] for line in item.split("\n") if line.startswith("event: "))
            return names

        with patch("kiro.streaming_core.EMPTY_RESPONSE_AS_ERROR", True):
            guarded = await self._run(_real_stream_response(chunks))
        with patch("kiro.streaming_core.EMPTY_RESPONSE_AS_ERROR", False):
            plain = await self._run(_real_stream_response(chunks))

        assert shape(guarded) == shape(plain)

    @pytest.mark.asyncio
    async def test_metered_empty_answer_streams_a_legal_empty_message(self, aws_frames, anthropic_sse):
        """
        What it does: contextUsageEvent + meteringEvent through the streaming wrapper.
        Purpose: Passed through as a legal, empty Anthropic message (no error).
        """
        response = _real_stream_response([aws_frames.context_usage(2.0) + aws_frames.metering(0.1)])

        with patch("kiro.streaming_core.EMPTY_RESPONSE_AS_ERROR", True):
            items = await self._run(response)

        events = anthropic_sse.parse(items[1:])
        assert anthropic_sse.assert_legal(events) == []


class TestStreamKiroToAnthropicEmitMessageStart:
    """stream_kiro_to_anthropic(emit_message_start=...)."""

    @pytest.mark.asyncio
    async def test_false_skips_message_start_only(self, aws_frames, mock_model_cache, mock_auth_manager):
        """
        What it does: Runs with emit_message_start=False and True on the same bytes.
        Purpose: The only difference is the absent message_start event.
        """
        data = _normal_bytes(aws_frames)

        async def run(flag):
            response = _real_stream_response([data])
            return [
                chunk async for chunk in stream_kiro_to_anthropic(
                    response, "m", mock_model_cache, mock_auth_manager,
                    first_token_timeout=5, emit_message_start=flag,
                )
            ]

        import re

        def normalize(chunks):
            # Placeholder thinking signatures are random per stream
            return [re.sub(r"sig_[0-9a-f]{32}", "sig_X", c) for c in chunks]

        with_start = await run(True)
        without_start = await run(False)

        assert with_start[0].startswith("event: message_start")
        assert normalize(with_start[1:]) == normalize(without_start)
        assert not any(c.startswith("event: message_start") for c in without_start)


# ==================================================================================================
# Incomplete upstream response (reasoning only, no meteringEvent, no text, no tool_use)
# ==================================================================================================

INCOMPLETE_THOUGHT = "SECRET_THOUGHT_FOR_INCOMPLETE_8e1b"


def _incomplete_bytes(aws_frames, count=6, with_context_usage=True):
    """Reasoning frames (+ contextUsageEvent) without meteringEvent, text or tool call."""
    data = b"".join(aws_frames.reasoning(f"{INCOMPLETE_THOUGHT} {i}") for i in range(count))
    if with_context_usage:
        data += aws_frames.context_usage(3.5)
    return data


async def _anthropic_stream(chunks, mode="as_reasoning_content", **kwargs):
    """Runs stream_kiro_to_anthropic over real bytes and returns the SSE strings."""
    cache = MagicMock()
    cache.get_max_input_tokens.return_value = 200000
    with _patch_handling(mode):
        return [
            item async for item in stream_kiro_to_anthropic(
                _real_stream_response(chunks), "claude-sonnet-4-5", cache, MagicMock(),
                first_token_timeout=5, **kwargs,
            )
        ]


class TestStreamingAnthropicIncompleteResponse:
    """stream_kiro_to_anthropic(): incomplete responses end with an error event."""

    @pytest.mark.asyncio
    async def test_thinking_block_is_closed_then_error_event(self, aws_frames, anthropic_sse):
        """
        What it does: 6 reasoning frames + contextUsageEvent, no metering (reported failure).
        Purpose: message_start -> thinking block -> content_block_stop(same index) -> error,
        and neither message_delta nor message_stop.
        """
        chunks = await _anthropic_stream([_incomplete_bytes(aws_frames)])

        events = anthropic_sse.parse(chunks)
        names = [n for n, _ in events]
        print(names)
        assert anthropic_sse.assert_legal_aborted(events) == ["thinking"]
        assert names[-2:] == ["content_block_stop", "error"]
        assert names.count("content_block_delta") == 6
        assert events[-2][1]["index"] == 0
        assert events[-1][1]["error"]["type"] == "overloaded_error"

    @pytest.mark.asyncio
    async def test_error_message_is_friendly_and_has_no_content(self, aws_frames):
        """
        What it does: Reads the error event text.
        Purpose: Mentions retry, claims no cause, leaks no reasoning text.
        """
        chunks = await _anthropic_stream([_incomplete_bytes(aws_frames)])

        error_chunk = chunks[-1]
        assert error_chunk.startswith("event: error\n")
        message = json.loads(error_chunk.split("\n")[1][len("data: "):])["error"]["message"].lower()
        assert "retry" in message and "incomplete" in message
        for forbidden in ("overloaded", "rate limit", "capacity", "quota", "throttl"):
            assert forbidden not in message
        assert INCOMPLETE_THOUGHT.lower() not in message

    @pytest.mark.asyncio
    async def test_configured_error_type_is_used(self, aws_frames):
        """
        What it does: INCOMPLETE_RESPONSE_ERROR_TYPE=api_error.
        Purpose: The error event carries the configured type.
        """
        with patch("kiro.empty_response.INCOMPLETE_RESPONSE_ERROR_TYPE", "api_error"):
            chunks = await _anthropic_stream([_incomplete_bytes(aws_frames)])

        assert json.loads(chunks[-1].split("\n")[1][len("data: "):])["error"]["type"] == "api_error"

    @pytest.mark.asyncio
    async def test_only_reasoning_frames_and_byte_by_byte_chunks(self, aws_frames, anthropic_sse):
        """
        What it does: 49 reasoning frames, no contextUsageEvent (17:15:07 shape), 1-byte chunks
        and 13-byte chunks.
        Purpose: Same error ending whatever the chunking.
        """
        data = _incomplete_bytes(aws_frames, count=49, with_context_usage=False)

        for sizes in ([1], [13], [len(data)]):
            events = anthropic_sse.parse(await _anthropic_stream(aws_frames.split(data, sizes)))
            assert anthropic_sse.assert_legal_aborted(events) == ["thinking"], sizes

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mode", ["include_as_text", "strip_tags", "pass"])
    async def test_text_modes_close_the_text_block_then_error(self, aws_frames, anthropic_sse, mode):
        """
        What it does: Reasoning rendered as text (FAKE_REASONING_HANDLING text modes).
        Purpose: The reasoning-as-text block is closed before the error; the verdict does not
        treat that text as a real answer.
        """
        events = anthropic_sse.parse(await _anthropic_stream([_incomplete_bytes(aws_frames)], mode=mode))

        assert anthropic_sse.assert_legal_aborted(events) == ["text"]

    @pytest.mark.asyncio
    async def test_remove_mode_opens_no_block(self, aws_frames, anthropic_sse):
        """
        What it does: FAKE_REASONING_HANDLING=remove discards the reasoning frames.
        Purpose: message_start is followed directly by the error event.
        """
        events = anthropic_sse.parse(await _anthropic_stream([_incomplete_bytes(aws_frames)], mode="remove"))

        assert [n for n, _ in events] == ["message_start", "error"]
        assert anthropic_sse.assert_legal_aborted(events) == []

    @pytest.mark.asyncio
    async def test_emit_message_start_false_still_ends_with_error(self, aws_frames):
        """
        What it does: Retry attempt (message_start was sent by an earlier attempt).
        Purpose: No second message_start; the stream still ends with the error event.
        """
        chunks = await _anthropic_stream([_incomplete_bytes(aws_frames)], emit_message_start=False)

        assert not any(c.startswith("event: message_start") for c in chunks)
        assert chunks[-1].startswith("event: error")

    @pytest.mark.asyncio
    async def test_metering_makes_it_a_normal_end_turn(self, aws_frames, anthropic_sse):
        """
        What it does: Same reasoning + a meteringEvent, no body.
        Purpose: Complete: legal stream ending with message_stop (pass through verbatim).
        """
        events = anthropic_sse.parse(
            await _anthropic_stream([_incomplete_bytes(aws_frames) + aws_frames.metering(0.3)])
        )

        assert anthropic_sse.assert_legal(events) == ["thinking"]
        delta = next(d for n, d in events if n == "message_delta")
        assert delta["delta"]["stop_reason"] == "end_turn"

    @pytest.mark.asyncio
    async def test_reasoning_and_text_without_metering_is_unchanged_with_warning(
        self, aws_frames, anthropic_sse, loguru_messages
    ):
        """
        What it does: Reasoning + text, no metering (possibly truncated text).
        Purpose: Behavior unchanged (normal end); exactly one WARNING, no content in it.
        """
        data = aws_frames.reasoning("r") + aws_frames.content("partial answer") + aws_frames.context_usage(2.0)

        events = anthropic_sse.parse(await _anthropic_stream([data]))

        assert anthropic_sse.assert_legal(events) == ["thinking", "text"]
        lines = [m for m in loguru_messages if "[IncompleteUpstreamResponse]" in m]
        assert len(lines) == 1 and "partial answer" not in lines[0]

    @pytest.mark.asyncio
    async def test_reasoning_and_tool_use_without_metering_is_unchanged(self, aws_frames, anthropic_sse):
        """
        What it does: Reasoning + tool call, no metering.
        Purpose: Normal end with stop_reason=tool_use.
        """
        data = (
            aws_frames.reasoning("r")
            + aws_frames.tool_start("Bash", "toolu_1")
            + aws_frames.tool_input('{"command": "ls"}', "Bash", "toolu_1")
            + aws_frames.tool_stop("Bash", "toolu_1")
        )

        events = anthropic_sse.parse(await _anthropic_stream([data]))

        assert anthropic_sse.assert_legal(events) == ["thinking", "tool_use"]
        assert next(d for n, d in events if n == "message_delta")["delta"]["stop_reason"] == "tool_use"

    @pytest.mark.asyncio
    async def test_switch_off_restores_the_old_ending(self, aws_frames, anthropic_sse, loguru_messages):
        """
        What it does: INCOMPLETE_RESPONSE_AS_ERROR=False with an incomplete stream.
        Purpose: Old behavior: thinking-only message ending with end_turn + message_stop.
        """
        with patch("kiro.streaming_core.INCOMPLETE_RESPONSE_AS_ERROR", False):
            chunks = await _anthropic_stream([_incomplete_bytes(aws_frames)])

        events = anthropic_sse.parse(chunks)
        assert anthropic_sse.assert_legal(events) == ["thinking"]
        assert next(d for n, d in events if n == "message_delta")["delta"]["stop_reason"] == "end_turn"
        assert len([m for m in loguru_messages if "[IncompleteUpstreamResponse]" in m]) == 1

    @pytest.mark.asyncio
    async def test_mid_stream_failure_keeps_the_old_error_path(self, aws_frames, loguru_messages):
        """
        What it does: Reasoning frames, then the upstream connection breaks.
        Purpose: The existing api_error event + re-raise; exactly one error event; no
        [IncompleteUpstreamResponse] (not a normal end).
        """
        cache = MagicMock()
        cache.get_max_input_tokens.return_value = 200000
        response = _real_stream_response([aws_frames.reasoning("r")])
        original = response.aiter_bytes

        def broken(*args, **kwargs):
            inner = original()

            async def gen():
                async for chunk in inner:
                    yield chunk
                raise httpx.RemoteProtocolError("peer closed connection")
            return gen()

        response.aiter_bytes = broken
        out = []

        with _patch_handling("as_reasoning_content"):
            with pytest.raises(httpx.RemoteProtocolError):
                async for item in stream_kiro_to_anthropic(
                    response, "claude-sonnet-4-5", cache, MagicMock(), first_token_timeout=5
                ):
                    out.append(item)

        assert sum(1 for c in out if c.startswith("event: error")) == 1
        assert "api_error" in out[-1]
        assert not [m for m in loguru_messages if "[IncompleteUpstreamResponse]" in m]

    @pytest.mark.asyncio
    async def test_consumer_close_sends_no_error_event(self, aws_frames):
        """
        What it does: The consumer closes the generator after the first chunks.
        Purpose: No error event for an abandoned stream; the upstream response is closed.
        """
        cache = MagicMock()
        cache.get_max_input_tokens.return_value = 200000
        response = _real_stream_response([aws_frames.reasoning("a"), aws_frames.reasoning("b")])
        out = []

        with _patch_handling("as_reasoning_content"):
            gen = stream_kiro_to_anthropic(response, "m", cache, MagicMock(), first_token_timeout=5)
            out.append(await gen.__anext__())
            await gen.aclose()

        assert not any(c.startswith("event: error") for c in out)
        response.aclose.assert_awaited()

    @pytest.mark.asyncio
    async def test_response_is_closed_after_error_event(self, aws_frames):
        """
        What it does: Runs an incomplete stream to its end.
        Purpose: The upstream response is closed (no leaked connection).
        """
        cache = MagicMock()
        cache.get_max_input_tokens.return_value = 200000
        response = _real_stream_response([_incomplete_bytes(aws_frames)])

        with _patch_handling("as_reasoning_content"):
            [c async for c in stream_kiro_to_anthropic(response, "m", cache, MagicMock(), first_token_timeout=5)]

        response.aclose.assert_awaited()

    @pytest.mark.asyncio
    async def test_warning_logged_once_without_content(self, aws_frames, loguru_messages):
        """
        What it does: Incomplete stream, reads the log.
        Purpose: One [IncompleteUpstreamResponse] WARNING with action=error_event; no content.
        """
        await _anthropic_stream([_incomplete_bytes(aws_frames)])

        lines = [m for m in loguru_messages if "[IncompleteUpstreamResponse]" in m]
        assert len(lines) == 1
        assert lines[0].startswith("WARNING|") and "action=error_event" in lines[0]
        assert INCOMPLETE_THOUGHT not in lines[0]
        assert not [m for m in loguru_messages if m.startswith("ERROR|")]


class TestStreamingAnthropicIncompleteViaRetryWrapper:
    """stream_with_first_token_retry_anthropic(): marker, single message_start, no retry."""

    @pytest.mark.asyncio
    async def test_marker_then_error_without_retry(self, aws_frames, anthropic_sse):
        """
        What it does: Incomplete upstream response through the retry wrapper with the marker.
        Purpose: STREAM_READY first, one message_start, error ending; make_request never called.
        """
        response = _real_stream_response([_incomplete_bytes(aws_frames)])
        make_request = AsyncMock()
        cache = MagicMock()
        cache.get_max_input_tokens.return_value = 200000

        with _patch_handling("as_reasoning_content"), patch("kiro.streaming_core.EMPTY_RESPONSE_AS_ERROR", True):
            items = [
                item async for item in stream_with_first_token_retry_anthropic(
                    make_request=make_request, model="m", model_cache=cache, auth_manager=MagicMock(),
                    initial_response=response, first_token_timeout=5, emit_ready_marker=True,
                )
            ]

        assert items[0] is STREAM_READY
        events = anthropic_sse.parse(items[1:])
        assert anthropic_sse.assert_legal_aborted(events) == ["thinking"]
        make_request.assert_not_awaited()


class TestCollectAnthropicResponseIncomplete:
    """collect_anthropic_response(): non-streaming incomplete responses raise."""

    @staticmethod
    async def _collect(chunks):
        cache = MagicMock()
        cache.get_max_input_tokens.return_value = 200000
        with _patch_handling("as_reasoning_content"):
            return await collect_anthropic_response(
                _real_stream_response(chunks), "claude-sonnet-4-5", cache, MagicMock()
            )

    @pytest.mark.asyncio
    async def test_incomplete_raises_with_diagnostics(self, aws_frames):
        """
        What it does: Reasoning + contextUsageEvent, no metering.
        Purpose: EmptyUpstreamResponseError(kind=incomplete) with the stream facts.
        """
        data = _incomplete_bytes(aws_frames)

        with pytest.raises(EmptyUpstreamResponseError) as exc_info:
            await self._collect([data])

        err = exc_info.value
        assert err.kind == "incomplete"
        assert err.bytes_received == len(data)
        assert err.frame_counts == {"reasoningContentEvent": 6, "contextUsageEvent": 1}
        assert err.produced_text is False and err.produced_tool_use is False

    @pytest.mark.asyncio
    async def test_metering_returns_thinking_only_message(self, aws_frames):
        """
        What it does: Reasoning + metering, no body.
        Purpose: Complete: message with only a thinking block, end_turn.
        """
        result = await self._collect([_incomplete_bytes(aws_frames) + aws_frames.metering(0.1)])

        assert [b["type"] for b in result["content"]] == ["thinking"]
        assert result["stop_reason"] == "end_turn"

    @pytest.mark.asyncio
    async def test_reasoning_and_text_without_metering_is_returned(self, aws_frames):
        """
        What it does: Reasoning + text, no metering.
        Purpose: Returned as before (WARNING only).
        """
        data = aws_frames.reasoning("r") + aws_frames.content("answer")

        result = await self._collect([data])

        assert [b["type"] for b in result["content"]] == ["thinking", "text"]

    @pytest.mark.asyncio
    async def test_switch_off_returns_thinking_only_message(self, aws_frames):
        """
        What it does: INCOMPLETE_RESPONSE_AS_ERROR=False.
        Purpose: Old behavior: thinking-only message with end_turn.
        """
        with patch("kiro.streaming_core.INCOMPLETE_RESPONSE_AS_ERROR", False):
            result = await self._collect([_incomplete_bytes(aws_frames)])

        assert [b["type"] for b in result["content"]] == ["thinking"]
        assert result["stop_reason"] == "end_turn"
