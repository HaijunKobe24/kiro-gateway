
# -*- coding: utf-8 -*-

"""
Unit tests for streaming_openai module.

Tests for:
- stream_kiro_to_openai() generator
- stream_kiro_to_openai_internal() generator
- stream_with_first_token_retry() function
- collect_stream_response() function
"""

import pytest
import json
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from kiro.streaming_openai import (
    stream_kiro_to_openai,
    stream_kiro_to_openai_internal,
    stream_with_first_token_retry,
    collect_stream_response,
    FirstTokenTimeoutError,
)
from kiro.streaming_core import KiroEvent


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
def mock_http_client():
    """Mock for httpx.AsyncClient."""
    client = AsyncMock()
    return client


@pytest.fixture
def mock_response():
    """Mock for httpx.Response."""
    response = AsyncMock()
    response.status_code = 200
    response.aclose = AsyncMock()
    return response


# ==================================================================================================
# Tests for stream_kiro_to_openai()
# ==================================================================================================

class TestStreamKiroToOpenai:
    """Tests for stream_kiro_to_openai() generator."""
    
    @pytest.mark.asyncio
    async def test_yields_content_chunks(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Yields content chunks in OpenAI format.
        Goal: Verify content streaming.
        """
        print("Setup: Mock stream with content events...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Hello")
            yield KiroEvent(type="content", content=" World")
        
        print("Action: Streaming to OpenAI format...")
        chunks = []
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                async for chunk in stream_kiro_to_openai(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager
                ):
                    chunks.append(chunk)
        
        print(f"Received {len(chunks)} chunks")
        
        # Should have content chunks
        content_chunks = [c for c in chunks if "content" in c and '"Hello"' in c or '" World"' in c]
        assert len(content_chunks) >= 2
        print("✓ Content chunks yielded correctly")
    
    @pytest.mark.asyncio
    async def test_first_chunk_has_role(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: First chunk includes role: assistant.
        Goal: Verify OpenAI streaming protocol.
        """
        print("Setup: Mock stream with content...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Hello")
        
        print("Action: Streaming to OpenAI format...")
        chunks = []
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                async for chunk in stream_kiro_to_openai(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager
                ):
                    chunks.append(chunk)
        
        print(f"Received {len(chunks)} chunks")
        
        # First content chunk should have role
        first_content_chunk = [c for c in chunks if '"content"' in c and '"Hello"' in c][0]
        assert '"role": "assistant"' in first_content_chunk
        print("✓ First chunk has role: assistant")
    
    @pytest.mark.asyncio
    async def test_yields_done_at_end(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Yields [DONE] at end of stream.
        Goal: Verify stream termination.
        """
        print("Setup: Mock stream with content...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Hello")
        
        print("Action: Streaming to OpenAI format...")
        chunks = []
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                async for chunk in stream_kiro_to_openai(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager
                ):
                    chunks.append(chunk)
        
        print(f"Received {len(chunks)} chunks")
        
        # Last chunk should be [DONE]
        assert chunks[-1] == "data: [DONE]\n\n"
        print("✓ [DONE] yielded at end")
    
    @pytest.mark.asyncio
    async def test_yields_final_chunk_with_usage(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Yields final chunk with usage info.
        Goal: Verify usage is included.
        """
        print("Setup: Mock stream with content...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Hello")
            yield KiroEvent(type="context_usage", context_usage_percentage=5.0)
        
        print("Action: Streaming to OpenAI format...")
        chunks = []
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                async for chunk in stream_kiro_to_openai(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager
                ):
                    chunks.append(chunk)
        
        print(f"Received {len(chunks)} chunks")
        
        # Should have chunk with usage before [DONE]
        usage_chunks = [c for c in chunks if '"usage"' in c]
        assert len(usage_chunks) >= 1
        print("✓ Final chunk with usage yielded")
    
    @pytest.mark.asyncio
    async def test_yields_tool_calls_chunk(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Yields tool_calls chunk when tools present.
        Goal: Verify tool call streaming.
        """
        print("Setup: Mock stream with tool call...")
        
        tool_use_data = {
            "id": "call_123",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "Moscow"}'}
        }
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Let me check")
            yield KiroEvent(type="tool_use", tool_use=tool_use_data)
        
        print("Action: Streaming to OpenAI format...")
        chunks = []
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                async for chunk in stream_kiro_to_openai(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager
                ):
                    chunks.append(chunk)
        
        print(f"Received {len(chunks)} chunks")
        
        # Should have tool_calls chunk
        tool_chunks = [c for c in chunks if '"tool_calls"' in c]
        assert len(tool_chunks) >= 1
        assert "get_weather" in tool_chunks[0]
        print("✓ Tool calls chunk yielded")
    
    @pytest.mark.asyncio
    async def test_tool_calls_have_index(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Tool calls have index field.
        Goal: Verify OpenAI streaming spec compliance.
        """
        print("Setup: Mock stream with multiple tool calls...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="tool_use", tool_use={
                "id": "call_1", "type": "function",
                "function": {"name": "func1", "arguments": "{}"}
            })
            yield KiroEvent(type="tool_use", tool_use={
                "id": "call_2", "type": "function",
                "function": {"name": "func2", "arguments": "{}"}
            })
        
        print("Action: Streaming to OpenAI format...")
        chunks = []
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                async for chunk in stream_kiro_to_openai(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager
                ):
                    chunks.append(chunk)
        
        print(f"Received {len(chunks)} chunks")
        
        # Find tool_calls chunk and verify indices
        tool_chunks = [c for c in chunks if '"tool_calls"' in c]
        assert len(tool_chunks) >= 1
        
        # Parse and check indices
        for chunk in tool_chunks:
            if chunk.startswith("data: "):
                json_str = chunk[6:].strip()
                if json_str != "[DONE]":
                    data = json.loads(json_str)
                    if "choices" in data and data["choices"]:
                        delta = data["choices"][0].get("delta", {})
                        if "tool_calls" in delta:
                            for tc in delta["tool_calls"]:
                                assert "index" in tc
        
        print("✓ Tool calls have index field")
    
    @pytest.mark.asyncio
    async def test_finish_reason_is_tool_calls_when_tools_present(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Sets finish_reason to tool_calls when tools present.
        Goal: Verify correct finish reason.
        """
        print("Setup: Mock stream with tool call...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="tool_use", tool_use={
                "id": "call_1", "type": "function",
                "function": {"name": "func1", "arguments": "{}"}
            })
        
        print("Action: Streaming to OpenAI format...")
        chunks = []
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                async for chunk in stream_kiro_to_openai(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager
                ):
                    chunks.append(chunk)
        
        print(f"Received {len(chunks)} chunks")
        
        # Final chunk before [DONE] should have finish_reason: tool_calls
        final_chunk = chunks[-2]  # Before [DONE]
        assert '"finish_reason": "tool_calls"' in final_chunk
        print("✓ finish_reason is tool_calls")
    
    @pytest.mark.asyncio
    async def test_finish_reason_is_stop_without_tools(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Sets finish_reason to stop without tools.
        Goal: Verify correct finish reason.
        """
        print("Setup: Mock stream without tool calls...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Hello")
            yield KiroEvent(type="usage", usage={"inputTokenCount": 10, "outputTokenCount": 1})
        
        print("Action: Streaming to OpenAI format...")
        chunks = []
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                async for chunk in stream_kiro_to_openai(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager
                ):
                    chunks.append(chunk)
        
        print(f"Received {len(chunks)} chunks")
        
        # Final chunk before [DONE] should have finish_reason: stop
        final_chunk = chunks[-2]  # Before [DONE]
        assert '"finish_reason": "stop"' in final_chunk
        print("✓ finish_reason is stop")
    
    @pytest.mark.asyncio
    async def test_closes_response_on_completion(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Closes response on completion.
        Goal: Verify resource cleanup.
        """
        print("Setup: Mock stream...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Hello")
        
        print("Action: Streaming to OpenAI format...")
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                async for chunk in stream_kiro_to_openai(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager
                ):
                    pass
        
        print("Check: response.aclose() should be called...")
        mock_response.aclose.assert_called()
        print("✓ Response closed on completion")
    
    @pytest.mark.asyncio
    async def test_closes_response_on_error(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Closes response on error.
        Goal: Verify resource cleanup on error.
        """
        print("Setup: Mock stream that raises error...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Hello")
            raise RuntimeError("Test error")
        
        print("Action: Streaming to OpenAI format with error...")
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                try:
                    async for chunk in stream_kiro_to_openai(
                        mock_http_client, mock_response, "claude-sonnet-4",
                        mock_model_cache, mock_auth_manager
                    ):
                        pass
                except RuntimeError:
                    pass
        
        print("Check: response.aclose() should be called...")
        mock_response.aclose.assert_called()
        print("✓ Response closed on error")


# ==================================================================================================
# Tests for thinking content handling
# ==================================================================================================

class TestStreamingOpenaiThinkingContent:
    """Tests for thinking content handling in OpenAI streaming."""
    
    @pytest.mark.asyncio
    async def test_yields_thinking_as_reasoning_content(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Yields thinking as reasoning_content when configured.
        Goal: Verify thinking content handling.
        """
        print("Setup: Mock stream with thinking content...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="thinking", thinking_content="Let me think...")
            yield KiroEvent(type="content", content="Here is my answer")
        
        print("Action: Streaming to OpenAI format with reasoning mode...")
        chunks = []
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                with patch('kiro.streaming_openai.FAKE_REASONING_HANDLING', 'as_reasoning_content'):
                    async for chunk in stream_kiro_to_openai(
                        mock_http_client, mock_response, "claude-sonnet-4",
                        mock_model_cache, mock_auth_manager
                    ):
                        chunks.append(chunk)
        
        print(f"Received {len(chunks)} chunks")
        
        # Should have reasoning_content
        reasoning_chunks = [c for c in chunks if '"reasoning_content"' in c]
        assert len(reasoning_chunks) >= 1
        assert "Let me think" in reasoning_chunks[0]
        print("✓ Thinking yielded as reasoning_content")
    
    @pytest.mark.asyncio
    async def test_yields_thinking_as_content_when_configured(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Yields thinking as content when configured.
        Goal: Verify thinking content handling.
        """
        print("Setup: Mock stream with thinking content...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="thinking", thinking_content="Let me think...")
            yield KiroEvent(type="content", content="Here is my answer")
        
        print("Action: Streaming to OpenAI format with content mode...")
        chunks = []
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                with patch('kiro.streaming_openai.FAKE_REASONING_HANDLING', 'include_as_text'):
                    async for chunk in stream_kiro_to_openai(
                        mock_http_client, mock_response, "claude-sonnet-4",
                        mock_model_cache, mock_auth_manager
                    ):
                        chunks.append(chunk)
        
        print(f"Received {len(chunks)} chunks")
        
        # Should have thinking as content
        content_chunks = [c for c in chunks if '"content"' in c and "Let me think" in c]
        assert len(content_chunks) >= 1
        print("✓ Thinking yielded as content")


# ==================================================================================================
# Tests for None protection in tool calls
# ==================================================================================================

class TestStreamingOpenaiNoneProtection:
    """Tests for None protection in tool calls."""
    
    @pytest.mark.asyncio
    async def test_handles_none_function_name(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Handles None in function.name.
        Goal: Verify None is replaced with empty string.
        """
        print("Setup: Mock stream with None function name...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="tool_use", tool_use={
                "id": "call_1", "type": "function",
                "function": {"name": None, "arguments": "{}"}
            })
        
        print("Action: Streaming to OpenAI format...")
        chunks = []
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                async for chunk in stream_kiro_to_openai(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager
                ):
                    chunks.append(chunk)
        
        print(f"Received {len(chunks)} chunks")
        
        # Should handle None gracefully
        tool_chunks = [c for c in chunks if '"tool_calls"' in c]
        assert len(tool_chunks) >= 1
        
        # Parse and verify name is empty string
        for chunk in tool_chunks:
            if chunk.startswith("data: "):
                json_str = chunk[6:].strip()
                if json_str != "[DONE]":
                    data = json.loads(json_str)
                    if "choices" in data and data["choices"]:
                        delta = data["choices"][0].get("delta", {})
                        if "tool_calls" in delta:
                            for tc in delta["tool_calls"]:
                                assert tc["function"]["name"] == ""
        
        print("✓ None function name handled")
    
    @pytest.mark.asyncio
    async def test_handles_none_function_arguments(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Handles None in function.arguments.
        Goal: Verify None is replaced with "{}".
        """
        print("Setup: Mock stream with None arguments...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="tool_use", tool_use={
                "id": "call_1", "type": "function",
                "function": {"name": "func1", "arguments": None}
            })
        
        print("Action: Streaming to OpenAI format...")
        chunks = []
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                async for chunk in stream_kiro_to_openai(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager
                ):
                    chunks.append(chunk)
        
        print(f"Received {len(chunks)} chunks")
        
        # Should handle None gracefully
        tool_chunks = [c for c in chunks if '"tool_calls"' in c]
        assert len(tool_chunks) >= 1
        
        # Parse and verify arguments is "{}"
        for chunk in tool_chunks:
            if chunk.startswith("data: "):
                json_str = chunk[6:].strip()
                if json_str != "[DONE]":
                    data = json.loads(json_str)
                    if "choices" in data and data["choices"]:
                        delta = data["choices"][0].get("delta", {})
                        if "tool_calls" in delta:
                            for tc in delta["tool_calls"]:
                                assert tc["function"]["arguments"] == "{}"
        
        print("✓ None function arguments handled")
    
    @pytest.mark.asyncio
    async def test_handles_none_function_object(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Handles None function object.
        Goal: Verify None function is handled.
        """
        print("Setup: Mock stream with None function...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="tool_use", tool_use={
                "id": "call_1", "type": "function",
                "function": None
            })
        
        print("Action: Streaming to OpenAI format...")
        chunks = []
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                async for chunk in stream_kiro_to_openai(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager
                ):
                    chunks.append(chunk)
        
        print(f"Received {len(chunks)} chunks")
        
        # Should handle None gracefully without error
        assert len(chunks) > 0
        print("✓ None function object handled")


# ==================================================================================================
# Tests for stream_with_first_token_retry()
# ==================================================================================================

@pytest.mark.usefixtures("no_empty_response_guard")
class TestStreamWithFirstTokenRetry:
    """Tests for stream_with_first_token_retry() function."""
    
    @pytest.mark.asyncio
    async def test_retries_on_first_token_timeout(self, mock_http_client, mock_model_cache, mock_auth_manager):
        """
        What it does: Retries on first token timeout.
        Goal: Verify retry logic.
        """
        print("Setup: Mock make_request that succeeds on second attempt...")
        
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.aclose = AsyncMock()
        
        call_count = 0
        
        async def mock_make_request():
            nonlocal call_count
            call_count += 1
            print(f"make_request called (attempt {call_count})")
            return mock_response
        
        # First call raises timeout, second succeeds
        timeout_raised = False
        
        async def mock_parse_kiro_stream_with_retry(*args, **kwargs):
            nonlocal timeout_raised
            if not timeout_raised:
                timeout_raised = True
                raise FirstTokenTimeoutError("Timeout!")
            yield KiroEvent(type="content", content="Success")
        
        print("Action: Running stream_with_first_token_retry...")
        chunks = []
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream_with_retry):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                async for chunk in stream_with_first_token_retry(
                    mock_make_request,
                    mock_http_client,
                    "claude-sonnet-4",
                    mock_model_cache,
                    mock_auth_manager,
                    max_retries=3,
                    first_token_timeout=15
                ):
                    chunks.append(chunk)
        
        print(f"Received {len(chunks)} chunks")
        print(f"make_request was called {call_count} times")
        
        assert call_count == 2
        assert len(chunks) > 0
        print("✓ Retry logic worked correctly")
    
    @pytest.mark.asyncio
    async def test_raises_504_after_all_retries_exhausted(self, mock_http_client, mock_model_cache, mock_auth_manager):
        """
        What it does: Raises 504 after all retries exhausted.
        Goal: Verify error handling.
        """
        from fastapi import HTTPException
        
        print("Setup: Mock make_request that always times out...")
        
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.aclose = AsyncMock()
        
        call_count = 0
        
        async def mock_make_request():
            nonlocal call_count
            call_count += 1
            return mock_response
        
        async def mock_parse_kiro_stream_always_timeout(*args, **kwargs):
            raise FirstTokenTimeoutError("Timeout!")
            yield  # Make it a generator
        
        max_retries = 3
        
        print(f"Action: Running stream_with_first_token_retry with max_retries={max_retries}...")
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream_always_timeout):
            with pytest.raises(HTTPException) as exc_info:
                async for chunk in stream_with_first_token_retry(
                    mock_make_request,
                    mock_http_client,
                    "claude-sonnet-4",
                    mock_model_cache,
                    mock_auth_manager,
                    max_retries=max_retries,
                    first_token_timeout=15
                ):
                    pass
        
        print(f"Caught HTTPException: {exc_info.value.status_code}")
        print(f"make_request was called {call_count} times")
        
        assert exc_info.value.status_code == 504
        assert call_count == max_retries
        print("✓ 504 raised after all retries exhausted")
    
    @pytest.mark.asyncio
    async def test_handles_api_error_response(self, mock_http_client, mock_model_cache, mock_auth_manager):
        """
        What it does: Handles API error response.
        Goal: Verify error response handling.
        """
        from fastapi import HTTPException
        
        print("Setup: Mock make_request that returns error...")
        
        mock_response = AsyncMock()
        mock_response.status_code = 500
        # Use simple error text without curly braces to avoid loguru format issues
        mock_response.aread = AsyncMock(return_value=b'Internal server error')
        mock_response.aclose = AsyncMock()
        
        async def mock_make_request():
            return mock_response
        
        print("Action: Running stream_with_first_token_retry with error response...")
        
        with pytest.raises(HTTPException) as exc_info:
            async for chunk in stream_with_first_token_retry(
                mock_make_request,
                mock_http_client,
                "claude-sonnet-4",
                mock_model_cache,
                mock_auth_manager,
                max_retries=3,
                first_token_timeout=15
            ):
                pass
        
        print(f"Caught HTTPException: {exc_info.value.status_code}")
        assert exc_info.value.status_code == 500
        print("✓ API error response handled")
    
    @pytest.mark.asyncio
    async def test_propagates_non_timeout_errors(self, mock_http_client, mock_model_cache, mock_auth_manager):
        """
        What it does: Propagates non-timeout errors without retry.
        Goal: Verify only timeout errors trigger retry.
        """
        print("Setup: Mock make_request that raises RuntimeError...")
        
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.aclose = AsyncMock()
        
        call_count = 0
        
        async def mock_make_request():
            nonlocal call_count
            call_count += 1
            return mock_response
        
        async def mock_parse_kiro_stream_error(*args, **kwargs):
            raise RuntimeError("Test error")
            yield  # Make it a generator
        
        print("Action: Running stream_with_first_token_retry with RuntimeError...")
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream_error):
            with pytest.raises(RuntimeError) as exc_info:
                async for chunk in stream_with_first_token_retry(
                    mock_make_request,
                    mock_http_client,
                    "claude-sonnet-4",
                    mock_model_cache,
                    mock_auth_manager,
                    max_retries=3,
                    first_token_timeout=15
                ):
                    pass
        
        print(f"Caught RuntimeError: {exc_info.value}")
        print(f"make_request was called {call_count} times")
        
        # Should only be called once - no retry for non-timeout errors
        assert call_count == 1
        assert "Test error" in str(exc_info.value)
        print("✓ Non-timeout errors propagated without retry")
    
    @pytest.mark.asyncio
    async def test_closes_response_on_retry(self, mock_http_client, mock_model_cache, mock_auth_manager):
        """
        What it does: Closes response when retrying.
        Goal: Verify resource cleanup on retry.
        """
        print("Setup: Mock responses for retry...")
        
        mock_response1 = AsyncMock()
        mock_response1.status_code = 200
        mock_response1.aclose = AsyncMock()
        
        mock_response2 = AsyncMock()
        mock_response2.status_code = 200
        mock_response2.aclose = AsyncMock()
        
        responses = [mock_response1, mock_response2]
        call_count = 0
        
        async def mock_make_request():
            nonlocal call_count
            response = responses[call_count]
            call_count += 1
            return response
        
        # First call raises timeout, second succeeds
        timeout_raised = False
        
        async def mock_parse_kiro_stream_with_retry(*args, **kwargs):
            nonlocal timeout_raised
            if not timeout_raised:
                timeout_raised = True
                raise FirstTokenTimeoutError("Timeout!")
            yield KiroEvent(type="content", content="Success")
        
        print("Action: Running stream_with_first_token_retry...")
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream_with_retry):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                async for chunk in stream_with_first_token_retry(
                    mock_make_request,
                    mock_http_client,
                    "claude-sonnet-4",
                    mock_model_cache,
                    mock_auth_manager,
                    max_retries=3,
                    first_token_timeout=15
                ):
                    pass
        
        print("Check: First response should be closed...")
        mock_response1.aclose.assert_called()
        print("✓ Response closed on retry")


# ==================================================================================================
# Tests for collect_stream_response()
# ==================================================================================================

@pytest.mark.usefixtures("no_empty_response_guard")
class TestCollectStreamResponse:
    """Tests for collect_stream_response() function."""
    
    @pytest.mark.asyncio
    async def test_collects_content(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Collects content from stream.
        Goal: Verify content accumulation.
        """
        print("Setup: Mock stream with content...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Hello")
            yield KiroEvent(type="content", content=" World")
        
        print("Action: Collecting stream response...")
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                result = await collect_stream_response(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager
                )
        
        print(f"Result: {result}")
        
        assert result["choices"][0]["message"]["content"] == "Hello World"
        print("✓ Content collected correctly")
    
    @pytest.mark.asyncio
    async def test_collects_reasoning_content(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Collects reasoning content from stream.
        Goal: Verify reasoning content accumulation.
        """
        print("Setup: Mock stream with thinking content...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="thinking", thinking_content="Let me think...")
            yield KiroEvent(type="content", content="Answer")
        
        print("Action: Collecting stream response with reasoning mode...")
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                with patch('kiro.streaming_openai.FAKE_REASONING_HANDLING', 'as_reasoning_content'):
                    result = await collect_stream_response(
                        mock_http_client, mock_response, "claude-sonnet-4",
                        mock_model_cache, mock_auth_manager
                    )
        
        print(f"Result: {result}")
        
        message = result["choices"][0]["message"]
        assert "reasoning_content" in message
        assert message["reasoning_content"] == "Let me think..."
        print("✓ Reasoning content collected correctly")
    
    @pytest.mark.asyncio
    async def test_collects_tool_calls(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Collects tool calls from stream.
        Goal: Verify tool call accumulation.
        """
        print("Setup: Mock stream with tool calls...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="tool_use", tool_use={
                "id": "call_1", "type": "function",
                "function": {"name": "func1", "arguments": '{"a": 1}'}
            })
        
        print("Action: Collecting stream response...")
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                result = await collect_stream_response(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager
                )
        
        print(f"Result: {result}")
        
        message = result["choices"][0]["message"]
        assert "tool_calls" in message
        assert len(message["tool_calls"]) == 1
        assert message["tool_calls"][0]["function"]["name"] == "func1"
        print("✓ Tool calls collected correctly")
    
    @pytest.mark.asyncio
    async def test_tool_calls_have_no_index(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Collected tool calls don't have index field.
        Goal: Verify index is removed for non-streaming.
        """
        print("Setup: Mock stream with tool calls...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="tool_use", tool_use={
                "id": "call_1", "type": "function",
                "function": {"name": "func1", "arguments": "{}"}
            })
        
        print("Action: Collecting stream response...")
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                result = await collect_stream_response(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager
                )
        
        print(f"Result: {result}")
        
        message = result["choices"][0]["message"]
        for tc in message.get("tool_calls", []):
            assert "index" not in tc
        
        print("✓ Tool calls have no index field")
    
    @pytest.mark.asyncio
    async def test_includes_usage(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Includes usage in response.
        Goal: Verify usage is included.
        """
        print("Setup: Mock stream with content...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Hello")
            yield KiroEvent(type="context_usage", context_usage_percentage=5.0)
        
        print("Action: Collecting stream response...")
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                result = await collect_stream_response(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager
                )
        
        print(f"Result: {result}")
        
        assert "usage" in result
        assert "prompt_tokens" in result["usage"]
        assert "completion_tokens" in result["usage"]
        assert "total_tokens" in result["usage"]
        print("✓ Usage included in response")
    
    @pytest.mark.asyncio
    async def test_sets_finish_reason_tool_calls(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Sets finish_reason to tool_calls when tools present.
        Goal: Verify correct finish reason.
        """
        print("Setup: Mock stream with tool calls...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="tool_use", tool_use={
                "id": "call_1", "type": "function",
                "function": {"name": "func1", "arguments": "{}"}
            })
        
        print("Action: Collecting stream response...")
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                result = await collect_stream_response(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager
                )
        
        print(f"Result: {result}")
        
        assert result["choices"][0]["finish_reason"] == "tool_calls"
        print("✓ finish_reason is tool_calls")
    
    @pytest.mark.asyncio
    async def test_sets_finish_reason_stop(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Sets finish_reason to stop without tools.
        Goal: Verify correct finish reason.
        """
        print("Setup: Mock stream without tool calls...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Hello")
            yield KiroEvent(type="usage", usage={"inputTokenCount": 10, "outputTokenCount": 1})
        
        print("Action: Collecting stream response...")
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                result = await collect_stream_response(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager
                )
        
        print(f"Result: {result}")
        
        assert result["choices"][0]["finish_reason"] == "stop"
        print("✓ finish_reason is stop")
    
    @pytest.mark.asyncio
    async def test_generates_completion_id(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Generates completion ID.
        Goal: Verify ID is present.
        """
        print("Setup: Mock stream...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Hello")
        
        print("Action: Collecting stream response...")
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                result = await collect_stream_response(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager
                )
        
        print(f"ID: {result['id']}")
        
        assert result["id"].startswith("chatcmpl-")
        print("✓ Completion ID generated")
    
    @pytest.mark.asyncio
    async def test_includes_model_name(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Includes model name in response.
        Goal: Verify model is included.
        """
        print("Setup: Mock stream...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Hello")
        
        print("Action: Collecting stream response...")
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                result = await collect_stream_response(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager
                )
        
        print(f"Model: {result['model']}")
        
        assert result["model"] == "claude-sonnet-4"
        print("✓ Model name included")
    
    @pytest.mark.asyncio
    async def test_object_is_chat_completion(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Sets object to chat.completion.
        Goal: Verify OpenAI format.
        """
        print("Setup: Mock stream...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Hello")
        
        print("Action: Collecting stream response...")
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                result = await collect_stream_response(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager
                )
        
        print(f"Object: {result['object']}")
        
        assert result["object"] == "chat.completion"
        print("✓ Object is chat.completion")


# ==================================================================================================
# Tests for error handling
# ==================================================================================================

class TestStreamingOpenaiErrorHandling:
    """Tests for error handling in streaming_openai."""
    
    @pytest.mark.asyncio
    async def test_propagates_first_token_timeout_error(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Propagates FirstTokenTimeoutError.
        Goal: Verify timeout error is propagated for retry.
        """
        print("Setup: Mock stream that raises timeout...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            raise FirstTokenTimeoutError("Timeout!")
            yield  # Make it a generator
        
        print("Action: Streaming to OpenAI format with timeout...")
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with pytest.raises(FirstTokenTimeoutError):
                async for chunk in stream_kiro_to_openai(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager
                ):
                    pass
        
        print("✓ FirstTokenTimeoutError propagated correctly")
    
    @pytest.mark.asyncio
    async def test_handles_generator_exit_gracefully(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Handles GeneratorExit gracefully without re-raising.
        Goal: Verify client disconnect is handled without error.
        """
        print("Setup: Mock stream that raises GeneratorExit...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Hello")
            raise GeneratorExit()
        
        print("Action: Streaming to OpenAI format with GeneratorExit...")
        chunks = []
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                # GeneratorExit is caught internally and not re-raised
                # This is correct behavior - client disconnect should be handled gracefully
                async for chunk in stream_kiro_to_openai(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager
                ):
                    chunks.append(chunk)
        
        print(f"Received {len(chunks)} chunks before disconnect")
        # Response should be closed
        mock_response.aclose.assert_called()
        print("✓ GeneratorExit handled gracefully")
    
    @pytest.mark.asyncio
    async def test_propagates_other_exceptions(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Propagates other exceptions.
        Goal: Verify errors are not swallowed.
        """
        print("Setup: Mock stream that raises RuntimeError...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Hello")
            raise RuntimeError("Test error")
        
        print("Action: Streaming to OpenAI format with RuntimeError...")
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                with pytest.raises(RuntimeError) as exc_info:
                    async for chunk in stream_kiro_to_openai(
                        mock_http_client, mock_response, "claude-sonnet-4",
                        mock_model_cache, mock_auth_manager
                    ):
                        pass
        
        print(f"Caught exception: {exc_info.value}")
        assert "Test error" in str(exc_info.value)
        print("✓ RuntimeError propagated correctly")
    
    @pytest.mark.asyncio
    async def test_aclose_error_does_not_mask_original(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: aclose() error doesn't mask original error.
        Goal: Verify original exception is propagated.
        """
        print("Setup: Mock response with error in aclose()...")
        
        mock_response.aclose = AsyncMock(side_effect=ConnectionError("Connection lost"))
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Hello")
            raise RuntimeError("Original error")
        
        print("Action: Streaming to OpenAI format with error and aclose error...")
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                with pytest.raises(RuntimeError) as exc_info:
                    async for chunk in stream_kiro_to_openai(
                        mock_http_client, mock_response, "claude-sonnet-4",
                        mock_model_cache, mock_auth_manager
                    ):
                        pass
        
        print(f"Caught exception: {exc_info.value}")
        assert "Original error" in str(exc_info.value)
        print("✓ Original error not masked by aclose error")


# ==================================================================================================
# Tests for bracket tool calls
# ==================================================================================================

class TestStreamingOpenaiBracketToolCalls:
    """Tests for bracket-style tool call handling."""
    
    @pytest.mark.asyncio
    async def test_detects_bracket_tool_calls(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Detects bracket-style tool calls in content.
        Goal: Verify bracket tool call detection.
        """
        print("Setup: Mock stream with bracket tool calls...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="[tool_call: func1]")
        
        bracket_tool_calls = [
            {"id": "call_1", "type": "function", "function": {"name": "func1", "arguments": "{}"}}
        ]
        
        print("Action: Streaming to OpenAI format...")
        chunks = []
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=bracket_tool_calls):
                async for chunk in stream_kiro_to_openai(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager
                ):
                    chunks.append(chunk)
        
        print(f"Received {len(chunks)} chunks")
        
        # Should have tool_calls chunk
        tool_chunks = [c for c in chunks if '"tool_calls"' in c]
        assert len(tool_chunks) >= 1
        print("✓ Bracket tool calls detected")
    
    @pytest.mark.asyncio
    async def test_deduplicates_tool_calls(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Deduplicates tool calls from stream and bracket.
        Goal: Verify deduplication.
        """
        print("Setup: Mock stream with duplicate tool calls...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="text")
            yield KiroEvent(type="tool_use", tool_use={
                "id": "call_1", "type": "function",
                "function": {"name": "func1", "arguments": "{}"}
            })
        
        # Same tool call from bracket detection
        bracket_tool_calls = [
            {"id": "call_1", "type": "function", "function": {"name": "func1", "arguments": "{}"}}
        ]
        
        print("Action: Streaming to OpenAI format...")
        chunks = []
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=bracket_tool_calls):
                with patch('kiro.streaming_openai.deduplicate_tool_calls') as mock_dedup:
                    mock_dedup.return_value = [
                        {"id": "call_1", "type": "function", "function": {"name": "func1", "arguments": "{}"}}
                    ]
                    async for chunk in stream_kiro_to_openai(
                        mock_http_client, mock_response, "claude-sonnet-4",
                        mock_model_cache, mock_auth_manager
                    ):
                        chunks.append(chunk)
                    
                    # Verify deduplicate was called
                    mock_dedup.assert_called()
        
        print("✓ Tool calls deduplicated")


# ==================================================================================================
# Tests for metering data
# ==================================================================================================

class TestStreamingOpenaiMeteringData:
    """Tests for metering data handling."""
    
    @pytest.mark.asyncio
    async def test_includes_credits_used_in_usage(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Includes credits_used in usage when metering data present.
        Goal: Verify metering data is included.
        """
        print("Setup: Mock stream with metering data...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Hello")
            yield KiroEvent(type="usage", usage={"credits": 0.001})
        
        print("Action: Streaming to OpenAI format...")
        chunks = []
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                async for chunk in stream_kiro_to_openai(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager
                ):
                    chunks.append(chunk)
        
        print(f"Received {len(chunks)} chunks")
        
        # Final chunk should have credits_used
        final_chunk = chunks[-2]  # Before [DONE]
        assert '"credits_used"' in final_chunk
        print("✓ credits_used included in usage")


# ==================================================================================================
# Tests for truncation detection
# ==================================================================================================

@pytest.mark.usefixtures("no_empty_response_guard")
class TestStreamingOpenaiTruncationDetection:
    """Tests for truncation detection in OpenAI streaming."""
    
    @pytest.mark.asyncio
    async def test_finish_reason_is_length_when_truncated(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Sets finish_reason to length when content is truncated.
        Goal: Verify truncation detection without completion signals.
        """
        print("Setup: Mock stream without completion signals (truncated)...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="This response was cut off mid-sentence because")
            # No usage event = truncation
        
        print("Action: Streaming to OpenAI format...")
        chunks = []
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                async for chunk in stream_kiro_to_openai(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager
                ):
                    chunks.append(chunk)
        
        print(f"Received {len(chunks)} chunks")
        
        # Final chunk before [DONE] should have finish_reason: length
        final_chunk = chunks[-2]  # Before [DONE]
        print(f"Comparing finish_reason: Expected 'length', Got chunk: {final_chunk}")
        assert '"finish_reason": "length"' in final_chunk
        print("✓ finish_reason is length when truncated")
    
    @pytest.mark.asyncio
    async def test_finish_reason_is_tool_calls_even_without_completion_signals(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Sets finish_reason to tool_calls when tool calls present.
        Goal: Verify tool_calls take priority (not confused with content truncation).
        """
        print("Setup: Mock stream with tool calls but no completion signals...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Let me call a tool")
            yield KiroEvent(type="tool_use", tool_use={
                "id": "call_1", "type": "function",
                "function": {"name": "get_weather", "arguments": "{}"}
            })
            # No usage event, but tool calls present = tool_calls finish_reason
        
        print("Action: Streaming to OpenAI format...")
        chunks = []
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                async for chunk in stream_kiro_to_openai(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager
                ):
                    chunks.append(chunk)
        
        print(f"Received {len(chunks)} chunks")
        
        # Tool calls take priority (not confused with content truncation)
        final_chunk = chunks[-2]  # Before [DONE]
        print(f"Comparing finish_reason: Expected 'tool_calls', Got chunk: {final_chunk}")
        assert '"finish_reason": "tool_calls"' in final_chunk
        print("✓ finish_reason is tool_calls (not confused with content truncation)")
    
    @pytest.mark.asyncio
    async def test_finish_reason_is_stop_with_completion_signals(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Sets finish_reason to stop when completion signals present.
        Goal: Verify normal completion is detected correctly.
        """
        print("Setup: Mock stream with completion signals (not truncated)...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Complete response")
            yield KiroEvent(type="usage", usage={"inputTokenCount": 10, "outputTokenCount": 5})
        
        print("Action: Streaming to OpenAI format...")
        chunks = []
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                async for chunk in stream_kiro_to_openai(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager
                ):
                    chunks.append(chunk)
        
        print(f"Received {len(chunks)} chunks")
        
        # With completion signals, should be stop
        final_chunk = chunks[-2]  # Before [DONE]
        print(f"Comparing finish_reason: Expected 'stop', Got chunk: {final_chunk}")
        assert '"finish_reason": "stop"' in final_chunk
        print("✓ finish_reason is stop with completion signals")
    
    @pytest.mark.asyncio
    async def test_collect_extracts_finish_reason_from_chunks(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Non-streaming extracts finish_reason from streaming chunks.
        Goal: Verify collect_stream_response correctly extracts finish_reason.
        """
        print("Setup: Mock stream without completion signals...")
        
        async def mock_parse_kiro_stream(*args, **kwargs):
            yield KiroEvent(type="content", content="Truncated")
            # No usage = truncation
        
        print("Action: Collecting stream response...")
        
        with patch('kiro.streaming_openai.parse_kiro_stream', mock_parse_kiro_stream):
            with patch('kiro.streaming_openai.parse_bracket_tool_calls', return_value=[]):
                result = await collect_stream_response(
                    mock_http_client, mock_response, "claude-sonnet-4",
                    mock_model_cache, mock_auth_manager
                )
        
        print(f"Result finish_reason: {result['choices'][0]['finish_reason']}")
        
        # Should extract "length" from streaming chunks
        assert result["choices"][0]["finish_reason"] == "length"
        print("✓ collect_stream_response extracts finish_reason correctly")

# ==================================================================================================
# Native reasoning (reasoningContentEvent) tests
# ==================================================================================================

from contextlib import ExitStack

from kiro.tokenizer import count_tokens

NATIVE_SECRET = "SECRET_REASONING_MARKER_7f3a"


def _patch_handling(mode, enabled=True):
    """Pins FAKE_REASONING_* in BOTH modules that read them (core parsing + OpenAI formatting)."""
    stack = ExitStack()
    stack.enter_context(patch("kiro.streaming_core.FAKE_REASONING_HANDLING", mode))
    stack.enter_context(patch("kiro.streaming_openai.FAKE_REASONING_HANDLING", mode))
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
    return KiroEvent(type="thinking", thinking_content=text, is_native_reasoning=True)


def _content(text):
    return KiroEvent(type="content", content=text)


def _tool(name="Bash", tool_id="toolu_1", args='{"command": "ls"}'):
    return KiroEvent(type="tool_use", tool_use={
        "id": tool_id, "type": "function", "function": {"name": name, "arguments": args},
    })


def _done():
    """Completion signal so the stream is not treated as truncated."""
    return KiroEvent(type="context_usage", context_usage_percentage=1.0)


def _parse_chunks(raw_chunks):
    """Returns (list of JSON chunk dicts, saw_done) from OpenAI SSE strings."""
    parsed, saw_done = [], False
    for raw in raw_chunks:
        assert raw.startswith("data: ") and raw.endswith("\n\n"), repr(raw)
        body = raw[len("data: "):].strip()
        if body == "[DONE]":
            saw_done = True
            continue
        parsed.append(json.loads(body))
    return parsed, saw_done


def _deltas(parsed, key):
    """Collects delta[key] for every chunk that carries it."""
    return [c["choices"][0]["delta"][key] for c in parsed if key in c["choices"][0]["delta"]]


class TestStreamingOpenaiNativeReasoning:
    """OpenAI streaming for native reasoning (KiroEvent level)."""

    async def _run(self, events, mock_http_client, mock_response, mock_model_cache, mock_auth_manager, mode="as_reasoning_content"):
        out = []
        with _patch_handling(mode), \
                patch("kiro.streaming_openai.parse_kiro_stream", _events_source(events)):
            async for item in stream_kiro_to_openai(
                mock_http_client, mock_response, "claude-sonnet-4", mock_model_cache, mock_auth_manager
            ):
                out.append(item)
        return _parse_chunks(out)

    @pytest.mark.asyncio
    async def test_reasoning_then_content(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: reasoning x3 -> content x2.
        Goal: First chunk is a reasoning_content delta WITH role; then content deltas;
        finish_reason=stop; [DONE] last; completion_tokens counts reasoning.
        """
        reasoning = "think " * 100
        parsed, saw_done = await self._run(
            [_native(reasoning), _native("b"), _native("c"), _content("X"), _content("Y"), _done()],
            mock_http_client, mock_response, mock_model_cache, mock_auth_manager,
        )

        first = parsed[0]["choices"][0]["delta"]
        assert first == {"reasoning_content": reasoning, "role": "assistant"}
        assert _deltas(parsed, "reasoning_content") == [reasoning, "b", "c"]
        assert _deltas(parsed, "content") == ["X", "Y"]
        assert [c["choices"][0]["delta"].get("role") for c in parsed].count("assistant") == 1
        assert parsed[-1]["choices"][0]["finish_reason"] == "stop"
        assert parsed[-1]["usage"]["completion_tokens"] >= count_tokens(reasoning)
        assert saw_done
        # reasoning chunks come strictly before content chunks
        kinds = ["r" if "reasoning_content" in c["choices"][0]["delta"] else "c" for c in parsed[:-1]]
        assert kinds == ["r", "r", "r", "c", "c"]

    @pytest.mark.asyncio
    async def test_reasoning_then_tool_calls(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: reasoning -> tool_use.
        Goal: tool_calls chunk arrives after reasoning; finish_reason=tool_calls.
        """
        parsed, saw_done = await self._run(
            [_native("plan"), _tool(), _done()], mock_http_client, mock_response, mock_model_cache, mock_auth_manager,
        )
        assert _deltas(parsed, "reasoning_content") == ["plan"]
        tool_chunks = _deltas(parsed, "tool_calls")
        assert tool_chunks and tool_chunks[0][0]["function"]["name"] == "Bash"
        assert parsed[-1]["choices"][0]["finish_reason"] == "tool_calls"
        assert saw_done

    @pytest.mark.asyncio
    async def test_reasoning_only_then_pure_tool_use(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Many reasoning chunks, tool call only at the end, no content.
        Goal: Every reasoning chunk is emitted before the tool_calls chunk; no content chunk.
        """
        parsed, _ = await self._run(
            [_native(str(i)) for i in range(15)] + [_tool(), _done()],
            mock_http_client, mock_response, mock_model_cache, mock_auth_manager,
        )
        kinds = []
        for c in parsed[:-1]:
            delta = c["choices"][0]["delta"]
            kinds.append("r" if "reasoning_content" in delta else "t" if "tool_calls" in delta else "?")
        assert kinds == ["r"] * 15 + ["t"]
        assert not _deltas(parsed, "content")

    @pytest.mark.asyncio
    async def test_reasoning_only(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Only reasoning.
        Goal: Stream completes with finish_reason=stop and [DONE].
        """
        parsed, saw_done = await self._run([_native("solo")], mock_http_client, mock_response, mock_model_cache, mock_auth_manager)
        assert _deltas(parsed, "reasoning_content") == ["solo"]
        assert parsed[-1]["choices"][0]["finish_reason"] == "stop"
        assert saw_done

    @pytest.mark.asyncio
    async def test_remove_mode_emits_no_reasoning_delta(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: remove (core drops native reasoning before it reaches the formatter).
        Goal: No reasoning anywhere in the output.
        """
        # With real parsing the core discards the frame; model that by feeding only content.
        parsed, _ = await self._run([_content("shown"), _done()], mock_http_client, mock_response, mock_model_cache, mock_auth_manager, mode="remove")
        assert not _deltas(parsed, "reasoning_content")
        assert _deltas(parsed, "content") == ["shown"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mode", ["strip_tags", "pass"])
    async def test_text_modes_emit_reasoning_as_content_delta(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager, mode):
        """
        What it does: strip_tags / pass.
        Goal: Reasoning is delivered as regular content deltas (existing OpenAI behavior).
        """
        parsed, _ = await self._run([_native("R"), _content("C"), _done()], mock_http_client, mock_response, mock_model_cache, mock_auth_manager, mode=mode)
        assert not _deltas(parsed, "reasoning_content")
        assert _deltas(parsed, "content") == ["R", "C"]
        assert parsed[0]["choices"][0]["delta"]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_chinese_and_emoji_unescaped(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Chinese / emoji reasoning.
        Goal: Raw characters in the SSE payload (ensure_ascii=False).
        """
        out = []
        with _patch_handling("as_reasoning_content"), \
                patch("kiro.streaming_openai.parse_kiro_stream", _events_source([_native("中文 😀 —"), _done()])):
            async for item in stream_kiro_to_openai(mock_http_client, mock_response, "m", mock_model_cache, mock_auth_manager):
                out.append(item)
        assert "中文 😀 —" in out[0]

    @pytest.mark.asyncio
    async def test_bracket_text_in_reasoning_is_not_a_tool_call(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager):
        """
        What it does: Reasoning quoting bracket tool-call syntax.
        Goal: No tool_calls chunk; finish_reason=stop.
        """
        parsed, _ = await self._run(
            [_native('[Called evil with args: {"a": 1}]'), _content("ok"), _done()],
            mock_http_client, mock_response, mock_model_cache, mock_auth_manager,
        )
        assert not _deltas(parsed, "tool_calls")
        assert parsed[-1]["choices"][0]["finish_reason"] == "stop"


class TestStreamingOpenaiNativeReasoningRealBytes:
    """AWS frames -> REAL parser -> OpenAI SSE / non-streaming response."""

    TEXTS = ["I", "'ll", " check", ".\n\n", "中文推理", " 😀", " — done"]

    def _data(self, aws_frames, with_tool=False, content=True):
        data = b"".join(aws_frames.reasoning(t) for t in self.TEXTS) + aws_frames.signature("QUJD" * 20)
        if content:
            data += aws_frames.content("Hello") + aws_frames.content(" world")
        if with_tool:
            data += (
                aws_frames.tool_start("Bash", "toolu_1")
                + aws_frames.tool_input('{"command": "ls"}', "Bash", "toolu_1")
                + aws_frames.tool_stop("Bash", "toolu_1")
            )
        return data + aws_frames.context_usage(2.0)

    async def _stream(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager, chunks, mode="as_reasoning_content"):
        _bytes_response(mock_response, chunks)
        out = []
        with _patch_handling(mode):
            async for item in stream_kiro_to_openai(mock_http_client, mock_response, "claude-sonnet-4", mock_model_cache, mock_auth_manager):
                out.append(item)
        return out

    @pytest.mark.asyncio
    async def test_one_reasoning_delta_per_frame(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager, aws_frames):
        """
        What it does: reasoning x7 -> signature -> content x2.
        Goal: reasoning_content deltas == frames in order; signature never forwarded; content intact.
        """
        raw = await self._stream(mock_http_client, mock_response, mock_model_cache, mock_auth_manager, [self._data(aws_frames)])
        parsed, saw_done = _parse_chunks(raw)

        assert _deltas(parsed, "reasoning_content") == self.TEXTS
        assert "".join(_deltas(parsed, "content")) == "Hello world"
        assert "QUJD" * 20 not in "".join(raw)
        assert saw_done

    @pytest.mark.asyncio
    async def test_chunking_does_not_change_output(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager, aws_frames):
        """
        What it does: Same bytes chunked 1-byte and with odd sizes.
        Goal: Same reasoning / content sequence.
        """
        data = self._data(aws_frames)
        for sizes in ([1], [3, 5, 11], [64]):
            raw = await self._stream(mock_http_client, mock_response, mock_model_cache, mock_auth_manager, aws_frames.split(data, sizes))
            parsed, _ = _parse_chunks(raw)
            assert _deltas(parsed, "reasoning_content") == self.TEXTS, sizes
            assert "".join(_deltas(parsed, "content")) == "Hello world", sizes

    @pytest.mark.asyncio
    async def test_reasoning_then_tool_call_frames(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager, aws_frames):
        """
        What it does: reasoning frames then tool call frames, no text.
        Goal: reasoning deltas then tool_calls chunk; finish_reason=tool_calls.
        """
        raw = await self._stream(mock_http_client, mock_response, mock_model_cache, mock_auth_manager, [self._data(aws_frames, with_tool=True, content=False)])
        parsed, _ = _parse_chunks(raw)
        assert _deltas(parsed, "reasoning_content") == self.TEXTS
        assert _deltas(parsed, "tool_calls")[0][0]["function"]["name"] == "Bash"
        assert parsed[-1]["choices"][0]["finish_reason"] == "tool_calls"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mode,expect_reasoning,expect_content", [
        ("as_reasoning_content", True, "Hello world"),
        ("remove", False, "Hello world"),
        ("strip_tags", False, "".join(TEXTS) + "Hello world"),
        ("pass", False, "".join(TEXTS) + "Hello world"),
    ])
    async def test_handling_modes(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager, aws_frames, mode, expect_reasoning, expect_content):
        """
        What it does: Each FAKE_REASONING_HANDLING mode over real bytes.
        Goal: as_reasoning_content -> reasoning_content; remove -> nothing; strip_tags/pass -> content.
        """
        raw = await self._stream(mock_http_client, mock_response, mock_model_cache, mock_auth_manager, [self._data(aws_frames)], mode=mode)
        parsed, _ = _parse_chunks(raw)
        assert bool(_deltas(parsed, "reasoning_content")) is expect_reasoning
        assert "".join(_deltas(parsed, "content")) == expect_content
        if mode == "remove":
            assert "".join(self.TEXTS) not in "".join(raw)

    @pytest.mark.asyncio
    async def test_logs_never_contain_reasoning_text(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager, aws_frames, loguru_messages):
        """
        What it does: Streams reasoning with a secret marker.
        Goal: No log line (parser, core, formatter) contains the reasoning text.
        """
        data = aws_frames.reasoning(NATIVE_SECRET) + aws_frames.content("ok") + aws_frames.context_usage(1.0)
        await self._stream(mock_http_client, mock_response, mock_model_cache, mock_auth_manager, [data])
        assert NATIVE_SECRET not in "\n".join(loguru_messages)


class TestCollectStreamResponseNativeReasoning:
    """Non-streaming OpenAI response with native reasoning."""

    async def _collect(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager, data, mode="as_reasoning_content"):
        _bytes_response(mock_response, [data])
        with _patch_handling(mode):
            return await collect_stream_response(mock_http_client, mock_response, "claude-sonnet-4", mock_model_cache, mock_auth_manager)

    @pytest.mark.asyncio
    async def test_reasoning_and_content_aggregated(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager, aws_frames):
        """
        What it does: reasoning + content.
        Goal: message.reasoning_content == all reasoning; content == answer; usage present.
        """
        data = aws_frames.reasoning("one ") + aws_frames.reasoning("中文") + aws_frames.content("answer") + aws_frames.context_usage(2.0)
        result = await self._collect(mock_http_client, mock_response, mock_model_cache, mock_auth_manager, data)
        message = result["choices"][0]["message"]
        assert message["reasoning_content"] == "one 中文"
        assert message["content"] == "answer"
        assert "usage" in result and result["usage"]["completion_tokens"] > 0

    @pytest.mark.asyncio
    async def test_with_tool_calls(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager, aws_frames):
        """
        What it does: reasoning + tool call.
        Goal: reasoning_content and tool_calls both present; finish_reason=tool_calls.
        """
        data = b"".join([
            aws_frames.reasoning("need tool"),
            aws_frames.tool_start("Bash", "toolu_1"),
            aws_frames.tool_input('{"command": "ls"}', "Bash", "toolu_1"),
            aws_frames.tool_stop("Bash", "toolu_1"),
            aws_frames.context_usage(1.0),
        ])
        result = await self._collect(mock_http_client, mock_response, mock_model_cache, mock_auth_manager, data)
        choice = result["choices"][0]
        assert choice["message"]["reasoning_content"] == "need tool"
        assert choice["message"]["tool_calls"][0]["function"]["name"] == "Bash"
        assert choice["finish_reason"] == "tool_calls"

    @pytest.mark.asyncio
    async def test_reasoning_only(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager, aws_frames):
        """
        What it does: Only reasoning, with a meteringEvent (complete response without a body).
        Goal: reasoning_content set, content empty string.
        """
        data = aws_frames.reasoning("solo") + aws_frames.metering(0.1)
        result = await self._collect(mock_http_client, mock_response, mock_model_cache, mock_auth_manager, data)
        message = result["choices"][0]["message"]
        assert message["reasoning_content"] == "solo"
        assert message["content"] == ""

    @pytest.mark.asyncio
    async def test_remove_mode_has_no_reasoning_key(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager, aws_frames):
        """
        What it does: remove.
        Goal: No reasoning_content key and no reasoning text anywhere.
        """
        data = aws_frames.reasoning(NATIVE_SECRET) + aws_frames.content("shown") + aws_frames.context_usage(1.0)
        result = await self._collect(mock_http_client, mock_response, mock_model_cache, mock_auth_manager, data, mode="remove")
        assert "reasoning_content" not in result["choices"][0]["message"]
        assert NATIVE_SECRET not in json.dumps(result)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mode", ["strip_tags", "pass"])
    async def test_text_modes_put_reasoning_in_content(self, mock_http_client, mock_response, mock_model_cache, mock_auth_manager, aws_frames, mode):
        """
        What it does: strip_tags / pass.
        Goal: Reasoning folded into content; no reasoning_content key.
        """
        data = aws_frames.reasoning("R.") + aws_frames.content("C") + aws_frames.context_usage(1.0)
        result = await self._collect(mock_http_client, mock_response, mock_model_cache, mock_auth_manager, data, mode=mode)
        message = result["choices"][0]["message"]
        assert message["content"] == "R.C"
        assert "reasoning_content" not in message


# ==================================================================================================
# Empty upstream response (guard)
# ==================================================================================================

import httpx
from fastapi import HTTPException

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


def _openai_cache():
    """Model cache double."""
    cache = MagicMock()
    cache.get_max_input_tokens.return_value = 200000
    return cache


class TestCollectStreamResponseEmptyUpstream:
    """collect_stream_response() with the empty-response guard (non-streaming)."""

    @pytest.mark.asyncio
    async def test_empty_upstream_response_raises(self, aws_frames):
        """
        What it does: Non-streaming collection of a lone contextUsageEvent frame.
        Purpose: The route gets EmptyUpstreamResponseError instead of an empty completion.
        """
        response = _real_stream_response([aws_frames.context_usage(60.7)])

        with patch("kiro.streaming_core.EMPTY_RESPONSE_AS_ERROR", True):
            with pytest.raises(EmptyUpstreamResponseError):
                await collect_stream_response(
                    MagicMock(), response, "claude-sonnet-4.5", _openai_cache(), MagicMock()
                )

    @pytest.mark.asyncio
    async def test_metered_empty_answer_is_a_normal_empty_completion(self, aws_frames):
        """
        What it does: contextUsageEvent + meteringEvent, no output.
        Purpose: The model's empty answer is passed through (no error).
        """
        response = _real_stream_response([aws_frames.context_usage(2.0) + aws_frames.metering(0.1)])

        with patch("kiro.streaming_core.EMPTY_RESPONSE_AS_ERROR", True):
            result = await collect_stream_response(
                MagicMock(), response, "claude-sonnet-4.5", _openai_cache(), MagicMock()
            )

        assert result["choices"][0]["message"]["content"] == ""
        assert result["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_switch_off_returns_empty_completion(self, aws_frames):
        """
        What it does: EMPTY_RESPONSE_AS_ERROR=False with an empty upstream response.
        Purpose: Native behavior restored.
        """
        response = _real_stream_response([aws_frames.context_usage(60.7)])

        with patch("kiro.streaming_core.EMPTY_RESPONSE_AS_ERROR", False):
            result = await collect_stream_response(
                MagicMock(), response, "claude-sonnet-4.5", _openai_cache(), MagicMock()
            )

        assert result["choices"][0]["message"]["content"] == ""

    @pytest.mark.asyncio
    async def test_normal_response_is_unchanged(self, aws_frames):
        """
        What it does: Normal response, guard on, awkward chunking.
        Purpose: Content and reasoning are collected as before.
        """
        response = _real_stream_response(aws_frames.split(_normal_bytes(aws_frames), [19]))

        with patch("kiro.streaming_core.EMPTY_RESPONSE_AS_ERROR", True):
            result = await collect_stream_response(
                MagicMock(), response, "claude-sonnet-4.5", _openai_cache(), MagicMock()
            )

        message = result["choices"][0]["message"]
        assert message["content"] == "Hello world"
        assert message["reasoning_content"] == "think"


class TestStreamingOpenaiEmptyUpstream:
    """stream_with_first_token_retry() (OpenAI) with the guard."""

    @staticmethod
    async def _run(response, make_request=None, timeout=5, ready=True):
        """Runs the OpenAI retry wrapper and returns all yielded items."""
        return [
            item async for item in stream_with_first_token_retry(
                make_request=make_request or AsyncMock(),
                client=MagicMock(),
                model="claude-sonnet-4.5",
                model_cache=_openai_cache(),
                auth_manager=MagicMock(),
                initial_response=response,
                first_token_timeout=timeout,
                emit_ready_marker=ready,
            )
        ]

    @pytest.mark.asyncio
    async def test_empty_response_raises_before_yielding_anything(self, aws_frames):
        """
        What it does: Empty upstream response through the OpenAI streaming wrapper.
        Purpose: Raised before any SSE chunk (and before the marker).
        """
        response = _real_stream_response([aws_frames.context_usage(60.7)])
        items = []

        with patch("kiro.streaming_core.EMPTY_RESPONSE_AS_ERROR", True):
            with pytest.raises(EmptyUpstreamResponseError):
                async for item in stream_with_first_token_retry(
                    make_request=AsyncMock(), client=MagicMock(), model="m",
                    model_cache=_openai_cache(), auth_manager=MagicMock(),
                    initial_response=response, first_token_timeout=5, emit_ready_marker=True,
                ):
                    items.append(item)

        assert items == []

    @pytest.mark.asyncio
    async def test_normal_stream_has_role_first_and_done_last(self, aws_frames):
        """
        What it does: Normal response, marker enabled.
        Purpose: Marker, then first chunk with role=assistant, last item [DONE].
        """
        response = _real_stream_response(aws_frames.split(_normal_bytes(aws_frames), [13]))

        with patch("kiro.streaming_core.EMPTY_RESPONSE_AS_ERROR", True):
            items = await self._run(response)

        assert items[0] is STREAM_READY
        chunks = items[1:]
        assert all(isinstance(c, str) for c in chunks)
        first = json.loads(chunks[0][len("data: "):])
        assert first["choices"][0]["delta"]["role"] == "assistant"
        assert chunks[-1].strip() == "data: [DONE]"

    @pytest.mark.asyncio
    async def test_sequence_identical_with_and_without_guard(self, aws_frames):
        """
        What it does: Same successful bytes with the guard on and off.
        Purpose: The guard must not change the SSE content (ids/timestamps aside).
        """
        chunks = aws_frames.split(_normal_bytes(aws_frames), [29])

        def shape(items):
            shaped = []
            for item in items:
                if not isinstance(item, str) or item.strip() == "data: [DONE]":
                    shaped.append(str(item).strip())
                    continue
                data = json.loads(item[len("data: "):])
                shaped.append(json.dumps(data["choices"], sort_keys=True))
            return shaped

        with patch("kiro.streaming_core.EMPTY_RESPONSE_AS_ERROR", True):
            guarded = await self._run(_real_stream_response(chunks))
        with patch("kiro.streaming_core.EMPTY_RESPONSE_AS_ERROR", False):
            plain = await self._run(_real_stream_response(chunks))

        assert shape(guarded) == shape(plain)

    @pytest.mark.asyncio
    async def test_timeout_inside_guard_is_retried(self, aws_frames):
        """
        What it does: Attempt 1 hangs inside the guard, attempt 2 is normal.
        Purpose: FIRST_TOKEN retry behavior is unchanged; one marker, complete stream.
        """
        good = _real_stream_response([_normal_bytes(aws_frames)])
        make_request = AsyncMock(return_value=good)

        with patch("kiro.streaming_core.EMPTY_RESPONSE_AS_ERROR", True):
            items = await self._run(_hanging_stream_response(), make_request=make_request, timeout=0.05)

        assert items[0] is STREAM_READY
        assert items[-1].strip() == "data: [DONE]"
        assert make_request.await_count == 1

    @pytest.mark.asyncio
    async def test_all_attempts_timing_out_still_raises_504(self):
        """
        What it does: Every attempt hangs.
        Purpose: HTTPException 504 after all retries, as before.
        """
        make_request = AsyncMock(side_effect=lambda: _hanging_stream_response())

        with patch("kiro.streaming_core.EMPTY_RESPONSE_AS_ERROR", True):
            with pytest.raises(HTTPException) as exc_info:
                await self._run(_hanging_stream_response(), make_request=make_request, timeout=0.02)

        assert exc_info.value.status_code == 504

    @pytest.mark.asyncio
    async def test_metered_empty_answer_streams_normally(self, aws_frames):
        """
        What it does: contextUsageEvent + meteringEvent through the OpenAI streaming wrapper.
        Purpose: Passed through as a normal (empty) stream that ends with [DONE].
        """
        response = _real_stream_response([aws_frames.context_usage(2.0) + aws_frames.metering(0.1)])

        with patch("kiro.streaming_core.EMPTY_RESPONSE_AS_ERROR", True):
            items = await self._run(response)

        assert items[0] is STREAM_READY
        assert items[-1].strip() == "data: [DONE]"


# ==================================================================================================
# Incomplete upstream response (reasoning only, no meteringEvent, no text, no tool_use)
# ==================================================================================================

INCOMPLETE_THOUGHT = "SECRET_THOUGHT_FOR_INCOMPLETE_4d7c"


def _incomplete_bytes(aws_frames, count=6, with_context_usage=True):
    """Reasoning frames (+ contextUsageEvent) without meteringEvent, text or tool call."""
    data = b"".join(aws_frames.reasoning(f"{INCOMPLETE_THOUGHT} {i}") for i in range(count))
    if with_context_usage:
        data += aws_frames.context_usage(3.5)
    return data


async def _openai_stream(chunks, mode="as_reasoning_content", response=None):
    """Runs stream_kiro_to_openai over real bytes and returns the SSE strings."""
    with _patch_handling(mode):
        return [
            item async for item in stream_kiro_to_openai(
                MagicMock(), response or _real_stream_response(chunks), "claude-sonnet-4-5",
                _openai_cache(), MagicMock(),
            )
        ]


class TestStreamingOpenaiIncompleteResponse:
    """stream_kiro_to_openai(): incomplete responses end with an error chunk and [DONE]."""

    @pytest.mark.asyncio
    async def test_error_chunk_then_done_without_finish_reason(self, aws_frames):
        """
        What it does: 6 reasoning frames + contextUsageEvent, no metering (reported failure).
        Purpose: reasoning_content deltas, then {"error": ...}, then [DONE]; no finish_reason
        chunk, no tool_calls chunk.
        """
        items = await _openai_stream([_incomplete_bytes(aws_frames)])

        body = [i[len("data: "):].strip() for i in items]
        print(body[-3:])
        assert body[-1] == "[DONE]"
        error = json.loads(body[-2])
        assert error == {
            "error": {
                "message": error["error"]["message"],
                "type": "server_error",
                "param": None,
                "code": "upstream_incomplete_response",
            }
        }
        assert "incomplete" in error["error"]["message"] and "retry" in error["error"]["message"]
        parsed = [json.loads(b) for b in body[:-2]]
        assert len(_deltas(parsed, "reasoning_content")) == 6
        assert all(c["choices"][0]["finish_reason"] is None for c in parsed)
        assert not any("tool_calls" in c["choices"][0]["delta"] for c in parsed)
        assert all(i.startswith("data: ") and i.endswith("\n\n") for i in items)

    @pytest.mark.asyncio
    async def test_no_content_leaks_into_the_error(self, aws_frames):
        """
        What it does: Reads the error chunk.
        Purpose: Friendly message, no cause claim, no reasoning text.
        """
        items = await _openai_stream([_incomplete_bytes(aws_frames)])

        message = json.loads(items[-2][len("data: "):])["error"]["message"].lower()
        for forbidden in ("overloaded", "rate limit", "capacity", "quota", "throttl", INCOMPLETE_THOUGHT.lower()):
            assert forbidden not in message

    @pytest.mark.asyncio
    async def test_only_reasoning_frames_and_any_chunking(self, aws_frames):
        """
        What it does: 49 reasoning frames, no contextUsageEvent (17:15:07 shape), byte-sized,
        13-byte and whole chunks.
        Purpose: Same ending whatever the chunking.
        """
        data = _incomplete_bytes(aws_frames, count=49, with_context_usage=False)

        for sizes in ([1], [13], [len(data)]):
            items = await _openai_stream(aws_frames.split(data, sizes))
            assert items[-1] == "data: [DONE]\n\n", sizes
            assert "upstream_incomplete_response" in items[-2], sizes

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mode", ["include_as_text", "strip_tags", "pass"])
    async def test_text_modes_end_with_error(self, mode, aws_frames):
        """
        What it does: Reasoning delivered as content deltas (text modes).
        Purpose: Those deltas never make the response count as a real answer.
        """
        items = await _openai_stream([_incomplete_bytes(aws_frames)], mode=mode)

        assert "upstream_incomplete_response" in items[-2]
        assert items[-1] == "data: [DONE]\n\n"
        parsed, _ = _parse_chunks([i for i in items if "error" not in i])
        assert all(c["choices"][0]["finish_reason"] is None for c in parsed)

    @pytest.mark.asyncio
    async def test_remove_mode_sends_only_the_error(self, aws_frames):
        """
        What it does: FAKE_REASONING_HANDLING=remove discards reasoning.
        Purpose: Nothing but the error chunk and [DONE].
        """
        items = await _openai_stream([_incomplete_bytes(aws_frames)], mode="remove")

        assert len(items) == 2 and "upstream_incomplete_response" in items[0]

    @pytest.mark.asyncio
    async def test_metering_makes_it_a_normal_stop(self, aws_frames):
        """
        What it does: Same reasoning + meteringEvent, no body.
        Purpose: Complete: finish_reason chunk with usage, then [DONE].
        """
        items = await _openai_stream([_incomplete_bytes(aws_frames) + aws_frames.metering(0.3)])

        parsed, saw_done = _parse_chunks(items)
        assert saw_done
        assert parsed[-1]["choices"][0]["finish_reason"] == "stop"
        assert not any("error" in c for c in parsed)

    @pytest.mark.asyncio
    async def test_reasoning_and_text_without_metering_is_unchanged_with_warning(self, loguru_messages, aws_frames):
        """
        What it does: Reasoning + text, no metering (possibly truncated).
        Purpose: Normal finish; one WARNING without content.
        """
        data = aws_frames.reasoning("r") + aws_frames.content("partial answer") + aws_frames.context_usage(1.0)

        items = await _openai_stream([data])

        parsed, saw_done = _parse_chunks(items)
        assert saw_done and parsed[-1]["choices"][0]["finish_reason"] == "stop"
        lines = [m for m in loguru_messages if "[IncompleteUpstreamResponse]" in m]
        assert len(lines) == 1 and "partial answer" not in lines[0]

    @pytest.mark.asyncio
    async def test_reasoning_and_tool_use_without_metering_is_unchanged(self, aws_frames):
        """
        What it does: Reasoning + tool call, no metering.
        Purpose: finish_reason=tool_calls.
        """
        data = (
            aws_frames.reasoning("r")
            + aws_frames.tool_start("Bash", "toolu_1")
            + aws_frames.tool_input('{"command": "ls"}', "Bash", "toolu_1")
            + aws_frames.tool_stop("Bash", "toolu_1")
            + aws_frames.context_usage(1.0)
        )

        items = await _openai_stream([data])

        parsed, saw_done = _parse_chunks(items)
        assert saw_done and parsed[-1]["choices"][0]["finish_reason"] == "tool_calls"

    @pytest.mark.asyncio
    async def test_switch_off_restores_the_old_ending(self, loguru_messages, aws_frames):
        """
        What it does: INCOMPLETE_RESPONSE_AS_ERROR=False with an incomplete stream.
        Purpose: finish_reason=stop + [DONE] as before; only a diagnostic WARNING.
        """
        with patch("kiro.streaming_core.INCOMPLETE_RESPONSE_AS_ERROR", False):
            items = await _openai_stream([_incomplete_bytes(aws_frames)])

        parsed, saw_done = _parse_chunks(items)
        assert saw_done and parsed[-1]["choices"][0]["finish_reason"] == "stop"
        assert len([m for m in loguru_messages if "[IncompleteUpstreamResponse]" in m]) == 1

    @pytest.mark.asyncio
    async def test_mid_stream_failure_keeps_the_old_error_path(self, loguru_messages, aws_frames):
        """
        What it does: Reasoning, then the upstream connection breaks.
        Purpose: Exception propagates (route handles it), no error chunk, ERROR logged,
        no [IncompleteUpstreamResponse].
        """
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
                async for item in stream_kiro_to_openai(
                    MagicMock(), response, "m", _openai_cache(), MagicMock()
                ):
                    out.append(item)

        assert not any("upstream_incomplete_response" in i for i in out)
        assert not [m for m in loguru_messages if "[IncompleteUpstreamResponse]" in m]
        assert [m for m in loguru_messages if m.startswith("ERROR|") and "Error during streaming" in m]

    @pytest.mark.asyncio
    async def test_response_closed_and_warning_logged_once(self, loguru_messages, aws_frames):
        """
        What it does: Runs an incomplete stream to the end.
        Purpose: Upstream response closed; one WARNING with action=error_event; no ERROR.
        """
        response = _real_stream_response([_incomplete_bytes(aws_frames)])

        await _openai_stream(None, response=response)

        response.aclose.assert_awaited()
        lines = [m for m in loguru_messages if "[IncompleteUpstreamResponse]" in m]
        assert len(lines) == 1 and "action=error_event" in lines[0]
        assert INCOMPLETE_THOUGHT not in lines[0]
        assert not [m for m in loguru_messages if m.startswith("ERROR|")]


class TestStreamingOpenaiIncompleteViaRetryWrapper:
    """stream_with_first_token_retry() (OpenAI): marker, error chunk, no retry."""

    @pytest.mark.asyncio
    async def test_marker_then_error_chunk_without_retry(self, aws_frames):
        """
        What it does: Incomplete upstream response through the retry wrapper with the marker.
        Purpose: STREAM_READY first, error chunk + [DONE] last, make_request never called.
        """
        make_request = AsyncMock()
        response = _real_stream_response([_incomplete_bytes(aws_frames)])

        with _patch_handling("as_reasoning_content"), patch("kiro.streaming_core.EMPTY_RESPONSE_AS_ERROR", True):
            items = [
                item async for item in stream_with_first_token_retry(
                    make_request=make_request, client=MagicMock(), model="m",
                    model_cache=_openai_cache(), auth_manager=MagicMock(),
                    initial_response=response, first_token_timeout=5, emit_ready_marker=True,
                )
            ]

        assert items[0] is STREAM_READY
        assert "upstream_incomplete_response" in items[-2] and items[-1] == "data: [DONE]\n\n"
        make_request.assert_not_awaited()


class TestCollectStreamResponseIncomplete:
    """collect_stream_response(): non-streaming incomplete responses raise."""

    @staticmethod
    async def _collect(chunks):
        with _patch_handling("as_reasoning_content"):
            return await collect_stream_response(
                MagicMock(), _real_stream_response(chunks), "claude-sonnet-4-5", _openai_cache(), MagicMock()
            )

    @pytest.mark.asyncio
    async def test_incomplete_raises_without_error_log(self, loguru_messages, aws_frames):
        """
        What it does: Reasoning + contextUsageEvent, no metering.
        Purpose: EmptyUpstreamResponseError(kind=incomplete) with the facts; no ERROR line
        with a stack trace (the route reports it).
        """
        data = _incomplete_bytes(aws_frames)

        with pytest.raises(EmptyUpstreamResponseError) as exc_info:
            await self._collect([data])

        err = exc_info.value
        assert err.kind == "incomplete"
        assert err.bytes_received == len(data)
        assert err.frame_counts == {"reasoningContentEvent": 6, "contextUsageEvent": 1}
        assert not [m for m in loguru_messages if "Error during streaming" in m]

    @pytest.mark.asyncio
    async def test_metering_returns_reasoning_only_completion(self, aws_frames):
        """
        What it does: Reasoning + metering, no body.
        Purpose: Complete: reasoning_content present, empty content, finish_reason stop.
        """
        result = await self._collect([_incomplete_bytes(aws_frames) + aws_frames.metering(0.1)])

        message = result["choices"][0]["message"]
        assert INCOMPLETE_THOUGHT in message["reasoning_content"] and message["content"] == ""
        assert result["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_reasoning_and_text_without_metering_is_returned(self, aws_frames):
        """
        What it does: Reasoning + text, no metering.
        Purpose: Returned as before (WARNING only).
        """
        result = await self._collect([aws_frames.reasoning("r") + aws_frames.content("answer")])

        assert result["choices"][0]["message"]["content"] == "answer"

    @pytest.mark.asyncio
    async def test_switch_off_returns_reasoning_only_completion(self, aws_frames):
        """
        What it does: INCOMPLETE_RESPONSE_AS_ERROR=False.
        Purpose: Old behavior: reasoning-only completion.
        """
        with patch("kiro.streaming_core.INCOMPLETE_RESPONSE_AS_ERROR", False):
            result = await self._collect([_incomplete_bytes(aws_frames)])

        assert result["choices"][0]["message"]["content"] == ""
        assert result["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_streaming_wrapper_default_does_not_raise(self, aws_frames):
        """
        What it does: stream_kiro_to_openai without raise_on_incomplete on an incomplete stream.
        Purpose: Streaming default stays False: an error chunk, never an exception.
        """
        items = await _openai_stream([_incomplete_bytes(aws_frames)])

        assert "upstream_incomplete_response" in items[-2]
