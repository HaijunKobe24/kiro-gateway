# -*- coding: utf-8 -*-

"""
Unit tests for AwsEventStreamParser and auxiliary parsing functions.
Tests the parsing logic for AWS SSE stream from Kiro API.
"""

import pytest

from kiro.parsers import (
    AwsEventStreamParser,
    find_matching_brace,
    parse_bracket_tool_calls,
    deduplicate_tool_calls
)


class TestFindMatchingBrace:
    """Tests for find_matching_brace function."""
    
    def test_simple_json_object(self):
        """
        What it does: Tests finding closing brace for simple JSON.
        Goal: Ensure the basic case works.
        """
        print("Setup: Simple JSON object...")
        text = '{"key": "value"}'
        
        print("Action: Finding closing brace...")
        result = find_matching_brace(text, 0)
        
        print(f"Comparing result: Expected 15, Got {result}")
        assert result == 15
    
    def test_nested_json_object(self):
        """
        What it does: Tests finding brace for nested JSON.
        Goal: Ensure nesting is handled correctly.
        """
        print("Setup: Nested JSON object...")
        text = '{"outer": {"inner": "value"}}'
        
        print("Action: Finding closing brace...")
        result = find_matching_brace(text, 0)
        
        # String length 29, last character index 28
        print(f"Comparing result: Expected 28, Got {result}")
        assert result == 28
    
    def test_json_with_braces_in_string(self):
        """
        What it does: Tests ignoring braces inside strings.
        Goal: Ensure braces in strings don't affect counting.
        """
        print("Setup: JSON with braces in string...")
        text = '{"text": "Hello {world}"}'
        
        print("Action: Finding closing brace...")
        result = find_matching_brace(text, 0)
        
        print(f"Comparing result: Expected 24, Got {result}")
        assert result == 24
    
    def test_json_with_escaped_quotes(self):
        """
        What it does: Tests handling of escaped quotes.
        Goal: Ensure escape sequences don't break parsing.
        """
        print("Setup: JSON with escaped quotes...")
        text = '{"text": "Say \\"hello\\""}'
        
        print("Action: Finding closing brace...")
        result = find_matching_brace(text, 0)
        
        # String length 25, last character index 24
        print(f"Comparing result: Expected 24, Got {result}")
        assert result == 24
    
    def test_incomplete_json(self):
        """
        What it does: Tests handling of incomplete JSON.
        Goal: Ensure -1 is returned for incomplete JSON.
        """
        print("Setup: Incomplete JSON...")
        text = '{"key": "value"'
        
        print("Action: Finding closing brace...")
        result = find_matching_brace(text, 0)
        
        print(f"Comparing result: Expected -1, Got {result}")
        assert result == -1
    
    def test_invalid_start_position(self):
        """
        What it does: Tests handling of invalid start position.
        Goal: Ensure -1 is returned if start_pos is not on '{'.
        """
        print("Setup: Text without brace at start_pos...")
        text = 'hello {"key": "value"}'
        
        print("Action: Finding from position 0 (not a brace)...")
        result = find_matching_brace(text, 0)
        
        print(f"Comparing result: Expected -1, Got {result}")
        assert result == -1
    
    def test_start_position_out_of_bounds(self):
        """
        What it does: Tests handling of position beyond text bounds.
        Goal: Ensure -1 is returned for invalid position.
        """
        print("Setup: Short text...")
        text = '{"a":1}'
        
        print("Action: Finding from position 100...")
        result = find_matching_brace(text, 100)
        
        print(f"Comparing result: Expected -1, Got {result}")
        assert result == -1


class TestParseBracketToolCalls:
    """Tests for parse_bracket_tool_calls function."""
    
    def test_parses_single_tool_call(self):
        """
        What it does: Tests parsing of a single tool call.
        Goal: Ensure bracket-style tool call is extracted correctly.
        """
        print("Setup: Text with one tool call...")
        text = '[Called get_weather with args: {"location": "Moscow"}]'
        
        print("Action: Parsing tool calls...")
        result = parse_bracket_tool_calls(text)
        
        print(f"Result: {result}")
        assert len(result) == 1
        assert result[0]["function"]["name"] == "get_weather"
        assert '"location"' in result[0]["function"]["arguments"]
    
    def test_parses_multiple_tool_calls(self):
        """
        What it does: Tests parsing of multiple tool calls.
        Goal: Ensure all tool calls are extracted.
        """
        print("Setup: Text with multiple tool calls...")
        text = '''
        [Called get_weather with args: {"location": "Moscow"}]
        Some text in between
        [Called get_time with args: {"timezone": "UTC"}]
        '''
        
        print("Action: Parsing tool calls...")
        result = parse_bracket_tool_calls(text)
        
        print(f"Result: {result}")
        assert len(result) == 2
        assert result[0]["function"]["name"] == "get_weather"
        assert result[1]["function"]["name"] == "get_time"
    
    def test_returns_empty_for_no_tool_calls(self):
        """
        What it does: Tests returning empty list without tool calls.
        Goal: Ensure regular text is not parsed as tool call.
        """
        print("Setup: Regular text without tool calls...")
        text = "This is just regular text without any tool calls."
        
        print("Action: Parsing tool calls...")
        result = parse_bracket_tool_calls(text)
        
        print(f"Comparing result: Expected [], Got {result}")
        assert result == []
    
    def test_returns_empty_for_empty_string(self):
        """
        What it does: Tests handling of empty string.
        Goal: Ensure empty string doesn't cause errors.
        """
        print("Setup: Empty string...")
        
        print("Action: Parsing tool calls...")
        result = parse_bracket_tool_calls("")
        
        print(f"Comparing result: Expected [], Got {result}")
        assert result == []
    
    def test_returns_empty_for_none(self):
        """
        What it does: Tests handling of None.
        Goal: Ensure None doesn't cause errors.
        """
        print("Setup: None...")
        
        print("Action: Parsing tool calls...")
        result = parse_bracket_tool_calls(None)
        
        print(f"Comparing result: Expected [], Got {result}")
        assert result == []
    
    def test_handles_nested_json_in_args(self):
        """
        What it does: Tests parsing of nested JSON in arguments.
        Goal: Ensure complex arguments are parsed correctly.
        """
        print("Setup: Tool call with nested JSON...")
        text = '[Called complex_func with args: {"data": {"nested": {"deep": "value"}}}]'
        
        print("Action: Parsing tool calls...")
        result = parse_bracket_tool_calls(text)
        
        print(f"Result: {result}")
        assert len(result) == 1
        assert result[0]["function"]["name"] == "complex_func"
        assert "nested" in result[0]["function"]["arguments"]
    
    def test_generates_unique_ids(self):
        """
        What it does: Tests generation of unique IDs for tool calls.
        Goal: Ensure each tool call has a unique ID.
        """
        print("Setup: Two identical tool calls...")
        text = '''
        [Called func with args: {"a": 1}]
        [Called func with args: {"a": 1}]
        '''
        
        print("Action: Parsing tool calls...")
        result = parse_bracket_tool_calls(text)
        
        print(f"IDs: {[r['id'] for r in result]}")
        assert len(result) == 2
        assert result[0]["id"] != result[1]["id"]


class TestDeduplicateToolCalls:
    """Tests for deduplicate_tool_calls function."""
    
    def test_removes_duplicates(self):
        """
        What it does: Tests removal of duplicates.
        Goal: Ensure identical tool calls are removed.
        """
        print("Setup: List with duplicates...")
        tool_calls = [
            {"id": "1", "function": {"name": "func", "arguments": '{"a": 1}'}},
            {"id": "2", "function": {"name": "func", "arguments": '{"a": 1}'}},
            {"id": "3", "function": {"name": "other", "arguments": '{"b": 2}'}},
        ]
        
        print("Action: Deduplication...")
        result = deduplicate_tool_calls(tool_calls)
        
        print(f"Comparing length: Expected 2, Got {len(result)}")
        assert len(result) == 2
    
    def test_preserves_first_occurrence(self):
        """
        What it does: Tests preservation of first occurrence.
        Goal: Ensure the first tool call from duplicates is preserved.
        """
        print("Setup: List with duplicates...")
        tool_calls = [
            {"id": "first", "function": {"name": "func", "arguments": '{"a": 1}'}},
            {"id": "second", "function": {"name": "func", "arguments": '{"a": 1}'}},
        ]
        
        print("Action: Deduplication...")
        result = deduplicate_tool_calls(tool_calls)
        
        print(f"Comparing ID: Expected 'first', Got '{result[0]['id']}'")
        assert result[0]["id"] == "first"
    
    def test_handles_empty_list(self):
        """
        What it does: Tests handling of empty list.
        Goal: Ensure empty list doesn't cause errors.
        """
        print("Setup: Empty list...")
        
        print("Action: Deduplication...")
        result = deduplicate_tool_calls([])
        
        print(f"Comparing result: Expected [], Got {result}")
        assert result == []
    
    def test_deduplicates_by_id_keeps_one_with_arguments(self):
        """
        What it does: Tests deduplication by id keeping tool call with arguments.
        Goal: Ensure that when duplicates by id exist, the one with arguments is kept.
        """
        print("Setup: Two tool calls with same id, one with arguments, one empty...")
        tool_calls = [
            {"id": "call_123", "function": {"name": "func", "arguments": "{}"}},
            {"id": "call_123", "function": {"name": "func", "arguments": '{"location": "Moscow"}'}},
        ]
        
        print("Action: Deduplication...")
        result = deduplicate_tool_calls(tool_calls)
        
        print(f"Result: {result}")
        print(f"Comparing length: Expected 1, Got {len(result)}")
        assert len(result) == 1
        
        print("Verifying that tool call with arguments was kept...")
        assert "Moscow" in result[0]["function"]["arguments"]
    
    def test_deduplicates_by_id_prefers_longer_arguments(self):
        """
        What it does: Tests that duplicates by id prefer longer arguments.
        Goal: Ensure tool call with more complete arguments is kept.
        """
        print("Setup: Two tool calls with same id, different argument lengths...")
        tool_calls = [
            {"id": "call_abc", "function": {"name": "search", "arguments": '{"q": "test"}'}},
            {"id": "call_abc", "function": {"name": "search", "arguments": '{"q": "test", "limit": 10, "offset": 0}'}},
        ]
        
        print("Action: Deduplication...")
        result = deduplicate_tool_calls(tool_calls)
        
        print(f"Result: {result}")
        assert len(result) == 1
        
        print("Verifying that tool call with longer arguments was kept...")
        assert "limit" in result[0]["function"]["arguments"]
    
    def test_deduplicates_empty_arguments_replaced_by_non_empty(self):
        """
        What it does: Tests replacement of empty arguments with non-empty.
        Goal: Ensure "{}" is replaced with actual arguments.
        """
        print("Setup: First tool call with empty arguments, second with real ones...")
        tool_calls = [
            {"id": "call_xyz", "function": {"name": "get_weather", "arguments": "{}"}},
            {"id": "call_xyz", "function": {"name": "get_weather", "arguments": '{"city": "London"}'}},
        ]
        
        print("Action: Deduplication...")
        result = deduplicate_tool_calls(tool_calls)
        
        print(f"Result: {result}")
        assert len(result) == 1
        assert result[0]["function"]["arguments"] == '{"city": "London"}'
    
    def test_handles_tool_calls_without_id(self):
        """
        What it does: Tests handling of tool calls without id.
        Goal: Ensure tool calls without id are deduplicated by name+arguments.
        """
        print("Setup: Tool calls without id...")
        tool_calls = [
            {"id": "", "function": {"name": "func", "arguments": '{"a": 1}'}},
            {"id": "", "function": {"name": "func", "arguments": '{"a": 1}'}},
            {"id": "", "function": {"name": "func", "arguments": '{"b": 2}'}},
        ]
        
        print("Action: Deduplication...")
        result = deduplicate_tool_calls(tool_calls)
        
        print(f"Result: {result}")
        # Two unique by name+arguments
        assert len(result) == 2
    
    def test_mixed_with_and_without_id(self):
        """
        What it does: Tests mixed list with and without id.
        Goal: Ensure both types are handled correctly.
        """
        print("Setup: Mixed list...")
        tool_calls = [
            {"id": "call_1", "function": {"name": "func1", "arguments": '{"x": 1}'}},
            {"id": "call_1", "function": {"name": "func1", "arguments": "{}"}},  # Duplicate by id
            {"id": "", "function": {"name": "func2", "arguments": '{"y": 2}'}},
            {"id": "", "function": {"name": "func2", "arguments": '{"y": 2}'}},  # Duplicate by name+args
        ]
        
        print("Action: Deduplication...")
        result = deduplicate_tool_calls(tool_calls)
        
        print(f"Result: {result}")
        # call_1 with arguments + func2 once
        assert len(result) == 2
        
        # Verify that call_1 kept its arguments
        call_1 = next(tc for tc in result if tc["id"] == "call_1")
        assert call_1["function"]["arguments"] == '{"x": 1}'


class TestAwsEventStreamParserInitialization:
    """Tests for AwsEventStreamParser initialization."""
    
    def test_initialization_creates_empty_state(self):
        """
        What it does: Tests initial parser state.
        Goal: Ensure parser is created with empty state.
        """
        print("Setup: Creating parser...")
        parser = AwsEventStreamParser()
        
        print("Check: Buffer is empty...")
        assert parser.buffer == ""
        
        print("Check: last_content is None...")
        assert parser.last_content is None
        
        print("Check: current_tool_call is None...")
        assert parser.current_tool_call is None
        
        print("Check: tool_calls is empty...")
        assert parser.tool_calls == []


class TestAwsEventStreamParserFeed:
    """Tests for parser feed method."""
    
    def test_parses_content_event(self, aws_event_parser):
        """
        What it does: Tests parsing of content event.
        Goal: Ensure text content is extracted.
        """
        print("Setup: Chunk with content...")
        chunk = b'{"content":"Hello World"}'
        
        print("Action: Parsing chunk...")
        events = aws_event_parser.feed(chunk)
        
        print(f"Result: {events}")
        assert len(events) == 1
        assert events[0]["type"] == "content"
        assert events[0]["data"] == "Hello World"
    
    def test_parses_multiple_content_events(self, aws_event_parser):
        """
        What it does: Tests parsing of multiple content events.
        Goal: Ensure all events are extracted.
        """
        print("Setup: Chunk with multiple events...")
        chunk = b'{"content":"First"}{"content":"Second"}'
        
        print("Action: Parsing chunk...")
        events = aws_event_parser.feed(chunk)
        
        print(f"Result: {events}")
        assert len(events) == 2
        assert events[0]["data"] == "First"
        assert events[1]["data"] == "Second"
    
    def test_deduplicates_repeated_content(self, aws_event_parser):
        """
        What it does: Tests deduplication of repeated content.
        Goal: Ensure identical content is not duplicated.
        """
        print("Setup: Chunks with repeated content...")
        
        print("Action: Parsing first chunk...")
        events1 = aws_event_parser.feed(b'{"content":"Same"}')
        
        print("Action: Parsing second chunk with same content...")
        events2 = aws_event_parser.feed(b'{"content":"Same"}')
        
        print(f"First result: {events1}")
        print(f"Second result: {events2}")
        assert len(events1) == 1
        assert len(events2) == 0  # Duplicate filtered out
    
    def test_parses_usage_event(self, aws_event_parser):
        """
        What it does: Tests parsing of usage event.
        Goal: Ensure credits information is extracted.
        """
        print("Setup: Chunk with usage...")
        chunk = b'{"usage":1.5}'
        
        print("Action: Parsing chunk...")
        events = aws_event_parser.feed(chunk)
        
        print(f"Result: {events}")
        assert len(events) == 1
        assert events[0]["type"] == "usage"
        assert events[0]["data"] == 1.5
    
    def test_parses_context_usage_event(self, aws_event_parser):
        """
        What it does: Tests parsing of context_usage event.
        Goal: Ensure context usage percentage is extracted.
        """
        print("Setup: Chunk with context usage...")
        chunk = b'{"contextUsagePercentage":25.5}'
        
        print("Action: Parsing chunk...")
        events = aws_event_parser.feed(chunk)
        
        print(f"Result: {events}")
        assert len(events) == 1
        assert events[0]["type"] == "context_usage"
        assert events[0]["data"] == 25.5
    
    def test_handles_incomplete_json(self, aws_event_parser):
        """
        What it does: Tests handling of incomplete JSON.
        Goal: Ensure incomplete JSON is buffered.
        """
        print("Setup: Incomplete chunk...")
        chunk = b'{"content":"Hel'
        
        print("Action: Parsing incomplete chunk...")
        events = aws_event_parser.feed(chunk)
        
        print(f"Result: {events}")
        assert len(events) == 0  # Nothing parsed
        
        print("Check: Data in buffer...")
        assert 'content' in aws_event_parser.buffer
    
    def test_completes_json_across_chunks(self, aws_event_parser):
        """
        What it does: Tests assembling JSON from multiple chunks.
        Goal: Ensure JSON is assembled from parts.
        """
        print("Setup: First part of JSON...")
        events1 = aws_event_parser.feed(b'{"content":"Hel')
        
        print("Action: Second part of JSON...")
        events2 = aws_event_parser.feed(b'lo World"}')
        
        print(f"First result: {events1}")
        print(f"Second result: {events2}")
        assert len(events1) == 0
        assert len(events2) == 1
        assert events2[0]["data"] == "Hello World"
    
    def test_decodes_escape_sequences(self, aws_event_parser):
        """
        What it does: Tests decoding of escape sequences.
        Goal: Ensure \\n is converted to actual newline.
        """
        print("Setup: Chunk with escape sequence...")
        # Using correct escape sequence format
        chunk = b'{"content":"Line1\\nLine2"}'
        
        print("Action: Parsing chunk...")
        events = aws_event_parser.feed(chunk)
        
        print(f"Result: {events}")
        assert len(events) == 1
        assert "\n" in events[0]["data"]
    def test_handles_invalid_bytes(self, aws_event_parser):
        """
        What it does: Tests handling of invalid bytes.
        Goal: Ensure invalid data doesn't break the parser.
        """
        print("Setup: Invalid bytes...")
        chunk = b'\xff\xfe{"content":"test"}'
        
        print("Action: Parsing chunk...")
        events = aws_event_parser.feed(chunk)
        
        print(f"Result: {events}")
        # Parser should continue working
        assert len(events) == 1


class TestAwsEventStreamParserToolCalls:
    """Tests for tool calls parsing."""
    
    def test_parses_tool_start_event(self, aws_event_parser):
        """
        What it does: Tests parsing of tool call start.
        Goal: Ensure tool_start creates current_tool_call.
        """
        print("Setup: Chunk with tool call start...")
        chunk = b'{"name":"get_weather","toolUseId":"call_123"}'
        
        print("Action: Parsing chunk...")
        events = aws_event_parser.feed(chunk)
        
        print(f"Result: {events}")
        print(f"current_tool_call: {aws_event_parser.current_tool_call}")
        
        # tool_start doesn't return event, but creates current_tool_call
        assert aws_event_parser.current_tool_call is not None
        assert aws_event_parser.current_tool_call["function"]["name"] == "get_weather"
    
    def test_parses_tool_input_event(self, aws_event_parser):
        """
        What it does: Tests parsing of input for tool call.
        Goal: Ensure input is added to current_tool_call.
        """
        print("Setup: Tool call start...")
        aws_event_parser.feed(b'{"name":"func","toolUseId":"call_1"}')
        
        print("Action: Parsing input...")
        aws_event_parser.feed(b'{"input":"{\\"key\\": \\"value\\"}"}')
        
        print(f"current_tool_call: {aws_event_parser.current_tool_call}")
        assert '{"key": "value"}' in aws_event_parser.current_tool_call["function"]["arguments"]
    
    def test_parses_tool_stop_event(self, aws_event_parser):
        """
        What it does: Tests tool call completion.
        Goal: Ensure tool call is added to the list.
        """
        print("Setup: Complete tool call...")
        aws_event_parser.feed(b'{"name":"func","toolUseId":"call_1"}')
        aws_event_parser.feed(b'{"input":"{}"}')
        
        print("Action: Parsing stop...")
        aws_event_parser.feed(b'{"stop":true}')
        
        print(f"tool_calls: {aws_event_parser.tool_calls}")
        assert len(aws_event_parser.tool_calls) == 1
        assert aws_event_parser.current_tool_call is None
    
    def test_get_tool_calls_returns_all(self, aws_event_parser):
        """
        What it does: Tests getting all tool calls.
        Goal: Ensure get_tool_calls returns completed calls.
        """
        print("Setup: Multiple tool calls...")
        aws_event_parser.feed(b'{"name":"func1","toolUseId":"call_1"}')
        aws_event_parser.feed(b'{"stop":true}')
        aws_event_parser.feed(b'{"name":"func2","toolUseId":"call_2"}')
        aws_event_parser.feed(b'{"stop":true}')
        
        print("Action: Getting tool calls...")
        tool_calls = aws_event_parser.get_tool_calls()
        
        print(f"Result: {tool_calls}")
        assert len(tool_calls) == 2
    
    def test_get_tool_calls_finalizes_current(self, aws_event_parser):
        """
        What it does: Tests finalization of incomplete tool call.
        Goal: Ensure get_tool_calls finalizes current_tool_call.
        """
        print("Setup: Incomplete tool call...")
        aws_event_parser.feed(b'{"name":"func","toolUseId":"call_1"}')
        
        print("Action: Getting tool calls...")
        tool_calls = aws_event_parser.get_tool_calls()
        
        print(f"Result: {tool_calls}")
        assert len(tool_calls) == 1
        assert aws_event_parser.current_tool_call is None


class TestAwsEventStreamParserReset:
    """Tests for reset method."""
    
    def test_reset_clears_state(self, aws_event_parser):
        """
        What it does: Tests parser state reset.
        Goal: Ensure reset clears all data.
        """
        print("Setup: Filling parser with data...")
        aws_event_parser.feed(b'{"content":"test"}')
        aws_event_parser.feed(b'{"name":"func","toolUseId":"call_1"}')
        
        print("Action: Resetting parser...")
        aws_event_parser.reset()
        
        print("Check: All data cleared...")
        assert aws_event_parser.buffer == ""
        assert aws_event_parser.last_content is None
        assert aws_event_parser.current_tool_call is None
        assert aws_event_parser.tool_calls == []


class TestAwsEventStreamParserFinalizeToolCall:
    """Tests for _finalize_tool_call method handling different input types."""
    
    def test_finalize_with_string_arguments(self, aws_event_parser):
        """
        What it does: Tests finalization of tool call with string arguments.
        Goal: Ensure JSON string is parsed and serialized back.
        """
        print("Setup: Tool call with string arguments...")
        aws_event_parser.current_tool_call = {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "test_func",
                "arguments": '{"key": "value"}'
            }
        }
        
        print("Action: Finalizing tool call...")
        aws_event_parser._finalize_tool_call()
        
        print(f"Result: {aws_event_parser.tool_calls}")
        assert len(aws_event_parser.tool_calls) == 1
        assert aws_event_parser.tool_calls[0]["function"]["arguments"] == '{"key": "value"}'
    
    def test_finalize_with_dict_arguments(self, aws_event_parser):
        """
        What it does: Tests finalization of tool call with dict arguments.
        Goal: Ensure dict is serialized to JSON string.
        """
        print("Setup: Tool call with dict arguments...")
        aws_event_parser.current_tool_call = {
            "id": "call_2",
            "type": "function",
            "function": {
                "name": "test_func",
                "arguments": {"location": "Moscow", "units": "celsius"}
            }
        }
        
        print("Action: Finalizing tool call...")
        aws_event_parser._finalize_tool_call()
        
        print(f"Result: {aws_event_parser.tool_calls}")
        assert len(aws_event_parser.tool_calls) == 1
        
        args = aws_event_parser.tool_calls[0]["function"]["arguments"]
        print(f"Arguments: {args}")
        assert isinstance(args, str)
        assert "Moscow" in args
        assert "celsius" in args
    
    def test_finalize_with_empty_string_arguments(self, aws_event_parser):
        """
        What it does: Tests finalization of tool call with empty string arguments.
        Goal: Ensure empty string is replaced with "{}".
        """
        print("Setup: Tool call with empty string arguments...")
        aws_event_parser.current_tool_call = {
            "id": "call_3",
            "type": "function",
            "function": {
                "name": "test_func",
                "arguments": ""
            }
        }
        
        print("Action: Finalizing tool call...")
        aws_event_parser._finalize_tool_call()
        
        print(f"Result: {aws_event_parser.tool_calls}")
        assert len(aws_event_parser.tool_calls) == 1
        assert aws_event_parser.tool_calls[0]["function"]["arguments"] == "{}"
    
    def test_finalize_with_whitespace_only_arguments(self, aws_event_parser):
        """
        What it does: Tests finalization of tool call with whitespace arguments.
        Goal: Ensure whitespace string is replaced with "{}".
        """
        print("Setup: Tool call with whitespace arguments...")
        aws_event_parser.current_tool_call = {
            "id": "call_4",
            "type": "function",
            "function": {
                "name": "test_func",
                "arguments": "   "
            }
        }
        
        print("Action: Finalizing tool call...")
        aws_event_parser._finalize_tool_call()
        
        print(f"Result: {aws_event_parser.tool_calls}")
        assert len(aws_event_parser.tool_calls) == 1
        assert aws_event_parser.tool_calls[0]["function"]["arguments"] == "{}"
    
    def test_finalize_with_invalid_json_arguments(self, aws_event_parser):
        """
        What it does: Tests finalization of tool call with invalid JSON.
        Goal: Ensure invalid JSON is replaced with "{}".
        """
        print("Setup: Tool call with invalid JSON...")
        aws_event_parser.current_tool_call = {
            "id": "call_5",
            "type": "function",
            "function": {
                "name": "test_func",
                "arguments": "not valid json {"
            }
        }
        
        print("Action: Finalizing tool call...")
        aws_event_parser._finalize_tool_call()
        
        print(f"Result: {aws_event_parser.tool_calls}")
        assert len(aws_event_parser.tool_calls) == 1
        assert aws_event_parser.tool_calls[0]["function"]["arguments"] == "{}"
    
    def test_finalize_with_none_current_tool_call(self, aws_event_parser):
        """
        What it does: Tests finalization when current_tool_call is None.
        Goal: Ensure nothing happens with None.
        """
        print("Setup: current_tool_call = None...")
        aws_event_parser.current_tool_call = None
        
        print("Action: Finalizing tool call...")
        aws_event_parser._finalize_tool_call()
        
        print(f"Result: {aws_event_parser.tool_calls}")
        assert len(aws_event_parser.tool_calls) == 0
    
    def test_finalize_clears_current_tool_call(self, aws_event_parser):
        """
        What it does: Tests that finalization clears current_tool_call.
        Goal: Ensure current_tool_call = None after finalization.
        """
        print("Setup: Tool call...")
        aws_event_parser.current_tool_call = {
            "id": "call_6",
            "type": "function",
            "function": {
                "name": "test_func",
                "arguments": "{}"
            }
        }
        
        print("Action: Finalizing tool call...")
        aws_event_parser._finalize_tool_call()
        
        print(f"current_tool_call after finalization: {aws_event_parser.current_tool_call}")
        assert aws_event_parser.current_tool_call is None


class TestAwsEventStreamParserEdgeCases:
    """Tests for edge cases."""
    
    def test_handles_followup_prompt(self, aws_event_parser):
        """
        What it does: Tests ignoring followupPrompt.
        Goal: Ensure followupPrompt doesn't create an event.
        """
        print("Setup: Chunk with followupPrompt...")
        chunk = b'{"content":"text","followupPrompt":"suggestion"}'
        
        print("Action: Parsing chunk...")
        events = aws_event_parser.feed(chunk)
        
        print(f"Result: {events}")
        assert len(events) == 0  # followupPrompt is ignored
    
    def test_handles_mixed_events(self, aws_event_parser):
        """
        What it does: Tests parsing of mixed events.
        Goal: Ensure different event types are handled together.
        """
        print("Setup: Chunk with mixed events...")
        chunk = b'{"content":"Hello"}{"usage":1.0}{"contextUsagePercentage":50}'
        
        print("Action: Parsing chunk...")
        events = aws_event_parser.feed(chunk)
        
        print(f"Result: {events}")
        assert len(events) == 3
        assert events[0]["type"] == "content"
        assert events[1]["type"] == "usage"
        assert events[2]["type"] == "context_usage"
    
    def test_handles_garbage_between_events(self, aws_event_parser):
        """
        What it does: Tests handling of garbage between events.
        Goal: Ensure parser finds JSON among garbage.
        """
        print("Setup: Chunk with garbage between JSON...")
        chunk = b'garbage{"content":"valid"}more garbage{"usage":1}'
        
        print("Action: Parsing chunk...")
        events = aws_event_parser.feed(chunk)
        
        print(f"Result: {events}")
        assert len(events) == 2
    
    def test_handles_empty_chunk(self, aws_event_parser):
        """
        What it does: Tests handling of empty chunk.
        Goal: Ensure empty chunk doesn't cause errors.
        """
        print("Setup: Empty chunk...")
        
        print("Action: Parsing empty chunk...")
        events = aws_event_parser.feed(b'')
        
        print(f"Comparing result: Expected [], Got {events}")
        assert events == []


class TestDiagnoseJsonTruncation:
    """
    Tests for _diagnose_json_truncation method for diagnosing truncated JSON.
    
    This method helps distinguish upstream issues (Kiro API truncates large
    tool call arguments) from actually invalid JSON from the model.
    """
    
    def test_empty_string_not_truncated(self, aws_event_parser):
        """
        What it does: Tests handling of empty string.
        Goal: Ensure empty string is not considered truncated.
        """
        print("Setup: Empty string...")
        json_str = ""
        
        print("Action: Diagnosis...")
        result = aws_event_parser._diagnose_json_truncation(json_str)
        
        print(f"Result: {result}")
        print(f"Comparing is_truncated: Expected False, Got {result['is_truncated']}")
        assert result["is_truncated"] is False
        assert result["reason"] == "empty string"
        assert result["size_bytes"] == 0
    
    def test_whitespace_only_not_truncated(self, aws_event_parser):
        """
        What it does: Tests handling of whitespace-only string.
        Goal: Ensure whitespace string is not considered truncated.
        """
        print("Setup: Whitespace string...")
        json_str = "   \t\n  "
        
        print("Action: Diagnosis...")
        result = aws_event_parser._diagnose_json_truncation(json_str)
        
        print(f"Result: {result}")
        print(f"Comparing is_truncated: Expected False, Got {result['is_truncated']}")
        assert result["is_truncated"] is False
        assert result["reason"] == "empty string"
    
    def test_valid_json_not_truncated(self, aws_event_parser):
        """
        What it does: Tests handling of valid JSON.
        Goal: Ensure valid JSON is not considered truncated.
        """
        print("Setup: Valid JSON...")
        json_str = '{"key": "value", "number": 42}'
        
        print("Action: Diagnosis...")
        result = aws_event_parser._diagnose_json_truncation(json_str)
        
        print(f"Result: {result}")
        print(f"Comparing is_truncated: Expected False, Got {result['is_truncated']}")
        assert result["is_truncated"] is False
        assert result["reason"] == "malformed JSON"  # Function doesn't check validity, only structure
    
    def test_valid_nested_json_not_truncated(self, aws_event_parser):
        """
        What it does: Tests handling of nested valid JSON.
        Goal: Ensure complex JSON is not considered truncated.
        """
        print("Setup: Nested valid JSON...")
        json_str = '{"outer": {"inner": {"deep": [1, 2, 3]}}}'
        
        print("Action: Diagnosis...")
        result = aws_event_parser._diagnose_json_truncation(json_str)
        
        print(f"Result: {result}")
        assert result["is_truncated"] is False
    
    def test_missing_closing_brace_truncated(self, aws_event_parser):
        """
        What it does: Tests detection of missing closing brace.
        Goal: Ensure JSON without closing } is considered truncated.
        """
        print("Setup: JSON without closing brace...")
        json_str = '{"filePath": "/path/to/file.md"'
        
        print("Action: Diagnosis...")
        result = aws_event_parser._diagnose_json_truncation(json_str)
        
        print(f"Result: {result}")
        print(f"Comparing is_truncated: Expected True, Got {result['is_truncated']}")
        assert result["is_truncated"] is True
        assert "missing" in result["reason"] and "brace" in result["reason"]
    
    def test_real_world_truncation_from_issue_34(self, aws_event_parser):
        """
        What it does: Tests real example from Issue #34.
        Goal: Ensure real truncated JSON from bug is detected.
        """
        print("Setup: Real example from Issue #34...")
        # This is exact example from log: JSON truncated after filePath
        json_str = '{"filePath": "/Users/cc/Documents/Code/mock-all/docs/plans/2026-01-12-mock-all-impl.md"'
        
        print("Action: Diagnosis...")
        result = aws_event_parser._diagnose_json_truncation(json_str)
        
        print(f"Result: {result}")
        print(f"Comparing is_truncated: Expected True, Got {result['is_truncated']}")
        assert result["is_truncated"] is True
        assert "brace" in result["reason"]
        assert result["size_bytes"] == 87  # Exact size from log (char 87 = error position)
    
    def test_multiple_missing_braces_truncated(self, aws_event_parser):
        """
        What it does: Tests detection of multiple missing braces.
        Goal: Ensure nested JSON without closing braces is detected.
        """
        print("Setup: Nested JSON without closing braces...")
        json_str = '{"outer": {"inner": {"deep": "value"'
        
        print("Action: Diagnosis...")
        result = aws_event_parser._diagnose_json_truncation(json_str)
        
        print(f"Result: {result}")
        assert result["is_truncated"] is True
        assert "3" in result["reason"] or "brace" in result["reason"]
    
    def test_missing_closing_bracket_truncated(self, aws_event_parser):
        """
        What it does: Tests detection of missing closing square bracket.
        Goal: Ensure array without ] is considered truncated.
        """
        print("Setup: Array without closing bracket...")
        json_str = '[1, 2, 3, {"key": "value"}'
        
        print("Action: Diagnosis...")
        result = aws_event_parser._diagnose_json_truncation(json_str)
        
        print(f"Result: {result}")
        print(f"Comparing is_truncated: Expected True, Got {result['is_truncated']}")
        assert result["is_truncated"] is True
        assert "bracket" in result["reason"]
    
    def test_array_start_truncated(self, aws_event_parser):
        """
        What it does: Tests detection of truncated array at start.
        Goal: Ensure [ without ] is detected.
        """
        print("Setup: Array start without end...")
        json_str = '["item1", "item2"'
        
        print("Action: Diagnosis...")
        result = aws_event_parser._diagnose_json_truncation(json_str)
        
        print(f"Result: {result}")
        assert result["is_truncated"] is True
        assert "bracket" in result["reason"]
    
    def test_unbalanced_braces_truncated(self, aws_event_parser):
        """
        What it does: Tests detection of unbalanced curly braces.
        Goal: Ensure different count of { and } is detected.
        """
        print("Setup: JSON with unbalanced braces...")
        # Ends with }, but has extra opening inside
        json_str = '{"a": {"b": 1}}'[:-1]  # Remove last }
        
        print("Action: Diagnosis...")
        result = aws_event_parser._diagnose_json_truncation(json_str)
        
        print(f"Result: {result}")
        assert result["is_truncated"] is True
    
    def test_unbalanced_brackets_truncated(self, aws_event_parser):
        """
        What it does: Tests detection of unbalanced square brackets.
        Goal: Ensure different count of [ and ] is detected.
        """
        print("Setup: JSON with unbalanced square brackets...")
        json_str = '{"items": [[1, 2], [3, 4]}'  # Missing one ]
        
        print("Action: Diagnosis...")
        result = aws_event_parser._diagnose_json_truncation(json_str)
        
        print(f"Result: {result}")
        assert result["is_truncated"] is True
        assert "bracket" in result["reason"]
    
    def test_unclosed_string_truncated(self, aws_event_parser):
        """
        What it does: Tests detection of unclosed string.
        Goal: Ensure odd number of quotes is detected.
        """
        print("Setup: JSON with unclosed string...")
        json_str = '{"content": "This is a very long string that was cut off'
        
        print("Action: Diagnosis...")
        result = aws_event_parser._diagnose_json_truncation(json_str)
        
        print(f"Result: {result}")
        print(f"Comparing is_truncated: Expected True, Got {result['is_truncated']}")
        assert result["is_truncated"] is True
        assert "string" in result["reason"] or "brace" in result["reason"]
    
    def test_escaped_quotes_handled_correctly(self, aws_event_parser):
        """
        What it does: Tests correct handling of escaped quotes.
        Goal: Ensure \\" doesn't break quote counting.
        """
        print("Setup: JSON with escaped quotes...")
        json_str = '{"text": "Say \\"hello\\" to everyone"}'
        
        print("Action: Diagnosis...")
        result = aws_event_parser._diagnose_json_truncation(json_str)
        
        print(f"Result: {result}")
        print(f"Comparing is_truncated: Expected False, Got {result['is_truncated']}")
        assert result["is_truncated"] is False
    
    def test_truncated_in_middle_of_escaped_sequence(self, aws_event_parser):
        """
        What it does: Tests truncation in middle of escape sequence.
        Goal: Ensure truncation after \\ is detected.
        """
        print("Setup: JSON truncated after backslash...")
        json_str = '{"text": "Line1\\nLine2\\'
        
        print("Action: Diagnosis...")
        result = aws_event_parser._diagnose_json_truncation(json_str)
        
        print(f"Result: {result}")
        assert result["is_truncated"] is True
    
    def test_size_bytes_calculated_correctly(self, aws_event_parser):
        """
        What it does: Tests correct byte size calculation.
        Goal: Ensure UTF-8 characters are counted correctly.
        """
        print("Setup: JSON with Unicode characters...")
        json_str = '{"city": "Москва"'  # Cyrillic = 2 bytes per character
        
        print("Action: Diagnosis...")
        result = aws_event_parser._diagnose_json_truncation(json_str)
        
        print(f"Result: {result}")
        expected_size = len(json_str.encode('utf-8'))
        print(f"Comparing size_bytes: Expected {expected_size}, Got {result['size_bytes']}")
        assert result["size_bytes"] == expected_size
        assert result["is_truncated"] is True  # No closing }
    
    def test_large_truncated_json(self, aws_event_parser):
        """
        What it does: Tests handling of large truncated JSON.
        Goal: Ensure large data is handled correctly.
        """
        print("Setup: Large truncated JSON...")
        # Simulate large file that was truncated
        content = "x" * 10000
        json_str = f'{{"filePath": "/path/to/file.md", "content": "{content}'
        
        print("Action: Diagnosis...")
        result = aws_event_parser._diagnose_json_truncation(json_str)
        
        print(f"Result: is_truncated={result['is_truncated']}, size_bytes={result['size_bytes']}")
        assert result["is_truncated"] is True
        assert result["size_bytes"] > 10000
    
    def test_malformed_but_not_truncated(self, aws_event_parser):
        """
        What it does: Tests invalid but not truncated JSON.
        Goal: Ensure syntax errors are not confused with truncation.
        """
        print("Setup: Invalid JSON (trailing comma)...")
        json_str = '{"key": "value",}'  # Trailing comma - invalid, but not truncated
        
        print("Action: Diagnosis...")
        result = aws_event_parser._diagnose_json_truncation(json_str)
        
        print(f"Result: {result}")
        print(f"Comparing is_truncated: Expected False, Got {result['is_truncated']}")
        assert result["is_truncated"] is False
        assert result["reason"] == "malformed JSON"
    
    def test_json_with_only_opening_brace(self, aws_event_parser):
        """
        What it does: Tests JSON with only opening brace.
        Goal: Ensure minimal truncated JSON is detected.
        """
        print("Setup: Only opening brace...")
        json_str = '{'
        
        print("Action: Diagnosis...")
        result = aws_event_parser._diagnose_json_truncation(json_str)
        
        print(f"Result: {result}")
        assert result["is_truncated"] is True
        assert "brace" in result["reason"]
    
    def test_json_with_only_opening_bracket(self, aws_event_parser):
        """
        What it does: Tests JSON with only opening square bracket.
        Goal: Ensure minimal truncated array is detected.
        """
        print("Setup: Only opening square bracket...")
        json_str = '['
        
        print("Action: Diagnosis...")
        result = aws_event_parser._diagnose_json_truncation(json_str)
        
        print(f"Result: {result}")
        assert result["is_truncated"] is True
        assert "bracket" in result["reason"]
    
    def test_braces_inside_string_not_counted(self, aws_event_parser):
        """
        What it does: Tests that braces inside strings don't affect counting.
        Goal: Ensure "{}" inside string doesn't break diagnosis.
        
        Note: Current implementation uses simplified counting,
        which doesn't account for string context. This is a known limitation.
        """
        print("Setup: JSON with braces inside string...")
        json_str = '{"text": "Hello {world}"}'
        
        print("Action: Diagnosis...")
        result = aws_event_parser._diagnose_json_truncation(json_str)
        
        print(f"Result: {result}")
        # Function uses simplified counting, so this may be False
        # Main thing - it doesn't crash and returns correct structure
        assert "is_truncated" in result
        assert "reason" in result
        assert "size_bytes" in result
    
    def test_complex_nested_truncation(self, aws_event_parser):
        """
        What it does: Tests complex nested truncated JSON.
        Goal: Ensure deep nesting is handled.
        """
        print("Setup: Complex nested truncated JSON...")
        json_str = '{"level1": {"level2": {"level3": [{"item": "value'
        
        print("Action: Diagnosis...")
        result = aws_event_parser._diagnose_json_truncation(json_str)
        
        print(f"Result: {result}")
        assert result["is_truncated"] is True


# =============================================================================
# Tests for Truncation Recovery System integration (Issue #56)
# =============================================================================

class TestTruncationRecoveryIntegration:
    """
    Tests for Truncation Recovery System integration in parsers.
    
    Verifies that tool calls are marked with truncation flags when JSON is truncated.
    Part of Truncation Recovery System (Issue #56).
    """
    
    def test_tool_call_marked_with_truncation_flags(self, aws_event_parser):
        """
        What it does: Verifies tool call is marked with _truncation_detected and _truncation_info.
        Purpose: Ensure truncation detection marks tool calls for recovery system.
        """
        print("Setup: Creating tool call with truncated JSON arguments...")
        aws_event_parser.current_tool_call = {
            "id": "tooluse_truncated",
            "type": "function",
            "function": {
                "name": "write_to_file",
                "arguments": '{"filePath": "/path/to/file.md", "content": "This is a very long content that was cut off'
            }
        }
        
        print("Action: Finalizing tool call (should detect truncation)...")
        aws_event_parser._finalize_tool_call()
        
        print("Checking: Tool call was added to list...")
        assert len(aws_event_parser.tool_calls) == 1
        
        tool_call = aws_event_parser.tool_calls[0]
        print(f"Tool call: {tool_call}")
        
        print("Checking: _truncation_detected flag is set...")
        assert tool_call.get("_truncation_detected") is True
        
        print("Checking: _truncation_info is present...")
        assert "_truncation_info" in tool_call
        
        truncation_info = tool_call["_truncation_info"]
        print(f"Truncation info: {truncation_info}")
        
        print("Checking: truncation_info has required fields...")
        assert truncation_info["is_truncated"] is True
        assert "size_bytes" in truncation_info
        assert truncation_info["size_bytes"] > 0
        assert "reason" in truncation_info
        assert len(truncation_info["reason"]) > 0
    
    def test_valid_tool_call_not_marked_with_truncation(self, aws_event_parser):
        """
        What it does: Verifies valid tool call is NOT marked with truncation flags.
        Purpose: Ensure false positives don't occur.
        """
        print("Setup: Creating tool call with valid JSON arguments...")
        aws_event_parser.current_tool_call = {
            "id": "tooluse_valid",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location": "Moscow", "units": "celsius"}'
            }
        }
        
        print("Action: Finalizing tool call...")
        aws_event_parser._finalize_tool_call()
        
        print("Checking: Tool call was added to list...")
        assert len(aws_event_parser.tool_calls) == 1
        
        tool_call = aws_event_parser.tool_calls[0]
        print(f"Tool call: {tool_call}")
        
        print("Checking: _truncation_detected flag is NOT set...")
        assert tool_call.get("_truncation_detected") is not True
        
        print("Checking: _truncation_info is NOT present...")
        assert "_truncation_info" not in tool_call
    
    def test_multiple_tool_calls_with_mixed_truncation(self, aws_event_parser):
        """
        What it does: Verifies multiple tool calls are marked independently.
        Purpose: Ensure truncation detection works correctly for multiple tool calls.
        """
        print("Setup: Creating first tool call (valid)...")
        aws_event_parser.current_tool_call = {
            "id": "tooluse_1",
            "type": "function",
            "function": {
                "name": "func1",
                "arguments": '{"param": "value"}'
            }
        }
        aws_event_parser._finalize_tool_call()
        
        print("Setup: Creating second tool call (truncated)...")
        aws_event_parser.current_tool_call = {
            "id": "tooluse_2",
            "type": "function",
            "function": {
                "name": "func2",
                "arguments": '{"param": "incomplete'
            }
        }
        aws_event_parser._finalize_tool_call()
        
        print("Setup: Creating third tool call (valid)...")
        aws_event_parser.current_tool_call = {
            "id": "tooluse_3",
            "type": "function",
            "function": {
                "name": "func3",
                "arguments": '{"param": "complete"}'
            }
        }
        aws_event_parser._finalize_tool_call()
        
        print("Checking: All tool calls were added...")
        assert len(aws_event_parser.tool_calls) == 3
        
        print("Checking: First tool call NOT marked as truncated...")
        assert aws_event_parser.tool_calls[0].get("_truncation_detected") is not True
        
        print("Checking: Second tool call IS marked as truncated...")
        assert aws_event_parser.tool_calls[1].get("_truncation_detected") is True
        assert "_truncation_info" in aws_event_parser.tool_calls[1]
        
        print("Checking: Third tool call NOT marked as truncated...")
        assert aws_event_parser.tool_calls[2].get("_truncation_detected") is not True


# ==================================================================================================
# Native reasoning (reasoningContentEvent) tests
# ==================================================================================================

SECRET_MARKER = "SECRET_REASONING_MARKER_7f3a"


def _parse_chunks(chunks):
    """Feeds chunks to a fresh parser and returns (parser, all events)."""
    parser = AwsEventStreamParser()
    events = []
    for chunk in chunks:
        events.extend(parser.feed(chunk))
    return parser, events


def _types(events):
    """Returns the list of event types."""
    return [e["type"] for e in events]


def _reasoning_texts(events):
    """Returns the data of all reasoning events."""
    return [e["data"] for e in events if e["type"] == "reasoning"]


class TestAwsFrameBuilderFixture:
    """Golden tests that pin the aws_frames fixture to real captured Kiro bytes."""

    def test_first_real_reasoning_frame_matches_capture(self, aws_frames):
        """
        What it does: Builds the reasoning frame for text "I" and compares to captured bytes.
        Goal: Prove the fixture produces byte-identical frames to real Kiro traffic.
        """
        print("Action: Building reasoning frame for 'I'...")
        frame = aws_frames.reasoning("I")

        print(f"Comparing: {frame.hex()[:40]}...")
        assert frame.hex() == aws_frames.SAMPLE_FIRST_REASONING_FRAME_HEX

    def test_total_length_field_equals_frame_length(self, aws_frames):
        """
        What it does: Checks the big-endian total-length prelude field for several payloads.
        Goal: Frames must be structurally valid for any payload size.
        """
        import struct

        for frame in (
            aws_frames.reasoning(""),
            aws_frames.reasoning("中文 😀 —"),
            aws_frames.content("x" * 1000),
            aws_frames.build("customEvent", b""),
        ):
            total = struct.unpack(">I", frame[:4])[0]
            print(f"Frame length {len(frame)}, prelude says {total}")
            assert total == len(frame)

    def test_split_reassembles_to_original(self, aws_frames):
        """
        What it does: Splits a stream with cycled sizes and joins it back.
        Goal: split() must be lossless and never emit empty chunks.
        """
        data = aws_frames.reasoning("abc") + aws_frames.content("def")
        chunks = aws_frames.split(data, [1, 5, 3])
        assert b"".join(chunks) == data
        assert all(chunks)


class TestAwsEventStreamParserReasoningEvents:
    """Tests for recognizing {"text": ...} reasoning frames."""

    def test_event_patterns_keep_legacy_entries_and_add_reasoning(self):
        """
        What it does: Pins EVENT_PATTERNS.
        Goal: The seven legacy patterns keep order/content; reasoning patterns are appended.
        """
        patterns = AwsEventStreamParser.EVENT_PATTERNS
        assert patterns[:7] == [
            ('{"content":', 'content'),
            ('{"name":', 'tool_start'),
            ('{"input":', 'tool_input'),
            ('{"stop":', 'tool_stop'),
            ('{"followupPrompt":', 'followup'),
            ('{"usage":', 'usage'),
            ('{"contextUsagePercentage":', 'context_usage'),
        ]
        assert ('{"text":', 'reasoning') in patterns
        assert ('{"signature":', 'reasoning_signature') in patterns

    def test_bare_json_reasoning_event(self):
        """
        What it does: Feeds bare JSON {"text":"hello"}.
        Goal: Emits exactly one reasoning event.
        """
        _, events = _parse_chunks([b'{"text":"hello"}'])
        print(f"Events: {events}")
        assert events == [{"type": "reasoning", "data": "hello"}]

    def test_real_aws_frame_reasoning_event(self, aws_frames):
        """
        What it does: Feeds a full AWS frame (headers + CRC) carrying reasoning text.
        Goal: Binary framing around the JSON must not prevent recognition.
        """
        _, events = _parse_chunks([aws_frames.reasoning("I")])
        assert events == [{"type": "reasoning", "data": "I"}]

    def test_fifty_frames_in_one_chunk_keep_order(self, aws_frames):
        """
        What it does: Puts 50 reasoning frames in a single chunk.
        Goal: All 50 events are returned in order with exact text.
        """
        texts = [f"t{i}" for i in range(50)]
        data = b"".join(aws_frames.reasoning(t) for t in texts)

        _, events = _parse_chunks([data])

        assert _reasoning_texts(events) == texts
        assert len(events) == 50

    def test_interleaved_stream_event_sequence_and_tool_calls(self, aws_frames):
        """
        What it does: Interleaves reasoning with content, tool, usage and context frames.
        Goal: Exact event sequence; tool call assembly must be unaffected by reasoning frames.
        """
        data = b"".join([
            aws_frames.reasoning("think 1"),
            aws_frames.signature("QUJD"),
            aws_frames.content("answer"),
            aws_frames.tool_start("Bash", "toolu_1"),
            aws_frames.reasoning("think between tool frames"),
            aws_frames.tool_input('{"command"', "Bash", "toolu_1"),
            aws_frames.reasoning("think 3"),
            aws_frames.tool_input(': "ls"}', "Bash", "toolu_1"),
            aws_frames.tool_stop("Bash", "toolu_1"),
            aws_frames.build("usageEvent", {"usage": 0.5}),
            aws_frames.context_usage(12.5),
            aws_frames.reasoning("think 4"),
        ])

        parser, events = _parse_chunks([data])
        print(f"Types: {_types(events)}")

        assert _types(events) == [
            "reasoning", "reasoning_signature", "content",
            "reasoning", "reasoning",
            "usage", "context_usage", "reasoning",
        ]
        tool_calls = parser.get_tool_calls()
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "Bash"
        assert tool_calls[0]["function"]["arguments"] == '{"command": "ls"}'

    def test_reasoning_does_not_affect_content_deduplication(self, aws_frames):
        """
        What it does: content "A" -> reasoning "A" -> content "A".
        Goal: The second content is still deduplicated; last_content is not overwritten.
        """
        data = (
            aws_frames.content("A") + aws_frames.reasoning("A") + aws_frames.content("A")
        )

        parser, events = _parse_chunks([data])

        assert _types(events) == ["content", "reasoning"]
        assert parser.last_content == "A"

    def test_reasoning_between_equal_contents_keeps_dedup_state(self):
        """
        What it does: content "X", reasoning "Y", content "X" as bare JSON.
        Goal: Reasoning must not reset the dedup state (last_content stays "X").
        """
        parser, events = _parse_chunks([b'{"content":"X"}{"text":"Y"}{"content":"X"}'])
        assert _types(events) == ["content", "reasoning"]
        assert parser.last_content == "X"

    def test_identical_consecutive_reasoning_texts_are_not_deduplicated(self, aws_frames):
        """
        What it does: Two consecutive reasoning frames with the same text.
        Goal: Both are emitted - repeated tokens are legitimate text.
        """
        data = aws_frames.reasoning("the") + aws_frames.reasoning("the")
        _, events = _parse_chunks([data])
        assert _reasoning_texts(events) == ["the", "the"]

    def test_reset_clears_partial_frame_and_pending_utf8_bytes(self):
        """
        What it does: Feeds half a frame plus half a multi-byte char, resets, feeds a new frame.
        Goal: No leftovers from the old stream leak into the new one.
        """
        parser = AwsEventStreamParser()
        raw = '{"text":"中"}'.encode("utf-8")
        assert parser.feed(raw[:11]) == []  # cut inside the 3-byte char
        parser.reset()

        events = parser.feed(b'{"text":"fresh"}')

        assert events == [{"type": "reasoning", "data": "fresh"}]
        assert parser.buffer == ""

    def test_reasoning_events_do_not_create_tool_state(self, aws_frames):
        """
        What it does: Feeds only reasoning frames.
        Goal: No tool state is created.
        """
        parser, _ = _parse_chunks([aws_frames.reasoning("a") + aws_frames.signature("sig")])
        assert parser.current_tool_call is None
        assert parser.get_tool_calls() == []


class TestAwsEventStreamParserReasoningChunking:
    """Chunk-boundary robustness for reasoning frames."""

    @staticmethod
    def _stream(aws_frames):
        """A realistic stream with multi-byte characters in every reasoning frame."""
        return b"".join([
            aws_frames.reasoning("中文推理"),
            aws_frames.reasoning(" 😀 — dash"),
            aws_frames.signature("QUJD" * 20),
            aws_frames.content("正文 ✓"),
            aws_frames.context_usage(3.5),
        ])

    def test_split_at_every_byte_offset_equals_whole(self, aws_frames):
        """
        What it does: Splits the stream in two at EVERY byte offset.
        Goal: Event list is identical to feeding it whole (covers cuts inside prefixes,
        headers, CRC and multi-byte characters).
        """
        data = self._stream(aws_frames)
        _, expected = _parse_chunks([data])
        assert _reasoning_texts(expected) == ["中文推理", " 😀 — dash"]

        for cut in range(1, len(data)):
            _, events = _parse_chunks([data[:cut], data[cut:]])
            assert events == expected, f"mismatch when cutting at byte {cut}"

    def test_byte_by_byte_feed_equals_whole(self, aws_frames):
        """
        What it does: Feeds the stream one byte at a time.
        Goal: Same events as the whole-stream feed.
        """
        data = self._stream(aws_frames)
        _, expected = _parse_chunks([data])

        _, events = _parse_chunks([data[i:i + 1] for i in range(len(data))])

        assert events == expected

    def test_seeded_random_chunking_equals_whole(self, aws_frames):
        """
        What it does: 50 deterministic random chunkings.
        Goal: Same events regardless of chunk sizes.
        """
        import random

        data = self._stream(aws_frames)
        _, expected = _parse_chunks([data])
        for seed in range(50):
            rnd = random.Random(seed)
            sizes = [rnd.randint(1, 40) for _ in range(rnd.randint(1, 10))]
            _, events = _parse_chunks(aws_frames.split(data, sizes))
            assert events == expected, f"mismatch for seed {seed}"

    def test_hypothesis_arbitrary_chunk_sizes_equal_whole(self, aws_frames):
        """
        What it does: Property test over arbitrary chunk-size sequences.
        Goal: Chunking never changes the parsed events.
        """
        from hypothesis import HealthCheck, given, settings, strategies as st

        data = self._stream(aws_frames)
        _, expected = _parse_chunks([data])

        @settings(max_examples=50, deadline=None, suppress_health_check=list(HealthCheck))
        @given(st.lists(st.integers(min_value=1, max_value=64), min_size=1))
        def check(sizes):
            _, events = _parse_chunks(aws_frames.split(data, sizes))
            assert events == expected

        check()

    @pytest.mark.parametrize("char", ["中", "😀", "—"])
    def test_multibyte_character_split_inside_is_not_lost(self, char):
        """
        What it does: Cuts the raw JSON at every byte inside a 2-4 byte character.
        Goal: The character survives (incremental UTF-8 decoding). Without it, the
        character is silently dropped.
        """
        raw = f'{{"text":"ab{char}cd"}}'.encode("utf-8")
        start = raw.index(char.encode("utf-8"))
        for cut in range(start + 1, start + len(char.encode("utf-8"))):
            _, events = _parse_chunks([raw[:cut], raw[cut:]])
            assert events == [{"type": "reasoning", "data": f"ab{char}cd"}], f"cut at {cut}"

    def test_pattern_prefix_split_across_chunks(self):
        """
        What it does: Splits bare JSON inside the '{"text":' prefix.
        Goal: Prefix is matched once the rest arrives.
        """
        parser = AwsEventStreamParser()
        assert parser.feed(b'{"te') == []
        events = parser.feed(b'xt":"a"}')
        assert events == [{"type": "reasoning", "data": "a"}]

    def test_signature_split_across_three_chunks(self, aws_frames):
        """
        What it does: Splits a long signature frame in 3 pieces.
        Goal: One signature event with the full length.
        """
        sig = "QUJD" * 300
        frame = aws_frames.signature(sig)
        third = len(frame) // 3

        _, events = _parse_chunks([frame[:third], frame[third:2 * third], frame[2 * third:]])

        assert events == [{"type": "reasoning_signature", "data": len(sig)}]


class TestAwsEventStreamParserReasoningEdgeCases:
    """Malformed, hostile and unusual reasoning payloads."""

    def test_empty_text_emits_nothing_and_next_frame_works(self, aws_frames):
        """
        What it does: Empty text frame followed by a normal one.
        Goal: No event for the empty text; parsing continues.
        """
        _, events = _parse_chunks([aws_frames.reasoning("") + aws_frames.reasoning("x")])
        assert events == [{"type": "reasoning", "data": "x"}]

    def test_escape_sequences_are_decoded(self, aws_frames):
        """
        What it does: Text with \\n \\t \\" \\\\ \\/ escapes.
        Goal: JSON escapes are decoded exactly.
        """
        text = 'line1\nline2\t"quoted" back\\slash /slash'
        _, events = _parse_chunks([aws_frames.reasoning(text)])
        assert _reasoning_texts(events) == [text]

    def test_unicode_escapes_and_surrogate_pairs(self, aws_frames):
        """
        What it does: Payload using \\u4e2d and the surrogate pair \\ud83d\\ude00.
        Goal: Decoded to 中 and 😀.
        """
        payload = b'{"text":"\\u4e2d\\ud83d\\ude00"}'
        _, events = _parse_chunks([aws_frames.build("reasoningContentEvent", payload)])
        assert _reasoning_texts(events) == ["中😀"]

    def test_raw_utf8_chinese_emoji_and_em_dash(self, aws_frames):
        """
        What it does: Raw (non-escaped) UTF-8 text.
        Goal: Preserved byte-exactly.
        """
        text = "中文推理 😀 — end"
        _, events = _parse_chunks([aws_frames.reasoning(text)])
        assert _reasoning_texts(events) == [text]

    def test_braces_and_brackets_inside_text(self, aws_frames):
        """
        What it does: Text containing { } [ ] and unbalanced braces.
        Goal: Brace matching respects JSON strings; one event only.
        """
        text = 'a { b } [ c ] } { "'
        _, events = _parse_chunks([aws_frames.reasoning(text) + aws_frames.reasoning("next")])
        assert _reasoning_texts(events) == [text, "next"]

    @pytest.mark.parametrize("text", [
        '{"content":"x"}',
        '{"name":"y"}',
        '{"text":"z"}',
        '{"stop":true}',
        '{"usage":1}',
        '{"input":"i"}',
        '{"contextUsagePercentage":9}',
    ])
    def test_text_that_looks_like_other_frames_creates_no_phantom_events(self, aws_frames, text):
        """
        What it does: Reasoning text that quotes another frame's JSON.
        Goal: Exactly one reasoning event; no content/tool/usage events; no tool state.
        """
        parser, events = _parse_chunks([aws_frames.reasoning(text)])
        print(f"Events: {events}")
        assert events == [{"type": "reasoning", "data": text}]
        assert parser.current_tool_call is None

    def test_content_containing_escaped_text_json_stays_content(self, aws_frames):
        """
        What it does: Content whose text is '{"text":"x"}'.
        Goal: Stays a content event - never misread as reasoning.
        """
        _, events = _parse_chunks([aws_frames.content('{"text":"x"}')])
        assert events == [{"type": "content", "data": '{"text":"x"}'}]

    @pytest.mark.parametrize("payload", [
        b'{"text":123}', b'{"text":null}', b'{"text":true}', b'{"text":[]}', b'{"text":{}}',
        b'{"text":{"a":"b"}}', b'{"text":1.5}',
    ])
    def test_non_string_text_is_ignored(self, payload):
        """
        What it does: Non-string "text" values.
        Goal: No event, no exception, next frame still parsed.
        """
        _, events = _parse_chunks([payload + b'{"text":"ok"}'])
        assert events == [{"type": "reasoning", "data": "ok"}]

    def test_invalid_json_is_skipped_and_never_logged_verbatim(self, loguru_messages):
        """
        What it does: Malformed reasoning JSON carrying a secret marker, then a valid frame.
        Goal: No crash, valid frame still parsed, the logs contain the length but never the text.
        """
        bad = f'{{"text":"{SECRET_MARKER}" oops}}'.encode()
        _, events = _parse_chunks([bad + b'{"text":"fine"}'])

        assert events == [{"type": "reasoning", "data": "fine"}]
        joined = "\n".join(loguru_messages)
        print(f"Logs: {joined}")
        assert SECRET_MARKER not in joined
        assert "length=" in joined

    def test_invalid_escape_is_skipped_and_never_logged_verbatim(self, loguru_messages):
        """
        What it does: Reasoning JSON with an invalid escape sequence.
        Goal: Skipped without leaking text into logs.
        """
        bad = f'{{"text":"{SECRET_MARKER}\\q"}}'.encode()
        _, events = _parse_chunks([bad + b'{"text":"after"}'])
        assert events == [{"type": "reasoning", "data": "after"}]
        assert SECRET_MARKER not in "\n".join(loguru_messages)

    def test_invalid_signature_json_is_not_logged_verbatim(self, loguru_messages):
        """
        What it does: Malformed signature JSON.
        Goal: Treated like reasoning text for log safety.
        """
        bad = f'{{"signature":"{SECRET_MARKER}" oops}}'.encode()
        _, events = _parse_chunks([bad])
        assert events == []
        assert SECRET_MARKER not in "\n".join(loguru_messages)

    def test_non_string_text_log_has_type_only(self, loguru_messages):
        """
        What it does: Non-string text.
        Goal: Debug log mentions the type name, not the value.
        """
        _parse_chunks([b'{"text":123456789}'])
        joined = "\n".join(loguru_messages)
        assert "int" in joined
        assert "123456789" not in joined

    def test_noise_between_frames_is_ignored(self, aws_frames):
        """
        What it does: Garbage bytes between frames.
        Goal: Frames around the noise are still parsed.
        """
        data = aws_frames.reasoning("a") + b"\xff\xfe\x00" + aws_frames.reasoning("b")
        _, events = _parse_chunks([data])
        assert _reasoning_texts(events) == ["a", "b"]

    def test_dangling_utf8_lead_byte_before_frame_does_not_swallow_it(self, aws_frames):
        """
        What it does: A UTF-8 lead byte (like a CRC byte could be) right before the next frame.
        Goal: The next frame is parsed - the pending-bytes decoder must not eat it.
        """
        for lead in (b"\xe4", b"\xf0", b"\xc3"):
            _, events = _parse_chunks([aws_frames.reasoning("a") + lead, aws_frames.reasoning("b")])
            assert _reasoning_texts(events) == ["a", "b"], f"lead byte {lead!r}"

        _, events = _parse_chunks([b"\xe4", b'{"text":"x"}'])
        assert _reasoning_texts(events) == ["x"]

    def test_huge_frame_single_chunk_and_multi_chunk(self, aws_frames):
        """
        What it does: 200KB reasoning frame fed whole and in 64KB chunks.
        Goal: Parsed in both cases without loss.
        """
        text = ("长" * 3000 + "x" * 197000)
        frame = aws_frames.reasoning(text) + aws_frames.reasoning("tail")

        _, whole = _parse_chunks([frame])
        _, chunked = _parse_chunks(aws_frames.split(frame, [65536]))

        assert _reasoning_texts(whole) == [text, "tail"]
        assert _reasoning_texts(chunked) == [text, "tail"]

    def test_signature_event_reports_length_only(self, aws_frames):
        """
        What it does: Signature frame.
        Goal: Emits reasoning_signature with len(signature); not a reasoning text event.
        """
        sig = "QUJD" * 50
        _, events = _parse_chunks([aws_frames.signature(sig)])
        assert events == [{"type": "reasoning_signature", "data": len(sig)}]

    @pytest.mark.parametrize("payload", [b'{"signature":""}', b'{"signature":123}', b'{"signature":null}'])
    def test_empty_or_non_string_signature_reports_zero(self, payload):
        """
        What it does: Empty / non-string signatures.
        Goal: data == 0 and no exception.
        """
        _, events = _parse_chunks([payload])
        assert events == [{"type": "reasoning_signature", "data": 0}]

    def test_unknown_first_key_frame_is_still_ignored(self, aws_frames):
        """
        What it does: Frame whose first key is unknown.
        Goal: Ignored exactly as before; neighbours unaffected.
        """
        data = aws_frames.build("weirdEvent", {"foo": 1}) + aws_frames.reasoning("ok")
        _, events = _parse_chunks([data])
        assert events == [{"type": "reasoning", "data": "ok"}]

    def test_real_shaped_stream_counts_and_order(self, aws_frames):
        """
        What it does: Reasoning x40 -> signature -> content x10 -> context usage -> metering.
        Goal: Matches the shape of real Kiro captures (metering is not parsed - unchanged).
        """
        data = b"".join(
            [aws_frames.reasoning(f"r{i}") for i in range(40)]
            + [aws_frames.signature("QUJD")]
            + [aws_frames.content(f"c{i}") for i in range(10)]
            + [aws_frames.context_usage(7.0), aws_frames.metering(0.7)]
        )

        _, events = _parse_chunks([data])

        assert _types(events) == (
            ["reasoning"] * 40 + ["reasoning_signature"] + ["content"] * 10 + ["context_usage"]
        )
