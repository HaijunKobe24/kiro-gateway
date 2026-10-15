# -*- coding: utf-8 -*-

"""
Unit tests for kiro/empty_response.py.

Covers the client-facing error built for an empty upstream response: status code,
Anthropic / OpenAI error body shapes, retry hint header, WARNING log content (and the
absence of request content), and the request-context helper.
"""

import json
from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock, patch

import pytest

from kiro.empty_response import (
    EMPTY_RESPONSE_MESSAGE,
    INCOMPLETE_RESPONSE_MESSAGE,
    EmptyResponseContext,
    build_empty_response_context,
    build_incomplete_stream_error_anthropic,
    build_incomplete_stream_error_openai,
    handle_empty_response_anthropic,
    handle_empty_response_openai,
    log_incomplete_stream,
)
from kiro.upstream_guard import (
    KIND_EMPTY,
    KIND_INCOMPLETE,
    EmptyUpstreamResponseError,
    StreamOutcome,
)


def make_error() -> EmptyUpstreamResponseError:
    """The error raised for the real failure shape."""
    return EmptyUpstreamResponseError(147, {"contextUsageEvent": 1})


def make_ctx(**overrides) -> EmptyResponseContext:
    """Context with sensible defaults."""
    values = dict(model="claude-sonnet-4.5", stream=True, request_bytes=1234, input_tokens_estimate=321)
    values.update(overrides)
    return EmptyResponseContext(**values)


def body_of(response) -> dict:
    """Decoded JSON body of a JSONResponse."""
    return json.loads(response.body)


class TestAnthropicErrorResponse:
    """handle_empty_response_anthropic()."""

    def test_default_status_503_with_api_error_body(self):
        """
        What it does: Builds the Anthropic response with the default configuration.
        Purpose: HTTP 503 and the standard Anthropic error envelope.
        """
        response = handle_empty_response_anthropic(make_error(), make_ctx())

        print(body_of(response))
        assert response.status_code == 503
        assert body_of(response) == {
            "type": "error",
            "error": {"type": "api_error", "message": EMPTY_RESPONSE_MESSAGE},
        }

    def test_should_retry_header(self):
        """
        What it does: Reads the response headers.
        Purpose: x-should-retry: true is a hint for SDK clients.
        """
        response = handle_empty_response_anthropic(make_error(), make_ctx())

        assert response.headers["x-should-retry"] == "true"

    def test_status_529_uses_overloaded_error_type(self):
        """
        What it does: Configures status 529.
        Purpose: Only the Anthropic overload status maps to overloaded_error.
        """
        with patch("kiro.empty_response.EMPTY_RESPONSE_ERROR_STATUS", 529):
            response = handle_empty_response_anthropic(make_error(), make_ctx())

        assert response.status_code == 529
        assert body_of(response)["error"]["type"] == "overloaded_error"

    @pytest.mark.parametrize("status", [500, 502, 503, 504, 599])
    def test_other_statuses_use_api_error_type(self, status):
        """
        What it does: Configures other 5xx statuses.
        Purpose: Status follows the configuration; the type stays api_error.
        """
        with patch("kiro.empty_response.EMPTY_RESPONSE_ERROR_STATUS", status):
            response = handle_empty_response_anthropic(make_error(), make_ctx())

        assert response.status_code == status
        assert body_of(response)["error"]["type"] == "api_error"


class TestOpenAIErrorResponse:
    """handle_empty_response_openai()."""

    def test_default_status_503_with_server_error_body(self):
        """
        What it does: Builds the OpenAI response with the default configuration.
        Purpose: HTTP 503 and the standard OpenAI error envelope with a stable code.
        """
        response = handle_empty_response_openai(make_error(), make_ctx())

        print(body_of(response))
        assert response.status_code == 503
        assert body_of(response) == {
            "error": {
                "message": EMPTY_RESPONSE_MESSAGE,
                "type": "server_error",
                "param": None,
                "code": "upstream_empty_response",
            }
        }
        assert response.headers["x-should-retry"] == "true"

    def test_status_529_keeps_server_error_type(self):
        """
        What it does: Configures status 529 for the OpenAI body.
        Purpose: OpenAI has no overloaded type: it stays server_error.
        """
        with patch("kiro.empty_response.EMPTY_RESPONSE_ERROR_STATUS", 529):
            response = handle_empty_response_openai(make_error(), make_ctx())

        assert response.status_code == 529
        assert body_of(response)["error"]["type"] == "server_error"


class TestMessageContent:
    """The user-facing message."""

    def test_message_is_actionable(self):
        """
        What it does: Checks the message text.
        Purpose: Says what happened and what to do (retry).
        """
        assert "empty response" in EMPTY_RESPONSE_MESSAGE
        assert "retry" in EMPTY_RESPONSE_MESSAGE

    @pytest.mark.parametrize("forbidden", ["overloaded", "rate limit", "capacity", "quota", "throttl"])
    def test_message_does_not_claim_a_cause(self, forbidden):
        """
        What it does: Scans the message for cause claims.
        Purpose: The cause of empty upstream responses is unknown: never state one.
        """
        assert forbidden not in EMPTY_RESPONSE_MESSAGE.lower()


class TestWarningLog:
    """WARNING log per detected empty response."""

    def test_warning_contains_non_sensitive_facts(self, loguru_messages):
        """
        What it does: Handles an empty response and reads the log.
        Purpose: model, stream, input token estimate, request size, upstream bytes, frames, status.
        """
        handle_empty_response_anthropic(make_error(), make_ctx())

        lines = [m for m in loguru_messages if "[EmptyUpstreamResponse]" in m]
        print(lines)
        assert len(lines) == 1
        line = lines[0]
        assert line.startswith("WARNING|")
        assert "model=claude-sonnet-4.5" in line
        assert "stream=True" in line
        assert "input_tokens_estimate=321" in line
        assert "request_bytes=1234" in line
        assert "upstream_bytes=147" in line
        assert "contextUsageEvent" in line
        assert "status=503" in line

    def test_access_log_line_for_streaming_and_non_streaming(self, loguru_messages):
        """
        What it does: Handles one streaming and one non-streaming request.
        Purpose: Access log style line names the endpoint and the mode.
        """
        handle_empty_response_anthropic(make_error(), make_ctx(stream=True))
        handle_empty_response_openai(make_error(), make_ctx(stream=False))

        text = "\n".join(loguru_messages)
        assert "HTTP 503 - POST /v1/messages (streaming)" in text
        assert "HTTP 503 - POST /v1/chat/completions (non-streaming)" in text

    def test_unknown_values_are_logged_as_unknown(self, loguru_messages):
        """
        What it does: Context without request size / token estimate.
        Purpose: Missing facts must not break logging.
        """
        handle_empty_response_openai(make_error(), make_ctx(request_bytes=None, input_tokens_estimate=None))

        line = next(m for m in loguru_messages if "[EmptyUpstreamResponse]" in m)
        assert "input_tokens_estimate=unknown" in line
        assert "request_bytes=unknown" in line

    def test_log_never_contains_message_content(self, loguru_messages):
        """
        What it does: Builds the context from a request whose text contains a unique secret,
        then handles the error.
        Purpose: No request content may reach the logs.
        """
        secret = "TOP_SECRET_USER_TEXT_9f2c"
        messages = [{"role": "user", "content": secret}]
        payload = {"conversationState": {"currentMessage": {"userInputMessage": {"content": secret}}}}

        ctx = build_empty_response_context("claude-sonnet-4.5", False, payload, messages, None)
        handle_empty_response_anthropic(make_error(), ctx)
        handle_empty_response_openai(make_error(), ctx)

        assert not any(secret in m for m in loguru_messages)

    def test_flushes_debug_logger_with_status(self):
        """
        What it does: Replaces debug_logger with a mock.
        Purpose: DEBUG_MODE=errors must save the logs of this failed request.
        """
        fake_logger = MagicMock()
        with patch("kiro.empty_response.debug_logger", fake_logger):
            handle_empty_response_anthropic(make_error(), make_ctx())

        fake_logger.flush_on_error.assert_called_once_with(503, EMPTY_RESPONSE_MESSAGE)

    def test_works_without_debug_logger(self):
        """
        What it does: debug_logger is None.
        Purpose: The error response is still produced.
        """
        with patch("kiro.empty_response.debug_logger", None):
            response = handle_empty_response_openai(make_error(), make_ctx())

        assert response.status_code == 503


class TestBuildEmptyResponseContext:
    """build_empty_response_context()."""

    def test_measures_request_bytes_and_tokens(self):
        """
        What it does: Builds the context from a small request.
        Purpose: Both facts are populated for a normal request.
        """
        payload = {"a": "b"}
        messages = [{"role": "user", "content": "hello there"}]

        ctx = build_empty_response_context("m", True, payload, messages, None, "system prompt")

        assert ctx.model == "m"
        assert ctx.stream is True
        assert ctx.request_bytes == len(json.dumps(payload).encode("utf-8"))
        assert isinstance(ctx.input_tokens_estimate, int) and ctx.input_tokens_estimate > 0

    def test_unserializable_payload_gives_none_bytes(self):
        """
        What it does: Payload with a non-JSON-serializable value.
        Purpose: Measuring must never prevent the error response.
        """
        ctx = build_empty_response_context("m", False, {"x": object()}, [], None)

        assert ctx.request_bytes is None

    def test_circular_payload_gives_none_bytes(self):
        """
        What it does: Payload that references itself.
        Purpose: ValueError from json.dumps is handled too.
        """
        payload = {}
        payload["self"] = payload

        ctx = build_empty_response_context("m", False, payload, [], None)

        assert ctx.request_bytes is None

    @pytest.mark.parametrize("error", [TypeError("x"), ValueError("x"), KeyError("x"), AttributeError("x")])
    def test_token_estimate_failure_gives_none(self, error):
        """
        What it does: estimate_request_tokens raises.
        Purpose: Token estimate falls back to None, the rest of the context is intact.
        """
        with patch("kiro.empty_response.estimate_request_tokens", side_effect=error):
            ctx = build_empty_response_context("m", True, {"a": 1}, [], None)

        assert ctx.input_tokens_estimate is None
        assert ctx.request_bytes is not None

    def test_context_is_immutable(self):
        """
        What it does: Tries to mutate the frozen dataclass.
        Purpose: The context is a value object.
        """
        ctx = make_ctx()

        with pytest.raises(FrozenInstanceError):
            ctx.model = "other"


# =============================================================================
# Incomplete upstream response
# =============================================================================

def make_incomplete_error() -> EmptyUpstreamResponseError:
    """The error raised for the reported failure shape (reasoning only, no metering)."""
    return EmptyUpstreamResponseError(
        2048, {"reasoningContentEvent": 18, "contextUsageEvent": 1}, kind=KIND_INCOMPLETE
    )


class TestIncompleteErrorResponse:
    """handle_empty_response_*() for kind "incomplete" (non-streaming 503)."""

    def test_anthropic_body_status_and_header(self):
        """
        What it does: Handles an incomplete response for the Anthropic API.
        Purpose: Same 503 / api_error / x-should-retry as the empty response, new message.
        """
        response = handle_empty_response_anthropic(make_incomplete_error(), make_ctx(stream=False))

        assert response.status_code == 503
        assert body_of(response) == {
            "type": "error",
            "error": {"type": "api_error", "message": INCOMPLETE_RESPONSE_MESSAGE},
        }
        assert response.headers["x-should-retry"] == "true"

    def test_openai_body_uses_incomplete_code(self):
        """
        What it does: Handles an incomplete response for the OpenAI API.
        Purpose: Standard server_error type with the upstream_incomplete_response code.
        """
        response = handle_empty_response_openai(make_incomplete_error(), make_ctx(stream=False))

        assert response.status_code == 503
        assert body_of(response) == {
            "error": {
                "message": INCOMPLETE_RESPONSE_MESSAGE,
                "type": "server_error",
                "param": None,
                "code": "upstream_incomplete_response",
            }
        }

    def test_empty_kind_is_unchanged(self):
        """
        What it does: Handles kind "empty" after the change.
        Purpose: Message and code stay those of the empty-response feature.
        """
        anthropic = handle_empty_response_anthropic(make_error(), make_ctx())
        openai = handle_empty_response_openai(make_error(), make_ctx())

        assert make_error().kind == KIND_EMPTY
        assert body_of(anthropic)["error"]["message"] == EMPTY_RESPONSE_MESSAGE
        assert body_of(openai)["error"]["code"] == "upstream_empty_response"

    @pytest.mark.parametrize("status", [500, 502, 529, 599])
    def test_configured_status_is_reused(self, status):
        """
        What it does: Patches EMPTY_RESPONSE_ERROR_STATUS.
        Purpose: No new status setting: the incomplete response follows the existing one.
        """
        with patch("kiro.empty_response.EMPTY_RESPONSE_ERROR_STATUS", status):
            anthropic = handle_empty_response_anthropic(make_incomplete_error(), make_ctx())
            openai = handle_empty_response_openai(make_incomplete_error(), make_ctx())

        assert anthropic.status_code == openai.status_code == status
        assert body_of(anthropic)["error"]["type"] == ("overloaded_error" if status == 529 else "api_error")

    def test_message_is_friendly_and_claims_no_cause(self):
        """
        What it does: Scans the message.
        Purpose: Says what happened and what to do; never states a cause; no content.
        """
        text = INCOMPLETE_RESPONSE_MESSAGE.lower()
        assert "incomplete" in text and "retry" in text
        for forbidden in ("overloaded", "rate limit", "capacity", "quota", "throttl"):
            assert forbidden not in text

    def test_warning_uses_incomplete_label_and_no_content(self, loguru_messages):
        """
        What it does: Handles an incomplete response and reads the log.
        Purpose: [IncompleteUpstreamResponse] WARNING with facts; the empty label is NOT used.
        """
        handle_empty_response_anthropic(make_incomplete_error(), make_ctx(stream=False))

        lines = [m for m in loguru_messages if "[IncompleteUpstreamResponse]" in m]
        assert len(lines) == 1
        assert lines[0].startswith("WARNING|")
        assert "upstream_bytes=2048" in lines[0]
        assert "reasoningContentEvent" in lines[0]
        assert "text=False" in lines[0] and "tool_use=False" in lines[0]
        assert "status=503" in lines[0]
        assert not [m for m in loguru_messages if "[EmptyUpstreamResponse]" in m]
        assert "upstream returned an incomplete response" in "\n".join(loguru_messages)

    def test_empty_kind_still_logs_the_empty_label(self, loguru_messages):
        """
        What it does: Handles kind "empty".
        Purpose: Empty-response log line is byte-identical in shape (no text=/tool_use=).
        """
        handle_empty_response_anthropic(make_error(), make_ctx())

        line = next(m for m in loguru_messages if "[EmptyUpstreamResponse]" in m)
        assert "text=" not in line and "tool_use=" not in line
        assert "frames={'contextUsageEvent': 1}, status=503" in line

    def test_flushes_debug_logger_with_incomplete_message(self):
        """
        What it does: Replaces debug_logger with a mock.
        Purpose: DEBUG_MODE=errors saves the logs of this failed request with the right message.
        """
        fake_logger = MagicMock()
        with patch("kiro.empty_response.debug_logger", fake_logger):
            handle_empty_response_openai(make_incomplete_error(), make_ctx())

        fake_logger.flush_on_error.assert_called_once_with(503, INCOMPLETE_RESPONSE_MESSAGE)


class TestIncompleteStreamErrorEvents:
    """In-stream error events for streaming clients (HTTP status is already 200)."""

    def test_anthropic_event_is_legal_sse_with_default_type(self):
        """
        What it does: Builds the Anthropic error event.
        Purpose: "event: error" + a single data line with {"type": "error", "error": {...}}.
        """
        text = build_incomplete_stream_error_anthropic()

        assert text.startswith("event: error\ndata: ")
        assert text.endswith("\n\n")
        data = json.loads(text.split("\n")[1][len("data: "):])
        assert data == {
            "type": "error",
            "error": {"type": "overloaded_error", "message": INCOMPLETE_RESPONSE_MESSAGE},
        }

    @pytest.mark.parametrize("error_type", ["overloaded_error", "api_error"])
    def test_anthropic_error_type_follows_the_config(self, error_type):
        """
        What it does: Patches INCOMPLETE_RESPONSE_ERROR_TYPE.
        Purpose: The configured type is what the client receives.
        """
        with patch("kiro.empty_response.INCOMPLETE_RESPONSE_ERROR_TYPE", error_type):
            text = build_incomplete_stream_error_anthropic()

        assert json.loads(text.split("\n")[1][len("data: "):])["error"]["type"] == error_type

    def test_openai_chunk_is_a_data_line_with_error_key(self):
        """
        What it does: Builds the OpenAI in-stream error.
        Purpose: "data: {\"error\": {...}}" - the form the OpenAI SDKs raise APIError for.
        """
        text = build_incomplete_stream_error_openai()

        assert text.startswith("data: ") and text.endswith("\n\n")
        assert text.count("\n") == 2
        assert json.loads(text[len("data: "):]) == {
            "error": {
                "message": INCOMPLETE_RESPONSE_MESSAGE,
                "type": "server_error",
                "param": None,
                "code": "upstream_incomplete_response",
            }
        }

    def test_events_have_exact_line_structure(self):
        """
        What it does: Serializes both events.
        Purpose: ensure_ascii=False must not break the SSE framing (no stray newlines).
        """
        assert build_incomplete_stream_error_anthropic().count("\n") == 3
        assert build_incomplete_stream_error_openai().count("\n") == 2


class TestLogIncompleteStream:
    """log_incomplete_stream(): one WARNING per incomplete streaming response."""

    @staticmethod
    def _outcome(aws_frames) -> StreamOutcome:
        outcome = StreamOutcome()
        outcome.scanner.feed(aws_frames.reasoning("TOP_SECRET_THOUGHT_77") + aws_frames.context_usage(1.0))
        outcome.finished = True
        return outcome

    def test_warning_has_counts_and_booleans_only(self, aws_frames, loguru_messages):
        """
        What it does: Logs an incomplete streaming response.
        Purpose: Frame counts, bytes, text/tool_use booleans, action; never content.
        """
        log_incomplete_stream(self._outcome(aws_frames), "claude-sonnet-4-5")

        lines = [m for m in loguru_messages if "[IncompleteUpstreamResponse]" in m]
        assert len(lines) == 1
        line = lines[0]
        assert line.startswith("WARNING|")
        assert "model=claude-sonnet-4-5" in line and "stream=True" in line
        assert "reasoningContentEvent" in line and "upstream_bytes=" in line
        assert "text=False" in line and "tool_use=False" in line
        assert "action=error_event" in line
        assert "TOP_SECRET_THOUGHT_77" not in line

    def test_flushes_debug_logger(self, aws_frames):
        """
        What it does: Replaces debug_logger with a mock.
        Purpose: DEBUG_MODE=errors keeps the raw logs of the incomplete stream.
        """
        fake_logger = MagicMock()
        with patch("kiro.empty_response.debug_logger", fake_logger):
            log_incomplete_stream(self._outcome(aws_frames), "m")

        fake_logger.flush_on_error.assert_called_once_with(200, INCOMPLETE_RESPONSE_MESSAGE)

    def test_works_without_debug_logger(self, aws_frames):
        """
        What it does: debug_logger is None.
        Purpose: Logging still works.
        """
        with patch("kiro.empty_response.debug_logger", None):
            log_incomplete_stream(self._outcome(aws_frames), "m")
