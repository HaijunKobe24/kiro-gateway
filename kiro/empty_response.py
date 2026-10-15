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
Client-facing error for an empty or incomplete upstream response.

When Kiro answers HTTP 200 with an empty response (see kiro.upstream_guard), the gateway
does not retry by itself. It answers with an explicit 5xx error in the native error format
of the API the client speaks, so the client can recognise the failure and retry.

An incomplete response (the stream ended early while the model was reasoning: reasoning
frames only, no meteringEvent, no text, no tool call) is reported the same way when the
client did not ask for streaming (EmptyUpstreamResponseError with kind "incomplete").
When streaming, the HTTP status is already 200 and reasoning was already forwarded, so the
error travels inside the stream instead: an Anthropic "event: error" or an OpenAI
{"error": ...} chunk (the form the OpenAI SDKs raise an APIError for), followed by [DONE].

Status code: EMPTY_RESPONSE_ERROR_STATUS (default 503). Anthropic bodies use "api_error",
or "overloaded_error" when the configured status is 529 (Anthropic's overload status).
OpenAI bodies use the standard "server_error" type with the code "upstream_empty_response".
The response carries "x-should-retry: true" as a hint for SDK based clients.

The error message never claims a cause (the cause of empty upstream responses is unknown)
and never contains request or response content.
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from fastapi.responses import JSONResponse
from loguru import logger

from kiro.config import EMPTY_RESPONSE_ERROR_STATUS, INCOMPLETE_RESPONSE_ERROR_TYPE
from kiro.tokenizer import estimate_request_tokens
from kiro.upstream_guard import KIND_INCOMPLETE, EmptyUpstreamResponseError, StreamOutcome

# Import debug_logger for logging
try:
    from kiro.debug_logger import debug_logger
except ImportError:
    debug_logger = None


# User-facing message: actionable, no claim about the cause, no request content
EMPTY_RESPONSE_MESSAGE = (
    "The upstream service returned an empty response (HTTP 200 with no content and "
    "no usage metering). This is usually transient; please retry the request."
)

# User-facing message for an incomplete response: no claim about the cause, no content
INCOMPLETE_RESPONSE_MESSAGE = (
    "The upstream service returned an incomplete response (reasoning only, no body). "
    "This is usually transient; please retry the request."
)

# Anthropic's "overloaded" status code: only then the Anthropic error type is overloaded_error
ANTHROPIC_OVERLOADED_STATUS = 529

# Machine-readable code of the OpenAI error body
OPENAI_EMPTY_RESPONSE_CODE = "upstream_empty_response"
OPENAI_INCOMPLETE_RESPONSE_CODE = "upstream_incomplete_response"

# Type of the OpenAI error body and of the OpenAI in-stream error chunk
OPENAI_ERROR_TYPE = "server_error"

# Value of the "action" field in the incomplete-response WARNING of a streaming response
ACTION_ERROR_EVENT = "error_event"

# Header hint honoured by SDK clients that follow the x-should-retry convention
SHOULD_RETRY_HEADERS = {"x-should-retry": "true"}

# Endpoint paths used in the access log line
ANTHROPIC_PATH = "/v1/messages"
OPENAI_PATH = "/v1/chat/completions"

# Placeholder for values that could not be determined
UNKNOWN = "unknown"


@dataclass(frozen=True)
class EmptyResponseContext:
    """
    Non-sensitive facts about the request, used only for the WARNING log.

    Attributes:
        model: Requested model name
        stream: Whether the client asked for a streaming response
        request_bytes: Size of the Kiro request payload in bytes (None if unknown)
        input_tokens_estimate: Estimated input tokens of the request (None if unknown)
    """
    model: str
    stream: bool
    request_bytes: Optional[int] = None
    input_tokens_estimate: Optional[int] = None


def build_empty_response_context(
    model: str,
    stream: bool,
    kiro_payload: Dict[str, Any],
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
    system: Optional[Any] = None,
) -> EmptyResponseContext:
    """
    Collect the non-sensitive request facts for the empty-response log.

    Runs only on the rare empty-response path. Failures to measure never prevent the
    error response: the affected field is None.

    Args:
        model: Requested model name
        stream: Whether the client asked for streaming
        kiro_payload: Payload sent to Kiro (only its serialized size is used)
        messages: Request messages as dicts (for the token estimate)
        tools: Request tools as dicts (for the token estimate)
        system: System prompt (for the token estimate)

    Returns:
        EmptyResponseContext with the measured values
    """
    request_bytes: Optional[int]
    try:
        request_bytes = len(json.dumps(kiro_payload, ensure_ascii=False).encode("utf-8"))
    except (TypeError, ValueError):
        request_bytes = None

    input_tokens_estimate: Optional[int]
    try:
        stats = estimate_request_tokens(
            messages=messages or [],
            tools=tools,
            system_prompt=system,
            apply_claude_correction=False,
        )
        input_tokens_estimate = stats["total_tokens"]
    except (TypeError, ValueError, KeyError, AttributeError):
        input_tokens_estimate = None

    return EmptyResponseContext(
        model=model,
        stream=stream,
        request_bytes=request_bytes,
        input_tokens_estimate=input_tokens_estimate,
    )


def _message_for(exc: EmptyUpstreamResponseError) -> str:
    """
    Pick the user-facing message for an empty or incomplete response.

    Args:
        exc: The detected empty or incomplete response

    Returns:
        INCOMPLETE_RESPONSE_MESSAGE for kind "incomplete", otherwise EMPTY_RESPONSE_MESSAGE
    """
    return INCOMPLETE_RESPONSE_MESSAGE if exc.kind == KIND_INCOMPLETE else EMPTY_RESPONSE_MESSAGE


def _log_empty_response(
    exc: EmptyUpstreamResponseError,
    ctx: EmptyResponseContext,
    status: int,
    path: str,
) -> None:
    """
    Log one empty or incomplete upstream response (WARNING) and flush debug logs.

    Args:
        exc: The detected empty or incomplete response
        ctx: Non-sensitive request facts
        status: HTTP status that is returned to the client
        path: Endpoint path for the access log line
    """
    incomplete = exc.kind == KIND_INCOMPLETE
    label = "IncompleteUpstreamResponse" if incomplete else "EmptyUpstreamResponse"
    output_facts = (
        f", text={exc.produced_text}, tool_use={exc.produced_tool_use}" if incomplete else ""
    )
    logger.warning(
        "[{}] model={}, stream={}, input_tokens_estimate={}, "
        "request_bytes={}, upstream_bytes={}, frames={}{}, status={}",
        label,
        ctx.model,
        ctx.stream,
        UNKNOWN if ctx.input_tokens_estimate is None else ctx.input_tokens_estimate,
        UNKNOWN if ctx.request_bytes is None else ctx.request_bytes,
        exc.bytes_received,
        exc.frame_counts,
        output_facts,
        status,
    )
    mode = "streaming" if ctx.stream else "non-streaming"
    description = "an incomplete" if incomplete else "an empty"
    logger.warning(
        "HTTP {} - POST {} ({}) - upstream returned {} response", status, path, mode, description
    )
    if debug_logger:
        debug_logger.flush_on_error(status, _message_for(exc))


def handle_empty_response_anthropic(
    exc: EmptyUpstreamResponseError,
    ctx: EmptyResponseContext,
) -> JSONResponse:
    """
    Build the Anthropic-format error response for an empty or incomplete upstream response.

    Args:
        exc: The detected empty or incomplete response
        ctx: Non-sensitive request facts for the log

    Returns:
        JSONResponse with the configured 5xx status and an Anthropic error body
    """
    status = EMPTY_RESPONSE_ERROR_STATUS
    _log_empty_response(exc, ctx, status, ANTHROPIC_PATH)
    error_type = "overloaded_error" if status == ANTHROPIC_OVERLOADED_STATUS else "api_error"
    return JSONResponse(
        status_code=status,
        content={
            "type": "error",
            "error": {"type": error_type, "message": _message_for(exc)},
        },
        headers=SHOULD_RETRY_HEADERS,
    )


def handle_empty_response_openai(
    exc: EmptyUpstreamResponseError,
    ctx: EmptyResponseContext,
) -> JSONResponse:
    """
    Build the OpenAI-format error response for an empty or incomplete upstream response.

    Uses the standard OpenAI "server_error" type (not the gateway specific
    "kiro_api_error") so OpenAI SDKs classify it as a retryable server error.

    Args:
        exc: The detected empty or incomplete response
        ctx: Non-sensitive request facts for the log

    Returns:
        JSONResponse with the configured 5xx status and an OpenAI error body
    """
    status = EMPTY_RESPONSE_ERROR_STATUS
    _log_empty_response(exc, ctx, status, OPENAI_PATH)
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "message": _message_for(exc),
                "type": OPENAI_ERROR_TYPE,
                "param": None,
                "code": (
                    OPENAI_INCOMPLETE_RESPONSE_CODE
                    if exc.kind == KIND_INCOMPLETE
                    else OPENAI_EMPTY_RESPONSE_CODE
                ),
            }
        },
        headers=SHOULD_RETRY_HEADERS,
    )


def build_incomplete_stream_error_anthropic() -> str:
    """
    Build the Anthropic SSE error event that ends a stream with an incomplete response.

    Format (Anthropic streaming "Error events"): event: error, data: {"type": "error",
    "error": {"type": ..., "message": ...}}. The error type is INCOMPLETE_RESPONSE_ERROR_TYPE.

    Returns:
        SSE string "event: error\ndata: {...}\n\n"
    """
    data = {
        "type": "error",
        "error": {"type": INCOMPLETE_RESPONSE_ERROR_TYPE, "message": INCOMPLETE_RESPONSE_MESSAGE},
    }
    return f"event: error\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def build_incomplete_stream_error_openai() -> str:
    """
    Build the OpenAI SSE error chunk that ends a stream with an incomplete response.

    The OpenAI SDKs raise an APIError for any streamed data line whose JSON has a truthy
    "error" key, so the idiomatic in-stream error is a plain data line {"error": {...}}.
    The caller sends "data: [DONE]" after it.

    Returns:
        SSE string "data: {"error": {...}}\n\n"
    """
    data = {
        "error": {
            "message": INCOMPLETE_RESPONSE_MESSAGE,
            "type": OPENAI_ERROR_TYPE,
            "param": None,
            "code": OPENAI_INCOMPLETE_RESPONSE_CODE,
        }
    }
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def log_incomplete_stream(outcome: StreamOutcome, model: str, action: str = ACTION_ERROR_EVENT) -> None:
    """
    Log one incomplete streaming response (WARNING) and flush debug logs.

    Only counts, sizes and booleans are logged, never request or response content.

    Args:
        outcome: StreamOutcome of the finished stream
        model: Requested model name
        action: What the gateway did (ACTION_ERROR_EVENT)
    """
    logger.warning(
        "[IncompleteUpstreamResponse] model={}, stream=True, upstream_bytes={}, frames={}, "
        "text={}, tool_use={}, action={}",
        model,
        outcome.scanner.bytes_received,
        dict(outcome.scanner.counts),
        outcome.produced_text,
        outcome.produced_tool_use,
        action,
    )
    if debug_logger:
        debug_logger.flush_on_error(200, INCOMPLETE_RESPONSE_MESSAGE)
