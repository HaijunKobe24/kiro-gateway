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
Translation of client thinking / effort settings into Kiro additionalModelRequestFields.

Kiro models advertise an ``additionalModelRequestFieldsSchema`` in ListAvailableModels. This
module turns the client's settings (Anthropic ``thinking`` and ``output_config.effort``,
OpenAI ``reasoning_effort``) into the top-level ``additionalModelRequestFields`` object of the
Kiro request body, driven ONLY by that schema:

- Every allowed value is read from the schema's ``enum`` lists; no model name or enum value
  is hardcoded, and the gateway never invents a value (for example ``between_tools``).
- A client value is forwarded only when it is a string that is contained in the matching enum
  (exact match). Anything else is dropped with a de-duplicated WARNING.
- When the schema is missing, unparseable or uses structure this module does not understand,
  nothing is sent (the historical behavior).
- Only keys declared by the schema are produced, and the result is validated against the
  schema once before it is returned; a failed self-check returns None, so the translation can
  never turn a working request into an HTTP 400.
- max_tokens, context_management, budget_tokens and other client fields are never forwarded.

The translation functions are pure (no network, no cache types); only
resolve_additional_model_request_fields touches the model cache and the config switch.
"""

import json
import threading
from typing import Any, Dict, List, Optional, Set, Tuple

from loguru import logger

from kiro.config import FORWARD_MODEL_REQUEST_FIELDS

# Top-level key of the Kiro request body.
ADDITIONAL_FIELDS_KEY: str = "additionalModelRequestFields"

# Key of the schema inside a ListAvailableModels model entry.
SCHEMA_FIELD_NAME: str = "additionalModelRequestFieldsSchema"

# Upper bound of the WARNING de-duplication set (client-controlled strings must not grow it forever).
MAX_WARNED_KEYS: int = 512

# Values longer than this are truncated in log messages.
MAX_LOGGED_VALUE_LENGTH: int = 64

# Maximum schema / value nesting depth handled by the validator.
MAX_SCHEMA_DEPTH: int = 16

# Schema sections and keys this module knows how to fill.
SECTION_THINKING: str = "thinking"
SECTION_OUTPUT_CONFIG: str = "output_config"
SECTION_REASONING: str = "reasoning"
KEY_TYPE: str = "type"
KEY_DISPLAY: str = "display"
KEY_EFFORT: str = "effort"

# Structural keywords the self-check cannot evaluate. A schema using them is treated as unusable.
UNSUPPORTED_SCHEMA_KEYWORDS: Tuple[str, ...] = (
    "$ref", "oneOf", "anyOf", "allOf", "not", "if", "then", "else",
    "patternProperties", "propertyNames", "dependentSchemas", "dependentRequired",
    "dependencies", "unevaluatedProperties",
)

_JSON_TYPES: Tuple[str, ...] = ("object", "array", "string", "integer", "number", "boolean", "null")

_warned_keys: Set[Tuple[str, str, str]] = set()
_warned_lock = threading.Lock()


def reset_warning_dedup() -> None:
    """
    Clear the WARNING de-duplication state.

    Intended for tests, so that the "warn once" behavior does not leak between test cases.
    """
    with _warned_lock:
        _warned_keys.clear()


def _short_repr(value: Any) -> str:
    """
    Build a bounded, single-line representation of a value for log messages.

    Args:
        value: Any client-controlled value

    Returns:
        repr(value) truncated to MAX_LOGGED_VALUE_LENGTH characters (control characters stay escaped)
    """
    text = repr(value)
    if len(text) > MAX_LOGGED_VALUE_LENGTH:
        return text[:MAX_LOGGED_VALUE_LENGTH] + "..."
    return text


def _warn_once(model_id: str, field: str, value: Any, reason: str) -> None:
    """
    Log a WARNING the first time a (model, field, value) combination is dropped, DEBUG afterwards.

    The de-duplication set is bounded by MAX_WARNED_KEYS; once it is full, new combinations are
    logged at DEBUG only.

    Args:
        model_id: Internal model ID the request targets
        field: Dotted field path that is being dropped (e.g. "output_config.effort")
        value: The rejected client value (only a bounded repr is logged)
        reason: Short explanation of why it was dropped
    """
    shown = _short_repr(value)
    key = (str(model_id), field, shown)
    with _warned_lock:
        is_new = key not in _warned_keys and len(_warned_keys) < MAX_WARNED_KEYS
        if is_new:
            _warned_keys.add(key)
    if is_new:
        logger.warning(
            "Not forwarding {} for model {}: {} (value: {})", field, model_id, reason, shown
        )
    else:
        logger.debug(
            "Not forwarding {} for model {}: {} (value: {})", field, model_id, reason, shown
        )


def parse_schema(raw: Any) -> Optional[Dict[str, Any]]:
    """
    Normalize a raw additionalModelRequestFieldsSchema value into a dict.

    Args:
        raw: A dict (returned as is), a JSON string holding an object, or anything else

    Returns:
        The schema dict, or None when the value is missing, not a dict / JSON object string,
        or not valid JSON. Never raises.
    """
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (ValueError, RecursionError):
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _schema_supported(schema: Any, depth: int = 0) -> bool:
    """
    Check that a schema only uses keywords the self-check can evaluate.

    Args:
        schema: Schema node (dict or bool)
        depth: Current nesting depth

    Returns:
        True when no unsupported structural keyword is present within MAX_SCHEMA_DEPTH
    """
    if isinstance(schema, bool):
        return True
    if not isinstance(schema, dict) or depth > MAX_SCHEMA_DEPTH:
        return False
    if any(keyword in schema for keyword in UNSUPPORTED_SCHEMA_KEYWORDS):
        return False
    properties = schema.get("properties")
    if properties is not None:
        if not isinstance(properties, dict):
            return False
        if not all(_schema_supported(sub, depth + 1) for sub in properties.values()):
            return False
    for keyword in ("additionalProperties", "items"):
        sub = schema.get(keyword)
        if isinstance(sub, dict) and not _schema_supported(sub, depth + 1):
            return False
    return True


def _json_equal(left: Any, right: Any) -> bool:
    """
    JSON-aware equality: booleans never equal numbers, containers compare recursively.

    Args:
        left: First value
        right: Second value

    Returns:
        True when both values are equal under JSON Schema semantics
    """
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_json_equal(left[k], right[k]) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_json_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, (dict, list)) or isinstance(right, (dict, list)):
        return False
    return left == right


def _matches_json_type(value: Any, json_type: Any) -> bool:
    """
    Check a value against one JSON Schema primitive type name.

    Args:
        value: Value to check
        json_type: Type name (unknown names never match)

    Returns:
        True when the value has that JSON type (booleans are not numbers)
    """
    if json_type == "object":
        return isinstance(value, dict)
    if json_type == "array":
        return isinstance(value, list)
    if json_type == "string":
        return isinstance(value, str)
    if json_type == "boolean":
        return isinstance(value, bool)
    if json_type == "null":
        return value is None
    if json_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if json_type == "integer":
        if isinstance(value, bool):
            return False
        return isinstance(value, int) or (isinstance(value, float) and value.is_integer())
    return False


def _validate_node(value: Any, schema: Any, depth: int) -> bool:
    """
    Recursive worker of validate_against_schema.

    Args:
        value: Value to validate
        schema: Schema node
        depth: Current nesting depth

    Returns:
        True when the value satisfies the schema node
    """
    if depth > MAX_SCHEMA_DEPTH:
        return False
    if isinstance(schema, bool):
        return schema
    if not isinstance(schema, dict):
        return False
    if any(keyword in schema for keyword in UNSUPPORTED_SCHEMA_KEYWORDS):
        return False

    expected = schema.get("type")
    if expected is not None:
        types: List[Any] = expected if isinstance(expected, list) else [expected]
        if not any(_matches_json_type(value, t) for t in types):
            return False

    if "enum" in schema:
        options = schema["enum"]
        if not isinstance(options, list) or not any(_json_equal(value, option) for option in options):
            return False
    if "const" in schema and not _json_equal(value, schema["const"]):
        return False

    if isinstance(value, dict):
        properties = schema.get("properties")
        if properties is not None and not isinstance(properties, dict):
            return False
        properties = properties or {}
        required = schema.get("required")
        if required is not None:
            if not isinstance(required, list) or any(name not in value for name in required):
                return False
        additional = schema.get("additionalProperties", True)
        for name, sub_value in value.items():
            if name in properties:
                if not _validate_node(sub_value, properties[name], depth + 1):
                    return False
            elif additional is False:
                return False
            elif isinstance(additional, dict) and not _validate_node(sub_value, additional, depth + 1):
                return False

    if isinstance(value, list):
        items = schema.get("items")
        if isinstance(items, dict) or isinstance(items, bool):
            if not all(_validate_node(element, items, depth + 1) for element in value):
                return False

    return True


def validate_against_schema(value: Any, schema: Any) -> bool:
    """
    Validate a value against a minimal JSON Schema subset.

    Supported keywords: type, enum, const, properties, additionalProperties, required, items.
    Annotation keywords (default, description, title, ...) and unknown keywords are ignored.
    A schema using $ref / oneOf / anyOf / allOf / not / if / patternProperties and similar
    keywords is reported as not satisfied, because the result could not be trusted.

    Args:
        value: Value to validate (a Kiro additionalModelRequestFields dict)
        schema: Parsed schema (dict or bool)

    Returns:
        True only when the value is known to satisfy the schema. Never raises.
    """
    try:
        return _validate_node(value, schema, 0)
    except RecursionError:
        return False


def _enum_at(properties: Dict[str, Any], section: str, key: str) -> Optional[List[str]]:
    """
    Read the string enum declared for properties.<section>.properties.<key>.

    Args:
        properties: The schema's top-level "properties" dict
        section: Section name (e.g. "thinking")
        key: Key inside the section (e.g. "type")

    Returns:
        The enum as a list of strings, or None when the path is absent or the enum is not a
        non-empty list of strings (the section is then treated as unsupported)
    """
    section_schema = properties.get(section)
    if not isinstance(section_schema, dict):
        return None
    inner = section_schema.get("properties")
    if not isinstance(inner, dict):
        return None
    key_schema = inner.get(key)
    if not isinstance(key_schema, dict):
        return None
    options = key_schema.get("enum")
    if not isinstance(options, list) or not options:
        return None
    if not all(isinstance(option, str) for option in options):
        return None
    return options


def _translate_thinking(
    model_id: str, properties: Dict[str, Any], thinking: Any
) -> Optional[Dict[str, Any]]:
    """
    Build the "thinking" section from the client's thinking object.

    Args:
        model_id: Internal model ID (log context)
        properties: The schema's top-level "properties" dict
        thinking: Client thinking object (typically {"type": ..., "display": ...})

    Returns:
        {"type": ...} plus an optional "display", or None when the schema has no usable
        thinking.type enum, the client sent nothing usable, or the type is not allowed
    """
    type_enum = _enum_at(properties, SECTION_THINKING, KEY_TYPE)
    if type_enum is None or thinking is None:
        return None
    if not isinstance(thinking, dict):
        _warn_once(model_id, "thinking", thinking, "thinking is not an object")
        return None
    thinking_type = thinking.get(KEY_TYPE)
    if thinking_type is None:
        return None
    if not isinstance(thinking_type, str) or thinking_type not in type_enum:
        _warn_once(
            model_id, "thinking.type", thinking_type, "value is not allowed by the model schema"
        )
        return None

    result: Dict[str, Any] = {KEY_TYPE: thinking_type}
    display = thinking.get(KEY_DISPLAY)
    if display is None:
        return result
    display_enum = _enum_at(properties, SECTION_THINKING, KEY_DISPLAY)
    if display_enum is None:
        logger.debug("Model {} schema has no thinking.display enum; dropping display", model_id)
    elif isinstance(display, str) and display in display_enum:
        result[KEY_DISPLAY] = display
    else:
        _warn_once(
            model_id, "thinking.display", display, "value is not allowed by the model schema"
        )
    return result


def _translate_effort(
    model_id: str, properties: Dict[str, Any], effort: Any
) -> Dict[str, Any]:
    """
    Build the effort-bearing sections (output_config and/or reasoning) from the client effort.

    Args:
        model_id: Internal model ID (log context)
        properties: The schema's top-level "properties" dict
        effort: Client effort value (Anthropic output_config.effort or OpenAI reasoning_effort)

    Returns:
        A dict with "output_config" and/or "reasoning" entries (possibly empty)
    """
    result: Dict[str, Any] = {}
    if effort is None:
        return result
    declared = [
        (SECTION_OUTPUT_CONFIG, _enum_at(properties, SECTION_OUTPUT_CONFIG, KEY_EFFORT)),
        (SECTION_REASONING, _enum_at(properties, SECTION_REASONING, KEY_EFFORT)),
    ]
    declared = [(section, options) for section, options in declared if options is not None]
    if not declared:
        logger.debug("Model {} schema declares no effort enum; dropping effort", model_id)
        return result
    for section, options in declared:
        if isinstance(effort, str) and effort in options:
            result[section] = {KEY_EFFORT: effort}
    if not result:
        field = ".".join((declared[0][0], KEY_EFFORT))
        _warn_once(model_id, field, effort, "value is not allowed by the model schema")
    return result


def translate_model_request_fields(
    model_id: str,
    schema: Any,
    thinking: Any = None,
    effort: Any = None,
) -> Optional[Dict[str, Any]]:
    """
    Translate client thinking / effort into additionalModelRequestFields for one model.

    Pure function: ignores the config switch and the model cache.

    Args:
        model_id: Internal model ID (used for log context only)
        schema: Raw additionalModelRequestFieldsSchema (dict, JSON string, None, ...)
        thinking: Client thinking object (Anthropic "thinking"); None for OpenAI
        effort: Client effort (Anthropic output_config.effort or OpenAI reasoning_effort)

    Returns:
        The top-level additionalModelRequestFields dict, or None when nothing should be sent
        (no/unusable schema, nothing forwardable, or the result fails the schema self-check).
        The inputs are never modified.
    """
    parsed = parse_schema(schema)
    if parsed is None:
        if schema is not None:
            _warn_once(model_id, "schema", schema, "schema cannot be parsed")
        return None
    if not _schema_supported(parsed):
        _warn_once(model_id, "schema", "<unsupported keywords>", "schema uses unsupported keywords")
        return None
    properties = parsed.get("properties")
    if not isinstance(properties, dict):
        return None

    result: Dict[str, Any] = {}
    thinking_section = _translate_thinking(model_id, properties, thinking)
    if thinking_section is not None:
        result[SECTION_THINKING] = thinking_section
    result.update(_translate_effort(model_id, properties, effort))

    if not result:
        return None
    if not validate_against_schema(result, parsed):
        _warn_once(
            model_id, "additionalModelRequestFields", sorted(result),
            "result fails the model schema self-check",
        )
        return None
    return result


def _flatten(prefix: str, value: Any) -> List[str]:
    """
    Flatten a nested dict into "a.b=value" strings.

    Args:
        prefix: Dotted path so far
        value: Current node

    Returns:
        List of "path=value" strings for every leaf
    """
    if isinstance(value, dict):
        parts: List[str] = []
        for key, sub in value.items():
            parts.extend(_flatten(f"{prefix}.{key}" if prefix else str(key), sub))
        return parts
    return [f"{prefix}={value}"]


def summarize_fields(fields: Optional[Dict[str, Any]]) -> str:
    """
    Describe additionalModelRequestFields for logs (key paths and enum values only).

    Args:
        fields: The dict produced by translate_model_request_fields, or None

    Returns:
        e.g. "thinking.type=adaptive, output_config.effort=medium", or "none"
    """
    if not fields:
        return "none"
    return ", ".join(_flatten("", fields))


def resolve_additional_model_request_fields(
    model_id: str,
    model_cache: Any,
    thinking: Any = None,
    effort: Any = None,
) -> Optional[Dict[str, Any]]:
    """
    Entry point used by the request converters.

    Args:
        model_id: Internal Kiro model ID (the one written to userInputMessage.modelId)
        model_cache: ModelInfoCache (or any object with get_additional_fields_schema); None allowed
        thinking: Client thinking object (Anthropic) or None
        effort: Client effort (Anthropic output_config.effort / OpenAI reasoning_effort)

    Returns:
        additionalModelRequestFields dict, or None when FORWARD_MODEL_REQUEST_FIELDS is off,
        there is no cache, the model has no usable schema, or nothing is forwardable
    """
    if not FORWARD_MODEL_REQUEST_FIELDS or model_cache is None:
        return None
    getter = getattr(model_cache, "get_additional_fields_schema", None)
    if not callable(getter):
        return None
    result = translate_model_request_fields(model_id, getter(model_id), thinking, effort)
    if result:
        logger.debug(
            "additionalModelRequestFields for model={}: {}", model_id, summarize_fields(result)
        )
    return result
