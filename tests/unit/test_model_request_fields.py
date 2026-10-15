# -*- coding: utf-8 -*-

"""
Unit tests for kiro/model_request_fields.py.

The module translates client thinking / effort settings into the top-level
additionalModelRequestFields of the Kiro request body, driven only by the target model's
additionalModelRequestFieldsSchema. All tests are pure (no network).
"""

import copy
import json
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from kiro.model_request_fields import (
    ADDITIONAL_FIELDS_KEY,
    MAX_LOGGED_VALUE_LENGTH,
    MAX_WARNED_KEYS,
    SCHEMA_FIELD_NAME,
    parse_schema,
    reset_warning_dedup,
    resolve_additional_model_request_fields,
    summarize_fields,
    translate_model_request_fields,
    validate_against_schema,
)

CLAUDE_CODE_THINKING = {"type": "adaptive", "display": "updates"}


@pytest.fixture(autouse=True)
def _reset_dedup():
    """Keeps the WARNING de-duplication state isolated between tests."""
    reset_warning_dedup()
    yield
    reset_warning_dedup()


def _warnings(messages):
    """Returns only the WARNING lines of the captured loguru messages."""
    return [m for m in messages if m.startswith("WARNING|")]


# =============================================================================
# parse_schema
# =============================================================================

class TestParseSchema:
    """Tests for parse_schema()."""

    def test_dict_is_returned_as_is(self):
        """
        What it does: Passes a dict.
        Purpose: dict schemas are used unchanged (same object).
        """
        schema = {"type": "object"}
        assert parse_schema(schema) is schema

    def test_json_object_string_is_parsed(self):
        """
        What it does: Passes a JSON string holding an object.
        Purpose: Some APIs ship the schema as a string; it must be parsed.
        """
        assert parse_schema('{"type": "object"}') == {"type": "object"}

    @pytest.mark.parametrize("raw", ["[]", "1", '"x"', "null", "not json", "", "   ", "{", "[1,2"])
    def test_non_object_or_invalid_json_string_returns_none(self, raw):
        """
        What it does: Strings that are not a JSON object.
        Purpose: Never raise, return None.
        """
        assert parse_schema(raw) is None

    @pytest.mark.parametrize("raw", [None, 42, 3.5, True, [], ["a"], MagicMock(), object(), b"{}"])
    def test_other_types_return_none(self, raw):
        """
        What it does: Non dict / non str inputs including MagicMock (route tests use MagicMock caches).
        Purpose: Treated as "no usable schema".
        """
        assert parse_schema(raw) is None

    def test_deeply_nested_json_string_does_not_raise(self):
        """
        What it does: A JSON string nested far beyond the recursion limit.
        Purpose: RecursionError from json.loads must be swallowed.
        """
        assert parse_schema("[" * 100000 + "]" * 100000) is None


# =============================================================================
# validate_against_schema
# =============================================================================

class TestValidateAgainstSchema:
    """Tests for the minimal JSON Schema subset validator."""

    @pytest.mark.parametrize("value,json_type,expected", [
        ({}, "object", True), ([], "object", False),
        ([], "array", True), ({}, "array", False),
        ("a", "string", True), (1, "string", False),
        (1, "integer", True), (1.0, "integer", True), (1.5, "integer", False), (True, "integer", False),
        (1, "number", True), (1.5, "number", True), (True, "number", False),
        (True, "boolean", True), (0, "boolean", False),
        (None, "null", True), (0, "null", False),
        ("a", "unknown-type", False),
    ])
    def test_type(self, value, json_type, expected):
        """
        What it does: Each JSON type against matching and mismatching values.
        Purpose: bool is never an integer/number; unknown type names never match.
        """
        assert validate_against_schema(value, {"type": json_type}) is expected

    def test_type_list(self):
        """
        What it does: "type" given as a list.
        Purpose: Any listed type may match.
        """
        schema = {"type": ["string", "null"]}
        assert validate_against_schema("a", schema) is True
        assert validate_against_schema(None, schema) is True
        assert validate_against_schema(1, schema) is False

    def test_enum_is_type_aware(self):
        """
        What it does: enum containing 1 checked with True and 1.
        Purpose: True must not match 1 (Python's True == 1 trap).
        """
        schema = {"enum": [1, "a"]}
        assert validate_against_schema(1, schema) is True
        assert validate_against_schema(True, schema) is False
        assert validate_against_schema("b", schema) is False
        assert validate_against_schema("a", {"enum": "not-a-list"}) is False

    def test_const(self):
        """
        What it does: const with a nested value.
        Purpose: Deep JSON equality.
        """
        schema = {"const": {"a": [1, 2]}}
        assert validate_against_schema({"a": [1, 2]}, schema) is True
        assert validate_against_schema({"a": [1, True]}, schema) is False

    def test_properties_required_and_additional_properties_false(self):
        """
        What it does: Strict object schema.
        Purpose: Extra keys and missing required keys are rejected.
        """
        schema = {
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "required": ["a"],
            "additionalProperties": False,
        }
        assert validate_against_schema({"a": "x"}, schema) is True
        assert validate_against_schema({}, schema) is False
        assert validate_against_schema({"a": "x", "b": 1}, schema) is False
        assert validate_against_schema({"a": 1}, schema) is False

    def test_additional_properties_schema(self):
        """
        What it does: additionalProperties given as a schema.
        Purpose: Extra keys must satisfy it.
        """
        schema = {"additionalProperties": {"type": "integer"}}
        assert validate_against_schema({"x": 1}, schema) is True
        assert validate_against_schema({"x": "1"}, schema) is False

    def test_items(self):
        """
        What it does: Array items schema.
        Purpose: Every element is validated.
        """
        schema = {"type": "array", "items": {"type": "string"}}
        assert validate_against_schema(["a", "b"], schema) is True
        assert validate_against_schema(["a", 1], schema) is False

    @pytest.mark.parametrize("keyword", [
        "$ref", "oneOf", "anyOf", "allOf", "not", "if", "patternProperties", "propertyNames",
        "dependentSchemas", "dependentRequired", "dependencies", "unevaluatedProperties",
    ])
    def test_unsupported_keywords_are_never_satisfied(self, keyword):
        """
        What it does: A schema using a structural keyword the validator cannot evaluate.
        Purpose: Fail closed (False) instead of pretending to validate.
        """
        assert validate_against_schema({}, {keyword: {}}) is False
        assert validate_against_schema({"a": 1}, {"properties": {"a": {keyword: []}}}) is False

    def test_annotation_and_unknown_keywords_are_ignored(self):
        """
        What it does: default/description/title/$schema and an x- keyword.
        Purpose: They carry no validation semantics.
        """
        schema = {
            "$schema": "x", "title": "t", "description": "d", "default": 1,
            "x-vendor": {"oneOf": 1}, "type": "object",
        }
        assert validate_against_schema({}, schema) is True

    def test_boolean_schemas(self):
        """
        What it does: true / false schemas.
        Purpose: true accepts everything, false nothing.
        """
        assert validate_against_schema({"a": 1}, {"properties": {"a": True}}) is True
        assert validate_against_schema({"a": 1}, {"properties": {"a": False}}) is False

    def test_depth_limit_returns_false_without_raising(self):
        """
        What it does: A 100-level nested schema and value.
        Purpose: Deep input must give False, not RecursionError.
        """
        schema, value = {"type": "object"}, {}
        for _ in range(100):
            schema = {"type": "object", "properties": {"a": schema}}
            value = {"a": value}

        assert validate_against_schema(value, schema) is False

    def test_cyclic_schema_does_not_hang(self):
        """
        What it does: A schema dict that references itself.
        Purpose: The depth limit must stop the recursion.
        """
        schema = {"type": "object", "properties": {}}
        schema["properties"]["a"] = schema
        value = {}
        cursor = value
        for _ in range(50):
            cursor["a"] = {}
            cursor = cursor["a"]

        assert validate_against_schema(value, schema) is False

    @pytest.mark.parametrize("bad", [None, 5, "x", ["a"]])
    def test_invalid_schema_objects_return_false(self, bad):
        """
        What it does: Non dict / non bool schemas.
        Purpose: Never raise.
        """
        assert validate_against_schema({}, bad) is False

    def test_invalid_required_and_properties_shapes(self):
        """
        What it does: Malformed required / properties values.
        Purpose: Fail closed.
        """
        assert validate_against_schema({"a": 1}, {"required": "a"}) is False
        assert validate_against_schema({"a": 1}, {"properties": []}) is False


# =============================================================================
# translate_model_request_fields: schema shapes
# =============================================================================

class TestTranslateClaudeShapes:
    """Claude-shaped schemas (thinking + output_config)."""

    def test_sonnet_55_claude_code_request_drops_display_updates(self, model_schemas):
        """
        What it does: The real Claude Code request (adaptive + display "updates" + effort medium).
        Purpose: "updates" is not in the display enum -> dropped; the rest is forwarded.
        """
        result = translate_model_request_fields(
            "claude-sonnet-5.5", model_schemas.claude(), CLAUDE_CODE_THINKING, "medium"
        )

        assert result == {"thinking": {"type": "adaptive"}, "output_config": {"effort": "medium"}}

    @pytest.mark.parametrize("display", ["summarized", "omitted"])
    def test_valid_display_is_kept(self, model_schemas, display):
        """
        What it does: display values that are in the enum.
        Purpose: Forwarded unchanged.
        """
        result = translate_model_request_fields(
            "m", model_schemas.claude(), {"type": "adaptive", "display": display}, None
        )

        assert result == {"thinking": {"type": "adaptive", "display": display}}

    @pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
    def test_every_enum_effort_is_forwarded(self, model_schemas, effort):
        """
        What it does: Each effort in the enum.
        Purpose: Exact forwarding of all allowed values.
        """
        result = translate_model_request_fields("m", model_schemas.claude(), None, effort)

        assert result == {"output_config": {"effort": effort}}

    @pytest.mark.parametrize("thinking_type", ["disabled", "enabled", "ADAPTIVE", "", "auto"])
    def test_type_outside_enum_drops_thinking_and_warns(self, model_schemas, loguru_messages, thinking_type):
        """
        What it does: thinking.type values that sonnet-5.5 does not allow.
        Purpose: No thinking object at all, effort still forwarded, one WARNING.
        """
        result = translate_model_request_fields(
            "claude-sonnet-5.5", model_schemas.claude(),
            {"type": thinking_type, "display": "summarized"}, "high",
        )

        assert result == {"output_config": {"effort": "high"}}
        assert len([w for w in _warnings(loguru_messages) if "thinking.type" in w]) == 1

    def test_between_tools_only_when_client_sends_it(self, model_schemas):
        """
        What it does: Client sends between_tools explicitly, then sends no thinking.
        Purpose: Forwarded only when explicit and allowed; never invented by the gateway.
        """
        explicit = translate_model_request_fields(
            "m", model_schemas.claude(), {"type": "between_tools"}, None
        )
        implicit = translate_model_request_fields("m", model_schemas.claude(), None, "medium")

        assert explicit == {"thinking": {"type": "between_tools"}}
        assert "thinking" not in implicit

    def test_disabled_forwarded_only_when_enum_contains_it(self, model_schemas, loguru_messages):
        """
        What it does: thinking.disabled against sonnet-5 (allowed) and opus-5.5 (not allowed).
        Purpose: Forward when allowed, drop + WARNING otherwise.
        """
        allowed = translate_model_request_fields(
            "claude-sonnet-5", model_schemas.claude(type_enum=("adaptive", "disabled")),
            {"type": "disabled"}, None,
        )
        refused = translate_model_request_fields(
            "claude-opus-5.5", model_schemas.claude(type_enum=("adaptive",)),
            {"type": "disabled"}, None,
        )

        assert allowed == {"thinking": {"type": "disabled"}}
        assert refused is None
        assert len(_warnings(loguru_messages)) == 1

    @pytest.mark.parametrize("effort", ["ultra", "Medium", "MEDIUM", " medium", "medium ", "", "none", "minimal"])
    def test_invalid_effort_is_dropped_with_warning(self, model_schemas, loguru_messages, effort):
        """
        What it does: Effort values outside the enum, wrong case, padded or empty.
        Purpose: Exact match only; dropped with one WARNING; nothing is sent.
        """
        result = translate_model_request_fields("m", model_schemas.claude(), None, effort)

        assert result is None
        assert len(_warnings(loguru_messages)) == 1

    @pytest.mark.parametrize("effort", [123, 1, 0, True, False, ["medium"], {"x": 1}, 1.5, ("medium",)])
    def test_non_string_effort_is_dropped(self, model_schemas, effort):
        """
        What it does: Non-string effort values (True == 1 style traps included).
        Purpose: Only strings can be forwarded.
        """
        assert translate_model_request_fields("m", model_schemas.claude(), None, effort) is None

    def test_nothing_set_returns_none_silently(self, model_schemas, loguru_messages):
        """
        What it does: No thinking and no effort.
        Purpose: Nothing to forward, no WARNING.
        """
        assert translate_model_request_fields("m", model_schemas.claude(), None, None) is None
        assert not _warnings(loguru_messages)

    @pytest.mark.parametrize("thinking", [{}, {"type": None}, {"display": "summarized"}, {"budget_tokens": 100}])
    def test_thinking_without_type_is_not_forwarded(self, model_schemas, thinking):
        """
        What it does: thinking objects that carry no type.
        Purpose: A thinking object must always have a legal type, so none is sent.
        """
        assert translate_model_request_fields("m", model_schemas.claude(), thinking, None) is None

    @pytest.mark.parametrize("thinking", [
        "adaptive", ["adaptive"], 5, True, {"type": 123}, {"type": ["adaptive"]}, {"type": {"a": 1}},
        {"type": True}, {"type": "Adaptive"},
    ])
    def test_malformed_thinking_is_dropped(self, model_schemas, thinking):
        """
        What it does: Wrong-typed thinking objects and types.
        Purpose: Dropped without raising.
        """
        assert translate_model_request_fields("m", model_schemas.claude(), thinking, None) is None

    @pytest.mark.parametrize("display", [123, ["summarized"], "Summarized", "", "updates"])
    def test_invalid_display_is_dropped_but_type_kept(self, model_schemas, display):
        """
        What it does: Invalid display with a valid type.
        Purpose: Only display is dropped.
        """
        result = translate_model_request_fields(
            "m", model_schemas.claude(), {"type": "adaptive", "display": display}, None
        )

        assert result == {"thinking": {"type": "adaptive"}}

    def test_display_dropped_when_schema_has_no_display_enum(self, model_schemas):
        """
        What it does: Schema without thinking.display.
        Purpose: Display must not be generated.
        """
        schema = model_schemas.claude(display_enum=None)
        result = translate_model_request_fields("m", schema, {"type": "adaptive", "display": "summarized"}, None)

        assert result == {"thinking": {"type": "adaptive"}}

    def test_unrelated_client_keys_are_never_forwarded(self, model_schemas):
        """
        What it does: Thinking with budget_tokens and extra keys.
        Purpose: Only type/display travel; max_tokens, context_management, format never appear.
        """
        thinking = {"type": "adaptive", "budget_tokens": 4000, "max_tokens": 1, "context_management": {}}
        result = translate_model_request_fields("m", model_schemas.claude(), thinking, "high")

        assert result == {"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}}
        assert "budget_tokens" not in json.dumps(result)

    def test_inputs_are_not_mutated(self, model_schemas):
        """
        What it does: Deep-copies inputs before and compares after translation.
        Purpose: The client's thinking dict and the schema stay untouched.
        """
        thinking = dict(CLAUDE_CODE_THINKING)
        schema = model_schemas.claude()
        before = (copy.deepcopy(thinking), copy.deepcopy(schema))

        translate_model_request_fields("m", schema, thinking, "medium")

        assert (thinking, schema) == before

    def test_json_string_schema_is_supported(self, model_schemas):
        """
        What it does: Schema delivered as a JSON string.
        Purpose: Same result as the dict form.
        """
        schema = json.dumps(model_schemas.claude())
        result = translate_model_request_fields("m", schema, {"type": "adaptive"}, "low")

        assert result == {"thinking": {"type": "adaptive"}, "output_config": {"effort": "low"}}

    def test_sonnet_5_opus_55_fable_51_shapes(self, model_schemas):
        """
        What it does: Fixture catalogue shapes B and C.
        Purpose: Each model forwards what its own enum allows.
        """
        catalogue = model_schemas.catalogue()
        for model_id in ("claude-sonnet-5", "claude-opus-5.5", "claude-fable-5.1"):
            result = translate_model_request_fields(model_id, catalogue[model_id], {"type": "adaptive"}, "medium")
            assert result == {"thinking": {"type": "adaptive"}, "output_config": {"effort": "medium"}}


class TestTranslateGptShape:
    """GPT-shaped schema (reasoning.effort only)."""

    def test_anthropic_input_yields_only_reasoning(self, model_schemas):
        """
        What it does: Claude Code style input against gpt-5.6-luna.
        Purpose: Only reasoning.effort; never thinking / output_config.
        """
        result = translate_model_request_fields(
            "gpt-5.6-luna", model_schemas.gpt(), CLAUDE_CODE_THINKING, "max"
        )

        assert result == {"reasoning": {"effort": "max"}}
        assert "thinking" not in result and "output_config" not in result

    @pytest.mark.parametrize("effort", ["none", "low", "medium", "high", "xhigh", "max"])
    def test_every_enum_effort_is_forwarded(self, model_schemas, effort):
        """
        What it does: All GPT efforts (OpenAI reasoning_effort style, no thinking).
        Purpose: Exact forwarding.
        """
        assert translate_model_request_fields("g", model_schemas.gpt(), None, effort) == {
            "reasoning": {"effort": effort}
        }

    def test_minimal_is_dropped_with_warning(self, model_schemas, loguru_messages):
        """
        What it does: OpenAI "minimal" against an enum without it.
        Purpose: Dropped + WARNING, nothing sent.
        """
        assert translate_model_request_fields("g", model_schemas.gpt(), None, "minimal") is None
        assert len(_warnings(loguru_messages)) == 1

    def test_thinking_alone_yields_nothing(self, model_schemas):
        """
        What it does: Only a thinking object against a GPT schema.
        Purpose: No thinking section is declared, so nothing is produced.
        """
        assert translate_model_request_fields("g", model_schemas.gpt(), {"type": "adaptive"}, None) is None


class TestTranslateOpenAiEffortOnClaude:
    """OpenAI reasoning_effort against a Claude-shaped schema."""

    @pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh"])
    def test_effort_maps_to_output_config(self, model_schemas, effort):
        """
        What it does: reasoning_effort with names shared with the Claude enum.
        Purpose: Forwarded as output_config.effort, never as thinking.
        """
        result = translate_model_request_fields("m", model_schemas.claude(), None, effort)

        assert result == {"output_config": {"effort": effort}}

    @pytest.mark.parametrize("effort", ["none", "minimal"])
    def test_none_and_minimal_do_not_become_disabled_thinking(self, model_schemas, effort):
        """
        What it does: OpenAI-only effort names against a schema that allows thinking.disabled.
        Purpose: Dropped; the gateway must not guess thinking.disabled.
        """
        schema = model_schemas.claude(type_enum=("adaptive", "disabled"))

        assert translate_model_request_fields("m", schema, None, effort) is None


class TestTranslateNoUsableSchema:
    """Everything that is not a usable schema yields None."""

    @pytest.mark.parametrize("schema", [
        None, {}, {"properties": None}, {"properties": []}, {"properties": {"thinking": []}},
        {"properties": {"thinking": {"properties": {"type": {"enum": []}}}}},
        {"properties": {"thinking": {"properties": {"type": {"enum": [1, 2]}}}}},
        {"properties": {"thinking": {"properties": {"type": {"enum": "adaptive"}}}}},
        {"properties": {"thinking": {"properties": {"type": {}}}}},
        {"properties": {"thinking": {"properties": []}}},
        "not json", "[]", "", 42, 3.5, [], MagicMock(), object(),
    ])
    def test_returns_none_and_never_raises(self, schema):
        """
        What it does: Schemas that are missing, empty, malformed or of the wrong type.
        Purpose: Fall back to "send nothing".
        """
        assert translate_model_request_fields("m", schema, {"type": "adaptive"}, "medium") is None

    @pytest.mark.parametrize("keyword", ["$ref", "oneOf", "anyOf", "allOf", "not", "patternProperties"])
    def test_unsupported_keywords_disable_the_translation(self, model_schemas, keyword):
        """
        What it does: A valid-looking schema that also uses an unsupported keyword.
        Purpose: The self-check could not be trusted, so nothing is sent.
        """
        schema = model_schemas.claude()
        schema[keyword] = []

        assert translate_model_request_fields("m", schema, {"type": "adaptive"}, "medium") is None

    def test_property_named_like_a_keyword_is_not_a_keyword(self):
        """
        What it does: A property literally called "not" inside properties.
        Purpose: Property names must not be mistaken for schema keywords.
        """
        schema = {"type": "object", "properties": {
            "not": {"type": "string"},
            "reasoning": {"type": "object", "properties": {"effort": {"enum": ["low"]}}},
        }}

        assert translate_model_request_fields("m", schema, None, "low") == {"reasoning": {"effort": "low"}}

    def test_self_check_failure_returns_none_with_warning(self, model_schemas, loguru_messages):
        """
        What it does: Schema requires both sections but the client only sent an effort.
        Purpose: The result would violate the schema, so nothing is sent (never a 400).
        """
        schema = model_schemas.claude()
        schema["required"] = ["thinking", "output_config"]

        result = translate_model_request_fields("m", schema, None, "high")

        assert result is None
        assert len(_warnings(loguru_messages)) == 1

    def test_unparseable_schema_warns_once(self, loguru_messages):
        """
        What it does: Same broken schema string three times.
        Purpose: One WARNING (de-duplicated).
        """
        for _ in range(3):
            translate_model_request_fields("m", "not json", None, "high")

        assert len(_warnings(loguru_messages)) == 1

    def test_schema_property_without_effort_drops_effort_quietly(self, loguru_messages):
        """
        What it does: Schema that only declares thinking, client sends an effort.
        Purpose: Not a client error -> no WARNING, effort simply not forwarded.
        """
        schema = {"properties": {"thinking": {"properties": {"type": {"enum": ["adaptive"]}}}}}

        result = translate_model_request_fields("m", schema, {"type": "adaptive"}, "high")

        assert result == {"thinking": {"type": "adaptive"}}
        assert not _warnings(loguru_messages)

    def test_hybrid_schema_each_section_is_independent(self):
        """
        What it does: A future schema that declares all three sections.
        Purpose: Each section is filled only when its own enum allows the value.
        """
        schema = {"properties": {
            "thinking": {"properties": {"type": {"enum": ["adaptive"]}}},
            "output_config": {"properties": {"effort": {"enum": ["low", "high"]}}},
            "reasoning": {"properties": {"effort": {"enum": ["high", "max"]}}},
        }}

        assert translate_model_request_fields("m", schema, {"type": "adaptive"}, "high") == {
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "high"},
            "reasoning": {"effort": "high"},
        }
        assert translate_model_request_fields("m", schema, None, "max") == {"reasoning": {"effort": "max"}}


class TestOutputsAlwaysSatisfyTheSchema:
    """Property: whatever the client sends, the output obeys the schema."""

    @settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        model_id=st.sampled_from(["claude-sonnet-5.5", "claude-sonnet-5", "claude-opus-5.5",
                                  "claude-fable-5.1", "gpt-5.6-luna"]),
        thinking=st.one_of(
            st.none(), st.text(max_size=10), st.integers(),
            st.dictionaries(
                st.sampled_from(["type", "display", "budget_tokens", "x"]),
                st.one_of(st.none(), st.text(max_size=12), st.integers(), st.booleans(),
                          st.sampled_from(["adaptive", "disabled", "between_tools", "summarized",
                                           "omitted", "updates", "enabled"])),
                max_size=4,
            ),
        ),
        effort=st.one_of(
            st.none(), st.text(max_size=12), st.integers(), st.booleans(),
            st.sampled_from(["none", "minimal", "low", "medium", "high", "xhigh", "max", "Medium"]),
        ),
    )
    def test_output_is_none_or_schema_valid(self, model_schemas, model_id, thinking, effort):
        """
        What it does: Random thinking / effort against every fixture schema.
        Purpose: Output is None or satisfies the strict schema (additionalProperties: false included).
        """
        reset_warning_dedup()
        schema = model_schemas.catalogue()[model_id]

        result = translate_model_request_fields(model_id, schema, thinking, effort)

        assert result is None or validate_against_schema(result, schema)
        if result is not None:
            assert set(result) <= set(schema["properties"])
            if "thinking" in result:
                assert result["thinking"]["type"] in schema["properties"]["thinking"]["properties"]["type"]["enum"]

    def test_cross_validates_with_jsonschema(self, model_schemas):
        """
        What it does: Compares the home-made validator with the jsonschema package, when installed.
        Purpose: Guard against validator drift on the supported subset.
        """
        jsonschema = pytest.importorskip("jsonschema")
        candidates = [
            {"thinking": {"type": "adaptive"}}, {"thinking": {"type": "adaptive", "display": "updates"}},
            {"thinking": {"type": "x"}}, {"output_config": {"effort": "max"}},
            {"output_config": {"effort": True}}, {"output_config": {"effort": 1}},
            {"reasoning": {"effort": "none"}}, {"reasoning": {}}, {"other": 1}, {}, {"thinking": []},
        ]
        for schema in model_schemas.catalogue().values():
            if schema is None:
                continue
            for candidate in candidates:
                assert validate_against_schema(candidate, schema) == jsonschema.Draft202012Validator(schema).is_valid(candidate)


# =============================================================================
# Logging
# =============================================================================

class TestLogging:
    """WARNING de-duplication and log safety."""

    def test_same_drop_warns_once_then_debug(self, model_schemas, loguru_messages):
        """
        What it does: The same invalid effort three times.
        Purpose: One WARNING, the rest DEBUG.
        """
        for _ in range(3):
            translate_model_request_fields("m", model_schemas.claude(), None, "ultra")

        assert len(_warnings(loguru_messages)) == 1
        assert len([m for m in loguru_messages if m.startswith("DEBUG|") and "ultra" in m]) == 2

    def test_new_model_or_value_warns_again(self, model_schemas, loguru_messages):
        """
        What it does: Same value on another model, and another value on the same model.
        Purpose: De-duplication is per (model, field, value).
        """
        translate_model_request_fields("a", model_schemas.claude(), None, "ultra")
        translate_model_request_fields("b", model_schemas.claude(), None, "ultra")
        translate_model_request_fields("a", model_schemas.claude(), None, "mega")

        assert len(_warnings(loguru_messages)) == 3

    def test_dedup_set_is_bounded(self, model_schemas, loguru_messages):
        """
        What it does: Floods the module with MAX_WARNED_KEYS + 100 distinct invalid values.
        Purpose: Memory is bounded and warnings stop at the cap.
        """
        from kiro import model_request_fields as module

        for i in range(MAX_WARNED_KEYS + 100):
            translate_model_request_fields("m", model_schemas.claude(), None, f"bad-{i}")

        assert len(module._warned_keys) == MAX_WARNED_KEYS
        assert len(_warnings(loguru_messages)) == MAX_WARNED_KEYS

    def test_long_value_is_truncated_in_logs(self, model_schemas, loguru_messages):
        """
        What it does: A 5000-character effort value.
        Purpose: The log line stays bounded.
        """
        translate_model_request_fields("m", model_schemas.claude(), None, "x" * 5000)

        line = _warnings(loguru_messages)[0]
        assert "x" * (MAX_LOGGED_VALUE_LENGTH + 10) not in line
        assert len(line) < 400

    def test_control_characters_are_escaped(self, model_schemas, loguru_messages):
        """
        What it does: Effort with a newline and an ESC byte.
        Purpose: Log injection is impossible: repr escapes them.
        """
        translate_model_request_fields("m", model_schemas.claude(), None, "a\nb\x1b[31m")

        line = _warnings(loguru_messages)[0]
        assert "\n" not in line.strip("\n")
        assert "\x1b" not in line
        assert "\\n" in line

    def test_no_unrelated_content_is_logged(self, model_schemas, loguru_messages):
        """
        What it does: Sentinel text inside unrelated thinking keys, with an invalid type and effort.
        Purpose: Logs mention only the rejected value itself, never other client content.
        """
        thinking = {"type": "bad", "budget_tokens": "SECRET-SENTINEL", "note": "SECRET-SENTINEL"}
        translate_model_request_fields("m", model_schemas.claude(), thinking, "ultra")

        assert not [m for m in loguru_messages if "SECRET-SENTINEL" in m]


class TestSummarize:
    """summarize_fields() output."""

    def test_summary(self):
        """
        What it does: Summarizes a typical result and None.
        Purpose: Key paths and values only.
        """
        fields = {"thinking": {"type": "adaptive"}, "output_config": {"effort": "medium"}}

        assert summarize_fields(fields) == "thinking.type=adaptive, output_config.effort=medium"
        assert summarize_fields(None) == "none"
        assert summarize_fields({}) == "none"


# =============================================================================
# resolve_additional_model_request_fields
# =============================================================================

class TestResolve:
    """Config switch and cache handling of the converter entry point."""

    def test_uses_cache_schema(self, schema_model_cache):
        """
        What it does: Real cache, sonnet-5.5, Claude Code style input.
        Purpose: Same as the pure translation.
        """
        result = resolve_additional_model_request_fields(
            "claude-sonnet-5.5", schema_model_cache, CLAUDE_CODE_THINKING, "medium"
        )

        assert result == {"thinking": {"type": "adaptive"}, "output_config": {"effort": "medium"}}

    @pytest.mark.parametrize("model_id", ["claude-sonnet-4.5", "claude-haiku-4.5", "deepseek-3.2", "unknown-model"])
    def test_models_without_schema_get_nothing(self, schema_model_cache, model_id):
        """
        What it does: Models with a null schema and an uncached model.
        Purpose: No schema -> None.
        """
        assert resolve_additional_model_request_fields(
            model_id, schema_model_cache, CLAUDE_CODE_THINKING, "medium"
        ) is None

    def test_switch_off_returns_none_without_touching_the_cache(self):
        """
        What it does: Disables the switch and passes a MagicMock cache.
        Purpose: Old behavior exactly; the cache is not even consulted.
        """
        cache = MagicMock()
        with patch("kiro.model_request_fields.FORWARD_MODEL_REQUEST_FIELDS", False):
            assert resolve_additional_model_request_fields("m", cache, {"type": "adaptive"}, "high") is None

        cache.get_additional_fields_schema.assert_not_called()

    def test_none_cache_returns_none(self):
        """
        What it does: model_cache=None.
        Purpose: Callers without a cache keep the old behavior.
        """
        assert resolve_additional_model_request_fields("m", None, {"type": "adaptive"}, "high") is None

    def test_magicmock_cache_returns_none(self):
        """
        What it does: A MagicMock cache (the shape used by older route tests).
        Purpose: Its MagicMock schema is "unparseable" -> None, no exception.
        """
        assert resolve_additional_model_request_fields("m", MagicMock(), {"type": "adaptive"}, "high") is None

    def test_object_without_getter_returns_none(self):
        """
        What it does: A cache object lacking get_additional_fields_schema.
        Purpose: Duck typing must not raise.
        """
        assert resolve_additional_model_request_fields("m", object(), {"type": "adaptive"}, "high") is None

    def test_debug_log_summarizes_without_content(self, schema_model_cache, loguru_messages):
        """
        What it does: A request whose thinking has a sentinel in an unrelated key.
        Purpose: DEBUG shows effort=medium; the sentinel and the raw thinking dict never appear.
        """
        thinking = {"type": "adaptive", "display": "updates", "note": "SECRET-SENTINEL"}
        resolve_additional_model_request_fields("claude-sonnet-5.5", schema_model_cache, thinking, "medium")

        joined = "\n".join(loguru_messages)
        assert "effort=medium" in joined
        assert "SECRET-SENTINEL" not in joined
        assert "'display'" not in joined


def test_constants_match_the_kiro_contract():
    """
    What it does: Checks the two Kiro key names.
    Purpose: Typos here would silently disable the whole feature.
    """
    assert ADDITIONAL_FIELDS_KEY == "additionalModelRequestFields"
    assert SCHEMA_FIELD_NAME == "additionalModelRequestFieldsSchema"
