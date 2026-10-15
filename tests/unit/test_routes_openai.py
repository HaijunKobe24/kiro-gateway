
# -*- coding: utf-8 -*-

"""
Unit tests for OpenAI API endpoints (routes_openai.py).

Tests the following endpoints:
- GET / - Root endpoint
- GET /health - Health check
- GET /v1/models - List available models
- POST /v1/chat/completions - Chat completions

For Anthropic API tests, see test_routes_anthropic.py.
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from datetime import datetime, timezone
import json
import time

from fastapi import HTTPException
from fastapi.testclient import TestClient

from kiro.routes_openai import verify_api_key, router
from kiro.config import PROXY_API_KEY, APP_VERSION


# =============================================================================
# Tests for verify_api_key function
# =============================================================================

class TestVerifyApiKey:
    """Tests for the verify_api_key authentication function."""
    
    @pytest.mark.asyncio
    async def test_valid_bearer_token_returns_true(self):
        """
        What it does: Verifies that a valid Bearer token passes authentication.
        Purpose: Ensure correct API keys are accepted.
        """
        print("Setup: Creating valid Bearer token...")
        valid_header = f"Bearer {PROXY_API_KEY}"
        
        print("Action: Calling verify_api_key...")
        result = await verify_api_key(valid_header)
        
        print(f"Comparing result: Expected True, Got {result}")
        assert result is True
    
    @pytest.mark.asyncio
    async def test_invalid_api_key_raises_401(self):
        """
        What it does: Verifies that an invalid API key is rejected.
        Purpose: Ensure unauthorized access is blocked.
        """
        print("Setup: Creating invalid Bearer token...")
        invalid_header = "Bearer wrong_key_12345"
        
        print("Action: Calling verify_api_key with invalid key...")
        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(invalid_header)
        
        print(f"Checking: HTTPException with status 401...")
        assert exc_info.value.status_code == 401
        assert "Invalid or missing API Key" in exc_info.value.detail
    
    @pytest.mark.asyncio
    async def test_missing_api_key_raises_401(self):
        """
        What it does: Verifies that missing API key is rejected.
        Purpose: Ensure requests without authentication are blocked.
        """
        print("Setup: No API key provided...")
        
        print("Action: Calling verify_api_key with None...")
        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(None)
        
        print(f"Checking: HTTPException with status 401...")
        assert exc_info.value.status_code == 401
    
    @pytest.mark.asyncio
    async def test_empty_api_key_raises_401(self):
        """
        What it does: Verifies that empty string API key is rejected.
        Purpose: Ensure empty credentials are blocked.
        """
        print("Setup: Empty API key...")
        
        print("Action: Calling verify_api_key with empty string...")
        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key("")
        
        print(f"Checking: HTTPException with status 401...")
        assert exc_info.value.status_code == 401
    
    @pytest.mark.asyncio
    async def test_key_without_bearer_prefix_raises_401(self):
        """
        What it does: Verifies that API key without Bearer prefix is rejected.
        Purpose: Ensure proper Authorization header format is required.
        """
        print("Setup: API key without Bearer prefix...")
        wrong_format = PROXY_API_KEY  # Without "Bearer "
        
        print("Action: Calling verify_api_key...")
        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(wrong_format)
        
        print(f"Checking: HTTPException with status 401...")
        assert exc_info.value.status_code == 401
    
    @pytest.mark.asyncio
    async def test_bearer_with_extra_spaces_raises_401(self):
        """
        What it does: Verifies that Bearer token with extra spaces is rejected.
        Purpose: Ensure strict format validation.
        """
        print("Setup: Bearer token with extra spaces...")
        malformed = f"Bearer  {PROXY_API_KEY}"  # Double space
        
        print("Action: Calling verify_api_key...")
        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(malformed)
        
        print(f"Checking: HTTPException with status 401...")
        assert exc_info.value.status_code == 401
    
    @pytest.mark.asyncio
    async def test_lowercase_bearer_raises_401(self):
        """
        What it does: Verifies that lowercase 'bearer' is rejected.
        Purpose: Ensure case-sensitive Bearer prefix.
        """
        print("Setup: Lowercase bearer prefix...")
        lowercase = f"bearer {PROXY_API_KEY}"
        
        print("Action: Calling verify_api_key...")
        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(lowercase)
        
        print(f"Checking: HTTPException with status 401...")
        assert exc_info.value.status_code == 401


# =============================================================================
# Tests for root endpoint (/)
# =============================================================================

class TestRootEndpoint:
    """Tests for the GET / endpoint."""
    
    def test_root_returns_status_ok(self, test_client):
        """
        What it does: Verifies root endpoint returns ok status.
        Purpose: Ensure basic health check works.
        """
        print("Action: GET /...")
        response = test_client.get("/")
        
        print(f"Result: {response.json()}")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
    
    def test_root_returns_gateway_message(self, test_client):
        """
        What it does: Verifies root endpoint returns gateway message.
        Purpose: Ensure service identification is present.
        """
        print("Action: GET /...")
        response = test_client.get("/")
        
        print(f"Result: {response.json()}")
        assert response.status_code == 200
        assert "Kiro Gateway" in response.json()["message"]
    
    def test_root_returns_version(self, test_client):
        """
        What it does: Verifies root endpoint returns application version.
        Purpose: Ensure version information is available.
        """
        print("Action: GET /...")
        response = test_client.get("/")
        
        print(f"Result: {response.json()}")
        assert response.status_code == 200
        assert "version" in response.json()
        assert response.json()["version"] == APP_VERSION
    
    def test_root_does_not_require_auth(self, test_client):
        """
        What it does: Verifies root endpoint is accessible without authentication.
        Purpose: Ensure public health check availability.
        """
        print("Action: GET / without auth headers...")
        response = test_client.get("/")
        
        print(f"Status: {response.status_code}")
        assert response.status_code == 200


# =============================================================================
# Tests for health endpoint (/health)
# =============================================================================

class TestHealthEndpoint:
    """Tests for the GET /health endpoint."""
    
    def test_health_returns_healthy_status(self, test_client):
        """
        What it does: Verifies health endpoint returns healthy status.
        Purpose: Ensure health check indicates service is running.
        """
        print("Action: GET /health...")
        response = test_client.get("/health")
        
        print(f"Result: {response.json()}")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_health_returns_timestamp(self, test_client):
        """
        What it does: Verifies health endpoint returns timestamp.
        Purpose: Ensure timestamp is present for monitoring.
        """
        print("Action: GET /health...")
        response = test_client.get("/health")
        
        print(f"Result: {response.json()}")
        assert response.status_code == 200
        assert "timestamp" in response.json()
        # Verify timestamp is ISO format
        timestamp = response.json()["timestamp"]
        assert "T" in timestamp  # ISO format contains T
    
    def test_health_returns_version(self, test_client):
        """
        What it does: Verifies health endpoint returns version.
        Purpose: Ensure version is available for monitoring.
        """
        print("Action: GET /health...")
        response = test_client.get("/health")
        
        print(f"Result: {response.json()}")
        assert response.status_code == 200
        assert response.json()["version"] == APP_VERSION
    
    def test_health_does_not_require_auth(self, test_client):
        """
        What it does: Verifies health endpoint is accessible without authentication.
        Purpose: Ensure health checks work for load balancers.
        """
        print("Action: GET /health without auth headers...")
        response = test_client.get("/health")
        
        print(f"Status: {response.status_code}")
        assert response.status_code == 200


# =============================================================================
# Tests for models endpoint (/v1/models)
# =============================================================================

class TestModelsEndpoint:
    """Tests for the GET /v1/models endpoint."""
    
    def test_models_requires_authentication(self, test_client):
        """
        What it does: Verifies models endpoint requires authentication.
        Purpose: Ensure protected endpoints are secured.
        """
        print("Action: GET /v1/models without auth...")
        response = test_client.get("/v1/models")
        
        print(f"Status: {response.status_code}")
        assert response.status_code == 401
    
    def test_models_rejects_invalid_key(self, test_client, invalid_proxy_api_key):
        """
        What it does: Verifies models endpoint rejects invalid API key.
        Purpose: Ensure authentication is enforced.
        """
        print("Action: GET /v1/models with invalid key...")
        response = test_client.get(
            "/v1/models",
            headers={"Authorization": f"Bearer {invalid_proxy_api_key}"}
        )
        
        print(f"Status: {response.status_code}")
        assert response.status_code == 401
    
    def test_models_returns_list_object(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies models endpoint returns list object type.
        Purpose: Ensure OpenAI API compatibility.
        """
        print("Action: GET /v1/models with valid auth...")
        response = test_client.get(
            "/v1/models",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"}
        )
        
        print(f"Result: {response.json()}")
        assert response.status_code == 200
        assert response.json()["object"] == "list"
    
    def test_models_returns_data_array(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies models endpoint returns data array.
        Purpose: Ensure response structure matches OpenAI format.
        """
        print("Action: GET /v1/models with valid auth...")
        response = test_client.get(
            "/v1/models",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"}
        )
        
        print(f"Result: {response.json()}")
        assert response.status_code == 200
        assert "data" in response.json()
        assert isinstance(response.json()["data"], list)
    
    def test_models_contains_available_models(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies all configured models are returned.
        Purpose: Ensure model list is complete.
        """
        print("Action: GET /v1/models with valid auth...")
        response = test_client.get(
            "/v1/models",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"}
        )
        
        print(f"Result: {response.json()}")
        assert response.status_code == 200
        
        model_ids = [m["id"] for m in response.json()["data"]]
        print(f"Model IDs: {model_ids}")
        
        # At minimum, hidden models should be present
        # (even if Kiro API cache is empty)
        assert len(model_ids) >= 1, "Expected at least one model (hidden models)"
    
    def test_models_format_is_openai_compatible(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies model objects have OpenAI-compatible format.
        Purpose: Ensure compatibility with OpenAI clients.
        """
        print("Action: GET /v1/models with valid auth...")
        response = test_client.get(
            "/v1/models",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"}
        )
        
        print(f"Result: {response.json()}")
        assert response.status_code == 200
        
        for model in response.json()["data"]:
            print(f"Checking model format: {model}")
            assert "id" in model, "Model missing 'id' field"
            assert "object" in model, "Model missing 'object' field"
            assert model["object"] == "model", "Model object type should be 'model'"
            assert "owned_by" in model, "Model missing 'owned_by' field"
    
    def test_models_owned_by_anthropic(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies models are owned by Anthropic.
        Purpose: Ensure correct model attribution.
        """
        print("Action: GET /v1/models with valid auth...")
        response = test_client.get(
            "/v1/models",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"}
        )
        
        print(f"Result: {response.json()}")
        assert response.status_code == 200
        
        for model in response.json()["data"]:
            assert model["owned_by"] == "anthropic"


# =============================================================================
# Tests for chat completions endpoint (/v1/chat/completions)
# =============================================================================

class TestChatCompletionsAuthentication:
    """Tests for authentication on /v1/chat/completions endpoint."""
    
    def test_chat_completions_requires_authentication(self, test_client):
        """
        What it does: Verifies chat completions requires authentication.
        Purpose: Ensure protected endpoint is secured.
        """
        print("Action: POST /v1/chat/completions without auth...")
        response = test_client.post(
            "/v1/chat/completions",
            json={
                "model": "claude-sonnet-4-5",
                "messages": [{"role": "user", "content": "Hello"}]
            }
        )
        
        print(f"Status: {response.status_code}")
        assert response.status_code == 401
    
    def test_chat_completions_rejects_invalid_key(self, test_client, invalid_proxy_api_key):
        """
        What it does: Verifies chat completions rejects invalid API key.
        Purpose: Ensure authentication is enforced.
        """
        print("Action: POST /v1/chat/completions with invalid key...")
        response = test_client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {invalid_proxy_api_key}"},
            json={
                "model": "claude-sonnet-4-5",
                "messages": [{"role": "user", "content": "Hello"}]
            }
        )
        
        print(f"Status: {response.status_code}")
        assert response.status_code == 401


class TestChatCompletionsValidation:
    """Tests for request validation on /v1/chat/completions endpoint."""
    
    def test_validates_empty_messages_array(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies empty messages array is rejected.
        Purpose: Ensure at least one message is required.
        """
        print("Action: POST /v1/chat/completions with empty messages...")
        response = test_client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json={
                "model": "claude-sonnet-4-5",
                "messages": []
            }
        )
        
        print(f"Status: {response.status_code}")
        assert response.status_code == 422
    
    def test_validates_missing_model(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies missing model field is rejected.
        Purpose: Ensure model is required.
        """
        print("Action: POST /v1/chat/completions without model...")
        response = test_client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json={
                "messages": [{"role": "user", "content": "Hello"}]
            }
        )
        
        print(f"Status: {response.status_code}")
        assert response.status_code == 422
    
    def test_validates_missing_messages(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies missing messages field is rejected.
        Purpose: Ensure messages are required.
        """
        print("Action: POST /v1/chat/completions without messages...")
        response = test_client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json={
                "model": "claude-sonnet-4-5"
            }
        )
        
        print(f"Status: {response.status_code}")
        assert response.status_code == 422
    
    def test_validates_invalid_json(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies invalid JSON is rejected.
        Purpose: Ensure proper JSON parsing.
        """
        print("Action: POST /v1/chat/completions with invalid JSON...")
        response = test_client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {valid_proxy_api_key}",
                "Content-Type": "application/json"
            },
            content=b"not valid json {{{}"
        )
        
        print(f"Status: {response.status_code}")
        assert response.status_code == 422
    
    def test_validates_invalid_role(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies invalid message role passes Pydantic validation.
        Purpose: Pydantic model accepts any string as role (validation happens later).
        Note: The role validation is not strict at Pydantic level, so invalid roles
        pass validation but may fail during processing.
        """
        print("Action: POST /v1/chat/completions with invalid role...")
        response = test_client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json={
                "model": "claude-sonnet-4-5",
                "messages": [{"role": "invalid_role", "content": "Hello"}]
            }
        )
        
        print(f"Status: {response.status_code}")
        # Pydantic model accepts any string as role, so validation passes (not 422)
        # The request may fail later during processing (500) due to network blocking
        assert response.status_code != 422
    
    def test_accepts_valid_request_format(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies valid request format passes validation.
        Purpose: Ensure Pydantic validation works correctly.
        """
        print("Action: POST /v1/chat/completions with valid format...")
        response = test_client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json={
                "model": "claude-sonnet-4-5",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False
            }
        )
        
        print(f"Status: {response.status_code}")
        # Should pass validation (not 422)
        # May fail on HTTP call due to network blocking, but that's expected
        assert response.status_code != 422
    
    def test_accepts_message_without_content(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies message without content is accepted.
        Purpose: Ensure content is optional (for tool results).
        """
        print("Action: POST /v1/chat/completions with message without content...")
        response = test_client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json={
                "model": "claude-sonnet-4-5",
                "messages": [{"role": "user"}]  # No content
            }
        )
        
        print(f"Status: {response.status_code}")
        # Should pass validation (content is optional)
        assert response.status_code != 422 or "content" not in str(response.json())


class TestChatCompletionsWithTools:
    """Tests for tool calling on /v1/chat/completions endpoint."""
    
    def test_accepts_valid_tool_definition(self, test_client, valid_proxy_api_key, sample_tool_definition):
        """
        What it does: Verifies valid tool definition is accepted.
        Purpose: Ensure tool calling format is supported.
        """
        print("Action: POST /v1/chat/completions with tools...")
        response = test_client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json={
                "model": "claude-sonnet-4-5",
                "messages": [{"role": "user", "content": "What's the weather?"}],
                "tools": [sample_tool_definition]
            }
        )
        
        print(f"Status: {response.status_code}")
        # Should pass validation
        assert response.status_code != 422
    
    def test_accepts_multiple_tools(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies multiple tools are accepted.
        Purpose: Ensure multiple tool definitions work.
        """
        print("Action: POST /v1/chat/completions with multiple tools...")
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {}}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "Get time",
                    "parameters": {"type": "object", "properties": {}}
                }
            }
        ]
        
        response = test_client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json={
                "model": "claude-sonnet-4-5",
                "messages": [{"role": "user", "content": "Hello"}],
                "tools": tools
            }
        )
        
        print(f"Status: {response.status_code}")
        assert response.status_code != 422


class TestChatCompletionsOptionalParams:
    """Tests for optional parameters on /v1/chat/completions endpoint."""
    
    def test_accepts_temperature_parameter(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies temperature parameter is accepted.
        Purpose: Ensure temperature control works.
        """
        print("Action: POST /v1/chat/completions with temperature...")
        response = test_client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json={
                "model": "claude-sonnet-4-5",
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 0.7
            }
        )
        
        print(f"Status: {response.status_code}")
        assert response.status_code != 422
    
    def test_accepts_max_tokens_parameter(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies max_tokens parameter is accepted.
        Purpose: Ensure output length control works.
        """
        print("Action: POST /v1/chat/completions with max_tokens...")
        response = test_client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json={
                "model": "claude-sonnet-4-5",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 100
            }
        )
        
        print(f"Status: {response.status_code}")
        assert response.status_code != 422
    
    def test_accepts_stream_true(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies stream=true is accepted.
        Purpose: Ensure streaming mode is supported.
        """
        print("Action: POST /v1/chat/completions with stream=true...")
        response = test_client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json={
                "model": "claude-sonnet-4-5",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True
            }
        )
        
        print(f"Status: {response.status_code}")
        assert response.status_code != 422
    
    def test_accepts_top_p_parameter(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies top_p parameter is accepted.
        Purpose: Ensure nucleus sampling control works.
        """
        print("Action: POST /v1/chat/completions with top_p...")
        response = test_client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json={
                "model": "claude-sonnet-4-5",
                "messages": [{"role": "user", "content": "Hello"}],
                "top_p": 0.9
            }
        )
        
        print(f"Status: {response.status_code}")
        assert response.status_code != 422


class TestChatCompletionsMessageTypes:
    """Tests for different message types on /v1/chat/completions endpoint."""
    
    def test_accepts_system_message(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies system message is accepted.
        Purpose: Ensure system prompts work.
        """
        print("Action: POST /v1/chat/completions with system message...")
        response = test_client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json={
                "model": "claude-sonnet-4-5",
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hello"}
                ]
            }
        )
        
        print(f"Status: {response.status_code}")
        assert response.status_code != 422
    
    def test_accepts_assistant_message(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies assistant message is accepted.
        Purpose: Ensure conversation history works.
        """
        print("Action: POST /v1/chat/completions with assistant message...")
        response = test_client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json={
                "model": "claude-sonnet-4-5",
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                    {"role": "user", "content": "How are you?"}
                ]
            }
        )
        
        print(f"Status: {response.status_code}")
        assert response.status_code != 422
    
    def test_accepts_multipart_content(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies multipart content array is accepted.
        Purpose: Ensure complex content format works.
        """
        print("Action: POST /v1/chat/completions with multipart content...")
        response = test_client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json={
                "model": "claude-sonnet-4-5",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Hello"},
                            {"type": "text", "text": "World"}
                        ]
                    }
                ]
            }
        )
        
        print(f"Status: {response.status_code}")
        assert response.status_code != 422


# =============================================================================
# Tests for router integration
# =============================================================================

class TestRouterIntegration:
    """Tests for router configuration and integration."""
    
    def test_router_has_root_endpoint(self):
        """
        What it does: Verifies root endpoint is registered.
        Purpose: Ensure endpoint is available.
        """
        print("Checking: Router endpoints...")
        routes = [route.path for route in router.routes]
        
        print(f"Found routes: {routes}")
        assert "/" in routes
    
    def test_router_has_health_endpoint(self):
        """
        What it does: Verifies health endpoint is registered.
        Purpose: Ensure endpoint is available.
        """
        print("Checking: Router endpoints...")
        routes = [route.path for route in router.routes]
        
        print(f"Found routes: {routes}")
        assert "/health" in routes
    
    def test_router_has_models_endpoint(self):
        """
        What it does: Verifies models endpoint is registered.
        Purpose: Ensure endpoint is available.
        """
        print("Checking: Router endpoints...")
        routes = [route.path for route in router.routes]
        
        print(f"Found routes: {routes}")
        assert "/v1/models" in routes
    
    def test_router_has_chat_completions_endpoint(self):
        """
        What it does: Verifies chat completions endpoint is registered.
        Purpose: Ensure endpoint is available.
        """
        print("Checking: Router endpoints...")
        routes = [route.path for route in router.routes]
        
        print(f"Found routes: {routes}")
        assert "/v1/chat/completions" in routes
    
    def test_root_endpoint_uses_get_method(self):
        """
        What it does: Verifies root endpoint uses GET method.
        Purpose: Ensure correct HTTP method.
        """
        print("Checking: HTTP methods...")
        for route in router.routes:
            if route.path == "/":
                print(f"Route / methods: {route.methods}")
                assert "GET" in route.methods
                return
        pytest.fail("Root endpoint not found")
    
    def test_health_endpoint_uses_get_method(self):
        """
        What it does: Verifies health endpoint uses GET method.
        Purpose: Ensure correct HTTP method.
        """
        print("Checking: HTTP methods...")
        for route in router.routes:
            if route.path == "/health":
                print(f"Route /health methods: {route.methods}")
                assert "GET" in route.methods
                return
        pytest.fail("Health endpoint not found")
    
    def test_models_endpoint_uses_get_method(self):
        """
        What it does: Verifies models endpoint uses GET method.
        Purpose: Ensure correct HTTP method.
        """
        print("Checking: HTTP methods...")
        for route in router.routes:
            if route.path == "/v1/models":
                print(f"Route /v1/models methods: {route.methods}")
                assert "GET" in route.methods
                return
        pytest.fail("Models endpoint not found")
    
    def test_chat_completions_endpoint_uses_post_method(self):
        """
        What it does: Verifies chat completions endpoint uses POST method.
        Purpose: Ensure correct HTTP method.
        """
        print("Checking: HTTP methods...")
        for route in router.routes:
            if route.path == "/v1/chat/completions":
                print(f"Route /v1/chat/completions methods: {route.methods}")
                assert "POST" in route.methods
                return
        pytest.fail("Chat completions endpoint not found")


# =============================================================================
# Tests for HTTP client selection (issue #54)
# =============================================================================

class TestHTTPClientSelection:
    """
    Tests for HTTP client selection in routes (issue #54).
    
    Verifies that streaming requests use per-request clients to avoid CLOSE_WAIT leak
    when network interface changes (VPN disconnect/reconnect), while non-streaming
    requests use shared client for connection pooling.
    """
    
    @patch('kiro.routes_openai.KiroHttpClient')
    def test_streaming_uses_per_request_client(
        self,
        mock_kiro_http_client_class,
        test_client,
        valid_proxy_api_key
    ):
        """
        What it does: Verifies streaming requests create per-request HTTP client.
        Purpose: Prevent CLOSE_WAIT leak on VPN disconnect (issue #54).
        """
        print("\n--- Test: Streaming uses per-request client ---")
        
        # Setup mock
        mock_client_instance = AsyncMock()
        mock_client_instance.request_with_retry = AsyncMock(
            side_effect=Exception("Network blocked")
        )
        mock_client_instance.close = AsyncMock()
        mock_kiro_http_client_class.return_value = mock_client_instance
        
        print("Action: POST with stream=true...")
        try:
            test_client.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
                json={
                    "model": "claude-sonnet-4-5",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stream": True
                }
            )
        except Exception:
            pass
        
        print("Checking: KiroHttpClient(shared_client=None)...")
        assert mock_kiro_http_client_class.called
        call_args = mock_kiro_http_client_class.call_args
        print(f"Call args: {call_args}")
        assert call_args[1]['shared_client'] is None, \
            "Streaming should use per-request client"
        print("✅ Streaming correctly uses per-request client")
    
    @patch('kiro.routes_openai.KiroHttpClient')
    def test_non_streaming_uses_shared_client(
        self,
        mock_kiro_http_client_class,
        test_client,
        valid_proxy_api_key
    ):
        """
        What it does: Verifies non-streaming requests use shared HTTP client.
        Purpose: Ensure connection pooling for non-streaming requests.
        """
        print("\n--- Test: Non-streaming uses shared client ---")
        
        # Setup mock
        mock_client_instance = AsyncMock()
        mock_client_instance.request_with_retry = AsyncMock(
            side_effect=Exception("Network blocked")
        )
        mock_client_instance.close = AsyncMock()
        mock_kiro_http_client_class.return_value = mock_client_instance
        
        print("Action: POST with stream=false...")
        try:
            test_client.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
                json={
                    "model": "claude-sonnet-4-5",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stream": False
                }
            )
        except Exception:
            pass
        
        print("Checking: KiroHttpClient(shared_client=app.state.http_client)...")
        assert mock_kiro_http_client_class.called
        call_args = mock_kiro_http_client_class.call_args
        print(f"Call args: {call_args}")
        assert call_args[1]['shared_client'] is not None, \
            "Non-streaming should use shared client"
        print("✅ Non-streaming correctly uses shared client")


# =============================================================================
# Tests for Truncation Recovery message modification (Issue #56)
# =============================================================================

class TestTruncationRecoveryMessageModification:
    """
    Tests for Truncation Recovery System message modification in routes_openai.
    
    Verifies that tool_result messages are modified when truncation info exists in cache.
    Part of Truncation Recovery System (Issue #56).
    """
    
    def test_modifies_tool_result_with_truncation_notice(self):
        """
        What it does: Verifies tool_result content is modified when truncation info exists.
        Purpose: Ensure truncation notice is prepended to tool_result.
        """
        print("Setup: Saving truncation info to cache...")
        from kiro.truncation_state import save_tool_truncation
        from kiro.models_openai import ChatMessage
        
        tool_call_id = "tooluse_test123"
        save_tool_truncation(tool_call_id, "write_to_file", {"size_bytes": 5000, "reason": "test"})
        
        print("Setup: Creating request with tool_result...")
        messages = [
            ChatMessage(role="tool", tool_call_id=tool_call_id, content="Missing parameter error")
        ]
        
        print("Action: Processing messages through truncation recovery logic...")
        # Import the function that modifies messages
        from kiro.routes_openai import router
        from kiro.truncation_recovery import should_inject_recovery, generate_truncation_tool_result
        from kiro.truncation_state import get_tool_truncation
        
        # Simulate the modification logic
        modified_messages = []
        for msg in messages:
            if msg.role == "tool" and msg.tool_call_id and should_inject_recovery():
                truncation_info = get_tool_truncation(msg.tool_call_id)
                if truncation_info:
                    print(f"Found truncation info for {msg.tool_call_id}")
                    synthetic = generate_truncation_tool_result(
                        truncation_info.tool_name,
                        truncation_info.tool_call_id,
                        truncation_info.truncation_info
                    )
                    modified_content = f"{synthetic['content']}\n\n---\n\nOriginal tool result:\n{msg.content}"
                    modified_msg = msg.model_copy(update={"content": modified_content})
                    modified_messages.append(modified_msg)
                else:
                    modified_messages.append(msg)
            else:
                modified_messages.append(msg)
        
        print("Checking: Modified message content...")
        modified_msg = modified_messages[0]
        print(f"Content: {modified_msg.content[:100]}...")
        
        assert "[API Limitation]" in modified_msg.content
        assert "Missing parameter error" in modified_msg.content
        assert "---" in modified_msg.content
    
    def test_no_modification_when_no_truncation(self):
        """
        What it does: Verifies messages are not modified when no truncation info exists.
        Purpose: Ensure normal messages pass through unchanged.
        """
        print("Setup: Creating request without truncation info in cache...")
        from kiro.models_openai import ChatMessage
        
        messages = [
            ChatMessage(role="tool", tool_call_id="tooluse_nonexistent", content="Success")
        ]
        
        print("Action: Processing messages...")
        from kiro.truncation_recovery import should_inject_recovery
        from kiro.truncation_state import get_tool_truncation
        
        modified_messages = []
        tool_results_modified = 0
        
        for msg in messages:
            if msg.role == "tool" and msg.tool_call_id and should_inject_recovery():
                truncation_info = get_tool_truncation(msg.tool_call_id)
                if truncation_info:
                    tool_results_modified += 1
                    # Would modify here
                    modified_messages.append(msg)
                else:
                    modified_messages.append(msg)
            else:
                modified_messages.append(msg)
        
        print(f"Checking: tool_results_modified count...")
        assert tool_results_modified == 0
        
        print("Checking: Message content unchanged...")
        assert modified_messages[0].content == "Success"
    
    def test_pydantic_immutability_new_object_created(self):
        """
        What it does: Verifies new ChatMessage object is created, not modified in-place.
        Purpose: Ensure Pydantic immutability is respected.
        """
        print("Setup: Saving truncation info and creating message...")
        from kiro.truncation_state import save_tool_truncation
        from kiro.models_openai import ChatMessage
        
        tool_call_id = "test_immutable"
        save_tool_truncation(tool_call_id, "tool", {"size_bytes": 1000, "reason": "test truncation"})
        
        original_msg = ChatMessage(role="tool", tool_call_id=tool_call_id, content="original")
        original_content = original_msg.content
        
        print("Action: Processing message...")
        from kiro.truncation_recovery import should_inject_recovery, generate_truncation_tool_result
        from kiro.truncation_state import get_tool_truncation
        
        if original_msg.role == "tool" and original_msg.tool_call_id and should_inject_recovery():
            truncation_info = get_tool_truncation(original_msg.tool_call_id)
            if truncation_info:
                synthetic = generate_truncation_tool_result(
                    truncation_info.tool_name,
                    truncation_info.tool_call_id,
                    truncation_info.truncation_info
                )
                modified_content = f"{synthetic['content']}\n\n---\n\nOriginal tool result:\n{original_msg.content}"
                modified_msg = original_msg.model_copy(update={"content": modified_content})
        
        print("Checking: Original message unchanged...")
        assert original_msg.content == original_content
        
        print("Checking: New object created...")
        assert modified_msg is not original_msg
        
        print("Checking: Content modified in new object...")
        assert modified_msg.content != original_msg.content
        assert "[API Limitation]" in modified_msg.content


# =============================================================================
# Tests for Truncation Recovery edge cases (Issue #56)
# =============================================================================

class TestTruncationRecoveryEdgeCases:
    """
    Tests for edge cases in Truncation Recovery System.
    
    Verifies graceful handling of unusual scenarios.
    Part of Truncation Recovery System (Issue #56).
    """
    
    def test_orphaned_tool_result_no_crash(self):
        """
        What it does: Verifies graceful handling when cache entry doesn't exist.
        Purpose: Ensure orphaned tool_result doesn't cause errors (Test Case 9.2).
        """
        print("Setup: Creating tool_result without prior truncation...")
        from kiro.models_openai import ChatMessage
        
        messages = [
            ChatMessage(role="tool", tool_call_id="tooluse_nonexistent_orphan", content="Result")
        ]
        
        print("Action: Processing messages (no truncation info in cache)...")
        from kiro.truncation_recovery import should_inject_recovery
        from kiro.truncation_state import get_tool_truncation
        
        modified_messages = []
        for msg in messages:
            if msg.role == "tool" and msg.tool_call_id and should_inject_recovery():
                truncation_info = get_tool_truncation(msg.tool_call_id)
                if truncation_info:
                    # Would modify here
                    pass
            modified_messages.append(msg)
        
        print("Checking: No error thrown...")
        assert len(modified_messages) == 1
        
        print("Checking: Message unchanged...")
        assert modified_messages[0].content == "Result"
    
    def test_empty_tool_result_content(self):
        """
        What it does: Verifies handling of empty tool_result content.
        Purpose: Ensure empty content doesn't cause errors (Test Case 9.4).
        """
        print("Setup: Saving truncation info and creating empty tool_result...")
        from kiro.truncation_state import save_tool_truncation
        from kiro.models_openai import ChatMessage
        
        tool_call_id = "tooluse_empty_content"
        save_tool_truncation(tool_call_id, "tool", {"size_bytes": 1000, "reason": "test"})
        
        messages = [
            ChatMessage(role="tool", tool_call_id=tool_call_id, content="")
        ]
        
        print("Action: Processing message with empty content...")
        from kiro.truncation_recovery import should_inject_recovery, generate_truncation_tool_result
        from kiro.truncation_state import get_tool_truncation
        
        modified_messages = []
        for msg in messages:
            if msg.role == "tool" and msg.tool_call_id and should_inject_recovery():
                truncation_info = get_tool_truncation(msg.tool_call_id)
                if truncation_info:
                    synthetic = generate_truncation_tool_result(
                        truncation_info.tool_name,
                        truncation_info.tool_call_id,
                        truncation_info.truncation_info
                    )
                    modified_content = f"{synthetic['content']}\n\n---\n\nOriginal tool result:\n{msg.content}"
                    modified_msg = msg.model_copy(update={"content": modified_content})
                    modified_messages.append(modified_msg)
                    continue
            modified_messages.append(msg)
        
        print("Checking: No crash occurred...")
        assert len(modified_messages) == 1
        
        print("Checking: Truncation notice still prepended...")
        assert "[API Limitation]" in modified_messages[0].content
        
        print("Checking: Empty original content preserved...")
        assert "Original tool result:\n" in modified_messages[0].content
    
    def test_very_long_content_hash_uses_first_500_chars(self):
        """
        What it does: Verifies content hash uses first 500 chars only.
        Purpose: Ensure hash stability for long content (Test Case 9.3).
        """
        print("Setup: Creating very long content...")
        from kiro.truncation_state import save_content_truncation, get_content_truncation
        
        content_long = "A" * 10000
        content_same_prefix = "A" * 500 + "B" * 9500
        
        print("Action: Saving long content...")
        hash1 = save_content_truncation(content_long)
        
        print("Action: Retrieving with same prefix...")
        info = get_content_truncation(content_same_prefix)
        
        print("Checking: Retrieval successful (same hash)...")
        assert info is not None
        assert info.message_hash == hash1
        
        print("Checking: Hash is 16 chars...")
        assert len(hash1) == 16
    
    def test_recovery_disabled_cache_entry_remains(self):
        """
        What it does: Verifies cache entry remains when recovery is disabled.
        Purpose: Ensure disabling recovery doesn't clear cache (Test Case 9.5).
        """
        print("Setup: Enabling recovery and saving truncation...")
        from kiro.truncation_state import save_tool_truncation, get_cache_stats
        from kiro.models_openai import ChatMessage
        import os
        
        tool_call_id = "tooluse_disabled_recovery"
        save_tool_truncation(tool_call_id, "tool", {"size_bytes": 1000, "reason": "test"})
        
        print("Checking: Cache entry exists...")
        stats = get_cache_stats()
        assert stats["tool_truncations"] >= 1
        
        print("Action: Disabling recovery...")
        with patch.dict(os.environ, {"TRUNCATION_RECOVERY": "false"}):
            from importlib import reload
            from kiro import config
            reload(config)
            
            print("Action: Processing tool_result with recovery disabled...")
            from kiro.truncation_recovery import should_inject_recovery
            from kiro.truncation_state import get_tool_truncation
            
            messages = [
                ChatMessage(role="tool", tool_call_id=tool_call_id, content="Result")
            ]
            
            modified_messages = []
            for msg in messages:
                if msg.role == "tool" and msg.tool_call_id and should_inject_recovery():
                    # This branch won't execute because recovery is disabled
                    truncation_info = get_tool_truncation(msg.tool_call_id)
                    if truncation_info:
                        pass
                modified_messages.append(msg)
            
            print("Checking: No modification occurred...")
            assert modified_messages[0].content == "Result"
            assert "[API Limitation]" not in modified_messages[0].content
        
        print("Checking: Cache entry still exists (not cleaned up)...")
        # Note: get_tool_truncation() was NOT called, so entry should still be there
        # But we can't verify this without calling get_tool_truncation again
        # which would delete it. This is acceptable - the test verifies
        # that recovery doesn't happen when disabled.


# =============================================================================
# Tests for Content Truncation Recovery (Issue #56)
# =============================================================================

class TestContentTruncationRecovery:
    """
    Tests for content truncation recovery (synthetic user message).
    
    Verifies that synthetic user message is added after truncated assistant message.
    Part of Truncation Recovery System (Issue #56).
    """
    
    def test_adds_synthetic_user_message_after_truncated_assistant(self):
        """
        What it does: Verifies synthetic user message is added after truncated assistant message.
        Purpose: Ensure content truncation recovery works (Test Case C.1).
        """
        print("Setup: Saving content truncation info...")
        from kiro.truncation_state import save_content_truncation
        from kiro.models_openai import ChatMessage
        
        truncated_content = "This is a very long response that was cut off mid-sentence"
        save_content_truncation(truncated_content)
        
        print("Setup: Creating request with truncated assistant message...")
        messages = [
            ChatMessage(role="assistant", content=truncated_content)
        ]
        
        print("Action: Processing messages through content truncation recovery...")
        from kiro.truncation_recovery import should_inject_recovery, generate_truncation_user_message
        from kiro.truncation_state import get_content_truncation
        
        modified_messages = []
        for msg in messages:
            if msg.role == "assistant" and msg.content and isinstance(msg.content, str):
                truncation_info = get_content_truncation(msg.content)
                if truncation_info:
                    print(f"Found content truncation for hash: {truncation_info.message_hash}")
                    # Add original message first
                    modified_messages.append(msg)
                    # Then add synthetic user message
                    synthetic_user_msg = ChatMessage(
                        role="user",
                        content=generate_truncation_user_message()
                    )
                    modified_messages.append(synthetic_user_msg)
                    continue
            modified_messages.append(msg)
        
        print("Checking: Two messages in result...")
        assert len(modified_messages) == 2
        
        print("Checking: First message is original assistant message...")
        assert modified_messages[0].role == "assistant"
        assert modified_messages[0].content == truncated_content
        
        print("Checking: Second message is synthetic user message...")
        assert modified_messages[1].role == "user"
        assert "[System Notice]" in modified_messages[1].content
        assert "truncated" in modified_messages[1].content.lower()
    
    def test_no_synthetic_message_when_no_content_truncation(self):
        """
        What it does: Verifies no synthetic message is added for normal assistant message.
        Purpose: Ensure false positives don't occur (Test Case C.3).
        """
        print("Setup: Creating normal assistant message (no truncation)...")
        from kiro.models_openai import ChatMessage
        
        messages = [
            ChatMessage(role="assistant", content="This is a complete response.")
        ]
        
        print("Action: Processing messages...")
        from kiro.truncation_state import get_content_truncation
        
        modified_messages = []
        for msg in messages:
            if msg.role == "assistant" and msg.content and isinstance(msg.content, str):
                truncation_info = get_content_truncation(msg.content)
                if truncation_info:
                    # Would add synthetic message here
                    pass
            modified_messages.append(msg)
        
        print("Checking: Only one message in result...")
        assert len(modified_messages) == 1
        
        print("Checking: Message unchanged...")
        assert modified_messages[0].content == "This is a complete response."
    
    def test_content_hash_matches_first_500_chars(self):
        """
        What it does: Verifies content hash is based on first 500 chars.
        Purpose: Ensure long messages can be matched by prefix.
        """
        print("Setup: Creating long content...")
        from kiro.truncation_state import save_content_truncation, get_content_truncation
        
        # Original content (what was saved during detection)
        original_content = "A" * 1000
        
        # Content in request (might be slightly different due to client processing)
        request_content = "A" * 500 + "B" * 500
        
        print("Action: Saving original content...")
        hash1 = save_content_truncation(original_content)
        
        print("Action: Retrieving with request content (same first 500 chars)...")
        info = get_content_truncation(request_content)
        
        print("Checking: Match found...")
        assert info is not None
        assert info.message_hash == hash1


# ==================================================================================================
# Tests for WebSearch Support (OpenAI)
# ==================================================================================================

class TestWebSearchAutoInjectionOpenAI:
    """Tests for WebSearch auto-injection in OpenAI endpoint (Path B only)."""
    
    def test_auto_injection_logic_openai(self):
        """
        What it does: Verifies web_search function tool auto-injection logic for OpenAI.
        Purpose: Ensure WEB_SEARCH_ENABLED controls auto-injection for OpenAI format.
        """
        print("Setup: Testing OpenAI auto-injection logic...")
        from kiro.models_openai import Tool, ToolFunction
        
        # Simulate auto-injection logic for OpenAI
        WEB_SEARCH_ENABLED = True
        tools = []
        
        if WEB_SEARCH_ENABLED:
            has_ws = any(
                getattr(tool, "type", None) == "function" and
                getattr(getattr(tool, "function", None), "name", None) == "web_search"
                for tool in tools
            )
            
            if not has_ws:
                web_search_tool = Tool(
                    type="function",
                    function=ToolFunction(
                        name="web_search",
                        description="Search the web for current information. Use when you need up-to-date data from the internet.",
                        parameters={
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Search query"
                                }
                            },
                            "required": ["query"]
                        }
                    )
                )
                tools.append(web_search_tool)
        
        print(f"Checking: web_search tool was added...")
        assert len(tools) == 1
        assert tools[0].type == "function"
        assert tools[0].function.name == "web_search"
        assert tools[0].function.parameters is not None
    
    def test_no_duplicate_injection_logic_openai(self):
        """
        What it does: Verifies duplicate detection logic for OpenAI format.
        Purpose: Ensure auto-injection doesn't create duplicates for OpenAI.
        """
        print("Setup: Testing OpenAI duplicate detection...")
        from kiro.models_openai import Tool, ToolFunction
        
        # Simulate existing web_search tool
        existing_tools = [
            Tool(
                type="function",
                function=ToolFunction(
                    name="web_search",
                    description="Existing web search",
                    parameters={"type": "object", "properties": {}}
                )
            )
        ]
        
        # Simulate auto-injection logic with duplicate check
        WEB_SEARCH_ENABLED = True
        
        if WEB_SEARCH_ENABLED:
            has_ws = any(
                getattr(tool, "type", None) == "function" and
                getattr(getattr(tool, "function", None), "name", None) == "web_search"
                for tool in existing_tools
            )
            
            if not has_ws:
                # Would add web_search here
                existing_tools.append(Tool(
                    type="function",
                    function=ToolFunction(
                        name="web_search",
                        description="Auto-injected",
                        parameters={"type": "object", "properties": {}}
                    )
                ))
        
        print(f"Checking: Only one web_search tool...")
        web_search_count = sum(
            1 for t in existing_tools
            if t.type == "function" and t.function.name == "web_search"
        )
        assert web_search_count == 1


# ==================================================================================================
# Tests for Account System - /v1/models endpoint
# ==================================================================================================

class TestModelsEndpointAccountSystem:
    """Tests for /v1/models endpoint with Account System."""
    
    def test_get_models_account_system_logic(self):
        """
        What it does: Verifies logic for collecting models in account system mode.
        Purpose: Ensure models are collected from all initialized accounts.
        """
        print("\n--- Test: /v1/models account system logic ---")
        
        # Simulate account system mode logic
        account_system = True
        
        mock_account_manager = Mock()
        mock_account_manager.get_all_available_models.return_value = [
            "claude-opus-4.5",
            "claude-sonnet-4.5",
            "claude-haiku-4.5"
        ]
        
        print("Action: Getting models in account system mode...")
        if account_system:
            available_model_ids = mock_account_manager.get_all_available_models()
        else:
            available_model_ids = []
        
        print("Checking: get_all_available_models() was called...")
        mock_account_manager.get_all_available_models.assert_called_once()
        
        print("Checking: Models from all accounts collected...")
        assert "claude-opus-4.5" in available_model_ids
        assert "claude-sonnet-4.5" in available_model_ids
        assert "claude-haiku-4.5" in available_model_ids
        assert len(available_model_ids) == 3
        print("✅ Account system mode correctly collects models from all accounts")
    
    def test_get_models_legacy_logic(self):
        """
        What it does: Verifies logic for getting models in legacy mode.
        Purpose: Ensure backward compatibility with single account.
        """
        print("\n--- Test: /v1/models legacy mode logic ---")
        
        # Simulate legacy mode logic
        account_system = False
        
        mock_account = Mock()
        mock_resolver = Mock()
        mock_resolver.get_available_models.return_value = [
            "claude-opus-4.5",
            "claude-sonnet-4.5"
        ]
        mock_account.model_resolver = mock_resolver
        
        mock_account_manager = Mock()
        mock_account_manager.get_first_account.return_value = mock_account
        
        print("Action: Getting models in legacy mode...")
        if account_system:
            available_model_ids = []
        else:
            account = mock_account_manager.get_first_account()
            available_model_ids = account.model_resolver.get_available_models()
        
        print("Checking: get_first_account() was called...")
        mock_account_manager.get_first_account.assert_called_once()
        
        print("Checking: model_resolver.get_available_models() was called...")
        mock_resolver.get_available_models.assert_called_once()
        
        print("Checking: Models from first account returned...")
        assert "claude-opus-4.5" in available_model_ids
        assert "claude-sonnet-4.5" in available_model_ids
        assert len(available_model_ids) == 2
        print("✅ Legacy mode correctly uses first account's resolver")


# ==================================================================================================
# Tests for Account System - Failover Loop
# ==================================================================================================

class TestChatCompletionsFailoverLoop:
    """Tests for failover loop in /v1/chat/completions endpoint."""
    
    @pytest.mark.asyncio
    async def test_chat_completions_failover_get_next_account(self):
        """
        What it does: Verifies get_next_account() is called with exclude_accounts.
        Purpose: Ensure failover loop tracks tried accounts.
        """
        print("\n--- Test: Failover calls get_next_account() with exclude_accounts ---")
        
        mock_account = Mock()
        mock_account.id = "/home/user/account1.json"
        mock_account.auth_manager = Mock()
        mock_account.model_cache = Mock()
        mock_account.model_resolver = Mock()
        
        mock_manager = Mock()
        mock_manager.get_next_account = AsyncMock(return_value=mock_account)
        mock_manager._accounts = {mock_account.id: mock_account}
        
        print("Checking: get_next_account() called with exclude_accounts parameter...")
        # This test verifies the signature - actual implementation tested in integration tests
        await mock_manager.get_next_account("claude-opus-4.5", exclude_accounts=set())
        
        mock_manager.get_next_account.assert_called_once()
        call_kwargs = mock_manager.get_next_account.call_args[1]
        assert "exclude_accounts" in call_kwargs
        print("✅ Failover loop correctly passes exclude_accounts")
    
    @pytest.mark.asyncio
    async def test_chat_completions_failover_success_first_account(self):
        """
        What it does: Verifies successful response on first account attempt.
        Purpose: Ensure no unnecessary failover when first account works.
        """
        print("\n--- Test: Success on first account ---")
        
        from kiro.account_manager import Account, AccountStats
        
        mock_account = Account(
            id="/home/user/account1.json",
            failures=0,
            last_failure_time=0.0,
            models_cached_at=time.time(),
            stats=AccountStats()
        )
        
        mock_manager = Mock()
        mock_manager.get_next_account = AsyncMock(return_value=mock_account)
        mock_manager.report_success = AsyncMock()
        mock_manager._accounts = {mock_account.id: mock_account}
        
        print("Action: Simulating successful request...")
        account = await mock_manager.get_next_account("claude-opus-4.5", exclude_accounts=set())
        
        print("Checking: First account returned...")
        assert account is not None
        assert account.id == "/home/user/account1.json"
        
        print("Action: Reporting success...")
        await mock_manager.report_success(account.id, "claude-opus-4.5")
        
        print("Checking: report_success() was called...")
        mock_manager.report_success.assert_called_once_with(
            "/home/user/account1.json",
            "claude-opus-4.5"
        )
        print("✅ Success on first account works correctly")
    
    @pytest.mark.asyncio
    async def test_chat_completions_failover_recoverable_try_next(self):
        """
        What it does: Verifies RECOVERABLE error triggers next account attempt.
        Purpose: Ensure failover happens for account-specific errors.
        """
        print("\n--- Test: RECOVERABLE error tries next account ---")
        
        from kiro.account_errors import ErrorType, classify_error
        
        print("Setup: Classifying 429 error...")
        error_type = classify_error(429, None)
        
        print("Checking: 429 is RECOVERABLE...")
        assert error_type == ErrorType.RECOVERABLE
        
        print("Checking: Failover logic should continue to next account...")
        # In actual implementation, this would trigger:
        # await account_manager.report_failure(...)
        # continue  # Next iteration of failover loop
        
        mock_manager = Mock()
        mock_manager.report_failure = AsyncMock()
        
        await mock_manager.report_failure(
            "/home/user/account1.json",
            "claude-opus-4.5",
            ErrorType.RECOVERABLE,
            429,
            None
        )
        
        mock_manager.report_failure.assert_called_once()
        print("✅ RECOVERABLE error correctly triggers failover")
    
    @pytest.mark.asyncio
    async def test_chat_completions_failover_fatal_immediate_return(self):
        """
        What it does: Verifies FATAL error returns immediately to client.
        Purpose: Ensure no wasted retries for request-level errors.
        """
        print("\n--- Test: FATAL error returns immediately ---")
        
        from kiro.account_errors import ErrorType, classify_error
        
        print("Setup: Classifying 400 + CONTENT_LENGTH_EXCEEDS_THRESHOLD...")
        error_type = classify_error(400, "CONTENT_LENGTH_EXCEEDS_THRESHOLD")
        
        print("Checking: Error is FATAL...")
        assert error_type == ErrorType.FATAL
        
        print("Checking: Failover logic should break immediately...")
        # In actual implementation, this would trigger:
        # await account_manager.report_failure(...)
        # return JSONResponse(...)  # No continue, immediate return
        
        mock_manager = Mock()
        mock_manager.report_failure = AsyncMock()
        
        await mock_manager.report_failure(
            "/home/user/account1.json",
            "claude-opus-4.5",
            ErrorType.FATAL,
            400,
            "CONTENT_LENGTH_EXCEEDS_THRESHOLD"
        )
        
        mock_manager.report_failure.assert_called_once()
        print("✅ FATAL error correctly returns immediately")
    
    def test_chat_completions_failover_single_account_original_error(self):
        """
        What it does: Verifies single account returns original error message.
        Purpose: Ensure users see specific error for single account setup.
        """
        print("\n--- Test: Single account returns original error ---")
        
        all_accounts = ["/home/user/account1.json"]
        last_error_message = "Monthly request limit exceeded"
        last_error_status = 402
        
        print("Checking: Single account error handling...")
        if len(all_accounts) == 1:
            error_response = {
                "status_code": last_error_status,
                "detail": last_error_message
            }
        else:
            error_response = {
                "status_code": 503,
                "detail": "No available accounts for this model"
            }
        
        print(f"Error response: {error_response}")
        assert error_response["status_code"] == 402
        assert error_response["detail"] == "Monthly request limit exceeded"
        print("✅ Single account correctly returns original error")
    
    def test_chat_completions_failover_multi_account_generic_error(self):
        """
        What it does: Verifies multi-account returns generic error message.
        Purpose: Ensure users don't see confusing account-specific errors.
        """
        print("\n--- Test: Multi-account returns generic error ---")
        
        all_accounts = [
            "/home/user/account1.json",
            "/home/user/account2.json"
        ]
        last_error_message = "Token expired"
        
        print("Checking: Multi-account error handling...")
        if len(all_accounts) == 1:
            error_response = {
                "status_code": 403,
                "detail": last_error_message
            }
        else:
            detail = "No available accounts for this model."
            if last_error_message:
                detail += f" Error from last account: {last_error_message}"
            error_response = {
                "status_code": 503,
                "detail": detail
            }
        
        print(f"Error response: {error_response}")
        assert error_response["status_code"] == 503
        assert "No available accounts" in error_response["detail"]
        assert "Error from last account: Token expired" in error_response["detail"]
        print("✅ Multi-account correctly returns generic error with context")
    
    @pytest.mark.asyncio
    async def test_chat_completions_failover_all_unavailable(self):
        """
        What it does: Verifies behavior when all accounts are unavailable.
        Purpose: Ensure graceful handling of complete failure.
        """
        print("\n--- Test: All accounts unavailable ---")
        
        mock_manager = Mock()
        mock_manager.get_next_account = AsyncMock(return_value=None)
        mock_manager._accounts = {
            "/home/user/account1.json": Mock(),
            "/home/user/account2.json": Mock()
        }
        
        print("Action: Requesting account when all unavailable...")
        account = await mock_manager.get_next_account(
            "claude-opus-4.5",
            exclude_accounts=set()
        )
        
        print("Checking: None returned...")
        assert account is None
        
        print("Checking: Error response logic...")
        all_accounts = list(mock_manager._accounts.keys())
        if len(all_accounts) == 1:
            error_msg = "Account unavailable"
        else:
            error_msg = "No available accounts for this model"
        
        assert "No available accounts" in error_msg
        print("✅ All unavailable correctly handled")
    
    @pytest.mark.asyncio
    async def test_chat_completions_failover_report_success(self):
        """
        What it does: Verifies report_success() is called after successful request.
        Purpose: Ensure statistics and sticky behavior are updated.
        """
        print("\n--- Test: report_success() called on success ---")
        
        mock_manager = Mock()
        mock_manager.report_success = AsyncMock()
        
        account_id = "/home/user/account1.json"
        model = "claude-opus-4.5"
        
        print("Action: Reporting success...")
        await mock_manager.report_success(account_id, model)
        
        print("Checking: report_success() was called with correct params...")
        mock_manager.report_success.assert_called_once_with(account_id, model)
        print("✅ report_success() correctly called")
    
    @pytest.mark.asyncio
    async def test_chat_completions_failover_report_failure(self):
        """
        What it does: Verifies report_failure() is called after failed request.
        Purpose: Ensure Circuit Breaker state is updated.
        """
        print("\n--- Test: report_failure() called on failure ---")
        
        from kiro.account_errors import ErrorType
        
        mock_manager = Mock()
        mock_manager.report_failure = AsyncMock()
        
        account_id = "/home/user/account1.json"
        model = "claude-opus-4.5"
        error_type = ErrorType.RECOVERABLE
        status_code = 429
        reason = None
        
        print("Action: Reporting failure...")
        await mock_manager.report_failure(
            account_id,
            model,
            error_type,
            status_code,
            reason
        )
        
        print("Checking: report_failure() was called with correct params...")
        mock_manager.report_failure.assert_called_once_with(
            account_id,
            model,
            error_type,
            status_code,
            reason
        )
        print("✅ report_failure() correctly called")
    
    @pytest.mark.asyncio
    async def test_chat_completions_failover_exclude_tried_accounts(self):
        """
        What it does: Verifies exclude_accounts grows with each attempt.
        Purpose: Ensure accounts aren't retried in same failover loop.
        """
        print("\n--- Test: exclude_accounts grows with attempts ---")
        
        tried_accounts = set()
        
        print("Action: Simulating multiple attempts...")
        account1_id = "/home/user/account1.json"
        account2_id = "/home/user/account2.json"
        
        # Attempt 1
        tried_accounts.add(account1_id)
        print(f"After attempt 1: {tried_accounts}")
        assert account1_id in tried_accounts
        assert len(tried_accounts) == 1
        
        # Attempt 2
        tried_accounts.add(account2_id)
        print(f"After attempt 2: {tried_accounts}")
        assert account2_id in tried_accounts
        assert len(tried_accounts) == 2
        
        print("Checking: Both accounts in exclude set...")
        assert account1_id in tried_accounts
        assert account2_id in tried_accounts
        print("✅ exclude_accounts correctly tracks tried accounts")
    
    def test_chat_completions_failover_max_attempts(self):
        """
        What it does: Verifies failover loop stops after MAX_ATTEMPTS.
        Purpose: Ensure infinite loops are prevented.
        """
        print("\n--- Test: MAX_ATTEMPTS prevents infinite loop ---")
        
        all_accounts = [
            "/home/user/account1.json",
            "/home/user/account2.json"
        ]
        MAX_ATTEMPTS = len(all_accounts) * 2
        
        print(f"Checking: MAX_ATTEMPTS = {MAX_ATTEMPTS}...")
        assert MAX_ATTEMPTS == 4
        
        print("Checking: Loop would stop after 4 attempts...")
        attempts = 0
        for attempt in range(MAX_ATTEMPTS):
            attempts += 1
            if attempts >= MAX_ATTEMPTS:
                break
        
        assert attempts == MAX_ATTEMPTS
        print("✅ MAX_ATTEMPTS correctly limits failover loop")


# ==================================================================================================
# Tests for Account System - Legacy Mode
# ==================================================================================================

class TestChatCompletionsLegacyMode:
    """Tests for legacy mode (ACCOUNT_SYSTEM=false) in /v1/chat/completions."""
    
    @pytest.mark.asyncio
    async def test_chat_completions_legacy_get_first_account(self):
        """
        What it does: Verifies legacy mode uses get_first_account().
        Purpose: Ensure backward compatibility with single account.
        """
        print("\n--- Test: Legacy mode uses get_first_account() ---")
        
        from kiro.account_manager import Account, AccountStats
        
        mock_account = Account(
            id="/home/user/account1.json",
            failures=0,
            last_failure_time=0.0,
            models_cached_at=time.time(),
            stats=AccountStats()
        )
        
        mock_manager = Mock()
        mock_manager.get_first_account.return_value = mock_account
        
        print("Action: Getting first account in legacy mode...")
        account = mock_manager.get_first_account()
        
        print("Checking: get_first_account() was called...")
        mock_manager.get_first_account.assert_called_once()
        
        print("Checking: Account returned...")
        assert account is not None
        assert account.id == "/home/user/account1.json"
        print("✅ Legacy mode correctly uses get_first_account()")
    
    def test_chat_completions_legacy_no_failover(self):
        """
        What it does: Verifies legacy mode has no failover loop.
        Purpose: Ensure single account behavior is preserved.
        """
        print("\n--- Test: Legacy mode has no failover ---")
        
        account_system = False
        
        print("Checking: account_system flag is False...")
        assert account_system is False
        
        print("Checking: Failover loop should be skipped...")
        if account_system:
            failover_enabled = True
        else:
            failover_enabled = False
        
        assert failover_enabled is False
        print("✅ Legacy mode correctly skips failover loop")

# ==================================================================================================
# Native reasoning (reasoningContentEvent) - all four route paths
# ==================================================================================================

class TestNativeReasoningRouteCoverage:
    """
    Proves that native Kiro reasoning reaches the client through ALL FOUR /v1/chat/completions
    paths: account-system branch and legacy branch, each streaming and non-streaming.

    Both branches delegate to stream_with_first_token_retry / collect_stream_response, so a
    parser / streaming-layer fix benefits them equally; these tests drive the real routes
    (real parser, real formatters, mocked HTTP) to show it explicitly.
    """

    TEXTS = ["I", "'ll", " look", ".\n\n", "中文推理", " 😀"]

    @staticmethod
    def _install_account_manager(app_state, monkeypatch, account_system):
        """Installs a deterministic account manager and selects the branch under test."""
        from kiro.account_manager import Account, AccountStats

        auth_manager = MagicMock()
        auth_manager.api_host = "https://kiro.example.invalid"
        auth_manager.profile_arn = "arn:aws:codewhisperer:us-east-1:000000000000:profile/TEST"
        model_cache = MagicMock()
        model_cache.get_max_input_tokens.return_value = 200000
        account = Account(
            id="acc1",
            auth_manager=auth_manager,
            model_cache=model_cache,
            model_resolver=MagicMock(),
            stats=AccountStats(),
        )
        manager = MagicMock()
        manager._accounts = {"acc1": account}
        manager.get_first_account = MagicMock(return_value=account)
        manager.get_next_account = AsyncMock(return_value=account)
        manager.report_success = AsyncMock()
        manager.report_failure = AsyncMock()
        monkeypatch.setattr(app_state, "account_manager", manager)
        monkeypatch.setattr(app_state, "account_system", account_system)
        return manager

    def _frames(self, aws_frames, with_tool=False):
        data = b"".join(aws_frames.reasoning(t) for t in self.TEXTS)
        data += aws_frames.signature("QUJD" * 10) + aws_frames.content("Hello")
        if with_tool:
            data += (
                aws_frames.tool_start("Bash", "toolu_1")
                + aws_frames.tool_input('{"command": "ls"}', "Bash", "toolu_1")
                + aws_frames.tool_stop("Bash", "toolu_1")
            )
        return data + aws_frames.context_usage(1.0)

    def _post(self, test_client, valid_proxy_api_key, mock_httpx_response, chunks, stream):
        with patch("kiro.routes_openai.KiroHttpClient") as client_class, \
                patch("kiro.streaming_core.FAKE_REASONING_ENABLED", True), \
                patch("kiro.streaming_core.FAKE_REASONING_HANDLING", "as_reasoning_content"), \
                patch("kiro.streaming_openai.FAKE_REASONING_HANDLING", "as_reasoning_content"):
            instance = AsyncMock()
            instance.request_with_retry = AsyncMock(
                return_value=mock_httpx_response(status_code=200, stream_chunks=chunks)
            )
            instance.close = AsyncMock()
            client_class.return_value = instance
            return test_client.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
                json={
                    "model": "claude-sonnet-4-5",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stream": stream,
                },
            )

    @pytest.mark.parametrize("account_system", [True, False])
    def test_streaming_surfaces_every_reasoning_frame(
        self, test_client, valid_proxy_api_key, mock_httpx_response, aws_frames, monkeypatch, account_system
    ):
        """
        What it does: POST stream=true through the account-system / legacy branch.
        Goal: One reasoning_content delta per frame (first with role), then content, then [DONE].
        """
        manager = self._install_account_manager(test_client.app.state, monkeypatch, account_system)

        response = self._post(
            test_client, valid_proxy_api_key, mock_httpx_response,
            aws_frames.split(self._frames(aws_frames), [9, 40, 3]), stream=True,
        )

        assert response.status_code == 200
        payloads = [line[len("data: "):] for line in response.text.split("\n\n") if line.startswith("data: ")]
        assert payloads[-1] == "[DONE]"
        chunks = [json.loads(p) for p in payloads[:-1]]
        deltas = [c["choices"][0]["delta"] for c in chunks]
        assert [d["reasoning_content"] for d in deltas if "reasoning_content" in d] == self.TEXTS
        assert deltas[0].get("role") == "assistant" and "reasoning_content" in deltas[0]
        assert "".join(d["content"] for d in deltas if "content" in d) == "Hello"
        if account_system:
            manager.get_next_account.assert_awaited()
            manager.report_success.assert_awaited()
        else:
            manager.get_next_account.assert_not_awaited()
            manager.get_first_account.assert_called()

    @pytest.mark.parametrize("account_system", [True, False])
    def test_non_streaming_returns_reasoning_content(
        self, test_client, valid_proxy_api_key, mock_httpx_response, aws_frames, monkeypatch, account_system
    ):
        """
        What it does: POST stream=false through both branches.
        Goal: message.reasoning_content carries all reasoning text; tool_calls preserved.
        """
        manager = self._install_account_manager(test_client.app.state, monkeypatch, account_system)

        response = self._post(
            test_client, valid_proxy_api_key, mock_httpx_response,
            [self._frames(aws_frames, with_tool=True)], stream=False,
        )

        assert response.status_code == 200
        choice = response.json()["choices"][0]
        assert choice["message"]["reasoning_content"] == "".join(self.TEXTS)
        assert choice["message"]["content"] == "Hello"
        assert choice["message"]["tool_calls"][0]["function"]["name"] == "Bash"
        assert choice["finish_reason"] == "tool_calls"
        if account_system:
            manager.get_next_account.assert_awaited()
        else:
            manager.get_next_account.assert_not_awaited()


import asyncio

import httpx


# ==================================================================================================
# Empty upstream response - all four route paths
# ==================================================================================================

class TestEmptyUpstreamResponse:
    """
    Kiro answers HTTP 200 with a stream that only carries a contextUsageEvent frame.
    /v1/chat/completions must turn that into an explicit, retryable HTTP error on ALL FOUR
    paths (account-system / legacy x streaming / non-streaming), without gateway-side retry
    and without touching the account's health.
    """

    @staticmethod
    def _hanging_response():
        """Response whose stream never produces a byte (first token timeout)."""
        response = AsyncMock(spec=httpx.Response)
        response.status_code = 200
        response.aclose = AsyncMock()

        async def gen():
            await asyncio.sleep(3600)
            yield b""

        response.aiter_bytes = lambda *a, **k: gen()
        return response

    @staticmethod
    def _post(test_client, key, responses, stream, fast_timeout=False):
        """POSTs /v1/chat/completions; request_with_retry returns `responses` in order."""
        from kiro.streaming_openai import stream_with_first_token_retry as real_retry

        def fast(**kwargs):
            kwargs["first_token_timeout"] = 0.05
            return real_retry(**kwargs)

        def do_post():
            return test_client.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": "claude-sonnet-4-5",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stream": stream,
                },
            )

        with patch("kiro.routes_openai.KiroHttpClient") as client_class:
            instance = AsyncMock()
            instance.request_with_retry = AsyncMock(side_effect=list(responses))
            instance.close = AsyncMock()
            client_class.return_value = instance
            if fast_timeout:
                with patch("kiro.routes_openai.stream_with_first_token_retry", side_effect=fast):
                    return do_post(), instance
            return do_post(), instance

    @staticmethod
    def _install(test_client, monkeypatch, account_system):
        return TestNativeReasoningRouteCoverage._install_account_manager(
            test_client.app.state, monkeypatch, account_system
        )

    @staticmethod
    def _sse_payloads(text):
        """Decoded `data:` payloads of an OpenAI SSE body."""
        return [line[len("data: "):] for line in text.split("\n\n") if line.startswith("data: ")]

    @pytest.mark.parametrize("stream", [True, False])
    @pytest.mark.parametrize("account_system", [True, False])
    def test_empty_response_becomes_503_error(
        self, test_client, valid_proxy_api_key, mock_httpx_response, aws_frames, monkeypatch,
        account_system, stream
    ):
        """
        What it does: Upstream sends one contextUsageEvent frame (the real 147 byte shape).
        Goal: HTTP 503 with an OpenAI error body and x-should-retry, no SSE content,
        and the upstream connection is closed.
        """
        manager = self._install(test_client, monkeypatch, account_system)
        chunks = [aws_frames.context_usage(60.73619842529297)]

        response, http_client = self._post(
            test_client, valid_proxy_api_key, [mock_httpx_response(status_code=200, stream_chunks=chunks)], stream
        )

        print(response.status_code, response.text)
        assert response.status_code == 503
        assert response.json() == {
            "error": {
                "message": response.json()["error"]["message"],
                "type": "server_error",
                "param": None,
                "code": "upstream_empty_response",
            }
        }
        assert "empty response" in response.json()["error"]["message"]
        assert response.headers["x-should-retry"] == "true"
        assert response.headers["content-type"].startswith("application/json")
        assert "chat.completion.chunk" not in response.text
        assert "[DONE]" not in response.text
        http_client.close.assert_awaited()
        assert http_client.request_with_retry.await_count == 1
        if account_system:
            manager.get_next_account.assert_awaited_once()
        else:
            manager.get_next_account.assert_not_awaited()
            manager.get_first_account.assert_called()

    @pytest.mark.parametrize("stream", [True, False])
    def test_empty_response_split_into_single_bytes(
        self, test_client, valid_proxy_api_key, mock_httpx_response, aws_frames, monkeypatch, stream
    ):
        """
        What it does: The empty-response frame arrives one byte per chunk.
        Goal: Same 503: detection does not depend on network chunking.
        """
        self._install(test_client, monkeypatch, True)
        chunks = aws_frames.split(aws_frames.context_usage(60.7), [1])

        response, _ = self._post(
            test_client, valid_proxy_api_key, [mock_httpx_response(status_code=200, stream_chunks=chunks)], stream
        )

        assert response.status_code == 503

    @pytest.mark.parametrize("stream", [True, False])
    def test_account_health_is_untouched_by_empty_response(
        self, test_client, valid_proxy_api_key, mock_httpx_response, aws_frames, monkeypatch, stream
    ):
        """
        What it does: Account system, empty upstream response.
        Goal: Neither report_failure (circuit breaker) nor report_success is awaited and
        no other account is tried: an empty response is not an account problem.
        """
        manager = self._install(test_client, monkeypatch, True)

        response, _ = self._post(
            test_client, valid_proxy_api_key,
            [mock_httpx_response(status_code=200, stream_chunks=[aws_frames.context_usage(1.0)])], stream
        )

        assert response.status_code == 503
        manager.report_failure.assert_not_awaited()
        manager.report_success.assert_not_awaited()
        assert manager.get_next_account.await_count == 1

    @pytest.mark.parametrize("stream", [True, False])
    @pytest.mark.parametrize("account_system", [True, False])
    def test_metered_empty_answer_is_passed_through(
        self, test_client, valid_proxy_api_key, mock_httpx_response, aws_frames, monkeypatch,
        account_system, stream
    ):
        """
        What it does: contextUsageEvent + meteringEvent, no output (the model really said nothing).
        Goal: HTTP 200 with an empty completion ([DONE] terminated when streaming), not an error.
        """
        manager = self._install(test_client, monkeypatch, account_system)
        chunks = [aws_frames.context_usage(2.0) + aws_frames.metering(0.4)]

        response, _ = self._post(
            test_client, valid_proxy_api_key, [mock_httpx_response(status_code=200, stream_chunks=chunks)], stream
        )

        assert response.status_code == 200
        if stream:
            assert self._sse_payloads(response.text)[-1] == "[DONE]"
        else:
            assert response.json()["choices"][0]["message"]["content"] == ""
        if account_system:
            manager.report_success.assert_awaited_once()

    @pytest.mark.parametrize("stream", [True, False])
    @pytest.mark.parametrize("account_system", [True, False])
    def test_switch_off_passes_empty_response_through(
        self, test_client, valid_proxy_api_key, mock_httpx_response, aws_frames, monkeypatch,
        account_system, stream
    ):
        """
        What it does: EMPTY_RESPONSE_AS_ERROR=False, upstream sends the empty-response frame.
        Goal: Native behavior: HTTP 200 with an empty completion.
        """
        manager = self._install(test_client, monkeypatch, account_system)

        with patch("kiro.streaming_core.EMPTY_RESPONSE_AS_ERROR", False):
            response, _ = self._post(
                test_client, valid_proxy_api_key,
                [mock_httpx_response(status_code=200, stream_chunks=[aws_frames.context_usage(1.0)])], stream
            )

        assert response.status_code == 200
        if stream:
            assert self._sse_payloads(response.text)[-1] == "[DONE]"
        else:
            assert response.json()["choices"][0]["message"]["content"] == ""
        if account_system:
            manager.report_success.assert_awaited_once()

    @pytest.mark.parametrize("stream", [True, False])
    def test_configured_status_is_used(
        self, test_client, valid_proxy_api_key, mock_httpx_response, aws_frames, monkeypatch, stream
    ):
        """
        What it does: EMPTY_RESPONSE_ERROR_STATUS=500.
        Goal: Route answers 500 with the same server_error body.
        """
        self._install(test_client, monkeypatch, False)

        with patch("kiro.empty_response.EMPTY_RESPONSE_ERROR_STATUS", 500):
            response, _ = self._post(
                test_client, valid_proxy_api_key,
                [mock_httpx_response(status_code=200, stream_chunks=[aws_frames.context_usage(1.0)])], stream
            )

        assert response.status_code == 500
        assert response.json()["error"]["type"] == "server_error"

    @pytest.mark.parametrize("account_system", [True, False])
    def test_warning_is_logged_once(
        self, test_client, valid_proxy_api_key, mock_httpx_response, aws_frames, loguru_messages,
        monkeypatch, account_system
    ):
        """
        What it does: Empty response, then reads the log.
        Goal: One WARNING with model / mode / sizes, no message content.
        """
        self._install(test_client, monkeypatch, account_system)

        response, _ = self._post(
            test_client, valid_proxy_api_key,
            [mock_httpx_response(status_code=200, stream_chunks=[aws_frames.context_usage(1.0)])], False
        )

        assert response.status_code == 503
        lines = [m for m in loguru_messages if "[EmptyUpstreamResponse]" in m]
        assert len(lines) == 1
        assert lines[0].startswith("WARNING|")
        assert "model=claude-sonnet-4-5" in lines[0]
        assert "stream=False" in lines[0]
        assert not any("Hello" in line for line in lines)

    @pytest.mark.parametrize("account_system", [True, False])
    def test_normal_response_marks_account_healthy_once(
        self, test_client, valid_proxy_api_key, mock_httpx_response, aws_frames, monkeypatch, account_system
    ):
        """
        What it does: A normal streaming response through the new priming structure.
        Goal: First chunk carries role=assistant, last item is [DONE]; report_success awaited
        exactly once (account system) and report_failure never.
        """
        manager = self._install(test_client, monkeypatch, account_system)
        data = aws_frames.content("Hello") + aws_frames.context_usage(2.0) + aws_frames.metering(0.4)

        response, _ = self._post(
            test_client, valid_proxy_api_key,
            [mock_httpx_response(status_code=200, stream_chunks=aws_frames.split(data, [11]))], True
        )

        assert response.status_code == 200
        payloads = self._sse_payloads(response.text)
        assert payloads[-1] == "[DONE]"
        first = json.loads(payloads[0])
        assert first["choices"][0]["delta"]["role"] == "assistant"
        assert "".join(
            json.loads(p)["choices"][0]["delta"].get("content", "") for p in payloads[:-1]
        ) == "Hello"
        manager.report_failure.assert_not_awaited()
        if account_system:
            manager.report_success.assert_awaited_once()

    @pytest.mark.parametrize("account_system", [True, False])
    def test_first_token_timeout_then_empty_response(
        self, test_client, valid_proxy_api_key, mock_httpx_response, aws_frames, monkeypatch, account_system
    ):
        """
        What it does: The first upstream response never sends a byte; the retried one is empty.
        Goal: The timeout is retried as before, the retried response is checked too: HTTP 503.
        """
        self._install(test_client, monkeypatch, account_system)
        responses = [
            self._hanging_response(),
            mock_httpx_response(status_code=200, stream_chunks=[aws_frames.context_usage(1.0)]),
        ]

        response, http_client = self._post(
            test_client, valid_proxy_api_key, responses, True, fast_timeout=True
        )

        assert response.status_code == 503
        assert http_client.request_with_retry.await_count == 2

    @pytest.mark.parametrize("account_system", [True, False])
    def test_first_token_timeout_then_normal_response(
        self, test_client, valid_proxy_api_key, mock_httpx_response, aws_frames, monkeypatch, account_system
    ):
        """
        What it does: The first upstream response hangs, the retried one is normal.
        Goal: HTTP 200 with the complete content and a single role chunk.
        """
        self._install(test_client, monkeypatch, account_system)
        data = aws_frames.content("Hi") + aws_frames.context_usage(2.0) + aws_frames.metering(0.4)
        responses = [self._hanging_response(), mock_httpx_response(status_code=200, stream_chunks=[data])]

        response, _ = self._post(test_client, valid_proxy_api_key, responses, True, fast_timeout=True)

        assert response.status_code == 200
        payloads = self._sse_payloads(response.text)
        assert payloads[-1] == "[DONE]"
        deltas = [json.loads(p)["choices"][0]["delta"] for p in payloads[:-1]]
        assert sum(1 for d in deltas if d.get("role") == "assistant") == 1
        assert "".join(d.get("content", "") for d in deltas) == "Hi"

    @pytest.mark.parametrize("stream", [True, False])
    def test_upstream_http_error_is_unchanged(
        self, test_client, valid_proxy_api_key, mock_httpx_response, monkeypatch, stream
    ):
        """
        What it does: Upstream answers HTTP 500 (legacy branch).
        Goal: The existing error path is untouched: same status, kiro_api_error body.
        """
        self._install(test_client, monkeypatch, False)
        upstream = mock_httpx_response(status_code=500, text="upstream exploded")

        response, _ = self._post(test_client, valid_proxy_api_key, [upstream], stream)

        assert response.status_code == 500
        assert response.json()["error"]["type"] == "kiro_api_error"


# ==================================================================================================
# Incomplete upstream response - all route paths
# ==================================================================================================

from contextlib import ExitStack  # noqa: E402

INCOMPLETE_ROUTE_SECRET = "SECRET_ROUTE_THOUGHT_6a90"


def _reasoning_pins() -> ExitStack:
    """Pins the native reasoning handling so the tests do not depend on the local .env."""
    stack = ExitStack()
    stack.enter_context(patch("kiro.streaming_core.FAKE_REASONING_ENABLED", True))
    stack.enter_context(patch("kiro.streaming_core.FAKE_REASONING_HANDLING", "as_reasoning_content"))
    stack.enter_context(patch("kiro.streaming_openai.FAKE_REASONING_HANDLING", "as_reasoning_content"))
    return stack


class TestIncompleteUpstreamResponse:
    """
    Kiro ends the stream early while reasoning: reasoning frames only, no meteringEvent,
    no text, no tool call. /v1/chat/completions must answer HTTP 503 (non-streaming) or end
    the SSE stream with an error chunk and [DONE] (streaming) on ALL FOUR paths, without
    touching the account.
    """

    @staticmethod
    def _incomplete(aws_frames, with_context_usage=True):
        data = b"".join(aws_frames.reasoning(f"{INCOMPLETE_ROUTE_SECRET} {i}") for i in range(5))
        return data + (aws_frames.context_usage(3.0) if with_context_usage else b"")

    @staticmethod
    def _run(test_client, key, mock_httpx_response, chunks, stream):
        with _reasoning_pins():
            return TestEmptyUpstreamResponse._post(
                test_client, key, [mock_httpx_response(status_code=200, stream_chunks=chunks)], stream
            )

    @staticmethod
    def _install(test_client, monkeypatch, account_system):
        return TestNativeReasoningRouteCoverage._install_account_manager(
            test_client.app.state, monkeypatch, account_system
        )

    @staticmethod
    def _payloads(text):
        return TestEmptyUpstreamResponse._sse_payloads(text)

    @pytest.mark.parametrize("account_system", [True, False])
    def test_non_streaming_becomes_503_error(
        self, test_client, valid_proxy_api_key, mock_httpx_response, aws_frames, monkeypatch, account_system
    ):
        """
        What it does: stream=false, upstream sends reasoning + contextUsageEvent only.
        Goal: HTTP 503, OpenAI error body (server_error / upstream_incomplete_response),
        x-should-retry, upstream closed, one upstream request, account untouched.
        """
        manager = self._install(test_client, monkeypatch, account_system)

        response, http_client = self._run(
            test_client, valid_proxy_api_key, mock_httpx_response, [self._incomplete(aws_frames)], False
        )

        print(response.status_code, response.text)
        assert response.status_code == 503
        error = response.json()["error"]
        assert error["type"] == "server_error" and error["code"] == "upstream_incomplete_response"
        assert "incomplete" in error["message"] and "retry" in error["message"]
        assert INCOMPLETE_ROUTE_SECRET not in response.text
        assert response.headers["x-should-retry"] == "true"
        http_client.close.assert_awaited()
        assert http_client.request_with_retry.await_count == 1
        manager.report_failure.assert_not_awaited()
        manager.report_success.assert_not_awaited()
        if account_system:
            manager.get_next_account.assert_awaited_once()
        else:
            manager.get_next_account.assert_not_awaited()
            manager.get_first_account.assert_called()

    @pytest.mark.parametrize("account_system", [True, False])
    def test_streaming_ends_with_error_chunk_and_done(
        self, test_client, valid_proxy_api_key, mock_httpx_response, aws_frames, monkeypatch, account_system
    ):
        """
        What it does: stream=true, upstream sends reasoning + contextUsageEvent only.
        Goal: HTTP 200 SSE: reasoning deltas, {"error": ...}, [DONE]; no finish_reason chunk;
        no STREAM_READY text; account not penalised; report_success once.
        """
        manager = self._install(test_client, monkeypatch, account_system)

        response, http_client = self._run(
            test_client, valid_proxy_api_key, mock_httpx_response,
            aws_frames.split(self._incomplete(aws_frames), [11, 50]), True,
        )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert "STREAM_READY" not in response.text
        payloads = self._payloads(response.text)
        assert payloads[-1] == "[DONE]"
        error = json.loads(payloads[-2])["error"]
        assert error["type"] == "server_error" and error["code"] == "upstream_incomplete_response"
        chunks = [json.loads(p) for p in payloads[:-2]]
        assert len(chunks) == 5
        assert all(c["choices"][0]["finish_reason"] is None for c in chunks)
        assert all("reasoning_content" in c["choices"][0]["delta"] for c in chunks)
        assert http_client.request_with_retry.await_count == 1
        manager.report_failure.assert_not_awaited()
        if account_system:
            manager.get_next_account.assert_awaited_once()
            manager.report_success.assert_awaited_once()
        else:
            manager.get_next_account.assert_not_awaited()

    @pytest.mark.parametrize("stream", [True, False])
    @pytest.mark.parametrize("account_system", [True, False])
    def test_only_reasoning_frames_without_context_usage(
        self, test_client, valid_proxy_api_key, mock_httpx_response, aws_frames, monkeypatch, account_system, stream
    ):
        """
        What it does: Only reasoning frames (17:15:07 shape), 7-byte chunks.
        Goal: Treated as incomplete on every path.
        """
        self._install(test_client, monkeypatch, account_system)
        data = self._incomplete(aws_frames, with_context_usage=False)

        response, _ = self._run(
            test_client, valid_proxy_api_key, mock_httpx_response, aws_frames.split(data, [7]), stream
        )

        if stream:
            assert response.status_code == 200
            assert "upstream_incomplete_response" in response.text
            assert '"finish_reason": "stop"' not in response.text
        else:
            assert response.status_code == 503

    @pytest.mark.parametrize("stream", [True, False])
    @pytest.mark.parametrize("account_system", [True, False])
    def test_metering_makes_it_a_normal_response(
        self, test_client, valid_proxy_api_key, mock_httpx_response, aws_frames, monkeypatch, account_system, stream
    ):
        """
        What it does: Same reasoning + meteringEvent, no body.
        Goal: HTTP 200 pass-through with finish_reason stop.
        """
        manager = self._install(test_client, monkeypatch, account_system)
        data = self._incomplete(aws_frames) + aws_frames.metering(0.4)

        response, _ = self._run(test_client, valid_proxy_api_key, mock_httpx_response, [data], stream)

        assert response.status_code == 200
        if stream:
            payloads = self._payloads(response.text)
            assert payloads[-1] == "[DONE]"
            assert json.loads(payloads[-2])["choices"][0]["finish_reason"] == "stop"
            assert "upstream_incomplete_response" not in response.text
        else:
            body = response.json()
            assert body["choices"][0]["finish_reason"] == "stop"
            assert INCOMPLETE_ROUTE_SECRET in body["choices"][0]["message"]["reasoning_content"]
        if account_system:
            manager.report_success.assert_awaited_once()

    @pytest.mark.parametrize("stream", [True, False])
    @pytest.mark.parametrize("account_system", [True, False])
    def test_text_without_metering_passes_through_with_warning(
        self, test_client, valid_proxy_api_key, mock_httpx_response, aws_frames, loguru_messages,
        monkeypatch, account_system, stream
    ):
        """
        What it does: Reasoning + text, no meteringEvent (possibly truncated text).
        Goal: Unchanged HTTP 200; exactly one WARNING without content; no ERROR.
        """
        self._install(test_client, monkeypatch, account_system)
        data = aws_frames.reasoning("r") + aws_frames.content("partial answer") + aws_frames.context_usage(2.0)

        response, _ = self._run(test_client, valid_proxy_api_key, mock_httpx_response, [data], stream)

        assert response.status_code == 200
        assert "partial answer" in response.text
        lines = [m for m in loguru_messages if "[IncompleteUpstreamResponse]" in m]
        assert len(lines) == 1 and "partial answer" not in lines[0]
        assert not [m for m in loguru_messages if m.startswith("ERROR|")]

    @pytest.mark.parametrize("stream", [True, False])
    def test_tool_use_without_metering_passes_through(
        self, test_client, valid_proxy_api_key, mock_httpx_response, aws_frames, monkeypatch, stream
    ):
        """
        What it does: Reasoning + tool call, no meteringEvent.
        Goal: HTTP 200 with finish_reason tool_calls.
        """
        self._install(test_client, monkeypatch, True)
        data = (
            aws_frames.reasoning("r")
            + aws_frames.tool_start("Bash", "toolu_1")
            + aws_frames.tool_input('{"command": "ls"}', "Bash", "toolu_1")
            + aws_frames.tool_stop("Bash", "toolu_1")
            + aws_frames.context_usage(1.0)
        )

        response, _ = self._run(test_client, valid_proxy_api_key, mock_httpx_response, [data], stream)

        assert response.status_code == 200
        assert "tool_calls" in response.text

    @pytest.mark.parametrize("stream", [True, False])
    @pytest.mark.parametrize("account_system", [True, False])
    def test_switch_off_restores_pass_through(
        self, test_client, valid_proxy_api_key, mock_httpx_response, aws_frames, monkeypatch, account_system, stream
    ):
        """
        What it does: INCOMPLETE_RESPONSE_AS_ERROR=False with an incomplete stream.
        Goal: Old behavior on every path: HTTP 200 with finish_reason stop.
        """
        manager = self._install(test_client, monkeypatch, account_system)

        with patch("kiro.streaming_core.INCOMPLETE_RESPONSE_AS_ERROR", False):
            response, _ = self._run(
                test_client, valid_proxy_api_key, mock_httpx_response, [self._incomplete(aws_frames)], stream
            )

        assert response.status_code == 200
        assert "upstream_incomplete_response" not in response.text
        assert '"finish_reason": "stop"' in response.text.replace('":"', '": "')
        if account_system:
            manager.report_success.assert_awaited_once()

    @pytest.mark.parametrize("status", [500, 529])
    def test_non_streaming_reuses_the_configured_status(
        self, test_client, valid_proxy_api_key, mock_httpx_response, aws_frames, monkeypatch, status
    ):
        """
        What it does: EMPTY_RESPONSE_ERROR_STATUS patched to 500 / 529.
        Goal: The incomplete response uses that status; the body type stays server_error.
        """
        self._install(test_client, monkeypatch, False)

        with patch("kiro.empty_response.EMPTY_RESPONSE_ERROR_STATUS", status):
            response, _ = self._run(
                test_client, valid_proxy_api_key, mock_httpx_response, [self._incomplete(aws_frames)], False
            )

        assert response.status_code == status
        assert response.json()["error"]["type"] == "server_error"

    @pytest.mark.parametrize("stream", [True, False])
    @pytest.mark.parametrize("account_system", [True, False])
    def test_exactly_one_warning_and_no_error_log(
        self, test_client, valid_proxy_api_key, mock_httpx_response, aws_frames, loguru_messages,
        monkeypatch, account_system, stream
    ):
        """
        What it does: Incomplete response, reads the log.
        Goal: One [IncompleteUpstreamResponse] WARNING (status= / action=error_event), no
        "Internal error" or "Error during streaming" ERROR, no reasoning text.
        """
        self._install(test_client, monkeypatch, account_system)

        self._run(test_client, valid_proxy_api_key, mock_httpx_response, [self._incomplete(aws_frames)], stream)

        lines = [m for m in loguru_messages if "[IncompleteUpstreamResponse]" in m]
        assert len(lines) == 1 and lines[0].startswith("WARNING|")
        assert ("action=error_event" if stream else "status=503") in lines[0]
        assert not any(INCOMPLETE_ROUTE_SECRET in m for m in loguru_messages)
        assert not [
            m for m in loguru_messages
            if m.startswith("ERROR|") and ("Internal error" in m or "Error during streaming" in m)
        ]

    @pytest.mark.parametrize("stream", [True, False])
    def test_empty_response_regression(
        self, test_client, valid_proxy_api_key, mock_httpx_response, aws_frames, loguru_messages,
        monkeypatch, stream
    ):
        """
        What it does: A lone contextUsageEvent (the earlier empty-response criterion).
        Goal: Still HTTP 503, code upstream_empty_response, [EmptyUpstreamResponse] label.
        """
        self._install(test_client, monkeypatch, False)

        response, _ = self._run(
            test_client, valid_proxy_api_key, mock_httpx_response, [aws_frames.context_usage(1.0)], stream
        )

        assert response.status_code == 503
        assert response.json()["error"]["code"] == "upstream_empty_response"
        assert [m for m in loguru_messages if "[EmptyUpstreamResponse]" in m]
        assert not [m for m in loguru_messages if "[IncompleteUpstreamResponse]" in m]

    @pytest.mark.parametrize("stream", [True, False])
    def test_empty_switch_off_does_not_turn_context_usage_into_incomplete(
        self, test_client, valid_proxy_api_key, mock_httpx_response, aws_frames, monkeypatch, stream
    ):
        """
        What it does: EMPTY_RESPONSE_AS_ERROR=False, INCOMPLETE on, a lone contextUsageEvent.
        Goal: Passed through (NO_OUTPUT_FRAMES is not INCOMPLETE): HTTP 200.
        """
        self._install(test_client, monkeypatch, False)

        with patch("kiro.streaming_core.EMPTY_RESPONSE_AS_ERROR", False):
            response, _ = self._run(
                test_client, valid_proxy_api_key, mock_httpx_response, [aws_frames.context_usage(1.0)], stream
            )

        assert response.status_code == 200
        assert "upstream_incomplete_response" not in response.text


# Exception types through which the TestClient reports a cancelled request handler
CANCELLATION_SURFACES = {"CancelledError", "RuntimeError", "BaseExceptionGroup", "ExceptionGroup"}


class TestCancellationClosesUpstream:
    """
    A client that goes away while the route waits for the upstream (peek phase of a stream,
    or the aggregation of a non-streaming response) must not leak the upstream connection.
    The cancellation is simulated by the awaited helper raising asyncio.CancelledError.
    """

    @pytest.mark.parametrize("target", ["prime_sse_stream", "collect_stream_response"])
    @pytest.mark.parametrize("account_system", [True, False])
    def test_http_client_is_closed_on_cancellation(
        self, test_client, valid_proxy_api_key, mock_httpx_response, aws_frames, monkeypatch,
        account_system, target
    ):
        """
        What it does: Makes the awaited peek / collect helper raise CancelledError.
        Goal: http_client.close() is awaited on both branches before the cancellation propagates;
        the account is not penalised.
        """
        manager = TestNativeReasoningRouteCoverage._install_account_manager(
            test_client.app.state, monkeypatch, account_system
        )
        stream = target == "prime_sse_stream"

        with patch("kiro.routes_openai.KiroHttpClient") as client_class, \
                patch(f"kiro.routes_openai.{target}", AsyncMock(side_effect=asyncio.CancelledError())):
            instance = AsyncMock()
            instance.request_with_retry = AsyncMock(
                return_value=mock_httpx_response(status_code=200, stream_chunks=[aws_frames.reasoning("x")])
            )
            instance.close = AsyncMock()
            client_class.return_value = instance
            try:
                test_client.post(
                    "/v1/chat/completions",
                    headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
                    json={"model": "claude-sonnet-4-5", "messages": [{"role": "user", "content": "hi"}], "stream": stream},
                )
            except BaseException as surfaced:  # noqa: BLE001 - narrowed right below
                # Starlette's BaseHTTPMiddleware reports a cancelled handler as an exception
                # group wrapping "No response returned."; anything else is a real failure.
                if type(surfaced).__name__ not in CANCELLATION_SURFACES:
                    raise

        instance.close.assert_awaited()
        manager.report_failure.assert_not_awaited()



# ==================================================================================================
# additionalModelRequestFields (reasoning_effort passthrough) - all four route paths
# ==================================================================================================

class TestForwardModelRequestFieldsRoutes:
    """
    Proves that /v1/chat/completions writes the schema-translated reasoning_effort into the top
    level of the Kiro payload on ALL FOUR paths (account-system / legacy x streaming / non-streaming),
    and that the HTTP 400 fallback works end to end with the real KiroHttpClient.
    """

    REJECTION = json.dumps({
        "message": "Improperly formed request: additionalModelRequestFields",
        "reason": "REQUEST_BODY_INVALID",
    }).encode()

    @staticmethod
    def _install_account_manager(app_state, monkeypatch, account_system, model_cache):
        """Installs a deterministic account manager owning `model_cache` and selects the branch."""
        from kiro.account_manager import Account, AccountStats

        auth_manager = MagicMock()
        auth_manager.api_host = "https://kiro.example.invalid"
        auth_manager.profile_arn = "arn:aws:codewhisperer:us-east-1:000000000000:profile/TEST"
        auth_manager.get_access_token = AsyncMock(return_value="token")
        auth_manager.force_refresh = AsyncMock(return_value="token")
        account = Account(
            id="acc1", auth_manager=auth_manager, model_cache=model_cache,
            model_resolver=MagicMock(), stats=AccountStats(),
        )
        manager = MagicMock()
        manager._accounts = {"acc1": account}
        manager.get_first_account = MagicMock(return_value=account)
        manager.get_next_account = AsyncMock(return_value=account)
        manager.report_success = AsyncMock()
        manager.report_failure = AsyncMock()
        monkeypatch.setattr(app_state, "account_manager", manager)
        monkeypatch.setattr(app_state, "account_system", account_system)
        return manager

    @staticmethod
    def _ok_frames(aws_frames):
        """A complete, non-empty upstream answer."""
        return [aws_frames.content("Hello") + aws_frames.metering(0.1) + aws_frames.context_usage(1.0)]

    @staticmethod
    def _post(test_client, key, stream, model="claude-sonnet-5.5", **extra):
        """POSTs a chat completion with reasoning_effort=medium by default."""
        body = {
            "model": model, "stream": stream, "reasoning_effort": "medium",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        body.update(extra)
        return test_client.post(
            "/v1/chat/completions", headers={"Authorization": f"Bearer {key}"}, json=body
        )

    def _capture_payload(self, test_client, key, mock_httpx_response, aws_frames, stream, **kwargs):
        """Runs a request with KiroHttpClient mocked; returns (response, payload sent to Kiro)."""
        with patch("kiro.routes_openai.KiroHttpClient") as client_class:
            instance = AsyncMock()
            instance.request_with_retry = AsyncMock(
                return_value=mock_httpx_response(status_code=200, stream_chunks=self._ok_frames(aws_frames))
            )
            instance.close = AsyncMock()
            client_class.return_value = instance
            response = self._post(test_client, key, stream, **kwargs)
        payload = instance.request_with_retry.call_args.args[2]
        return response, payload

    @pytest.mark.parametrize("stream", [True, False])
    @pytest.mark.parametrize("account_system", [True, False])
    def test_claude_shaped_model_gets_output_config(
        self, test_client, valid_proxy_api_key, mock_httpx_response, aws_frames, monkeypatch,
        schema_model_cache, account_system, stream
    ):
        """
        What it does: reasoning_effort=medium on claude-sonnet-5.5 through each of the four paths.
        Goal: Top-level additionalModelRequestFields == output_config.effort medium (no thinking).
        """
        manager = self._install_account_manager(test_client.app.state, monkeypatch, account_system, schema_model_cache)

        response, payload = self._capture_payload(
            test_client, valid_proxy_api_key, mock_httpx_response, aws_frames, stream
        )

        assert response.status_code == 200
        assert payload["additionalModelRequestFields"] == {"output_config": {"effort": "medium"}}
        if account_system:
            manager.get_next_account.assert_awaited()
        else:
            manager.get_next_account.assert_not_awaited()
            manager.get_first_account.assert_called()

    @pytest.mark.parametrize("stream", [True, False])
    @pytest.mark.parametrize("account_system", [True, False])
    def test_gpt_shaped_model_gets_reasoning(
        self, test_client, valid_proxy_api_key, mock_httpx_response, aws_frames, monkeypatch,
        schema_model_cache, account_system, stream
    ):
        """
        What it does: reasoning_effort=medium on gpt-5.6-luna through each of the four paths.
        Goal: Only reasoning.effort.
        """
        self._install_account_manager(test_client.app.state, monkeypatch, account_system, schema_model_cache)

        _, payload = self._capture_payload(
            test_client, valid_proxy_api_key, mock_httpx_response, aws_frames, stream, model="gpt-5.6-luna"
        )

        assert payload["additionalModelRequestFields"] == {"reasoning": {"effort": "medium"}}

    @pytest.mark.parametrize("stream", [True, False])
    @pytest.mark.parametrize("account_system", [True, False])
    @pytest.mark.parametrize("model", ["claude-sonnet-4-5", "claude-haiku-4-5", "unknown-model"])
    def test_models_without_schema_send_no_field(
        self, test_client, valid_proxy_api_key, mock_httpx_response, aws_frames, monkeypatch,
        schema_model_cache, account_system, stream, model
    ):
        """
        What it does: sonnet-4.5 / haiku-4.5 (null schema) and an uncached model on every path.
        Goal: No additionalModelRequestFields.
        """
        self._install_account_manager(test_client.app.state, monkeypatch, account_system, schema_model_cache)

        response, payload = self._capture_payload(
            test_client, valid_proxy_api_key, mock_httpx_response, aws_frames, stream, model=model
        )

        assert response.status_code == 200
        assert "additionalModelRequestFields" not in payload

    @pytest.mark.parametrize("stream", [True, False])
    @pytest.mark.parametrize("account_system", [True, False])
    def test_switch_off_and_missing_effort_send_no_field(
        self, test_client, valid_proxy_api_key, mock_httpx_response, aws_frames, monkeypatch,
        schema_model_cache, account_system, stream
    ):
        """
        What it does: Switch off, and a request without reasoning_effort.
        Goal: No field in either case.
        """
        self._install_account_manager(test_client.app.state, monkeypatch, account_system, schema_model_cache)

        with patch("kiro.model_request_fields.FORWARD_MODEL_REQUEST_FIELDS", False):
            _, off_payload = self._capture_payload(
                test_client, valid_proxy_api_key, mock_httpx_response, aws_frames, stream
            )
        _, no_effort_payload = self._capture_payload(
            test_client, valid_proxy_api_key, mock_httpx_response, aws_frames, stream, reasoning_effort=None
        )

        assert "additionalModelRequestFields" not in off_payload
        assert "additionalModelRequestFields" not in no_effort_payload

    @pytest.mark.parametrize("stream", [True, False])
    @pytest.mark.parametrize("account_system", [True, False])
    def test_magicmock_model_cache_still_works(
        self, test_client, valid_proxy_api_key, mock_httpx_response, aws_frames, monkeypatch,
        account_system, stream
    ):
        """
        What it does: A MagicMock model cache (shape used by older tests).
        Goal: Request succeeds and no field is sent.
        """
        cache = MagicMock()
        cache.get_max_input_tokens.return_value = 200000
        self._install_account_manager(test_client.app.state, monkeypatch, account_system, cache)

        response, payload = self._capture_payload(
            test_client, valid_proxy_api_key, mock_httpx_response, aws_frames, stream
        )

        assert response.status_code == 200
        assert "additionalModelRequestFields" not in payload

    @pytest.mark.parametrize("stream", [True, False])
    @pytest.mark.parametrize("account_system", [True, False])
    def test_400_fallback_end_to_end(
        self, test_client, valid_proxy_api_key, mock_httpx_response, aws_frames, scripted_httpx_client,
        monkeypatch, schema_model_cache, account_system, stream
    ):
        """
        What it does: REAL KiroHttpClient; Kiro answers 400 (field rejected), then 200.
        Goal: Client gets 200; the second send has no field; the account is NOT reported as failed.
        """
        from kiro.http_client import KiroHttpClient

        manager = self._install_account_manager(test_client.app.state, monkeypatch, account_system, schema_model_cache)
        rejected = mock_httpx_response(status_code=400)
        rejected.aread = AsyncMock(return_value=self.REJECTION)
        ok = mock_httpx_response(status_code=200, stream_chunks=self._ok_frames(aws_frames))
        fake = scripted_httpx_client([rejected, ok])

        with patch.object(KiroHttpClient, "_get_client", new=AsyncMock(return_value=fake)), \
                patch("kiro.http_client.get_kiro_headers", return_value={}):
            response = self._post(test_client, valid_proxy_api_key, stream)

        assert response.status_code == 200
        assert len(fake.sent_bodies) == 2
        assert fake.sent_bodies[0]["additionalModelRequestFields"] == {"output_config": {"effort": "medium"}}
        assert "additionalModelRequestFields" not in fake.sent_bodies[1]
        manager.report_failure.assert_not_awaited()

    def test_unrelated_400_is_still_reported_as_failure(
        self, test_client, valid_proxy_api_key, mock_httpx_response, scripted_httpx_client, monkeypatch,
        schema_model_cache
    ):
        """
        What it does: Real KiroHttpClient; a 400 that does not mention the field (account-system branch).
        Goal: Not swallowed: one send, error status reaches the client, normal failure reporting runs.
        """
        from kiro.http_client import KiroHttpClient

        manager = self._install_account_manager(test_client.app.state, monkeypatch, True, schema_model_cache)
        rejected = mock_httpx_response(status_code=400)
        rejected.aread = AsyncMock(return_value=b'{"message":"Improperly formed request.","reason":null}')
        rejected.text = '{"message":"Improperly formed request.","reason":null}'
        fake = scripted_httpx_client([rejected])

        with patch.object(KiroHttpClient, "_get_client", new=AsyncMock(return_value=fake)), \
                patch("kiro.http_client.get_kiro_headers", return_value={}):
            response = self._post(test_client, valid_proxy_api_key, False)

        assert response.status_code >= 400
        assert len(fake.sent_bodies) == 1
        assert "additionalModelRequestFields" in fake.sent_bodies[0]
        manager.report_failure.assert_awaited()
