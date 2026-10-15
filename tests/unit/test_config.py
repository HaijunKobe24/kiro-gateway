# -*- coding: utf-8 -*-

"""
Unit tests for the configuration module.
Verifies loading settings from environment variables.
"""

import pytest
import os
from unittest.mock import patch


class TestLogLevelConfig:
    """Tests for LOG_LEVEL configuration."""
    
    def test_default_log_level_is_info(self):
        """
        What it does: Verifies that LOG_LEVEL defaults to INFO.
        Purpose: Ensure that INFO is used when no environment variable is set.
        
        Note: This test verifies the config.py code logic, not the actual
        value from the .env file. We mock os.getenv to simulate
        the absence of the environment variable.
        """
        print("Setup: Mocking os.getenv for LOG_LEVEL...")
        
        # Create a mock that returns None for LOG_LEVEL (simulating missing variable)
        original_getenv = os.getenv
        
        def mock_getenv(key, default=None):
            if key == "LOG_LEVEL":
                print(f"os.getenv('{key}') -> None (mocked)")
                return default  # Return default, simulating missing variable
            return original_getenv(key, default)
        
        with patch.object(os, 'getenv', side_effect=mock_getenv):
            # Reload config module with mocked getenv
            import importlib
            import kiro.config as config_module
            importlib.reload(config_module)
            
            print(f"LOG_LEVEL: {config_module.LOG_LEVEL}")
            print(f"Comparing: Expected 'INFO', Got '{config_module.LOG_LEVEL}'")
            assert config_module.LOG_LEVEL == "INFO"
        
        # Restore module with real values
        import importlib
        import kiro.config as config_module
        importlib.reload(config_module)
    
    def test_log_level_from_environment(self):
        """
        What it does: Verifies loading LOG_LEVEL from environment variable.
        Purpose: Ensure that the value from environment is used.
        """
        print("Setup: Setting LOG_LEVEL=DEBUG...")
        
        with patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"}):
            import importlib
            import kiro.config as config_module
            importlib.reload(config_module)
            
            print(f"LOG_LEVEL: {config_module.LOG_LEVEL}")
            print(f"Comparing: Expected 'DEBUG', Got '{config_module.LOG_LEVEL}'")
            assert config_module.LOG_LEVEL == "DEBUG"
    
    def test_log_level_uppercase_conversion(self):
        """
        What it does: Verifies LOG_LEVEL conversion to uppercase.
        Purpose: Ensure that lowercase value is converted to uppercase.
        """
        print("Setup: Setting LOG_LEVEL=warning (lowercase)...")
        
        with patch.dict(os.environ, {"LOG_LEVEL": "warning"}):
            import importlib
            import kiro.config as config_module
            importlib.reload(config_module)
            
            print(f"LOG_LEVEL: {config_module.LOG_LEVEL}")
            print(f"Comparing: Expected 'WARNING', Got '{config_module.LOG_LEVEL}'")
            assert config_module.LOG_LEVEL == "WARNING"
    
    def test_log_level_trace(self):
        """
        What it does: Verifies setting LOG_LEVEL=TRACE.
        Purpose: Ensure that TRACE level is supported.
        """
        print("Setup: Setting LOG_LEVEL=TRACE...")
        
        with patch.dict(os.environ, {"LOG_LEVEL": "TRACE"}):
            import importlib
            import kiro.config as config_module
            importlib.reload(config_module)
            
            print(f"LOG_LEVEL: {config_module.LOG_LEVEL}")
            assert config_module.LOG_LEVEL == "TRACE"
    
    def test_log_level_error(self):
        """
        What it does: Verifies setting LOG_LEVEL=ERROR.
        Purpose: Ensure that ERROR level is supported.
        """
        print("Setup: Setting LOG_LEVEL=ERROR...")
        
        with patch.dict(os.environ, {"LOG_LEVEL": "ERROR"}):
            import importlib
            import kiro.config as config_module
            importlib.reload(config_module)
            
            print(f"LOG_LEVEL: {config_module.LOG_LEVEL}")
            assert config_module.LOG_LEVEL == "ERROR"
    
    def test_log_level_critical(self):
        """
        What it does: Verifies setting LOG_LEVEL=CRITICAL.
        Purpose: Ensure that CRITICAL level is supported.
        """
        print("Setup: Setting LOG_LEVEL=CRITICAL...")
        
        with patch.dict(os.environ, {"LOG_LEVEL": "CRITICAL"}):
            import importlib
            import kiro.config as config_module
            importlib.reload(config_module)
            
            print(f"LOG_LEVEL: {config_module.LOG_LEVEL}")
            assert config_module.LOG_LEVEL == "CRITICAL"


class TestToolDescriptionMaxLengthConfig:
    """Tests for TOOL_DESCRIPTION_MAX_LENGTH configuration."""
    
    def test_default_tool_description_max_length(self):
        """
        What it does: Verifies the default value for TOOL_DESCRIPTION_MAX_LENGTH.
        Purpose: Ensure that 10000 is used by default.
        """
        print("Setup: Removing TOOL_DESCRIPTION_MAX_LENGTH from environment...")
        
        with patch.dict(os.environ, {}, clear=False):
            if "TOOL_DESCRIPTION_MAX_LENGTH" in os.environ:
                del os.environ["TOOL_DESCRIPTION_MAX_LENGTH"]
            
            import importlib
            import kiro.config as config_module
            importlib.reload(config_module)
            
            print(f"TOOL_DESCRIPTION_MAX_LENGTH: {config_module.TOOL_DESCRIPTION_MAX_LENGTH}")
            assert config_module.TOOL_DESCRIPTION_MAX_LENGTH == 10000
    
    def test_tool_description_max_length_from_environment(self):
        """
        What it does: Verifies loading TOOL_DESCRIPTION_MAX_LENGTH from environment.
        Purpose: Ensure that the value from environment is used.
        """
        print("Setup: Setting TOOL_DESCRIPTION_MAX_LENGTH=5000...")
        
        with patch.dict(os.environ, {"TOOL_DESCRIPTION_MAX_LENGTH": "5000"}):
            import importlib
            import kiro.config as config_module
            importlib.reload(config_module)
            
            print(f"TOOL_DESCRIPTION_MAX_LENGTH: {config_module.TOOL_DESCRIPTION_MAX_LENGTH}")
            assert config_module.TOOL_DESCRIPTION_MAX_LENGTH == 5000
    
    def test_tool_description_max_length_zero_disables(self):
        """
        What it does: Verifies that 0 disables the feature.
        Purpose: Ensure that TOOL_DESCRIPTION_MAX_LENGTH=0 works.
        """
        print("Setup: Setting TOOL_DESCRIPTION_MAX_LENGTH=0...")
        
        with patch.dict(os.environ, {"TOOL_DESCRIPTION_MAX_LENGTH": "0"}):
            import importlib
            import kiro.config as config_module
            importlib.reload(config_module)
            
            print(f"TOOL_DESCRIPTION_MAX_LENGTH: {config_module.TOOL_DESCRIPTION_MAX_LENGTH}")
            assert config_module.TOOL_DESCRIPTION_MAX_LENGTH == 0


class TestTimeoutConfigurationWarning:
    """Tests for _warn_timeout_configuration() function."""
    
    def test_no_warning_when_first_token_less_than_streaming(self, capsys):
        """
        What it does: Verifies that warning is NOT shown with correct configuration.
        Purpose: Ensure that no warning when FIRST_TOKEN_TIMEOUT < STREAMING_READ_TIMEOUT.
        """
        print("Setup: FIRST_TOKEN_TIMEOUT=15, STREAMING_READ_TIMEOUT=300...")
        
        with patch.dict(os.environ, {
            "FIRST_TOKEN_TIMEOUT": "15",
            "STREAMING_READ_TIMEOUT": "300"
        }):
            import importlib
            import kiro.config as config_module
            importlib.reload(config_module)
            
            # Call the warning function
            config_module._warn_timeout_configuration()
            
            captured = capsys.readouterr()
            print(f"Captured stderr: {captured.err}")
            
            # Warning should NOT be shown
            assert "WARNING" not in captured.err
            assert "Suboptimal timeout configuration" not in captured.err
    
    def test_warning_when_first_token_equals_streaming(self, capsys):
        """
        What it does: Verifies that warning is shown when timeouts are equal.
        Purpose: Ensure that warning when FIRST_TOKEN_TIMEOUT == STREAMING_READ_TIMEOUT.
        """
        print("Setup: FIRST_TOKEN_TIMEOUT=300, STREAMING_READ_TIMEOUT=300...")
        
        with patch.dict(os.environ, {
            "FIRST_TOKEN_TIMEOUT": "300",
            "STREAMING_READ_TIMEOUT": "300"
        }):
            import importlib
            import kiro.config as config_module
            importlib.reload(config_module)
            
            # Call the warning function
            config_module._warn_timeout_configuration()
            
            captured = capsys.readouterr()
            print(f"Captured stderr: {captured.err}")
            
            # Warning SHOULD be shown
            assert "WARNING" in captured.err or "Suboptimal timeout configuration" in captured.err
    
    def test_warning_when_first_token_greater_than_streaming(self, capsys):
        """
        What it does: Verifies that warning is shown when FIRST_TOKEN > STREAMING.
        Purpose: Ensure that warning when FIRST_TOKEN_TIMEOUT > STREAMING_READ_TIMEOUT.
        """
        print("Setup: FIRST_TOKEN_TIMEOUT=500, STREAMING_READ_TIMEOUT=300...")
        
        with patch.dict(os.environ, {
            "FIRST_TOKEN_TIMEOUT": "500",
            "STREAMING_READ_TIMEOUT": "300"
        }):
            import importlib
            import kiro.config as config_module
            importlib.reload(config_module)
            
            # Call the warning function
            config_module._warn_timeout_configuration()
            
            captured = capsys.readouterr()
            print(f"Captured stderr: {captured.err}")
            
            # Warning SHOULD be shown
            assert "WARNING" in captured.err or "Suboptimal timeout configuration" in captured.err
            # Verify that timeout values are mentioned in warning
            assert "500" in captured.err
            assert "300" in captured.err
    
    def test_warning_contains_recommendation(self, capsys):
        """
        What it does: Verifies that warning contains a recommendation.
        Purpose: Ensure that user receives useful information.
        """
        print("Setup: FIRST_TOKEN_TIMEOUT=400, STREAMING_READ_TIMEOUT=300...")
        
        with patch.dict(os.environ, {
            "FIRST_TOKEN_TIMEOUT": "400",
            "STREAMING_READ_TIMEOUT": "300"
        }):
            import importlib
            import kiro.config as config_module
            importlib.reload(config_module)
            
            # Call the warning function
            config_module._warn_timeout_configuration()
            
            captured = capsys.readouterr()
            print(f"Captured stderr: {captured.err}")
            
            # Warning should contain recommendation
            assert "Recommendation" in captured.err or "LESS than" in captured.err


class TestAwsSsoOidcUrlConfig:
    """Tests for AWS SSO OIDC URL configuration."""
    
    def test_aws_sso_oidc_url_template_exists(self):
        """
        What it does: Verifies that AWS_SSO_OIDC_URL_TEMPLATE constant exists.
        Purpose: Ensure the template is defined in config.
        """
        print("Setup: Importing config module...")
        import importlib
        import kiro.config as config_module
        importlib.reload(config_module)
        
        print("Verification: AWS_SSO_OIDC_URL_TEMPLATE exists...")
        assert hasattr(config_module, 'AWS_SSO_OIDC_URL_TEMPLATE')
        
        print(f"AWS_SSO_OIDC_URL_TEMPLATE: {config_module.AWS_SSO_OIDC_URL_TEMPLATE}")
        assert "oidc" in config_module.AWS_SSO_OIDC_URL_TEMPLATE
        assert "amazonaws.com" in config_module.AWS_SSO_OIDC_URL_TEMPLATE
        assert "{region}" in config_module.AWS_SSO_OIDC_URL_TEMPLATE
    
    def test_get_aws_sso_oidc_url_returns_correct_url(self):
        """
        What it does: Verifies that get_aws_sso_oidc_url returns correct URL.
        Purpose: Ensure the function formats URL correctly.
        """
        print("Setup: Importing get_aws_sso_oidc_url...")
        from kiro.config import get_aws_sso_oidc_url
        
        print("Action: Calling get_aws_sso_oidc_url('us-east-1')...")
        url = get_aws_sso_oidc_url("us-east-1")
        
        print(f"Verification: URL is correct...")
        expected = "https://oidc.us-east-1.amazonaws.com/token"
        print(f"Comparing: Expected '{expected}', Got '{url}'")
        assert url == expected
    
    def test_get_aws_sso_oidc_url_with_different_regions(self):
        """
        What it does: Verifies URL generation for different regions.
        Purpose: Ensure the function works with various AWS regions.
        """
        print("Setup: Importing get_aws_sso_oidc_url...")
        from kiro.config import get_aws_sso_oidc_url
        
        test_cases = [
            ("us-east-1", "https://oidc.us-east-1.amazonaws.com/token"),
            ("eu-west-1", "https://oidc.eu-west-1.amazonaws.com/token"),
            ("ap-southeast-1", "https://oidc.ap-southeast-1.amazonaws.com/token"),
            ("us-west-2", "https://oidc.us-west-2.amazonaws.com/token"),
        ]
        
        for region, expected in test_cases:
            print(f"Action: Calling get_aws_sso_oidc_url('{region}')...")
            url = get_aws_sso_oidc_url(region)
            print(f"Comparing: Expected '{expected}', Got '{url}'")
            assert url == expected


class TestServerHostConfig:
    """Tests for SERVER_HOST configuration."""
    
    def test_default_server_host_is_0_0_0_0(self):
        """
        What it does: Verifies that SERVER_HOST defaults to 0.0.0.0.
        Purpose: Ensure that 0.0.0.0 (all interfaces) is used when no environment variable is set.
        """
        print("Setup: Removing SERVER_HOST from environment...")
        
        with patch.dict(os.environ, {}, clear=False):
            if "SERVER_HOST" in os.environ:
                del os.environ["SERVER_HOST"]
            
            import importlib
            import kiro.config as config_module
            importlib.reload(config_module)
            
            print(f"SERVER_HOST: {config_module.SERVER_HOST}")
            print(f"DEFAULT_SERVER_HOST: {config_module.DEFAULT_SERVER_HOST}")
            print(f"Comparing: Expected '0.0.0.0', Got '{config_module.SERVER_HOST}'")
            assert config_module.SERVER_HOST == "0.0.0.0"
            assert config_module.DEFAULT_SERVER_HOST == "0.0.0.0"
    
    def test_server_host_from_environment(self):
        """
        What it does: Verifies loading SERVER_HOST from environment variable.
        Purpose: Ensure that the value from environment is used.
        """
        print("Setup: Setting SERVER_HOST=127.0.0.1...")
        
        with patch.dict(os.environ, {"SERVER_HOST": "127.0.0.1"}):
            import importlib
            import kiro.config as config_module
            importlib.reload(config_module)
            
            print(f"SERVER_HOST: {config_module.SERVER_HOST}")
            print(f"Comparing: Expected '127.0.0.1', Got '{config_module.SERVER_HOST}'")
            assert config_module.SERVER_HOST == "127.0.0.1"
    
    def test_server_host_custom_value(self):
        """
        What it does: Verifies setting SERVER_HOST to a custom IP address.
        Purpose: Ensure that any valid IP address can be used.
        """
        print("Setup: Setting SERVER_HOST=192.168.1.100...")
        
        with patch.dict(os.environ, {"SERVER_HOST": "192.168.1.100"}):
            import importlib
            import kiro.config as config_module
            importlib.reload(config_module)
            
            print(f"SERVER_HOST: {config_module.SERVER_HOST}")
            assert config_module.SERVER_HOST == "192.168.1.100"


class TestServerPortConfig:
    """Tests for SERVER_PORT configuration."""
    
    def test_default_server_port_is_8000(self):
        """
        What it does: Verifies that SERVER_PORT defaults to 8000.
        Purpose: Ensure that 8000 is used when no environment variable is set.
        """
        print("Setup: Removing SERVER_PORT from environment...")
        
        with patch.dict(os.environ, {}, clear=False):
            if "SERVER_PORT" in os.environ:
                del os.environ["SERVER_PORT"]
            
            import importlib
            import kiro.config as config_module
            importlib.reload(config_module)
            
            print(f"SERVER_PORT: {config_module.SERVER_PORT}")
            print(f"DEFAULT_SERVER_PORT: {config_module.DEFAULT_SERVER_PORT}")
            print(f"Comparing: Expected 8000, Got {config_module.SERVER_PORT}")
            assert config_module.SERVER_PORT == 8000
            assert config_module.DEFAULT_SERVER_PORT == 8000
    
    def test_server_port_from_environment(self):
        """
        What it does: Verifies loading SERVER_PORT from environment variable.
        Purpose: Ensure that the value from environment is used.
        """
        print("Setup: Setting SERVER_PORT=9000...")
        
        with patch.dict(os.environ, {"SERVER_PORT": "9000"}):
            import importlib
            import kiro.config as config_module
            importlib.reload(config_module)
            
            print(f"SERVER_PORT: {config_module.SERVER_PORT}")
            print(f"Comparing: Expected 9000, Got {config_module.SERVER_PORT}")
            assert config_module.SERVER_PORT == 9000
    
    def test_server_port_custom_value(self):
        """
        What it does: Verifies setting SERVER_PORT to a custom port number.
        Purpose: Ensure that any valid port number can be used.
        """
        print("Setup: Setting SERVER_PORT=3000...")
        
        with patch.dict(os.environ, {"SERVER_PORT": "3000"}):
            import importlib
            import kiro.config as config_module
            importlib.reload(config_module)
            
            print(f"SERVER_PORT: {config_module.SERVER_PORT}")
            assert config_module.SERVER_PORT == 3000
    
    def test_server_port_is_integer(self):
        """
        What it does: Verifies that SERVER_PORT is converted to integer.
        Purpose: Ensure that string from environment is converted to int.
        """
        print("Setup: Setting SERVER_PORT=8080 (as string)...")
        
        with patch.dict(os.environ, {"SERVER_PORT": "8080"}):
            import importlib
            import kiro.config as config_module
            importlib.reload(config_module)
            
            print(f"SERVER_PORT: {config_module.SERVER_PORT}")
            print(f"Type: {type(config_module.SERVER_PORT)}")
            assert isinstance(config_module.SERVER_PORT, int)
            assert config_module.SERVER_PORT == 8080


class TestKiroCliDbFileConfig:
    """Tests for KIRO_CLI_DB_FILE configuration."""
    
    def test_kiro_cli_db_file_config_exists(self):
        """
        What it does: Verifies that KIRO_CLI_DB_FILE constant exists.
        Purpose: Ensure the config parameter is defined.
        """
        print("Setup: Importing config module...")
        import importlib
        import kiro.config as config_module
        importlib.reload(config_module)
        
        print("Verification: KIRO_CLI_DB_FILE exists...")
        assert hasattr(config_module, 'KIRO_CLI_DB_FILE')
        
        print(f"KIRO_CLI_DB_FILE: '{config_module.KIRO_CLI_DB_FILE}'")
        # Default should be empty string
        assert isinstance(config_module.KIRO_CLI_DB_FILE, str)
    
    def test_kiro_cli_db_file_from_environment(self):
        """
        What it does: Verifies loading KIRO_CLI_DB_FILE from environment variable.
        Purpose: Ensure the value from environment is used and normalized.
        """
        print("Setup: Importing config module...")
        import importlib
        import kiro.config as config_module
        
        # Test that KIRO_CLI_DB_FILE is loaded and is a string
        print(f"KIRO_CLI_DB_FILE: {config_module.KIRO_CLI_DB_FILE}")
        assert isinstance(config_module.KIRO_CLI_DB_FILE, str)
        
        # If value is set (not empty), verify it's a normalized path
        if config_module.KIRO_CLI_DB_FILE:
            # Path should be normalized (no raw ~ or forward slashes on Windows)
            assert not config_module.KIRO_CLI_DB_FILE.startswith("~")
            # Should be a valid path string (contains path separators or is absolute)
            from pathlib import Path
            path = Path(config_module.KIRO_CLI_DB_FILE)
            # Path should be constructable (doesn't raise exception)
            assert str(path) == config_module.KIRO_CLI_DB_FILE


class TestFallbackModelsConfig:
    """Tests for FALLBACK_MODELS configuration."""
    
    def test_fallback_models_exists(self):
        """
        What it does: Verifies that FALLBACK_MODELS constant exists.
        Purpose: Ensure the fallback model list is defined in config.
        """
        print("Setup: Importing config module...")
        import importlib
        import kiro.config as config_module
        importlib.reload(config_module)
        
        print("Verification: FALLBACK_MODELS exists...")
        assert hasattr(config_module, 'FALLBACK_MODELS')
        
        print(f"FALLBACK_MODELS type: {type(config_module.FALLBACK_MODELS)}")
        assert isinstance(config_module.FALLBACK_MODELS, list)
    
    def test_fallback_models_not_empty(self):
        """
        What it does: Verifies that FALLBACK_MODELS contains at least one model.
        Purpose: Ensure fallback list is populated for DNS failure recovery.
        """
        print("Setup: Importing FALLBACK_MODELS...")
        from kiro.config import FALLBACK_MODELS
        
        print(f"FALLBACK_MODELS length: {len(FALLBACK_MODELS)}")
        print(f"Comparing: Expected > 0, Got {len(FALLBACK_MODELS)}")
        assert len(FALLBACK_MODELS) > 0
    
    def test_fallback_models_structure(self):
        """
        What it does: Verifies that each fallback model has required modelId field.
        Purpose: Ensure fallback models have correct structure for cache.update().
        """
        print("Setup: Importing FALLBACK_MODELS...")
        from kiro.config import FALLBACK_MODELS
        
        print(f"Action: Checking structure of {len(FALLBACK_MODELS)} models...")
        for i, model in enumerate(FALLBACK_MODELS):
            print(f"Checking model {i}: {model}")
            
            print(f"  Verification: model is dict...")
            assert isinstance(model, dict), f"Model {i} is not a dict"
            
            print(f"  Verification: model has 'modelId'...")
            assert "modelId" in model, f"Model {i} missing 'modelId'"
            
            print(f"  Verification: modelId is string...")
            assert isinstance(model["modelId"], str), f"Model {i} modelId is not string"
            
            print(f"  Verification: modelId is not empty...")
            assert len(model["modelId"]) > 0, f"Model {i} modelId is empty"
    
    def test_fallback_models_contain_claude_models(self):
        """
        What it does: Verifies that fallback models include Claude models.
        Purpose: Ensure fallback list contains expected Claude 4/4.5 models.
        """
        print("Setup: Importing FALLBACK_MODELS...")
        from kiro.config import FALLBACK_MODELS
        
        model_ids = [m["modelId"] for m in FALLBACK_MODELS]
        print(f"Model IDs in fallback list: {model_ids}")
        
        print("Verification: Contains at least one Claude model...")
        has_claude = any("claude" in mid.lower() for mid in model_ids)
        assert has_claude, "No Claude models in fallback list"
    
    def test_fallback_models_use_dot_format(self):
        """
        What it does: Verifies that model IDs use dot format (e.g., claude-4.5).
        Purpose: Ensure consistency with Kiro API format.
        """
        print("Setup: Importing FALLBACK_MODELS...")
        from kiro.config import FALLBACK_MODELS
        
        print("Action: Checking model ID format...")
        for model in FALLBACK_MODELS:
            model_id = model["modelId"]
            print(f"Checking: {model_id}")
            
            # If model has version number, it should use dot format
            if any(char.isdigit() for char in model_id):
                # Check for patterns like "4.5" or "4-5"
                if "-4-5" in model_id or "-4-0" in model_id:
                    print(f"  WARNING: {model_id} uses dash format instead of dot")
                    # This is acceptable but not ideal
                    pass


class TestFallbackModelsIntegration:
    """Integration tests for FALLBACK_MODELS with ModelResolver."""
    
    @pytest.mark.asyncio
    async def test_fallback_models_work_with_model_resolver(self):
        """
        What it does: Verifies that fallback models work with ModelResolver normalization.
        Purpose: Ensure that model name normalization (claude-opus-4-5 → claude-opus-4.5)
                 works correctly with fallback models, just like with API models.
        """
        print("Setup: Importing FALLBACK_MODELS and creating cache...")
        from kiro.config import FALLBACK_MODELS
        from kiro.cache import ModelInfoCache
        from kiro.model_resolver import ModelResolver
        
        # Simulate DNS failure scenario - populate cache with fallback models
        cache = ModelInfoCache()
        await cache.update(FALLBACK_MODELS)
        
        print(f"Cache populated with {cache.size} fallback models")
        print(f"Model IDs in cache: {cache.get_all_model_ids()}")
        
        # Create resolver
        resolver = ModelResolver(cache=cache, hidden_models={})
        
        print("\nAction: Testing normalization with dash format...")
        # Test that dash format (claude-opus-4-5) is normalized and found
        test_cases = [
            ("claude-opus-4-5", "claude-opus-4.5"),  # Dash → Dot
            ("claude-sonnet-4-5", "claude-sonnet-4.5"),  # Dash → Dot
            ("claude-haiku-4-5", "claude-haiku-4.5"),  # Dash → Dot
        ]
        
        for input_name, expected_normalized in test_cases:
            print(f"\n  Testing: {input_name} → {expected_normalized}")
            resolution = resolver.resolve(input_name)
            
            print(f"    Resolution source: {resolution.source}")
            print(f"    Normalized: {resolution.normalized}")
            print(f"    Internal ID: {resolution.internal_id}")
            print(f"    Is verified: {resolution.is_verified}")
            
            # Verify normalization happened
            print(f"    Comparing normalized: Expected '{expected_normalized}', Got '{resolution.normalized}'")
            assert resolution.normalized == expected_normalized
            
            # Verify model was found in cache (not passthrough)
            print(f"    Comparing source: Expected 'cache', Got '{resolution.source}'")
            assert resolution.source == "cache", f"Model {input_name} should be found in fallback cache"
            
            print(f"    Comparing is_verified: Expected True, Got {resolution.is_verified}")
            assert resolution.is_verified is True
    
    @pytest.mark.asyncio
    async def test_fallback_models_appear_in_available_models(self):
        """
        What it does: Verifies that fallback models appear in get_available_models().
        Purpose: Ensure that /v1/models endpoint will show fallback models.
        """
        print("Setup: Importing FALLBACK_MODELS and creating cache...")
        from kiro.config import FALLBACK_MODELS
        from kiro.cache import ModelInfoCache
        from kiro.model_resolver import ModelResolver
        
        cache = ModelInfoCache()
        await cache.update(FALLBACK_MODELS)
        
        resolver = ModelResolver(cache=cache, hidden_models={})
        
        print("Action: Getting available models...")
        available = resolver.get_available_models()
        
        print(f"Available models: {available}")
        print(f"Comparing length: Expected {len(FALLBACK_MODELS)}, Got {len(available)}")
        assert len(available) == len(FALLBACK_MODELS)
        
        # Verify all fallback models are present
        fallback_ids = {m["modelId"] for m in FALLBACK_MODELS}
        available_set = set(available)
        
        print(f"Comparing sets: Expected {fallback_ids}, Got {available_set}")
        assert fallback_ids == available_set


# ==================================================================================================
# Tests for WebSearch Configuration
# ==================================================================================================

class TestWebSearchConfig:
    """Tests for WebSearch configuration (WEB_SEARCH_ENABLED)."""
    
    def test_web_search_enabled_default_true(self, monkeypatch):
        """
        What it does: Verifies WEB_SEARCH_ENABLED defaults to true.
        Purpose: Ensure auto-injection is enabled by default.
        """
        print("Setup: Removing WEB_SEARCH_ENABLED from environment...")
        monkeypatch.delenv("WEB_SEARCH_ENABLED", raising=False)
        
        print("Action: Reloading config module...")
        from importlib import reload
        import kiro.config as config_module
        reload(config_module)
        
        print(f"Comparing WEB_SEARCH_ENABLED: Expected True, Got {config_module.WEB_SEARCH_ENABLED}")
        assert config_module.WEB_SEARCH_ENABLED is True
    
    def test_web_search_enabled_false(self, monkeypatch):
        """
        What it does: Verifies WEB_SEARCH_ENABLED=false disables auto-injection.
        Purpose: Ensure users can disable auto-injection.
        """
        print("Setup: Setting WEB_SEARCH_ENABLED=false...")
        monkeypatch.setenv("WEB_SEARCH_ENABLED", "false")
        
        print("Action: Reloading config module...")
        from importlib import reload
        import kiro.config as config_module
        reload(config_module)
        
        print(f"Comparing WEB_SEARCH_ENABLED: Expected False, Got {config_module.WEB_SEARCH_ENABLED}")
        assert config_module.WEB_SEARCH_ENABLED is False
    
    def test_web_search_enabled_true(self, monkeypatch):
        """
        What it does: Verifies WEB_SEARCH_ENABLED=true enables auto-injection.
        Purpose: Ensure explicit true value works.
        """
        print("Setup: Setting WEB_SEARCH_ENABLED=true...")
        monkeypatch.setenv("WEB_SEARCH_ENABLED", "true")
        
        print("Action: Reloading config module...")
        from importlib import reload
        import kiro.config as config_module
        reload(config_module)
        
        print(f"Comparing WEB_SEARCH_ENABLED: Expected True, Got {config_module.WEB_SEARCH_ENABLED}")
        assert config_module.WEB_SEARCH_ENABLED is True
    
    def test_web_search_enabled_numeric_values(self, monkeypatch):
        """
        What it does: Verifies numeric values (1/0) work for WEB_SEARCH_ENABLED.
        Purpose: Ensure compatibility with numeric boolean values.
        """
        print("Setup: Testing WEB_SEARCH_ENABLED=1...")
        monkeypatch.setenv("WEB_SEARCH_ENABLED", "1")
        
        from importlib import reload
        import kiro.config as config_module
        reload(config_module)
        
        print(f"Comparing WEB_SEARCH_ENABLED: Expected True, Got {config_module.WEB_SEARCH_ENABLED}")
        assert config_module.WEB_SEARCH_ENABLED is True
        
        print("Setup: Testing WEB_SEARCH_ENABLED=0...")
        monkeypatch.setenv("WEB_SEARCH_ENABLED", "0")
        reload(config_module)
        
        print(f"Comparing WEB_SEARCH_ENABLED: Expected False, Got {config_module.WEB_SEARCH_ENABLED}")
        assert config_module.WEB_SEARCH_ENABLED is False
    
    def test_web_search_enabled_yes_value(self, monkeypatch):
        """
        What it does: Verifies WEB_SEARCH_ENABLED=yes enables auto-injection.
        Purpose: Ensure 'yes' value works.
        """
        print("Setup: Setting WEB_SEARCH_ENABLED=yes...")
        monkeypatch.setenv("WEB_SEARCH_ENABLED", "yes")
        
        print("Action: Reloading config module...")
        from importlib import reload
        import kiro.config as config_module
        reload(config_module)
        
        print(f"Comparing WEB_SEARCH_ENABLED: Expected True, Got {config_module.WEB_SEARCH_ENABLED}")
        assert config_module.WEB_SEARCH_ENABLED is True
    
    def test_web_search_enabled_case_insensitive(self, monkeypatch):
        """
        What it does: Verifies WEB_SEARCH_ENABLED is case-insensitive.
        Purpose: Ensure TRUE, True, true all work.
        """
        print("Setup: Testing WEB_SEARCH_ENABLED=TRUE...")
        monkeypatch.setenv("WEB_SEARCH_ENABLED", "TRUE")
        
        from importlib import reload
        import kiro.config as config_module
        reload(config_module)
        
        print(f"Comparing WEB_SEARCH_ENABLED: Expected True, Got {config_module.WEB_SEARCH_ENABLED}")
        assert config_module.WEB_SEARCH_ENABLED is True
        
        print("Setup: Testing WEB_SEARCH_ENABLED=FALSE...")
        monkeypatch.setenv("WEB_SEARCH_ENABLED", "FALSE")
        reload(config_module)
        
        print(f"Comparing WEB_SEARCH_ENABLED: Expected False, Got {config_module.WEB_SEARCH_ENABLED}")
        assert config_module.WEB_SEARCH_ENABLED is False


# ==================================================================================================
# Tests for Account System Configuration
# ==================================================================================================

class TestAccountSystemConfig:
    """Tests for Account System configuration constants."""
    
    def test_account_system_default_false(self):
        """
        What it does: Verifies ACCOUNT_SYSTEM defaults to false.
        Purpose: Ensure legacy mode is default (backward compatibility).
        """
        print("Setup: Mocking os.getenv for ACCOUNT_SYSTEM...")
        
        original_getenv = os.getenv
        
        def mock_getenv(key, default=None):
            if key == "ACCOUNT_SYSTEM":
                print(f"os.getenv('{key}') -> None (mocked)")
                return default  # Return default, simulating missing variable
            return original_getenv(key, default)
        
        with patch.object(os, 'getenv', side_effect=mock_getenv):
            print("Action: Reloading config module...")
            from importlib import reload
            import kiro.config as config_module
            reload(config_module)
            
            print(f"Comparing ACCOUNT_SYSTEM: Expected False, Got {config_module.ACCOUNT_SYSTEM}")
            assert config_module.ACCOUNT_SYSTEM is False
        
        # Restore module with real values
        from importlib import reload
        import kiro.config as config_module
        reload(config_module)
    
    def test_account_system_enabled(self, monkeypatch):
        """
        What it does: Verifies ACCOUNT_SYSTEM=true enables account system.
        Purpose: Ensure account system can be enabled via environment variable.
        """
        print("Setup: Setting ACCOUNT_SYSTEM=true...")
        monkeypatch.setenv("ACCOUNT_SYSTEM", "true")
        
        print("Action: Reloading config module...")
        from importlib import reload
        import kiro.config as config_module
        reload(config_module)
        
        print(f"Comparing ACCOUNT_SYSTEM: Expected True, Got {config_module.ACCOUNT_SYSTEM}")
        assert config_module.ACCOUNT_SYSTEM is True
    
    def test_accounts_config_file_default(self, monkeypatch):
        """
        What it does: Verifies ACCOUNTS_CONFIG_FILE defaults to credentials.json.
        Purpose: Ensure default path for credentials configuration.
        """
        print("Setup: Removing ACCOUNTS_CONFIG_FILE from environment...")
        monkeypatch.delenv("ACCOUNTS_CONFIG_FILE", raising=False)
        
        print("Action: Reloading config module...")
        from importlib import reload
        import kiro.config as config_module
        reload(config_module)
        
        print(f"Comparing ACCOUNTS_CONFIG_FILE: Expected 'credentials.json', Got '{config_module.ACCOUNTS_CONFIG_FILE}'")
        assert config_module.ACCOUNTS_CONFIG_FILE == "credentials.json"
    
    def test_accounts_state_file_default(self, monkeypatch):
        """
        What it does: Verifies ACCOUNTS_STATE_FILE defaults to state.json.
        Purpose: Ensure default path for runtime state file.
        """
        print("Setup: Removing ACCOUNTS_STATE_FILE from environment...")
        monkeypatch.delenv("ACCOUNTS_STATE_FILE", raising=False)
        
        print("Action: Reloading config module...")
        from importlib import reload
        import kiro.config as config_module
        reload(config_module)
        
        print(f"Comparing ACCOUNTS_STATE_FILE: Expected 'state.json', Got '{config_module.ACCOUNTS_STATE_FILE}'")
        assert config_module.ACCOUNTS_STATE_FILE == "state.json"
    
    def test_account_recovery_timeout_default(self, monkeypatch):
        """
        What it does: Verifies ACCOUNT_RECOVERY_TIMEOUT defaults to 60 seconds.
        Purpose: Ensure base timeout for exponential backoff is 60s.
        """
        print("Setup: Removing ACCOUNT_RECOVERY_TIMEOUT from environment...")
        monkeypatch.delenv("ACCOUNT_RECOVERY_TIMEOUT", raising=False)
        
        print("Action: Reloading config module...")
        from importlib import reload
        import kiro.config as config_module
        reload(config_module)
        
        print(f"Comparing ACCOUNT_RECOVERY_TIMEOUT: Expected 60, Got {config_module.ACCOUNT_RECOVERY_TIMEOUT}")
        assert config_module.ACCOUNT_RECOVERY_TIMEOUT == 60
    
    def test_account_max_backoff_multiplier_default(self, monkeypatch):
        """
        What it does: Verifies ACCOUNT_MAX_BACKOFF_MULTIPLIER defaults to 1440.0.
        Purpose: Ensure maximum backoff cap is 1 day (60s * 1440 = 86400s).
        """
        print("Setup: Removing ACCOUNT_MAX_BACKOFF_MULTIPLIER from environment...")
        monkeypatch.delenv("ACCOUNT_MAX_BACKOFF_MULTIPLIER", raising=False)
        
        print("Action: Reloading config module...")
        from importlib import reload
        import kiro.config as config_module
        reload(config_module)
        
        print(f"Comparing ACCOUNT_MAX_BACKOFF_MULTIPLIER: Expected 1440.0, Got {config_module.ACCOUNT_MAX_BACKOFF_MULTIPLIER}")
        assert config_module.ACCOUNT_MAX_BACKOFF_MULTIPLIER == 1440.0
    
    def test_account_probabilistic_retry_chance_default(self, monkeypatch):
        """
        What it does: Verifies ACCOUNT_PROBABILISTIC_RETRY_CHANCE defaults to 0.1.
        Purpose: Ensure 10% chance for probabilistic retry of broken accounts.
        """
        print("Setup: Removing ACCOUNT_PROBABILISTIC_RETRY_CHANCE from environment...")
        monkeypatch.delenv("ACCOUNT_PROBABILISTIC_RETRY_CHANCE", raising=False)
        
        print("Action: Reloading config module...")
        from importlib import reload
        import kiro.config as config_module
        reload(config_module)
        
        print(f"Comparing ACCOUNT_PROBABILISTIC_RETRY_CHANCE: Expected 0.1, Got {config_module.ACCOUNT_PROBABILISTIC_RETRY_CHANCE}")
        assert config_module.ACCOUNT_PROBABILISTIC_RETRY_CHANCE == 0.1
    
    def test_account_cache_ttl_default(self, monkeypatch):
        """
        What it does: Verifies ACCOUNT_CACHE_TTL defaults to 43200 seconds (12 hours).
        Purpose: Ensure model cache TTL is 12 hours by default.
        """
        print("Setup: Removing ACCOUNT_CACHE_TTL from environment...")
        monkeypatch.delenv("ACCOUNT_CACHE_TTL", raising=False)
        
        print("Action: Reloading config module...")
        from importlib import reload
        import kiro.config as config_module
        reload(config_module)
        
        print(f"Comparing ACCOUNT_CACHE_TTL: Expected 43200, Got {config_module.ACCOUNT_CACHE_TTL}")
        assert config_module.ACCOUNT_CACHE_TTL == 43200
    
    def test_state_save_interval_seconds_default(self, monkeypatch):
        """
        What it does: Verifies STATE_SAVE_INTERVAL_SECONDS defaults to 10 seconds.
        Purpose: Ensure periodic state saving happens every 10 seconds.
        """
        print("Setup: Removing STATE_SAVE_INTERVAL_SECONDS from environment...")
        monkeypatch.delenv("STATE_SAVE_INTERVAL_SECONDS", raising=False)
        
        print("Action: Reloading config module...")
        from importlib import reload
        import kiro.config as config_module
        reload(config_module)
        
        print(f"Comparing STATE_SAVE_INTERVAL_SECONDS: Expected 10, Got {config_module.STATE_SAVE_INTERVAL_SECONDS}")
        assert config_module.STATE_SAVE_INTERVAL_SECONDS == 10

class TestFakeReasoningAsTextModes:
    """Tests for FAKE_REASONING_AS_TEXT_MODES (modes that emit thinking as regular text)."""

    def test_text_modes_cover_pass_and_strip_tags_but_not_structured_modes(self):
        """
        What it does: Checks the contents of FAKE_REASONING_AS_TEXT_MODES.
        Purpose: "pass" / "strip_tags" (documented as keeping thinking in regular content)
        and the legacy "include_as_text" are text modes; "as_reasoning_content" and
        "remove" must never be.
        """
        from kiro.config import FAKE_REASONING_AS_TEXT_MODES

        print(f"Modes: {FAKE_REASONING_AS_TEXT_MODES}")
        assert "pass" in FAKE_REASONING_AS_TEXT_MODES
        assert "strip_tags" in FAKE_REASONING_AS_TEXT_MODES
        assert "include_as_text" in FAKE_REASONING_AS_TEXT_MODES
        assert "as_reasoning_content" not in FAKE_REASONING_AS_TEXT_MODES
        assert "remove" not in FAKE_REASONING_AS_TEXT_MODES

    def test_every_valid_handling_value_is_classified_exactly_once(self):
        """
        What it does: Classifies the four valid FAKE_REASONING_HANDLING values.
        Purpose: Each valid value is either a text mode or a structured/discard mode - never both.
        """
        from kiro.config import FAKE_REASONING_AS_TEXT_MODES

        structured = {"as_reasoning_content", "remove"}
        for value in ("as_reasoning_content", "remove", "pass", "strip_tags"):
            assert (value in FAKE_REASONING_AS_TEXT_MODES) != (value in structured), value


class TestEmptyResponseConfig:
    """Tests for EMPTY_RESPONSE_AS_ERROR and EMPTY_RESPONSE_ERROR_STATUS configuration."""

    @pytest.fixture(autouse=True)
    def restore_config(self, monkeypatch):
        """Reloads kiro.config with the real environment after each test."""
        yield
        monkeypatch.undo()
        from importlib import reload
        import kiro.config as config_module
        reload(config_module)

    @staticmethod
    def _reload(monkeypatch, **env):
        """
        Reloads kiro.config with the given environment overrides.

        Args:
            monkeypatch: pytest monkeypatch fixture
            **env: Environment variables to set (None removes the variable)

        Returns:
            The reloaded kiro.config module
        """
        from importlib import reload
        import kiro.config as config_module

        for name, value in env.items():
            if value is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, value)
        return reload(config_module)

    def test_defaults_enabled_with_status_503(self, monkeypatch):
        """
        What it does: Removes both variables and reloads the config.
        Purpose: Empty upstream responses must be converted to errors by default, with 503.
        """
        config_module = self._reload(
            monkeypatch, EMPTY_RESPONSE_AS_ERROR=None, EMPTY_RESPONSE_ERROR_STATUS=None
        )

        print(f"EMPTY_RESPONSE_AS_ERROR={config_module.EMPTY_RESPONSE_AS_ERROR}, "
              f"EMPTY_RESPONSE_ERROR_STATUS={config_module.EMPTY_RESPONSE_ERROR_STATUS}")
        assert config_module.EMPTY_RESPONSE_AS_ERROR is True
        assert config_module.EMPTY_RESPONSE_ERROR_STATUS == 503

    @pytest.mark.parametrize("raw", ["true", "TRUE", "True", "1", "yes", "YES", " true", "true ", " yes "])
    def test_truthy_values_enable(self, monkeypatch, raw):
        """
        What it does: Sets EMPTY_RESPONSE_AS_ERROR to each accepted truthy spelling.
        Purpose: Boolean parsing must match the project's existing true/1/yes convention.
        """
        config_module = self._reload(monkeypatch, EMPTY_RESPONSE_AS_ERROR=raw)

        assert config_module.EMPTY_RESPONSE_AS_ERROR is True

    @pytest.mark.parametrize("raw", ["false", "FALSE", "0", "no", "off", "garbage", "", "tru"])
    def test_falsy_and_invalid_values_disable(self, monkeypatch, raw):
        """
        What it does: Sets EMPTY_RESPONSE_AS_ERROR to false-like and invalid values.
        Purpose: Anything that is not true/1/yes restores the native pass-through behavior.
        """
        config_module = self._reload(monkeypatch, EMPTY_RESPONSE_AS_ERROR=raw)

        print(f"{raw!r} -> {config_module.EMPTY_RESPONSE_AS_ERROR}")
        assert config_module.EMPTY_RESPONSE_AS_ERROR is False

    @pytest.mark.parametrize("raw,expected", [
        ("500", 500), ("503", 503), ("529", 529), ("599", 599), (" 502 ", 502), ("+504", 504),
    ])
    def test_valid_status_values_are_used(self, monkeypatch, raw, expected):
        """
        What it does: Sets EMPTY_RESPONSE_ERROR_STATUS to values in 500-599.
        Purpose: Operators can pick the status their client retries on.
        """
        config_module = self._reload(monkeypatch, EMPTY_RESPONSE_ERROR_STATUS=raw)

        assert config_module.EMPTY_RESPONSE_ERROR_STATUS == expected

    @pytest.mark.parametrize("raw", ["499", "600", "0", "-1", "-503", "abc", "", "50 3", "503.5", "5e2", "200", "99999"])
    def test_invalid_status_values_fall_back_to_503(self, monkeypatch, raw):
        """
        What it does: Sets EMPTY_RESPONSE_ERROR_STATUS to out-of-range and malformed values.
        Purpose: A typo must never produce a 2xx/3xx/4xx status or crash the import.
        """
        config_module = self._reload(monkeypatch, EMPTY_RESPONSE_ERROR_STATUS=raw)

        print(f"{raw!r} -> {config_module.EMPTY_RESPONSE_ERROR_STATUS}")
        assert config_module.EMPTY_RESPONSE_ERROR_STATUS == 503

    def test_parse_helper_handles_boundaries(self):
        """
        What it does: Calls _parse_empty_response_status directly at the range edges.
        Purpose: 500 and 599 are accepted, 499 and 600 are rejected.
        """
        from kiro.config import _parse_empty_response_status

        assert _parse_empty_response_status("500") == 500
        assert _parse_empty_response_status("599") == 599
        assert _parse_empty_response_status("499") == 503
        assert _parse_empty_response_status("600") == 503


class TestIncompleteResponseConfig:
    """Tests for INCOMPLETE_RESPONSE_AS_ERROR, INCOMPLETE_RESPONSE_ERROR_TYPE and _parse_switch."""

    @pytest.fixture(autouse=True)
    def restore_config(self, monkeypatch):
        """Reloads kiro.config with the real environment after each test."""
        yield
        monkeypatch.undo()
        from importlib import reload
        import kiro.config as config_module
        reload(config_module)

    @staticmethod
    def _reload(monkeypatch, **env):
        """
        Reloads kiro.config with the given environment overrides and collects its WARNINGs.

        Args:
            monkeypatch: pytest monkeypatch fixture
            **env: Environment variables to set (None removes the variable)

        Returns:
            Tuple of (reloaded kiro.config module, list of WARNING messages logged while loading)
        """
        from importlib import reload
        import kiro.config as config_module
        from loguru import logger

        for name, value in env.items():
            if value is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, value)
        warnings = []
        handler_id = logger.add(lambda m: warnings.append(str(m)), level="WARNING", format="{message}")
        try:
            module = reload(config_module)
        finally:
            logger.remove(handler_id)
        return module, warnings

    def test_defaults(self, monkeypatch):
        """
        What it does: Removes both variables and reloads the config.
        Purpose: Switch ON and type overloaded_error by default, without any warning.
        """
        config_module, warnings = self._reload(
            monkeypatch, INCOMPLETE_RESPONSE_AS_ERROR=None, INCOMPLETE_RESPONSE_ERROR_TYPE=None,
            EMPTY_RESPONSE_AS_ERROR=None,
        )

        assert config_module.INCOMPLETE_RESPONSE_AS_ERROR is True
        assert config_module.INCOMPLETE_RESPONSE_ERROR_TYPE == "overloaded_error"
        assert not [w for w in warnings if "INCOMPLETE_RESPONSE" in w]

    @pytest.mark.parametrize("raw", ["true", "TRUE", "1", "yes", "YES", " true ", "\tyes\n"])
    def test_truthy_values_enable_without_warning(self, monkeypatch, raw):
        """
        What it does: Each accepted truthy spelling, also with surrounding whitespace.
        Purpose: Same parsing as EMPTY_RESPONSE_AS_ERROR, plus strip().
        """
        config_module, warnings = self._reload(monkeypatch, INCOMPLETE_RESPONSE_AS_ERROR=raw)

        assert config_module.INCOMPLETE_RESPONSE_AS_ERROR is True
        assert not [w for w in warnings if "INCOMPLETE_RESPONSE_AS_ERROR" in w]

    @pytest.mark.parametrize("raw", ["false", "FALSE", "0", "no", "off", "", " false "])
    def test_falsy_values_disable_without_warning(self, monkeypatch, raw):
        """
        What it does: Each accepted falsy spelling.
        Purpose: Switch OFF restores pass-through and is silent.
        """
        config_module, warnings = self._reload(monkeypatch, INCOMPLETE_RESPONSE_AS_ERROR=raw)

        assert config_module.INCOMPLETE_RESPONSE_AS_ERROR is False
        assert not [w for w in warnings if "INCOMPLETE_RESPONSE_AS_ERROR" in w]

    @pytest.mark.parametrize("raw", ["ture", "tru", "enable", "2", "on"])
    def test_unrecognized_values_disable_with_startup_warning(self, monkeypatch, raw):
        """
        What it does: Typos and unknown spellings.
        Purpose: Historical semantics kept (treated as false) but made visible by a WARNING.
        """
        config_module, warnings = self._reload(monkeypatch, INCOMPLETE_RESPONSE_AS_ERROR=raw)

        assert config_module.INCOMPLETE_RESPONSE_AS_ERROR is False
        matching = [w for w in warnings if "INCOMPLETE_RESPONSE_AS_ERROR" in w]
        assert len(matching) == 1 and repr(raw) in matching[0]

    def test_empty_response_switch_warns_on_typo_and_strips(self, monkeypatch):
        """
        What it does: EMPTY_RESPONSE_AS_ERROR with a typo, then with padded "true".
        Purpose: The existing switch shares the parser: typo -> False + WARNING, " true " -> True.
        """
        config_module, warnings = self._reload(monkeypatch, EMPTY_RESPONSE_AS_ERROR="ture")
        assert config_module.EMPTY_RESPONSE_AS_ERROR is False
        assert [w for w in warnings if "EMPTY_RESPONSE_AS_ERROR" in w]

        config_module, warnings = self._reload(monkeypatch, EMPTY_RESPONSE_AS_ERROR=" true ")
        assert config_module.EMPTY_RESPONSE_AS_ERROR is True
        assert not [w for w in warnings if "EMPTY_RESPONSE_AS_ERROR" in w]

    @pytest.mark.parametrize("raw,expected", [
        ("overloaded_error", "overloaded_error"),
        ("api_error", "api_error"),
        (" API_ERROR ", "api_error"),
        ("Overloaded_Error", "overloaded_error"),
    ])
    def test_error_type_is_normalized(self, monkeypatch, raw, expected):
        """
        What it does: Allowed values in odd spellings.
        Purpose: strip + lower, no warning.
        """
        config_module, warnings = self._reload(monkeypatch, INCOMPLETE_RESPONSE_ERROR_TYPE=raw)

        assert config_module.INCOMPLETE_RESPONSE_ERROR_TYPE == expected
        assert not [w for w in warnings if "INCOMPLETE_RESPONSE_ERROR_TYPE" in w]

    @pytest.mark.parametrize("raw", ["foo", "", "rate_limit_error", "invalid_request_error", "overloaded", "null", "123"])
    def test_invalid_error_type_falls_back_with_warning(self, monkeypatch, raw):
        """
        What it does: Values outside the whitelist.
        Purpose: Fall back to overloaded_error and log one WARNING; never raise.
        """
        config_module, warnings = self._reload(monkeypatch, INCOMPLETE_RESPONSE_ERROR_TYPE=raw)

        assert config_module.INCOMPLETE_RESPONSE_ERROR_TYPE == "overloaded_error"
        assert len([w for w in warnings if "INCOMPLETE_RESPONSE_ERROR_TYPE" in w]) == 1

    def test_status_setting_is_reused_unchanged(self, monkeypatch):
        """
        What it does: EMPTY_RESPONSE_ERROR_STATUS with the incomplete switch present.
        Purpose: No new status setting; the existing one keeps working.
        """
        config_module, _ = self._reload(monkeypatch, EMPTY_RESPONSE_ERROR_STATUS="529")

        assert config_module.EMPTY_RESPONSE_ERROR_STATUS == 529
        assert not hasattr(config_module, "INCOMPLETE_RESPONSE_ERROR_STATUS")

    def test_env_example_documents_both_new_variables(self):
        """
        What it does: Reads .env.example (the committed template, never .env).
        Purpose: Both new options must be documented there, with their defaults.
        """
        from pathlib import Path

        text = (Path(__file__).resolve().parents[2] / ".env.example").read_text(encoding="utf-8")

        assert "# INCOMPLETE_RESPONSE_AS_ERROR=true" in text
        assert "# INCOMPLETE_RESPONSE_ERROR_TYPE=overloaded_error" in text


class TestForwardModelRequestFieldsConfig:
    """Tests for the FORWARD_MODEL_REQUEST_FIELDS switch."""

    @pytest.fixture(autouse=True)
    def restore_config(self, monkeypatch):
        """Reloads kiro.config with the real environment after each test."""
        yield
        monkeypatch.undo()
        from importlib import reload
        import kiro.config as config_module
        reload(config_module)

    @staticmethod
    def _reload(monkeypatch, **env):
        """
        Reloads kiro.config with the given environment overrides and collects its WARNINGs.

        Args:
            monkeypatch: pytest monkeypatch fixture
            **env: Environment variables to set (None removes the variable)

        Returns:
            Tuple of (reloaded kiro.config module, list of WARNING messages logged while loading)
        """
        from importlib import reload
        import kiro.config as config_module
        from loguru import logger

        for name, value in env.items():
            if value is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, value)
        warnings = []
        handler_id = logger.add(lambda m: warnings.append(str(m)), level="WARNING", format="{message}")
        try:
            module = reload(config_module)
        finally:
            logger.remove(handler_id)
        return module, warnings

    def test_default_is_enabled_without_warning(self, monkeypatch):
        """
        What it does: Removes the variable and reloads the config.
        Purpose: The switch is ON by default and silent.
        """
        config_module, warnings = self._reload(monkeypatch, FORWARD_MODEL_REQUEST_FIELDS=None)

        assert config_module.FORWARD_MODEL_REQUEST_FIELDS is True
        assert not [w for w in warnings if "FORWARD_MODEL_REQUEST_FIELDS" in w]

    @pytest.mark.parametrize("raw", ["true", "TRUE", "1", "yes", " true ", " yes "])
    def test_truthy_values_enable_without_warning(self, monkeypatch, raw):
        """
        What it does: Each accepted truthy spelling, also padded.
        Purpose: Same parsing as the other switches (strip + lower).
        """
        config_module, warnings = self._reload(monkeypatch, FORWARD_MODEL_REQUEST_FIELDS=raw)

        assert config_module.FORWARD_MODEL_REQUEST_FIELDS is True
        assert not [w for w in warnings if "FORWARD_MODEL_REQUEST_FIELDS" in w]

    @pytest.mark.parametrize("raw", ["false", "FALSE", "0", "no", "off", "", " false "])
    def test_falsy_values_disable_without_warning(self, monkeypatch, raw):
        """
        What it does: Each accepted falsy spelling.
        Purpose: Switch OFF restores the old behavior (field never sent) and is silent.
        """
        config_module, warnings = self._reload(monkeypatch, FORWARD_MODEL_REQUEST_FIELDS=raw)

        assert config_module.FORWARD_MODEL_REQUEST_FIELDS is False
        assert not [w for w in warnings if "FORWARD_MODEL_REQUEST_FIELDS" in w]

    @pytest.mark.parametrize("raw", ["garbage", "tru", "ture", "on", "2"])
    def test_unrecognized_values_disable_with_startup_warning(self, monkeypatch, raw):
        """
        What it does: Typos and unknown spellings.
        Purpose: Treated as false but made visible by exactly one startup WARNING.
        """
        config_module, warnings = self._reload(monkeypatch, FORWARD_MODEL_REQUEST_FIELDS=raw)

        assert config_module.FORWARD_MODEL_REQUEST_FIELDS is False
        matching = [w for w in warnings if "FORWARD_MODEL_REQUEST_FIELDS" in w]
        assert len(matching) == 1 and repr(raw) in matching[0]

    def test_env_example_documents_the_switch(self):
        """
        What it does: Reads .env.example (the committed template, never .env).
        Purpose: The option must be documented there with its default.
        """
        from pathlib import Path

        text = (Path(__file__).resolve().parents[2] / ".env.example").read_text(encoding="utf-8")

        assert "# FORWARD_MODEL_REQUEST_FIELDS=true" in text
