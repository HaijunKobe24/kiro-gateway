# -*- coding: utf-8 -*-

"""
Unit tests for Kiro request headers (client version in User-Agent).
"""

from unittest.mock import MagicMock

from kiro.config import KIRO_IDE_VERSION
from kiro.utils import get_kiro_headers


class TestKiroIdeVersionHeaders:
    """Tests that the configured Kiro IDE version is used in headers."""

    def test_user_agent_headers_use_configured_version(self):
        """
        What it does: Builds headers and inspects both User-Agent headers.
        Purpose: Kiro API rejects outdated client versions with 403, so the
                 version must come from config, not a hardcoded literal.
        """
        # Arrange
        auth_manager = MagicMock()
        auth_manager.fingerprint = "abc123"

        # Act
        headers = get_kiro_headers(auth_manager, "token")

        # Assert
        expected = f"KiroIDE-{KIRO_IDE_VERSION}-abc123"
        assert expected in headers["User-Agent"]
        assert expected in headers["x-amz-user-agent"]

    def test_default_version_is_not_the_rejected_legacy_version(self):
        """
        What it does: Checks the default version constant.
        Purpose: 0.7.45 is rejected upstream (FEATURE_NOT_SUPPORTED) and must not return.
        """
        assert KIRO_IDE_VERSION != "0.7.45"
        assert KIRO_IDE_VERSION
