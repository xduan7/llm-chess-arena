"""Unit tests for configuration module."""

import os
from unittest.mock import patch

from llm_chess_arena import config


class TestLoadEnv:
    """Tests for load_env function."""

    def setup_method(self):
        """Reset global state before each test."""
        config._ENV_LOADED = False
        # Clear any test env vars
        for key in list(os.environ.keys()):
            if key.startswith("TEST_"):
                del os.environ[key]

    def test_load_env__given_valid_env_file__when_called__then_loads_variables(
        self, tmp_path
    ):
        """Test that load_env successfully loads variables from a .env file."""
        env_file = tmp_path / ".env"
        env_file.write_text("TEST_VAR=test_value\nTEST_NUMBER=42")

        # Ensure vars don't exist yet
        assert "TEST_VAR" not in os.environ
        assert "TEST_NUMBER" not in os.environ

        # Load the env file
        loaded_path = config.load_env(str(env_file))

        # Verify it was loaded
        assert loaded_path == env_file
        assert os.environ.get("TEST_VAR") == "test_value"
        assert os.environ.get("TEST_NUMBER") == "42"
        assert config._ENV_LOADED is True

    def test_load_env__when_called_twice__then_skips_second_load(self, tmp_path):
        """Test that load_env doesn't reload on second call unless override=True."""
        env_file = tmp_path / ".env"
        env_file.write_text("TEST_VAR=first_value")

        # First load
        first_path = config.load_env(str(env_file))
        assert first_path == env_file
        assert os.environ.get("TEST_VAR") == "first_value"

        # Modify the file
        env_file.write_text("TEST_VAR=second_value")

        # Second load without override - should skip
        second_path = config.load_env(str(env_file))
        assert second_path is None  # Skipped
        assert os.environ.get("TEST_VAR") == "first_value"  # Unchanged

    def test_load_env__given_override_true__when_already_loaded__then_reloads(
        self, tmp_path
    ):
        """Test that override=True forces a reload of environment variables."""
        env_file = tmp_path / ".env"
        env_file.write_text("TEST_VAR=initial")

        # First load
        config.load_env(str(env_file))
        assert os.environ.get("TEST_VAR") == "initial"

        # Update file
        env_file.write_text("TEST_VAR=updated")

        # Reload with override
        reloaded_path = config.load_env(str(env_file), override=True)
        assert reloaded_path == env_file
        assert os.environ.get("TEST_VAR") == "updated"

    def test_load_env__given_nonexistent_file__when_called__then_returns_none(self):
        """Test that load_env returns None when file doesn't exist."""
        result = config.load_env("nonexistent.env")
        assert result is None
        assert config._ENV_LOADED is False

    def test_load_env__given_env_file_envvar__when_no_filename__then_uses_envvar(
        self, tmp_path, monkeypatch
    ):
        """Test that ENV_FILE environment variable is used when no filename provided."""
        env_file = tmp_path / "custom.env"
        env_file.write_text("TEST_FROM_CUSTOM=yes")

        # Set ENV_FILE to point to our custom file
        monkeypatch.setenv("ENV_FILE", str(env_file))

        # Call without filename
        loaded_path = config.load_env()

        assert loaded_path == env_file
        assert os.environ.get("TEST_FROM_CUSTOM") == "yes"

    def test_load_env__given_empty_file__when_called__then_loads_successfully(
        self, tmp_path
    ):
        """Test that empty .env file is handled gracefully."""
        env_file = tmp_path / ".env"
        env_file.write_text("")

        loaded_path = config.load_env(str(env_file))

        assert loaded_path == env_file
        assert config._ENV_LOADED is True

    def test_load_env__given_comments_and_whitespace__when_called__then_parses_correctly(
        self, tmp_path
    ):
        """Test that .env file with comments and whitespace is parsed correctly."""
        env_file = tmp_path / ".env"
        env_file.write_text(
            """
# This is a comment
TEST_KEY=value  # inline comment

  # Another comment
TEST_SPACES  =  spaced_value  
TEST_QUOTES="quoted value"
"""
        )

        config.load_env(str(env_file))

        assert os.environ.get("TEST_KEY") == "value"
        assert os.environ.get("TEST_SPACES") == "spaced_value"
        assert os.environ.get("TEST_QUOTES") == "quoted value"

    def test_load_env__given_override_false__when_vars_exist__then_preserves_existing(
        self, tmp_path
    ):
        """Test that existing env vars are preserved when override=False."""
        env_file = tmp_path / ".env"
        env_file.write_text("TEST_EXISTING=from_file")

        # Set existing value
        os.environ["TEST_EXISTING"] = "from_environ"

        # Load without override
        config.load_env(str(env_file), override=False)

        # Existing value should be preserved
        assert os.environ.get("TEST_EXISTING") == "from_environ"

    def test_load_env__given_override_true__when_vars_exist__then_overwrites(
        self, tmp_path
    ):
        """Test that existing env vars are overwritten when override=True."""
        env_file = tmp_path / ".env"
        env_file.write_text("TEST_OVERRIDE=from_file")

        # Set existing value
        os.environ["TEST_OVERRIDE"] = "from_environ"

        # Load with override
        config.load_env(str(env_file), override=True)

        # Should be overwritten
        assert os.environ.get("TEST_OVERRIDE") == "from_file"


class TestConfigIntegration:
    """Integration tests for config module behavior."""

    def test_config_module_import_does_not_load_env(self):
        """Test that importing config module doesn't automatically load env."""
        # Reset state
        config._ENV_LOADED = False

        # Re-import shouldn't trigger load
        import importlib

        importlib.reload(config)

        assert config._ENV_LOADED is False

    def test_config_with_api_keys_pattern(self, tmp_path):
        """Test typical API key configuration pattern."""
        env_file = tmp_path / ".env"
        env_file.write_text(
            """
OPENAI_API_KEY=sk-test123
ANTHROPIC_API_KEY=ant-test456
GOOGLE_API_KEY=goog-test789
"""
        )

        config._ENV_LOADED = False
        # Use override=True to overwrite any existing values
        config.load_env(str(env_file), override=True)

        # Verify all keys are loaded
        assert os.environ.get("OPENAI_API_KEY") == "sk-test123"
        assert os.environ.get("ANTHROPIC_API_KEY") == "ant-test456"
        assert os.environ.get("GOOGLE_API_KEY") == "goog-test789"

    @patch("llm_chess_arena.config.logger")
    def test_load_env_logs_appropriately(self, mock_logger, tmp_path):
        """Test that load_env logs debug messages appropriately."""
        env_file = tmp_path / ".env"
        env_file.write_text("TEST=value")

        # Reset state
        config._ENV_LOADED = False

        # Load existing file
        config.load_env(str(env_file))
        mock_logger.debug.assert_called_with(f"Loaded environment from: {env_file}")

        # Reset mock
        mock_logger.reset_mock()
        config._ENV_LOADED = False

        # Try loading non-existent file
        config.load_env("nonexistent.env")
        mock_logger.debug.assert_called_with("No .env file found: nonexistent.env")
