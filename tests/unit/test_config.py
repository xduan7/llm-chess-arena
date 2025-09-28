"""Unit tests for configuration module."""

import os
from pathlib import Path

from unittest.mock import patch

from omegaconf import OmegaConf

from llm_chess_arena import config
from llm_chess_arena.config import _ensure_color, RandomPlayerConfig
from llm_chess_arena.game import Game
from llm_chess_arena.player.base_player import BasePlayer
from llm_chess_arena.types import PlayerDecision


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
        assert os.environ.get("OPENAI_API_KEY") == "sk-test123"
        assert os.environ.get("ANTHROPIC_API_KEY") == "ant-test456"
        assert os.environ.get("GOOGLE_API_KEY") == "goog-test789"


class CloseCountingPlayer(BasePlayer):
    """Test helper that tracks how many times close() is invoked."""

    def __init__(self, name: str, color: str) -> None:
        """Initialize the stub player and reset close() counters."""
        super().__init__(name=name, color=color)
        self.close_calls = 0

    def _make_decision(self, context) -> PlayerDecision:  # type: ignore[override]
        """Always resign so the game loop terminates quickly in tests."""
        return PlayerDecision(action="resign")

    def close(self) -> None:  # noqa: D401
        """Increment the counter to track cleanup calls."""
        self.close_calls += 1


def test_run_game_from_config_closes_players_once(monkeypatch):
    """Ensure run_game_from_config relies on Game.play for cleanup."""

    white_player = CloseCountingPlayer("White", "white")
    black_player = CloseCountingPlayer("Black", "black")
    game = Game(white_player, black_player, enable_metrics=False)

    captured_config: dict[str, config.AppConfig] = {}

    def fake_create_game(app_config: config.AppConfig) -> Game:
        """Capture the config used to build a game and return the stub instance."""
        captured_config["app_config"] = app_config
        return game

    monkeypatch.setattr(
        "llm_chess_arena.factory.GameFactory.create_game", fake_create_game
    )

    app_config = config.AppConfig(
        env=config.EnvConfig(load_dotenv=False),
        game=config.GameConfig(enable_metrics=False, max_num_moves=1),
        metrics=config.MetricsConfig(),
        players=config.PlayersConfig(
            white=config.RandomPlayerConfig(color="white", name="Random White"),
            black=config.RandomPlayerConfig(color="black", name="Random Black"),
        ),
    )

    config.run_game_from_config(app_config)

    assert white_player.close_calls == 1
    assert black_player.close_calls == 1
    assert captured_config["app_config"].game.max_num_moves == 1


class TestHydraConfig:
    """Tests validating Hydra-backed configuration helpers."""

    def test_load_app_config__when_defaults_requested__then_returns_expected_players(
        self,
    ):
        """Default composition should surface expected player/metric settings."""
        cfg = config.load_app_config()

        assert cfg.game.display_board is True
        assert cfg.players.white.kind == "random"
        assert cfg.players.black.kind == "random"
        assert cfg.metrics.stockfish_depth == 20
        assert cfg.metrics.stockfish_engine_options == {"Threads": 4, "Hash": 1024}
        assert cfg.metrics.quality_thresholds.excellent == 50.0

    def test_load_app_config__when_overrides_supplied__then_applies_changes(self):
        """Hydra overrides should mutate the resulting AppConfig dataclasses."""
        overrides = [
            "players@players.white=stockfish",
            "+players.white.engine_limits.depth=16",
            "players@players.black=llm/default",
            "players.black.connector.model=gpt-4",
            "players.black.name=GPT-4",
            "metrics.stockfish_depth=18",
            "metrics.quality_thresholds.mistake=250",
        ]
        cfg = config.load_app_config(overrides=overrides)

        assert cfg.players.white.kind == "stockfish"
        assert cfg.players.white.engine_limits["depth"] == 16
        assert cfg.players.black.kind == "llm"
        assert cfg.players.black.connector.model == "gpt-4"
        assert cfg.players.black.name == "GPT-4"
        assert cfg.metrics.stockfish_depth == 18
        assert cfg.metrics.quality_thresholds.mistake == 250

    def test_load_app_config__when_using_budget_llm_player__then_sets_budget_defaults(
        self,
    ):
        """Budget LLM preset should hydrate connector defaults from YAML."""
        overrides = [
            "players@players.white=llm/default",
            "players.white.connector.model=gpt-4o-mini",
            "players.white.name=GPT-4o Mini",
            "players@players.black=random",
        ]

        cfg = config.load_app_config(overrides=overrides)

        default_player_cfg = OmegaConf.load(
            Path(__file__).resolve().parents[2]
            / "configs"
            / "players"
            / "llm"
            / "default.yaml"
        )
        default_connector_cfg = default_player_cfg.connector

        assert cfg.players.white.kind == "llm"
        assert cfg.players.white.name == "GPT-4o Mini"
        assert cfg.players.white.max_move_retries == 3
        assert cfg.players.white.num_votes == 1
        assert cfg.players.white.connector.model == "gpt-4o-mini"
        assert (
            cfg.players.white.connector.temperature == default_connector_cfg.temperature
        )
        # max_tokens should be resolved from fractional (0.8) to actual tokens
        from llm_chess_arena.config import BASE_MODEL_OUTPUT_TOKEN_LIMITS

        gpt_4o_mini_limit = BASE_MODEL_OUTPUT_TOKEN_LIMITS["gpt-4o-mini"]
        expected_max_tokens = int(gpt_4o_mini_limit * 0.8)
        assert cfg.players.white.connector.max_tokens == expected_max_tokens
        assert cfg.players.white.connector.timeout == default_connector_cfg.timeout
        assert (
            cfg.players.white.connector.max_retries == default_connector_cfg.max_retries
        )

    def test_load_app_config__when_using_stockfish_elo_profile__then_sets_engine_options(
        self,
    ):
        """Stockfish profile overrides should inject strength-specific options."""
        overrides = [
            "players@players.white=stockfish/elo_1320",
            "players@players.black=stockfish/elo_2800",
        ]
        cfg = config.load_app_config(overrides=overrides)

        assert cfg.players.white.engine_options == {
            "UCI_LimitStrength": True,
            "UCI_Elo": 1320,
        }
        assert cfg.players.black.engine_options == {
            "UCI_LimitStrength": True,
            "UCI_Elo": 2800,
        }

    def test_app_config_from_dictconfig__when_given_raw_dict__then_returns_dataclasses(
        self,
    ):
        """Manual DictConfig conversion should yield fully typed AppConfig objects."""
        dict_cfg = OmegaConf.create(
            {
                "env": {
                    "load_dotenv": False,
                    "log_level": "DEBUG",
                },
                "game": {
                    "display_board": True,
                    "enable_metrics": False,
                    "max_num_moves": 10,
                },
                "metrics": {
                    "stockfish_depth": 12,
                    "stockfish_binary_path": "/tmp/stockfish",
                    "stockfish_engine_options": {"Threads": 4},
                    "quality_thresholds": {
                        "excellent": 45,
                        "good": 90,
                        "inaccuracy": 180,
                        "mistake": 260,
                    },
                },
                "players": {
                    "white": {
                        "kind": "random",
                        "color": "white",
                        "name": "White",
                        "seed": 1,
                    },
                    "black": {
                        "kind": "random",
                        "color": "black",
                        "name": "Black",
                        "seed": 2,
                    },
                },
            }
        )

        app_cfg = config.app_config_from_dictconfig(dict_cfg)

        assert app_cfg.env.log_level == "DEBUG"
        assert app_cfg.game.display_board is True
        assert app_cfg.metrics.stockfish_depth == 12
        assert app_cfg.metrics.stockfish_binary_path == "/tmp/stockfish"
        assert app_cfg.metrics.stockfish_engine_options == {"Threads": 4}
        assert app_cfg.metrics.quality_thresholds.excellent == 45
        assert app_cfg.players.white.kind == "random"
        assert app_cfg.players.black.seed == 2

    @patch("llm_chess_arena.config.logger")
    def test_load_env_logs_appropriately(self, mock_logger, tmp_path):
        """Test that load_env logs debug messages appropriately."""
        env_file = tmp_path / ".env"
        env_file.write_text("TEST=value")

        # Reset state
        config._ENV_LOADED = False

        # Load existing file
        config.load_env(str(env_file))
        mock_logger.debug.assert_called_with(
            "Loaded environment from: {}", str(env_file)
        )

        # Reset mock
        mock_logger.reset_mock()
        config._ENV_LOADED = False

        # Try loading non-existent file
        config.load_env("nonexistent.env")
        mock_logger.debug.assert_called_with(
            "No environment configuration file found: {}", "nonexistent.env"
        )


class TestEnsureColor:
    """Tests for _ensure_color function immutability behavior."""

    def test_ensure_color__when_called__then_returns_new_instance_with_color_set(self):
        """Test that _ensure_color returns a new config instance with the color set."""
        # Create original config with existing default color
        original_config = RandomPlayerConfig(
            kind="random",
            name="Test Player",
            seed=42,
        )

        # Original config has default color "white"
        assert original_config.color == "white"

        # Apply different color using _ensure_color
        new_config = _ensure_color(original_config, "black")

        # Verify original config remains unchanged (immutability)
        assert original_config.color == "white"

        # Verify new config has the new color set
        assert new_config.color == "black"

        # Verify other attributes are preserved
        assert new_config.kind == "random"
        assert new_config.name == "Test Player"
        assert new_config.seed == 42

        # Verify they are different objects
        assert new_config is not original_config

    def test_ensure_color__when_config_already_has_color__then_overrides_with_fallback(
        self,
    ):
        """Test that _ensure_color overrides existing color with fallback."""
        # Create config with existing color
        original_config = RandomPlayerConfig(
            kind="random",
            name="Test Player",
            color="black",
            seed=42,
        )

        # Apply different color using _ensure_color
        new_config = _ensure_color(original_config, "white")

        # Verify original config remains unchanged
        assert original_config.color == "black"

        # Verify new config has the fallback color
        assert new_config.color == "white"

        # Verify they are different objects
        assert new_config is not original_config
