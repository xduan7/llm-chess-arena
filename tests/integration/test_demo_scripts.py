"""Integration tests for demo script configurations.

These tests verify that the demo scripts can successfully compose their
Hydra configurations without stubbing out the actual game components.
This ensures the demo configurations work end-to-end.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock
import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from llm_chess_arena.config.schema import AppConfig
from llm_chess_arena.config.loader import (
    app_config_from_dictconfig,
    run_game_from_config,
)


class TestDemoScriptConfigurations:
    """Test that demo script Hydra overrides compose successfully."""

    def setup_method(self) -> None:
        """Set up Hydra for each test."""
        GlobalHydra.instance().clear()
        # This imports and registers the config schemas with Hydra
        from llm_chess_arena.cli.main import run_tournament_cli  # noqa: F401

    def teardown_method(self) -> None:
        """Clean up Hydra after each test."""
        GlobalHydra.instance().clear()

    def test_random_vs_random_demo_config_composes(self) -> None:
        """Verify the random vs random demo configuration composes correctly."""
        with initialize(config_path="../../configs", version_base=None):
            # Simulate the demo/run_random_game.sh overrides
            cfg = compose(
                config_name="config",
                overrides=[
                    "game.display_board=false",
                    "game.enable_metrics=false",
                    "game.max_num_moves=100",
                    "players@players.white=random",
                    "players@players.black=random",
                ],
            )

        # Convert to AppConfig
        app_config = app_config_from_dictconfig(cfg)
        assert isinstance(app_config, AppConfig)
        assert app_config.game.display_board is False
        assert app_config.game.enable_metrics is False
        assert app_config.game.max_num_moves == 100
        assert app_config.players.white.kind == "random"
        assert app_config.players.black.kind == "random"

        # Verify the configuration can be used to create actual game components
        # Mock the game.play() method to avoid actually running a game
        with patch("llm_chess_arena.game.Game.play"):
            game = run_game_from_config(app_config)

        # Verify game was created successfully
        assert game is not None
        assert hasattr(game, "white_player")
        assert hasattr(game, "black_player")
        assert game.white_player.name.startswith("Random")
        assert game.black_player.name.startswith("Random")

    def test_stockfish_vs_random_demo_config_composes(self) -> None:
        """Verify the Stockfish vs random demo configuration composes correctly."""
        with initialize(config_path="../../configs", version_base=None):
            # Simulate the demo/run_stockfish_game.sh overrides
            cfg = compose(
                config_name="config",
                overrides=[
                    "game.display_board=false",
                    "game.enable_metrics=false",
                    "game.max_num_moves=100",
                    "players@players.white=stockfish/elo_1600",
                    "players@players.black=random",
                ],
            )

        # Convert to AppConfig
        app_config = app_config_from_dictconfig(cfg)
        assert isinstance(app_config, AppConfig)
        assert app_config.players.white.kind == "stockfish"
        assert app_config.players.black.kind == "random"

        # Verify the configuration can be used to create actual game components
        # Mock the game.play() method to avoid actually running a game
        with patch("llm_chess_arena.game.Game.play"):
            game = run_game_from_config(app_config)

        # Verify game was created successfully
        assert game is not None
        assert "Stockfish" in game.white_player.name
        assert game.black_player.name.startswith("Random")

    def test_llm_vs_random_demo_config_composes(self) -> None:
        """Verify the LLM vs random demo configuration composes correctly."""
        with initialize(config_path="../../configs", version_base=None):
            # Simulate the demo/run_llm_game.sh overrides
            cfg = compose(
                config_name="config",
                overrides=[
                    "game.enable_metrics=false",
                    "players@players.white=llm/default",
                    "players.white.connector.model=gpt-4o-mini",
                    'players.white.name="GPT-4o Mini"',
                    "players@players.black=random",
                ],
            )

        # Convert to AppConfig
        app_config = app_config_from_dictconfig(cfg)
        assert isinstance(app_config, AppConfig)
        assert app_config.players.white.kind == "llm"
        assert app_config.players.white.connector.model == "gpt-4o-mini"
        assert app_config.players.white.name == "GPT-4o Mini"
        assert app_config.players.black.kind == "random"

        # Verify the configuration can be used to create actual game components
        # Mock LLM calls and game.play() method to avoid actual API calls and gameplay
        mock_connector = MagicMock()
        mock_connector.model = "gpt-4o-mini"
        mock_connector.query.return_value = ["e2e4"]
        mock_connector.get_last_usage.return_value = None
        mock_connector.get_total_usage.return_value = MagicMock(
            prompt_tokens=0, completion_tokens=0, total_tokens=0, cost=0.0
        )

        with (
            patch(
                "llm_chess_arena.factory.player_factory.LLMConnector",
                return_value=mock_connector,
            ),
            patch("llm_chess_arena.game.Game.play"),
        ):
            game = run_game_from_config(app_config)

        # Verify game was created successfully
        assert game is not None
        assert "GPT-4o Mini" in game.white_player.name
        assert game.black_player.name.startswith("Random")

    def test_llm_config_includes_gpt_4o_mini_token_limits(self) -> None:
        """Verify that gpt-4o-mini has proper token limits configured."""
        # This test specifically addresses the bug where gpt-4o-mini was missing
        with initialize(config_path="../../configs", version_base=None):
            cfg = compose(
                config_name="config",
                overrides=[
                    "players@players.white=llm/default",
                    "players.white.connector.model=gpt-4o-mini",
                ],
            )

        # Verify the configuration resolves without errors
        app_config = app_config_from_dictconfig(cfg)

        # The LLM player should have proper token limits set
        assert app_config.players.white.connector.max_num_tokens is not None
        assert app_config.players.white.connector.max_num_tokens > 0

    @pytest.mark.parametrize(
        "model_name,expected_limit",
        [
            ("gpt-4o", 16_384),
            ("gpt-4o-mini", 16_384),
            ("gpt-4.1-mini", 16_384),
            ("gpt-5", 128_000),
        ],
    )
    def test_model_token_limits_are_defined(
        self, model_name: str, expected_limit: int
    ) -> None:
        """Verify common models have defined token limits."""
        from llm_chess_arena.config.schema import BASE_MODEL_OUTPUT_TOKEN_LIMITS

        assert model_name in BASE_MODEL_OUTPUT_TOKEN_LIMITS
        assert BASE_MODEL_OUTPUT_TOKEN_LIMITS[model_name] == expected_limit
