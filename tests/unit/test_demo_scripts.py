"""Unit tests for demo script Hydra configuration composition.

These tests verify that demo script configurations compose correctly and can
instantiate game components. They mock Game.play to avoid running actual games,
making them fast unit tests rather than slow integration tests.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from llm_chess_arena.config.schema import AppConfig
from llm_chess_arena.config.loader import app_cfg_from_dictconfig
from llm_chess_arena.tournament.types import TournamentConfig
from llm_chess_arena.tournament.executor import TournamentRunner


class TestDemoScriptConfigurations:
    """Test that demo script Hydra overrides compose successfully."""

    def setup_method(self) -> None:
        """Set up Hydra for each test."""
        GlobalHydra.instance().clear()
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
        app_config = app_cfg_from_dictconfig(cfg)
        assert isinstance(app_config, AppConfig)
        assert app_config.game.display_board is False
        assert app_config.game.enable_metrics is False
        assert app_config.game.max_num_moves == 100
        assert app_config.players.white.kind == "random"
        assert app_config.players.black.kind == "random"

        tournament_config = TournamentConfig(
            match_name="test_random",
            num_games=1,
            parallel_games=1,
            alternate_colors=False,  # Don't swap colors for single game
            output_dir=Path("/tmp/test_output"),
            display_summary=False,
        )
        runner = TournamentRunner(
            tournament_cfg=tournament_config,
            game_cfg=app_config.game,
            metrics_cfg=app_config.metrics,
            white_player_cfg=app_config.players.white,
            black_player_cfg=app_config.players.black,
        )

        with patch("llm_chess_arena.game.Game.play"):
            result = runner.run()

        assert result is not None
        assert result.total_games == 1
        assert len(result.games) == 1
        game_result = result.games[0]
        assert game_result.white_player_name.startswith("Random")
        assert game_result.black_player_name.startswith("Random")

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
        app_config = app_cfg_from_dictconfig(cfg)
        assert isinstance(app_config, AppConfig)
        assert app_config.players.white.kind == "stockfish"
        assert app_config.players.black.kind == "random"

        tournament_config = TournamentConfig(
            match_name="test_stockfish",
            num_games=1,
            parallel_games=1,
            alternate_colors=False,  # Don't swap colors for single game
            output_dir=Path("/tmp/test_output"),
            display_summary=False,
        )
        runner = TournamentRunner(
            tournament_cfg=tournament_config,
            game_cfg=app_config.game,
            metrics_cfg=app_config.metrics,
            white_player_cfg=app_config.players.white,
            black_player_cfg=app_config.players.black,
        )

        with patch("llm_chess_arena.game.Game.play"):
            result = runner.run()

        assert result is not None
        assert result.total_games == 1
        assert len(result.games) == 1
        game_result = result.games[0]
        assert "Stockfish" in game_result.white_player_name
        assert game_result.black_player_name.startswith("Random")

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
        app_config = app_cfg_from_dictconfig(cfg)
        assert isinstance(app_config, AppConfig)
        assert app_config.players.white.kind == "llm"
        assert app_config.players.white.connector.model == "gpt-4o-mini"
        assert app_config.players.white.name == "GPT-4o Mini"
        assert app_config.players.black.kind == "random"

        mock_connector = MagicMock()
        mock_connector.model = "gpt-4o-mini"
        mock_connector.query.return_value = ["e2e4"]
        mock_connector.get_last_usage.return_value = None
        mock_connector.get_total_usage.return_value = MagicMock(
            prompt_tokens=0, completion_tokens=0, total_tokens=0, cost=0.0
        )

        tournament_config = TournamentConfig(
            match_name="test_llm",
            num_games=1,
            parallel_games=1,
            alternate_colors=False,  # Don't swap colors for single game
            output_dir=Path("/tmp/test_output"),
            display_summary=False,
        )
        runner = TournamentRunner(
            tournament_cfg=tournament_config,
            game_cfg=app_config.game,
            metrics_cfg=app_config.metrics,
            white_player_cfg=app_config.players.white,
            black_player_cfg=app_config.players.black,
        )

        with (
            patch(
                "llm_chess_arena.factory.LLMConnector",
                return_value=mock_connector,
            ),
            patch("llm_chess_arena.game.Game.play"),
        ):
            result = runner.run()

        assert result is not None
        assert result.total_games == 1
        assert len(result.games) == 1
        game_result = result.games[0]
        assert "GPT-4o Mini" in game_result.white_player_name
        assert game_result.black_player_name.startswith("Random")

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

        app_config = app_cfg_from_dictconfig(cfg)

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
