"""Factory for assembling configured game instances."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llm_chess_arena.core.policies import config_operation
from llm_chess_arena.game import Game
from llm_chess_arena.factory.metrics_factory import MetricsFactory
from llm_chess_arena.factory.player_factory import PlayerFactory

if TYPE_CHECKING:  # pragma: no cover - typing only
    from llm_chess_arena.config import AppConfig


class GameFactory:
    """Create game objects from composed application configuration."""

    @staticmethod
    @config_operation
    def create_game(app_config: "AppConfig") -> Game:
        """Create a configured game instance with players and optional metrics.

        Args:
            app_config: Application configuration containing player and game settings.

        Returns:
            Game: Configured game instance ready to play.
        """
        white_player = PlayerFactory.create_player(app_config.players.white)
        black_player = PlayerFactory.create_player(app_config.players.black)

        metrics_tracker = None
        if app_config.game.enable_metrics:
            metrics_tracker = MetricsFactory.create_metrics_tracker(app_config.metrics)

        return Game(
            white_player=white_player,
            black_player=black_player,
            display_board=app_config.game.display_board,
            enable_metrics=app_config.game.enable_metrics,
            metrics_tracker=metrics_tracker,
            history_output_path=app_config.game.history_output_path,
        )
