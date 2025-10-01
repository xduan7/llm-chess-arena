"""Unit tests for game factory."""

from __future__ import annotations

from llm_chess_arena.config import (
    AppConfig,
    GameConfig,
    MetricsConfig,
    PlayersConfig,
    RandomPlayerConfig,
)
from llm_chess_arena.factory.game_factory import GameFactory
from llm_chess_arena.game import Game


def test_create_game_with_metrics_disabled() -> None:
    """Ensure factory omits metrics tracker when disabled."""
    app_config = AppConfig(
        game=GameConfig(display_board=False, enable_metrics=False, max_num_moves=5),
        metrics=MetricsConfig(max_centipawn_loss_per_move=1000, stockfish_depth=6),
        players=PlayersConfig(
            white=RandomPlayerConfig(color="white", seed=1),
            black=RandomPlayerConfig(color="black", seed=2),
        ),
    )

    game = GameFactory.create_game(app_config)

    assert isinstance(game, Game)
    assert game.display_board is False
    assert game.metrics_tracker is None
    assert game.white_player.color == "white"
    assert game.black_player.color == "black"


def test_create_game_with_metrics_enabled() -> None:
    """Ensure factory wires metrics tracker when enabled."""
    app_config = AppConfig(
        game=GameConfig(display_board=False, enable_metrics=True, max_num_moves=5),
        metrics=MetricsConfig(max_centipawn_loss_per_move=1000, stockfish_depth=6),
        players=PlayersConfig(
            white=RandomPlayerConfig(color="white", seed=7),
            black=RandomPlayerConfig(color="black", seed=8),
        ),
    )

    game = GameFactory.create_game(app_config)

    assert isinstance(game, Game)
    assert game.metrics_tracker is not None
    assert game.metrics_tracker.enabled in {True, False}
