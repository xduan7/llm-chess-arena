"""Golden master tests for deterministic gameplay baselines."""

from __future__ import annotations

from llm_chess_arena.config import load_app_config, run_game_from_config


def test_random_game_golden_master() -> None:
    """Fixed-seed random game should retain opening sequence."""
    app_config = load_app_config(
        "config",
        [
            "players.white.seed=12345",
            "players.black.seed=67890",
            "game.max_num_moves=20",
            "game.enable_metrics=false",
            "game.display_board=false",
        ],
    )

    game = run_game_from_config(app_config)
    moves = [move.uci() for move in game.board.move_stack]

    assert moves[:3] == ["g2g4", "f7f6", "g1h3"]
    assert len(moves) == 20
