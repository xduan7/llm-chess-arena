"""Golden master tests for deterministic gameplay baselines."""

from __future__ import annotations

from llm_chess_arena.game import Game
from llm_chess_arena.player.random_player import RandomPlayer


def test_random_game_golden_master() -> None:
    """Fixed-seed random game should retain opening sequence."""
    # Create players with fixed seeds for deterministic behavior
    white_player = RandomPlayer(name="White", color="white", seed=12345)
    black_player = RandomPlayer(name="Black", color="black", seed=67890)

    # Create and play game
    game = Game(
        white_player=white_player,
        black_player=black_player,
        display_board=False,
        display_summary=False,
        enable_metrics=False,
    )
    game.play(max_num_moves=20)

    moves = [move.uci() for move in game.board.move_stack]

    assert moves[:3] == ["g2g4", "f7f6", "g1h3"]
    assert len(moves) == 20
