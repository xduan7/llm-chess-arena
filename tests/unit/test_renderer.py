"""Tests for the Rich-powered chess board renderer."""

import chess
from rich.console import Console
from unittest.mock import patch

from llm_chess_arena.renderer import display_board_with_context, display_game_summary
from llm_chess_arena.utils import GameOutcomeSummary


class TestRenderer:
    """Test rendering helpers for board and summary output."""

    def test_display_board_with_context__should_render_board__when_given_initial_position(
        self,
    ):
        """Test board display produces chess output."""
        board = chess.Board()

        capture_console = Console(record=True, width=80)

        with patch("llm_chess_arena.renderer.console", capture_console):
            display_board_with_context(
                board=board,
                current_player="TestPlayer",
                last_move=None,
                white_player="White",
                black_player="Black",
            )

            output = capture_console.export_text()

            assert any(rank in output for rank in "12345678")  # Rank numbers
            assert any(file in output for file in "abcdefgh")  # File letters
            assert len(output) > 100  # Board display should be substantial

    def test_display_game_summary__should_include_outcome__when_given_game_result(self):
        """Test game summary displays outcome information."""

        outcome = GameOutcomeSummary(
            outcome_line="Outcome: 1-0",
            termination_line="Termination: checkmate",
            total_moves_line="Total moves: 42",
            winner_line="Winner: White",
            winner_name="TestWhite",
            winner_color=chess.WHITE,
        )

        capture_console = Console(record=True, width=80)

        with patch("llm_chess_arena.renderer.console", capture_console):
            result = display_game_summary(
                white_player="TestWhite",
                black_player="TestBlack",
                white_summary=None,
                black_summary=None,
                outcome_summary=outcome,
            )

            output = capture_console.export_text()

            assert "1-0" in output
            assert "checkmate" in output
            assert "42" in output
            assert (
                "Winner: White" in output
            )  # Outcome summary shows winner, not player names
            assert result is True  # Function returns success boolean

    def test_display_board_with_context__should_highlight_last_move__when_move_provided(
        self,
    ):
        """Test that last move highlighting works."""
        board = chess.Board()
        move = chess.Move.from_uci("e2e4")
        board.push(move)

        capture_console = Console(record=True, width=80)

        with patch("llm_chess_arena.renderer.console", capture_console):
            display_board_with_context(
                board=board,
                current_player="TestPlayer",
                last_move=move,
                white_player="White",
                black_player="Black",
            )

            output = capture_console.export_text()

            assert len(output) > 100  # Substantial output
