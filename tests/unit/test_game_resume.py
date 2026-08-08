"""Unit tests for game resume functionality."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import chess
import pytest

from llm_chess_arena.exceptions import (
    GameNotResumableError,
    InvalidGameRecordError,
)
from llm_chess_arena.game import Game
from llm_chess_arena.player.random_player import RandomPlayer


class TestGameResume:
    """Test suite for Game.resume_from_file() functionality."""

    @pytest.fixture
    def temp_record_file(self, tmp_path: Path) -> Path:
        """Create a temporary resumable game record file."""
        record = {
            "game_setup": {"initial_fen": chess.STARTING_FEN},
            "moves": [
                {
                    "move_number": 1,
                    "player": "white",
                    "final_decision": {
                        "action": "move",
                        "attempted_move_in_uci": "e2e4",
                    },
                },
                {
                    "move_number": 2,
                    "player": "black",
                    "final_decision": {
                        "action": "move",
                        "attempted_move_in_uci": "e7e5",
                    },
                },
            ],
            "termination_metadata": {
                "resumable": True,
                "error_type": "TimeoutError",
                "player_color": "white",
                "error_message": "Connection timeout",
                "halfmove_index": 2,
                "fullmove_number": 2,
                "fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
            },
            "game_outcome": {"result": None},
            "environment": {"timestamp_start": "2024-01-01T00:00:00.000Z"},
        }

        file_path = tmp_path / "test_game.json"
        with file_path.open("w") as f:
            json.dump(record, f)

        return file_path

    @pytest.fixture
    def non_resumable_record_file(self, tmp_path: Path) -> Path:
        """Create a non-resumable game record file."""
        record = {
            "game_setup": {"initial_fen": chess.STARTING_FEN},
            "moves": [],
            "termination_metadata": {
                "resumable": False,
                "error_type": "InvalidMoveError",
                "player_color": "white",
            },
            "game_outcome": {"result": "0-1"},
        }

        file_path = tmp_path / "non_resumable.json"
        with file_path.open("w") as f:
            json.dump(record, f)

        return file_path

    def test_resume_from_file__missing_file__raises_file_not_found(
        self, tmp_path: Path
    ) -> None:
        """Test that missing record file raises FileNotFoundError."""
        missing_file = tmp_path / "nonexistent.json"

        with pytest.raises(FileNotFoundError, match="Game record not found"):
            Game.resume_from_file(
                record_path=missing_file,
                white_player=RandomPlayer(name="White", color="white"),
                black_player=RandomPlayer(name="Black", color="black"),
            )

    def test_resume_from_file__invalid_json__raises_invalid_record_error(
        self, tmp_path: Path
    ) -> None:
        """Test that malformed JSON raises InvalidGameRecordError."""
        bad_json = tmp_path / "bad.json"
        bad_json.write_text("not valid json {")

        with pytest.raises(InvalidGameRecordError, match="Invalid JSON"):
            Game.resume_from_file(
                record_path=bad_json,
                white_player=RandomPlayer(name="White", color="white"),
                black_player=RandomPlayer(name="Black", color="black"),
            )

    def test_resume_from_file__missing_termination_metadata__raises_error(
        self, tmp_path: Path
    ) -> None:
        """Test that missing termination_metadata raises InvalidGameRecordError."""
        incomplete_record = tmp_path / "incomplete.json"
        with incomplete_record.open("w") as f:
            json.dump({"game_setup": {"initial_fen": chess.STARTING_FEN}}, f)

        with pytest.raises(
            InvalidGameRecordError, match="missing termination_metadata"
        ):
            Game.resume_from_file(
                record_path=incomplete_record,
                white_player=RandomPlayer(name="White", color="white"),
                black_player=RandomPlayer(name="Black", color="black"),
            )

    def test_resume_from_file__not_resumable__raises_error(
        self, non_resumable_record_file: Path
    ) -> None:
        """Test that non-resumable game raises GameNotResumableError."""
        with pytest.raises(GameNotResumableError, match="not marked as resumable"):
            Game.resume_from_file(
                record_path=non_resumable_record_file,
                white_player=RandomPlayer(name="White", color="white"),
                black_player=RandomPlayer(name="Black", color="black"),
            )

    def test_resume_from_file__missing_initial_fen__raises_error(
        self, tmp_path: Path
    ) -> None:
        """Test that missing initial_fen raises InvalidGameRecordError."""
        record = {
            "game_setup": {},  # Missing initial_fen
            "moves": [],
            "termination_metadata": {"resumable": True},
        }

        bad_record = tmp_path / "no_fen.json"
        with bad_record.open("w") as f:
            json.dump(record, f)

        with pytest.raises(InvalidGameRecordError, match="missing required field"):
            Game.resume_from_file(
                record_path=bad_record,
                white_player=RandomPlayer(name="White", color="white"),
                black_player=RandomPlayer(name="Black", color="black"),
            )

    def test_resume_from_file__with_players__restores_board_state(
        self, temp_record_file: Path
    ) -> None:
        """Test successful resume with provided players."""
        white_player = RandomPlayer(name="White", color="white")
        black_player = RandomPlayer(name="Black", color="black")

        game = Game.resume_from_file(
            record_path=temp_record_file,
            white_player=white_player,
            black_player=black_player,
            display_board=False,
            display_summary=False,
        )

        # Verify board state is restored correctly
        assert len(game.board.move_stack) == 2
        # Note: en passant target square is transient and won't be preserved after replaying moves
        assert game.board.fen().startswith(
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq"
        )
        assert game.board.turn == chess.WHITE

    def test_resume_from_file__validates_player_colors(
        self, temp_record_file: Path
    ) -> None:
        """Test that player color validation works."""
        # Both players have wrong colors
        white_player = RandomPlayer(name="Wrong", color="black")  # Should be white
        black_player = RandomPlayer(name="Wrong", color="white")  # Should be black

        with pytest.raises(ValueError, match="wrong color"):
            Game.resume_from_file(
                record_path=temp_record_file,
                white_player=white_player,
                black_player=black_player,
            )

    def test_resume_from_file__preserves_move_history(
        self, temp_record_file: Path
    ) -> None:
        """Test that move history is preserved in the record collector."""
        white_player = RandomPlayer(name="White", color="white")
        black_player = RandomPlayer(name="Black", color="black")

        game = Game.resume_from_file(
            record_path=temp_record_file,
            white_player=white_player,
            black_player=black_player,
            display_board=False,
            record_dir=temp_record_file.parent,
        )

        # Verify record collector has the moves
        assert game._record_collector is not None
        moves = game._record_collector.get_data()["moves"]
        assert len(moves) == 2
        assert moves[0]["final_decision"]["attempted_move_in_uci"] == "e2e4"
        assert moves[1]["final_decision"]["attempted_move_in_uci"] == "e7e5"

    def test_resume_from_file__tracks_resumption_source(
        self, temp_record_file: Path
    ) -> None:
        """Test that resumed game tracks the original file path."""
        white_player = RandomPlayer(name="White", color="white")
        black_player = RandomPlayer(name="Black", color="black")

        game = Game.resume_from_file(
            record_path=temp_record_file,
            white_player=white_player,
            black_player=black_player,
            display_board=False,
        )

        assert game._resumed_from == temp_record_file

    def test_resume_from_file__without_players__requires_hydra_config(
        self, temp_record_file: Path
    ) -> None:
        """Test that resuming without players requires hydra_config."""
        with pytest.raises(ValueError, match="Players must be provided"):
            Game.resume_from_file(
                record_path=temp_record_file,
                white_player=None,  # No players provided
                black_player=None,
                display_board=False,
            )

    @patch("llm_chess_arena.factory.PlayerFactory.create_player")
    def test_resume_from_file__recreates_players_from_config(
        self, mock_create_player: MagicMock, tmp_path: Path
    ) -> None:
        """Test that players are recreated from hydra_config when not provided."""
        record = {
            "game_setup": {"initial_fen": chess.STARTING_FEN},
            "moves": [],
            "termination_metadata": {
                "resumable": True,
                "fen": chess.STARTING_FEN,
            },
            "game_outcome": {"result": "Unfinished"},
            "hydra_config": {
                "players": {
                    "white": {
                        "kind": "random",
                        "name": "White",
                        "color": "white",
                    },
                    "black": {
                        "kind": "random",
                        "name": "Black",
                        "color": "black",
                    },
                },
            },
        }

        record_file = tmp_path / "with_config.json"
        with record_file.open("w") as f:
            json.dump(record, f)

        # Mock player creation
        white_mock = RandomPlayer(name="White", color="white")
        black_mock = RandomPlayer(name="Black", color="black")
        mock_create_player.side_effect = [white_mock, black_mock]

        game = Game.resume_from_file(
            record_path=record_file,
            white_player=None,  # Should be created from config
            black_player=None,  # Should be created from config
            display_board=False,
        )

        # Verify players were created from config
        assert mock_create_player.call_count == 2
        assert game.white_player == white_mock
        assert game.black_player == black_mock

    def test_resume_from_file__generates_resume_specific_filename(
        self, temp_record_file: Path
    ) -> None:
        """Test that resumed games get a new filename with -resumed suffix."""
        white_player = RandomPlayer(name="White", color="white")
        black_player = RandomPlayer(name="Black", color="black")

        game = Game.resume_from_file(
            record_path=temp_record_file,
            white_player=white_player,
            black_player=black_player,
            display_board=False,
            record_dir=temp_record_file.parent,
        )

        # The record name should contain the original name plus -resumed-
        assert game._record_name is not None
        assert "test_game" in game._record_name
        assert "-resumed-" in game._record_name

    def test_resume_from_file__invalid_moves_raise_error(self, tmp_path: Path) -> None:
        """Test that invalid moves in history raise InvalidGameRecordError."""
        record = {
            "game_setup": {"initial_fen": chess.STARTING_FEN},
            "moves": [
                {
                    "move_number": 1,
                    "player": "white",
                    "final_decision": {
                        "action": "move",
                        "attempted_move_in_uci": "e2e4",
                    },
                },
                {
                    "move_number": 2,
                    "player": "black",
                    "final_decision": {
                        "action": "move",
                        "attempted_move_in_uci": "invalid_move",  # Invalid UCI
                    },
                },
            ],
            "termination_metadata": {
                "resumable": True,
                "fen": chess.STARTING_FEN,
            },
            "game_outcome": {"result": None},
        }

        record_file = tmp_path / "with_invalid.json"
        with record_file.open("w") as f:
            json.dump(record, f)

        # Should raise InvalidGameRecordError due to corrupted move history
        with pytest.raises(InvalidGameRecordError, match="invalid move.*at index 1"):
            Game.resume_from_file(
                record_path=record_file,
                white_player=RandomPlayer(name="White", color="white"),
                black_player=RandomPlayer(name="Black", color="black"),
                display_board=False,
            )

    def test_resume_from_file__preserves_original_start_timestamp(
        self, temp_record_file: Path
    ) -> None:
        """Test that original start timestamp is preserved."""
        game = Game.resume_from_file(
            record_path=temp_record_file,
            white_player=RandomPlayer(name="White", color="white"),
            black_player=RandomPlayer(name="Black", color="black"),
            display_board=False,
            record_dir=temp_record_file.parent,
        )

        # Verify original start timestamp is preserved
        assert game._record_collector is not None
        data = game._record_collector.get_data()
        assert data["start_timestamp"] == "2024-01-01T00:00:00.000Z"

    def test_resume_from_file__can_continue_playing(
        self, temp_record_file: Path
    ) -> None:
        """Test that resumed game can continue playing."""
        white_player = RandomPlayer(name="White", color="white", seed=42)
        black_player = RandomPlayer(name="Black", color="black", seed=43)

        game = Game.resume_from_file(
            record_path=temp_record_file,
            white_player=white_player,
            black_player=black_player,
            display_board=False,
        )

        # Initial state: 2 moves already played (e2e4, e7e5)
        initial_move_count = len(game.board.move_stack)
        assert initial_move_count == 2

        # Play a few more moves
        game.play(max_num_moves=5)

        # Should have played additional moves
        assert len(game.board.move_stack) > initial_move_count
