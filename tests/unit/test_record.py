"""Tests for game record collection and JSON serialization."""

import json
from datetime import datetime
from unittest.mock import patch

import chess

from llm_chess_arena.record import RecordCollector, RecordWriter, iso_timestamp
from llm_chess_arena.game import Game
from llm_chess_arena.player.random_player import RandomPlayer


class TestIsoTimestamp:
    """Test ISO timestamp formatting utility."""

    def test_iso_timestamp_format(self):
        """Verify ISO-8601 format with milliseconds and Z suffix."""
        dt = datetime(2024, 1, 15, 14, 30, 47, 456789)
        result = iso_timestamp(dt)
        assert result == "2024-01-15T14:30:47.456Z"

    def test_iso_timestamp_precision(self):
        """Verify millisecond precision is maintained."""
        dt = datetime(2024, 1, 15, 14, 30, 47, 123456)
        result = iso_timestamp(dt)
        assert result == "2024-01-15T14:30:47.123Z"

    def test_iso_timestamp_zero_microseconds(self):
        """Verify handling of zero microseconds."""
        dt = datetime(2024, 1, 15, 14, 30, 47, 0)
        result = iso_timestamp(dt)
        assert result == "2024-01-15T14:30:47.000Z"


class TestRecordCollector:
    """Test record data collection."""

    def test_init(self):
        """Test collector initialization."""
        collector = RecordCollector()
        data = collector.get_data()

        assert "moves" in data
        assert "start_timestamp" in data
        assert "end_timestamp" in data
        assert "outcome" in data
        assert data["moves"] == []
        assert data["start_timestamp"] is None
        assert data["end_timestamp"] is None
        assert data["outcome"] is None

    def test_set_timestamps(self):
        """Test timestamp setting."""
        collector = RecordCollector()
        start_time = "2024-01-15T14:30:47.456Z"
        end_time = "2024-01-15T14:35:47.789Z"

        collector.set_start_timestamp(start_time)
        collector.set_end_timestamp(end_time)

        data = collector.get_data()
        assert data["start_timestamp"] == start_time
        assert data["end_timestamp"] == end_time

    def test_add_move(self):
        """Test move data addition."""
        collector = RecordCollector()
        move_data = {
            "move_number": 1,
            "player": "white",
            "timestamp": "2024-01-15T14:30:47.456Z",
            "final_decision": {"action": "move", "move_uci": "e2e4"},
        }

        collector.add_move(move_data)

        data = collector.get_data()
        assert len(data["moves"]) == 1
        assert data["moves"][0] == move_data

    def test_set_outcome(self):
        """Test outcome setting."""
        collector = RecordCollector()
        outcome = chess.Outcome(
            termination=chess.Termination.CHECKMATE, winner=chess.WHITE
        )

        collector.set_outcome(outcome)

        data = collector.get_data()
        assert data["outcome"] == outcome


class TestRecordWriter:
    """Test record JSON serialization."""

    def test_calculate_summary_basic(self):
        """Test basic summary calculation."""
        moves = [
            {"player": "white", "thinking_time_in_sec": 2.5},
            {"player": "black", "thinking_time_in_sec": 1.8},
        ]
        outcome = chess.Outcome(
            termination=chess.Termination.CHECKMATE, winner=chess.WHITE
        )

        summary = RecordWriter._calculate_summary(moves, outcome)

        assert summary["result"] == "1-0"
        assert summary["termination"] == "checkmate"
        assert summary["total_moves"] == 2
        assert "players" in summary

    def test_calculate_summary_with_llm_data(self):
        """Test summary calculation with LLM player data."""
        moves = [
            {
                "player": "white",
                "thinking_time_in_sec": 2.5,
                "llm_decision_process": {
                    "api_calls": [
                        {
                            "response": {
                                "usage": {
                                    "prompt_tokens": 100,
                                    "completion_tokens": 50,
                                    "total_tokens": 150,
                                }
                            }
                        }
                    ]
                },
            },
        ]
        outcome = None  # Draw

        summary = RecordWriter._calculate_summary(moves, outcome)

        assert summary["result"] == "1/2-1/2"
        assert summary["termination"] == "unknown"
        assert summary["total_moves"] == 1
        assert "white" in summary["players"]
        assert summary["players"]["white"]["thinking_time_in_sec"] == 2.5
        assert summary["players"]["white"]["api_calls"] == 1
        assert summary["players"]["white"]["tokens_prompt"] == 100
        assert summary["players"]["white"]["tokens_completion"] == 50

    def test_capture_environment(self):
        """Test environment information capture."""
        env = RecordWriter._capture_environment()

        assert "python_version" in env
        assert "platform" in env
        assert "stockfish_available" in env
        assert isinstance(env["stockfish_available"], bool)

    def test_write_record(self, tmp_path):
        """Test complete record writing."""
        collector = RecordCollector()
        collector.set_start_timestamp("2024-01-15T14:30:47.456Z")
        collector.set_end_timestamp("2024-01-15T14:35:47.789Z")
        collector.add_move(
            {
                "move_number": 1,
                "player": "white",
                "final_decision": {"action": "move", "move_uci": "e2e4"},
            }
        )

        hydra_config = {
            "game": {"enable_metrics": True},
            "players": {"white": {"kind": "random"}},
        }

        output_path = tmp_path / "test_record.json"
        initial_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        RecordWriter.write(collector, hydra_config, output_path, initial_fen)

        assert output_path.exists()

        with output_path.open() as f:
            record = json.load(f)

        assert "summary" in record
        assert "environment" in record
        assert "hydra_config" in record
        assert "game_setup" in record
        assert "moves" in record
        assert "game_outcome" in record

        assert record["summary"]["total_moves"] == 1
        assert record["hydra_config"] == hydra_config
        assert len(record["moves"]) == 1
        assert record["environment"]["timestamp_start"] == "2024-01-15T14:30:47.456Z"
        assert record["environment"]["timestamp_end"] == "2024-01-15T14:35:47.789Z"


class TestGameRecordIntegration:
    """Integration test for game record generation."""

    @patch("llm_chess_arena.game.iso_timestamp")
    @patch("llm_chess_arena.record.iso_timestamp")
    def test_game_record_integration(self, mock_record_iso, mock_game_iso, tmp_path):
        """Run mocked game and verify key fields in output."""
        fixed_timestamp = "2024-01-15T14:30:47.456Z"
        mock_record_iso.return_value = fixed_timestamp
        mock_game_iso.return_value = fixed_timestamp

        white_player = RandomPlayer(name="Random White", color="white")
        black_player = RandomPlayer(name="Random Black", color="black")

        record_dir = tmp_path / "records"
        hydra_config = {
            "env": {"log_level": "INFO"},
            "game": {
                "enable_metrics": False,
                "record_dir": str(record_dir),
                "record_name": "test_game",
            },
            "players": {
                "white": {"kind": "random", "name": "Random White"},
                "black": {"kind": "random", "name": "Random Black"},
            },
        }

        game = Game(
            white_player=white_player,
            black_player=black_player,
            display_board=False,
            display_summary=False,
            enable_metrics=False,
            record_dir=record_dir,
            record_name="test_game",
            hydra_cfg=hydra_config,
        )

        game.play(max_num_moves=4)

        pgn_path = record_dir / "test_game.pgn"
        json_path = record_dir / "test_game.json"
        assert pgn_path.exists()
        assert json_path.exists()

        with json_path.open() as f:
            record = json.load(f)

        assert record["summary"]["total_moves"] == 4
        assert record["summary"]["result"] in ["1-0", "0-1", "1/2-1/2"]
        assert record["summary"]["termination"] in [
            "checkmate",
            "stalemate",
            "max_num_moves",
            "insufficient_material",
        ]

        assert "players" in record["summary"]

        assert record["hydra_config"] == hydra_config
        assert len(record["moves"]) == 4

        assert record["environment"]["timestamp_start"] == fixed_timestamp
        assert record["environment"]["timestamp_end"] == fixed_timestamp

        assert "game_outcome" in record
        assert record["game_outcome"]["total_moves"] == 4
        assert record["game_outcome"]["end_timestamp"] == fixed_timestamp

        for i, move in enumerate(record["moves"]):
            assert "move_number" in move
            assert "player" in move
            assert "timestamp" in move
            assert move["player"] in ["white", "black"]
            assert "final_decision" in move
            assert "position_after" in move
