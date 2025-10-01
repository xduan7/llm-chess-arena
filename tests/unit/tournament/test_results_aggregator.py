"""Unit tests for tournament results aggregation and export."""

import json
import csv
from datetime import datetime, UTC
from pathlib import Path

import pytest

from llm_chess_arena.tournament.export import ResultsExporter
from llm_chess_arena.tournament.types import TournamentResult, GameResult


@pytest.fixture
def sample_tournament_result() -> TournamentResult:
    """Create a sample tournament result for testing."""
    games = [
        GameResult(
            game_id=1,
            white_player_name="Player A",
            black_player_name="Player B",
            result="1-0",
            total_moves=40,
            termination_reason="checkmate",
            white_centipawn_loss=50.0,
            black_centipawn_loss=100.0,
            white_thinking_time=10.0,
            black_thinking_time=15.0,
            white_cost=0.01,
            black_cost=0.02,
            white_quality_counts={"best": 5, "good": 10, "inaccuracy": 3},
            black_quality_counts={"best": 3, "good": 8, "mistake": 5},
            timestamp=datetime.now(UTC),
            pgn_path=Path("/test/game1.pgn"),
            json_path=Path("/test/game1.json"),
        ),
        GameResult(
            game_id=2,
            white_player_name="Player B",
            black_player_name="Player A",
            result="0-1",
            total_moves=50,
            termination_reason="checkmate",
            white_centipawn_loss=120.0,
            black_centipawn_loss=60.0,
            white_thinking_time=12.0,
            black_thinking_time=18.0,
            white_cost=0.015,
            black_cost=0.025,
            white_quality_counts={"best": 4, "excellent": 6, "blunder": 2},
            black_quality_counts={"best": 6, "excellent": 8, "good": 10},
            timestamp=datetime.now(UTC),
            pgn_path=Path("/test/game2.pgn"),
            json_path=Path("/test/game2.json"),
        ),
    ]

    return TournamentResult(
        match_name="test_match",
        player1_name="Player A",
        player2_name="Player B",
        start_time=datetime.now(UTC),
        end_time=datetime.now(UTC),
        total_games=2,
        player1_wins=1,
        player2_wins=1,
        draws=0,
        total_cost=0.07,
        avg_game_length=45.0,
        player1_avg_centipawn_loss=80.0,
        player2_avg_centipawn_loss=80.0,
        player1_avg_thinking_time=14.0,
        player2_avg_thinking_time=13.5,
        player1_quality_counts={"best": 11, "good": 18, "inaccuracy": 3},
        player2_quality_counts={"best": 7, "excellent": 6, "mistake": 5},
        games=games,
    )


class TestResultsExporter:
    """Test results aggregator export functionality."""

    def test_export_json__given_result__then_creates_valid_json(
        self, sample_tournament_result: TournamentResult, tmp_path: Path
    ) -> None:
        """Test JSON export creates valid file."""
        output_file = tmp_path / "tournament.json"

        ResultsExporter.export_json(sample_tournament_result, output_file)

        # Verify file exists
        assert output_file.exists()

        # Verify JSON is valid and contains expected data
        with open(output_file) as f:
            data = json.load(f)

        assert data["match_name"] == "test_match"
        assert data["player1"] == "Player A"
        assert data["player2"] == "Player B"
        assert data["total_games"] == 2
        assert data["results"]["player1_wins"] == 1
        assert data["results"]["player2_wins"] == 1
        assert data["results"]["draws"] == 0
        assert data["total_cost"] == 0.07
        assert data["avg_game_length"] == 45.0

        # Verify move quality data is included
        assert "move_quality" in data
        assert "player1_counts" in data["move_quality"]
        assert "player1_percentages" in data["move_quality"]
        assert "player2_counts" in data["move_quality"]
        assert "player2_percentages" in data["move_quality"]

    def test_export_json__given_nested_dir__then_creates_directories(
        self, sample_tournament_result: TournamentResult, tmp_path: Path
    ) -> None:
        """Test JSON export creates parent directories."""
        output_file = tmp_path / "nested" / "path" / "tournament.json"

        ResultsExporter.export_json(sample_tournament_result, output_file)

        assert output_file.exists()
        assert output_file.parent.exists()

    def test_export_csv__given_result__then_creates_valid_csv(
        self, sample_tournament_result: TournamentResult, tmp_path: Path
    ) -> None:
        """Test CSV export creates valid file with game data."""
        output_file = tmp_path / "tournament.csv"

        ResultsExporter.export_csv(sample_tournament_result, output_file)

        # Verify file exists
        assert output_file.exists()

        # Verify CSV structure and content
        with open(output_file, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Should have 2 rows (one per game)
        assert len(rows) == 2

        # Check first game
        row1 = rows[0]
        assert row1["match_name"] == "test_match"
        assert row1["game_id"] == "1"
        assert row1["white_player"] == "Player A"
        assert row1["black_player"] == "Player B"
        assert row1["result"] == "1-0"
        assert row1["total_moves"] == "40"
        assert row1["termination_reason"] == "checkmate"
        assert row1["white_best"] == "5"
        assert row1["white_good"] == "10"
        assert row1["white_inaccuracy"] == "3"

        # Check second game
        row2 = rows[1]
        assert row2["game_id"] == "2"
        assert row2["white_player"] == "Player B"
        assert row2["black_player"] == "Player A"
        assert row2["result"] == "0-1"

    def test_export_csv__given_no_games__then_warns_and_skips(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test CSV export handles empty games list."""
        empty_result = TournamentResult(
            match_name="empty",
            player1_name="P1",
            player2_name="P2",
            start_time=datetime.now(UTC),
            games=[],
        )

        output_file = tmp_path / "empty.csv"

        ResultsExporter.export_csv(empty_result, output_file)

        # File should not be created
        assert not output_file.exists()

    def test_export_csv__given_nested_dir__then_creates_directories(
        self, sample_tournament_result: TournamentResult, tmp_path: Path
    ) -> None:
        """Test CSV export creates parent directories."""
        output_file = tmp_path / "nested" / "path" / "tournament.csv"

        ResultsExporter.export_csv(sample_tournament_result, output_file)

        assert output_file.exists()
        assert output_file.parent.exists()

    def test_export_csv__given_missing_metrics__then_exports_empty_strings(
        self, tmp_path: Path
    ) -> None:
        """Test CSV export handles missing optional metrics."""
        result = TournamentResult(
            match_name="partial",
            player1_name="P1",
            player2_name="P2",
            start_time=datetime.now(UTC),
            games=[
                GameResult(
                    game_id=1,
                    white_player_name="P1",
                    black_player_name="P2",
                    result="1-0",
                    total_moves=30,
                    termination_reason="resignation",
                    white_centipawn_loss=None,  # Missing
                    black_centipawn_loss=None,  # Missing
                    white_thinking_time=5.0,
                    black_thinking_time=6.0,
                    white_cost=0.01,
                    black_cost=0.02,
                    timestamp=datetime.now(UTC),
                )
            ],
        )

        output_file = tmp_path / "partial.csv"
        ResultsExporter.export_csv(result, output_file)

        with open(output_file, newline="") as f:
            reader = csv.DictReader(f)
            row = next(reader)

        # Should have empty strings for missing metrics
        assert row["white_centipawn_loss"] == ""
        assert row["black_centipawn_loss"] == ""

    def test_print_summary__given_result__then_prints_formatted_output(
        self, sample_tournament_result: TournamentResult, capsys: pytest.CaptureFixture
    ) -> None:
        """Test console summary printing."""
        ResultsExporter.print_summary(sample_tournament_result)

        captured = capsys.readouterr()
        output = captured.out

        # Verify key information is present
        assert "test_match" in output
        assert "Player A" in output
        assert "Player B" in output
        assert "Total Games: 2" in output
        assert "Total Cost: $0.0700" in output
        assert "P1=1 D=0 P2=1" in output
        assert "Avg Game Length: 45.0 moves" in output
        assert "Player 1 Avg CP Loss: 80.0" in output
        assert "Player 2 Avg CP Loss: 80.0" in output
