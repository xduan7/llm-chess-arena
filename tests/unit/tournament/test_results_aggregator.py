"""Unit tests for tournament results aggregation and export."""

import json
import csv
from datetime import datetime, UTC
from pathlib import Path

import pytest

from llm_chess_arena.tournament.aggregator import aggregate_tournament_results
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
            white_thinking_time_in_sec=10.0,
            black_thinking_time_in_sec=15.0,
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
            white_thinking_time_in_sec=12.0,
            black_thinking_time_in_sec=18.0,
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
        player1_total_thinking_time_in_sec=28.0,
        player2_total_thinking_time_in_sec=27.0,
        player1_avg_thinking_time_per_game_in_sec=14.0,
        player2_avg_thinking_time_per_game_in_sec=13.5,
        player1_avg_thinking_time_per_move_in_sec=0.5,
        player2_avg_thinking_time_per_move_in_sec=0.48,
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

        assert output_file.exists()

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

        assert output_file.exists()

        with open(output_file, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2

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
                    white_thinking_time_in_sec=5.0,
                    black_thinking_time_in_sec=6.0,
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

        assert row["white_centipawn_loss"] == ""
        assert row["black_centipawn_loss"] == ""

    def test_export_csv__given_numeric_values__then_preserves_types(
        self, tmp_path: Path
    ) -> None:
        """Test CSV export preserves numeric types for downstream analysis."""
        result = TournamentResult(
            match_name="numeric_test",
            player1_name="P1",
            player2_name="P2",
            start_time=datetime.now(UTC),
            games=[
                GameResult(
                    game_id=1,
                    white_player_name="P1",
                    black_player_name="P2",
                    result="1-0",
                    total_moves=45,
                    termination_reason="checkmate",
                    white_centipawn_loss=50.5,
                    black_centipawn_loss=120.75,
                    white_thinking_time_in_sec=10.5,
                    black_thinking_time_in_sec=15.25,
                    white_cost=0.012345,
                    black_cost=0.023456,
                    timestamp=datetime.now(UTC),
                )
            ],
        )

        output_file = tmp_path / "numeric.csv"
        ResultsExporter.export_csv(result, output_file)

        with open(output_file, newline="") as f:
            reader = csv.DictReader(f)
            row = next(reader)

        # CSV stores everything as strings, but they should be parseable as floats
        assert float(row["white_centipawn_loss"]) == 50.5
        assert float(row["black_centipawn_loss"]) == 120.75
        assert float(row["white_thinking_time_in_sec"]) == 10.5
        assert float(row["black_thinking_time_in_sec"]) == 15.25
        assert float(row["white_cost"]) == 0.012345
        assert float(row["black_cost"]) == 0.023456


class TestTournamentAggregator:
    """Test aggregator handling of failed games and edge cases."""

    def test_aggregate__given_failed_game__then_excludes_from_win_loss_stats(
        self,
    ) -> None:
        """Test that failed games (result='*') don't inflate draw counts."""
        start_time = datetime.now(UTC)
        games = [
            GameResult(
                game_id=1,
                white_player_name="Player A",
                black_player_name="Player B",
                result="1-0",
                total_moves=40,
                termination_reason="checkmate",
                timestamp=start_time,
            ),
            GameResult(
                game_id=2,
                white_player_name="Player A",
                black_player_name="Player B",
                result="*",  # Failed game
                total_moves=0,
                termination_reason="Error: Connection timeout",
                timestamp=start_time,
            ),
            GameResult(
                game_id=3,
                white_player_name="Player B",
                black_player_name="Player A",
                result="0-1",
                total_moves=50,
                termination_reason="checkmate",
                timestamp=start_time,
            ),
        ]

        result = aggregate_tournament_results(
            match_name="test",
            results=games,
            start_time=start_time,
            player1_name="Player A",
            player2_name="Player B",
        )

        # Total games includes failed game
        assert result.total_games == 3

        # Win/loss counts exclude failed game
        assert (
            result.player1_wins == 2
        )  # Player A won game 1 (as white) and game 3 (as black)
        assert result.player2_wins == 0  # Player B lost both completed games
        assert result.draws == 0  # Failed game should NOT be counted as draw

    def test_aggregate__given_only_failed_games__then_zero_wins_draws(self) -> None:
        """Test aggregator with only failed games."""
        start_time = datetime.now(UTC)
        games = [
            GameResult(
                game_id=1,
                white_player_name="Player A",
                black_player_name="Player B",
                result="*",
                total_moves=0,
                termination_reason="Error: API failure",
                timestamp=start_time,
            ),
            GameResult(
                game_id=2,
                white_player_name="Player A",
                black_player_name="Player B",
                result="*",
                total_moves=0,
                termination_reason="Error: Timeout",
                timestamp=start_time,
            ),
        ]

        result = aggregate_tournament_results(
            match_name="test",
            results=games,
            start_time=start_time,
            player1_name="Player A",
            player2_name="Player B",
        )

        assert result.total_games == 2
        assert result.player1_wins == 0
        assert result.player2_wins == 0
        assert result.draws == 0

    def test_aggregate__given_draws__then_counts_correctly(self) -> None:
        """Test that real draws (1/2-1/2) are counted correctly."""
        start_time = datetime.now(UTC)
        games = [
            GameResult(
                game_id=1,
                white_player_name="Player A",
                black_player_name="Player B",
                result="1/2-1/2",
                total_moves=60,
                termination_reason="stalemate",
                timestamp=start_time,
            ),
            GameResult(
                game_id=2,
                white_player_name="Player A",
                black_player_name="Player B",
                result="1/2-1/2",
                total_moves=75,
                termination_reason="fifty-move rule",
                timestamp=start_time,
            ),
        ]

        result = aggregate_tournament_results(
            match_name="test",
            results=games,
            start_time=start_time,
            player1_name="Player A",
            player2_name="Player B",
        )

        assert result.total_games == 2
        assert result.player1_wins == 0
        assert result.player2_wins == 0
        assert result.draws == 2  # Both games are real draws
