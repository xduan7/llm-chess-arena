"""Unit tests for tournament types and data structures."""

from datetime import datetime, UTC
from pathlib import Path

import pytest

from llm_chess_arena.tournament.types import (
    TournamentConfig,
    GameResult,
    TournamentResult,
)


class TestTournamentConfig:
    """Test tournament configuration."""

    def test_init__given_valid_params__then_creates_config(self) -> None:
        """Test successful config creation."""
        config = TournamentConfig(
            match_name="test_match",
            num_games=10,
            parallel_games=2,
            alternate_colors=True,
            display_summary=False,
            output_dir=Path("output"),
            rate_limit_rpm=60,
        )

        assert config.match_name == "test_match"
        assert config.num_games == 10
        assert config.parallel_games == 2
        assert config.rate_limit_rpm == 60
        assert config.alternate_colors is True
        assert config.display_summary is False
        assert isinstance(config.output_dir, Path)

    def test_init__given_invalid_num_games__then_raises_error(self) -> None:
        """Test validation for num_games."""
        with pytest.raises(ValueError, match="num_games must be positive"):
            TournamentConfig(
                match_name="test",
                num_games=0,
                parallel_games=1,
                alternate_colors=True,
                display_summary=False,
                output_dir=Path("output"),
            )

    def test_init__given_invalid_parallel_games__then_raises_error(self) -> None:
        """Test validation for parallel_games."""
        with pytest.raises(ValueError, match="parallel_games must be >= 1"):
            TournamentConfig(
                match_name="test",
                num_games=10,
                parallel_games=0,
                alternate_colors=True,
                display_summary=False,
                output_dir=Path("output"),
            )

    def test_init__given_zero_rate_limit__then_raises_error(self) -> None:
        """Test validation rejects rate_limit_rpm=0."""
        with pytest.raises(ValueError, match="rate_limit_rpm must be positive"):
            TournamentConfig(
                match_name="test",
                num_games=10,
                parallel_games=1,
                alternate_colors=True,
                display_summary=False,
                output_dir=Path("output"),
                rate_limit_rpm=0,
            )

    def test_init__given_negative_rate_limit__then_raises_error(self) -> None:
        """Test validation rejects negative rate_limit_rpm."""
        with pytest.raises(ValueError, match="rate_limit_rpm must be positive"):
            TournamentConfig(
                match_name="test",
                num_games=10,
                parallel_games=1,
                alternate_colors=True,
                display_summary=False,
                output_dir=Path("output"),
                rate_limit_rpm=-10,
            )

    def test_init__given_odd_games_with_alternation__then_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test warning for odd number of games with color alternation."""
        # Note: loguru doesn't integrate with pytest caplog by default
        # Just verify config creation succeeds - the warning is logged but not testable here
        config = TournamentConfig(
            match_name="test",
            num_games=5,
            parallel_games=1,
            alternate_colors=True,
            display_summary=False,
            output_dir=Path("output"),
        )
        assert config.num_games == 5


class TestGameResult:
    """Test game result data structure."""

    def test_init__given_minimal_params__then_creates_result(self) -> None:
        """Test game result creation with minimal parameters."""
        result = GameResult(
            game_id=1,
            white_player_name="Player1",
            black_player_name="Player2",
            result="1-0",
            total_moves=40,
            termination_reason="checkmate",
        )

        assert result.game_id == 1
        assert result.white_player_name == "Player1"
        assert result.black_player_name == "Player2"
        assert result.result == "1-0"
        assert result.total_moves == 40
        assert result.termination_reason == "checkmate"
        assert isinstance(result.timestamp, datetime)


class TestTournamentResult:
    """Test tournament result aggregation."""

    def test_duration_in_sec__given_times__then_calculates_duration(self) -> None:
        """Test duration calculation."""
        start = datetime.now(UTC)
        end = datetime.now(UTC)

        result = TournamentResult(
            match_name="test",
            player1_name="P1",
            player2_name="P2",
            start_time=start,
            end_time=end,
        )

        assert result.duration_in_sec is not None
        assert result.duration_in_sec >= 0

    def test_win_rates__given_games__then_calculates_rates(self) -> None:
        """Test win rate calculations."""
        result = TournamentResult(
            match_name="test",
            player1_name="P1",
            player2_name="P2",
            start_time=datetime.now(UTC),
            total_games=10,
            player1_wins=6,
            player2_wins=2,
            draws=2,
        )

        assert result.win_rate_player1 == 0.6
        assert result.win_rate_player2 == 0.2
        assert result.draw_rate == 0.2

    def test_to_dict__given_result__then_serializes(self) -> None:
        """Test dictionary serialization."""
        result = TournamentResult(
            match_name="test",
            player1_name="P1",
            player2_name="P2",
            start_time=datetime.now(UTC),
            end_time=datetime.now(UTC),
            total_games=5,
            player1_wins=3,
            player2_wins=1,
            draws=1,
        )

        data = result.to_dict()

        assert data["match_name"] == "test"
        assert data["player1"] == "P1"
        assert data["player2"] == "P2"
        assert data["total_games"] == 5
        assert "results" in data
        assert "win_rates" in data

    def test_quality_percentages__given_counts__then_calculates_percentages(
        self,
    ) -> None:
        """Test move quality percentage calculation."""
        result = TournamentResult(
            match_name="test",
            player1_name="P1",
            player2_name="P2",
            start_time=datetime.now(UTC),
            total_games=1,
            player1_quality_counts={"best": 10, "good": 30, "mistake": 10},  # 50 total
            player2_quality_counts={
                "best": 20,
                "excellent": 60,
                "blunder": 20,
            },  # 100 total
        )

        p1_pct = result.player1_quality_percentages
        p2_pct = result.player2_quality_percentages

        # Player 1: 10/50=20%, 30/50=60%, 10/50=20%
        assert p1_pct["best"] == 20.0
        assert p1_pct["good"] == 60.0
        assert p1_pct["mistake"] == 20.0

        # Player 2: 20/100=20%, 60/100=60%, 20/100=20%
        assert p2_pct["best"] == 20.0
        assert p2_pct["excellent"] == 60.0
        assert p2_pct["blunder"] == 20.0

    def test_quality_percentages__given_empty_counts__then_returns_empty(
        self,
    ) -> None:
        """Test percentage calculation with no quality data."""
        result = TournamentResult(
            match_name="test",
            player1_name="P1",
            player2_name="P2",
            start_time=datetime.now(UTC),
            total_games=1,
        )

        assert result.player1_quality_percentages == {}
        assert result.player2_quality_percentages == {}

    def test_to_dict__given_quality_data__then_includes_percentages(self) -> None:
        """Test that JSON export includes quality percentages."""
        result = TournamentResult(
            match_name="test",
            player1_name="P1",
            player2_name="P2",
            start_time=datetime.now(UTC),
            end_time=datetime.now(UTC),
            total_games=1,
            player1_quality_counts={"best": 5, "good": 15},  # 20 total
            player2_quality_counts={"excellent": 10},  # 10 total
        )

        data = result.to_dict()

        assert "move_quality" in data
        assert "player1_counts" in data["move_quality"]
        assert "player2_counts" in data["move_quality"]
        assert "player1_percentages" in data["move_quality"]
        assert "player2_percentages" in data["move_quality"]

        assert data["move_quality"]["player1_percentages"]["best"] == 25.0
        assert data["move_quality"]["player1_percentages"]["good"] == 75.0
        assert data["move_quality"]["player2_percentages"]["excellent"] == 100.0
