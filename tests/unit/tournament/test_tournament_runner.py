"""Unit tests for tournament runner."""

from datetime import datetime, UTC
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from dataclasses import replace

import pytest

from llm_chess_arena.tournament.executor import (
    TournamentRunner,
    _generate_game_schedule,
)
from llm_chess_arena.tournament.aggregator import aggregate_tournament_results
from llm_chess_arena.tournament.types import TournamentConfig, GameResult
from llm_chess_arena.config import (
    PlayerConfig,
    RandomPlayerConfig,
    GameConfig,
    MetricsConfig,
)


@pytest.fixture
def tournament_config() -> TournamentConfig:
    """Create a basic tournament config."""
    return TournamentConfig(
        match_name="test_match",
        num_games=4,
        parallel_games=1,
        rate_limit_rpm=None,  # No rate limiting in tests
        alternate_colors=True,
        display_summary=False,
        output_dir=Path("test_output"),
    )


@pytest.fixture
def game_config() -> GameConfig:
    """Create a basic game config."""
    return GameConfig(
        display_board=False,
        display_summary=False,
        enable_metrics=False,
        max_num_moves=100,
    )


@pytest.fixture
def metrics_config() -> MetricsConfig:
    """Create a basic metrics config."""
    from llm_chess_arena.config.schema import MoveQualityThresholdsConfig

    return MetricsConfig(
        max_centipawn_loss_per_move=1000,
        stockfish_depth=10,
        stockfish_engine_options={},
        quality_thresholds=MoveQualityThresholdsConfig(
            excellent=25.0,
            good=50.0,
            inaccuracy=100.0,
            mistake=300.0,
        ),
        stockfish_binary_path="stockfish",
    )


@pytest.fixture
def white_player_config() -> RandomPlayerConfig:
    """Create white player config."""
    return RandomPlayerConfig(
        name="White Player",
        color="white",
    )


@pytest.fixture
def black_player_config() -> RandomPlayerConfig:
    """Create black player config."""
    return RandomPlayerConfig(
        name="Black Player",
        color="black",
    )


class TestTournamentRunner:
    """Test tournament runner core logic."""

    def test_init__given_valid_config__then_creates_runner(
        self,
        tournament_config: TournamentConfig,
        game_config: GameConfig,
        metrics_config: MetricsConfig,
        white_player_config: PlayerConfig,
        black_player_config: PlayerConfig,
    ) -> None:
        """Test successful runner initialization."""
        runner = TournamentRunner(
            tournament_config=tournament_config,
            game_config=game_config,
            metrics_config=metrics_config,
            white_player_config=white_player_config,
            black_player_config=black_player_config,
        )

        assert runner.tournament_config == tournament_config
        assert runner.game_config == game_config
        assert runner.rate_limiter is None  # No rate limit configured

    def test_init__given_rate_limit__then_creates_limiter(
        self,
        tournament_config: TournamentConfig,
        game_config: GameConfig,
        metrics_config: MetricsConfig,
        white_player_config: PlayerConfig,
        black_player_config: PlayerConfig,
    ) -> None:
        """Test rate limiter creation when configured."""
        tournament_config = replace(tournament_config, rate_limit_rpm=60)

        runner = TournamentRunner(
            tournament_config=tournament_config,
            game_config=game_config,
            metrics_config=metrics_config,
            white_player_config=white_player_config,
            black_player_config=black_player_config,
        )

        assert runner.rate_limiter is not None
        assert runner.rate_limiter.rpm == 60.0

    def test_generate_schedule__given_alternation__then_swaps_colors(
        self,
        tournament_config: TournamentConfig,
        white_player_config: PlayerConfig,
        black_player_config: PlayerConfig,
    ) -> None:
        """Test schedule generation with color alternation."""
        schedule = _generate_game_schedule(
            tournament_config,
            white_player_config,
            black_player_config,
        )

        # 4 games with alternation: first 2 original, last 2 swapped
        assert len(schedule) == 4
        assert schedule[0] == (white_player_config, black_player_config)
        assert schedule[1] == (white_player_config, black_player_config)
        assert schedule[2] == (black_player_config, white_player_config)
        assert schedule[3] == (black_player_config, white_player_config)

    def test_generate_schedule__given_no_alternation__then_same_colors(
        self,
        tournament_config: TournamentConfig,
        white_player_config: PlayerConfig,
        black_player_config: PlayerConfig,
    ) -> None:
        """Test schedule generation without color alternation."""
        tournament_config = replace(tournament_config, alternate_colors=False)

        schedule = _generate_game_schedule(
            tournament_config,
            white_player_config,
            black_player_config,
        )

        # All games have same color assignment
        assert len(schedule) == 4
        assert all(s == (white_player_config, black_player_config) for s in schedule)

    def test_aggregate_results__given_games__then_calculates_stats(
        self,
        tournament_config: TournamentConfig,
    ) -> None:
        """Test results aggregation with sample games."""
        # Create sample game results
        games = [
            GameResult(
                game_id=1,
                white_player_name="White Player",
                black_player_name="Black Player",
                result="1-0",  # White wins
                total_moves=40,
                termination_reason="checkmate",
                white_centipawn_loss=50.0,
                black_centipawn_loss=100.0,
                white_thinking_time=10.0,
                black_thinking_time=15.0,
                white_cost=0.01,
                black_cost=0.02,
            ),
            GameResult(
                game_id=2,
                white_player_name="White Player",
                black_player_name="Black Player",
                result="0-1",  # Black wins
                total_moves=50,
                termination_reason="checkmate",
                white_centipawn_loss=120.0,
                black_centipawn_loss=60.0,
                white_thinking_time=12.0,
                black_thinking_time=18.0,
                white_cost=0.015,
                black_cost=0.025,
            ),
            GameResult(
                game_id=3,
                white_player_name="White Player",
                black_player_name="Black Player",
                result="1/2-1/2",  # Draw
                total_moves=30,
                termination_reason="stalemate",
                white_centipawn_loss=80.0,
                black_centipawn_loss=90.0,
                white_thinking_time=8.0,
                black_thinking_time=10.0,
                white_cost=0.01,
                black_cost=0.015,
            ),
        ]

        start_time = datetime.now(UTC)
        result = aggregate_tournament_results(
            match_name=tournament_config.match_name,
            results=games,
            start_time=start_time,
            player1_name="White Player",
            player2_name="Black Player",
        )

        # Verify basic counts
        assert result.total_games == 3
        assert result.player1_name == "White Player"
        assert result.player2_name == "Black Player"

        # White player: 1 win, 1 loss, 1 draw (as white)
        assert result.player1_wins == 1
        assert result.player2_wins == 1
        assert result.draws == 1

        # Verify averages
        assert result.total_cost == pytest.approx(0.1, abs=0.01)
        assert result.avg_game_length == pytest.approx(40.0)  # (40+50+30)/3

        # CP loss averages
        assert result.player1_avg_centipawn_loss == pytest.approx((50 + 120 + 80) / 3)
        assert result.player2_avg_centipawn_loss == pytest.approx((100 + 60 + 90) / 3)

        # Thinking time averages
        assert result.player1_avg_thinking_time == pytest.approx((10 + 12 + 8) / 3)
        assert result.player2_avg_thinking_time == pytest.approx((15 + 18 + 10) / 3)

    def test_aggregate_results__given_color_swapping__then_tracks_by_player(
        self,
        tournament_config: TournamentConfig,
    ) -> None:
        """Test aggregation correctly handles color swapping."""
        # Simulate color alternation: same players swap colors
        games = [
            GameResult(
                game_id=1,
                white_player_name="Player A",
                black_player_name="Player B",
                result="1-0",  # Player A wins as white
                total_moves=40,
                termination_reason="checkmate",
                white_cost=0.01,
                black_cost=0.02,
            ),
            GameResult(
                game_id=2,
                white_player_name="Player B",
                black_player_name="Player A",
                result="1-0",  # Player B wins as white (Player A loses)
                total_moves=45,
                termination_reason="checkmate",
                white_cost=0.015,
                black_cost=0.025,
            ),
        ]

        start_time = datetime.now(UTC)
        result = aggregate_tournament_results(
            match_name=tournament_config.match_name,
            results=games,
            start_time=start_time,
            player1_name="Player A",
            player2_name="Player B",
        )

        # Player A: 1 win, 1 loss
        # Player B: 1 win, 1 loss
        assert result.player1_name == "Player A"
        assert result.player2_name == "Player B"
        assert result.player1_wins == 1
        assert result.player2_wins == 1
        assert result.draws == 0

    @patch("llm_chess_arena.tournament.executor.PlayerFactory")
    @patch("llm_chess_arena.tournament.executor.MetricsFactory")
    @patch("llm_chess_arena.tournament.executor.Game")
    @patch("llm_chess_arena.tournament.executor.build_game_summary")
    def test_run_single_game__given_valid_config__then_creates_game(
        self,
        mock_build_summary: Mock,
        mock_game_class: Mock,
        mock_metrics_factory: Mock,
        mock_player_factory: Mock,
        tournament_config: TournamentConfig,
        game_config: GameConfig,
        metrics_config: MetricsConfig,
        white_player_config: PlayerConfig,
        black_player_config: PlayerConfig,
    ) -> None:
        """Test single game execution."""
        # Setup mocks
        mock_white_player = MagicMock()
        mock_white_player.name = "White"
        mock_black_player = MagicMock()
        mock_black_player.name = "Black"
        mock_player_factory.create_player.side_effect = [
            mock_white_player,
            mock_black_player,
        ]

        mock_game = MagicMock()
        mock_game_class.return_value = mock_game
        mock_game.metrics_tracker = None

        # Mock game summary
        mock_summary = MagicMock()
        mock_summary.result = "1-0"
        mock_summary.total_moves = 40
        mock_summary.termination = "checkmate"
        mock_summary.white_player = MagicMock(cost=0.01, thinking_time_in_seconds=10.0)
        mock_summary.black_player = MagicMock(cost=0.02, thinking_time_in_seconds=15.0)
        mock_build_summary.return_value = mock_summary

        runner = TournamentRunner(
            tournament_config=tournament_config,
            game_config=game_config,
            metrics_config=metrics_config,
            white_player_config=white_player_config,
            black_player_config=black_player_config,
        )

        result = runner._run_single_game(1, white_player_config, black_player_config)

        # Verify result
        assert result.game_id == 1
        assert result.white_player_name == "White"
        assert result.black_player_name == "Black"
        assert result.result == "1-0"
        assert result.total_moves == 40
        assert result.termination_reason == "checkmate"

        # Verify game was played
        mock_game.play.assert_called_once_with(max_num_moves=100)

    @patch("llm_chess_arena.tournament.executor.PlayerFactory")
    @patch("llm_chess_arena.tournament.executor.MetricsFactory")
    @patch("llm_chess_arena.tournament.executor.Game")
    @patch("llm_chess_arena.tournament.executor.build_game_summary")
    def test_run_parallel__given_game_failure__then_creates_failed_result(
        self,
        mock_build_summary: Mock,
        mock_game_class: Mock,
        mock_metrics_factory: Mock,
        mock_player_factory: Mock,
        tournament_config: TournamentConfig,
        game_config: GameConfig,
        metrics_config: MetricsConfig,
        white_player_config: PlayerConfig,
        black_player_config: PlayerConfig,
    ) -> None:
        """Test parallel execution handles game failures gracefully."""
        # Make _run_single_game raise an exception
        runner = TournamentRunner(
            tournament_config=tournament_config,
            game_config=game_config,
            metrics_config=metrics_config,
            white_player_config=white_player_config,
            black_player_config=black_player_config,
        )

        with patch.object(
            runner,
            "_run_single_game",
            side_effect=Exception("Simulated failure"),
        ):
            schedule = [(white_player_config, black_player_config)]
            results = runner._run_parallel(schedule)

            # Should create failed result instead of crashing
            assert len(results) == 1
            assert results[0].result == "*"
            assert "Error:" in results[0].termination_reason
            assert results[0].total_moves == 0
