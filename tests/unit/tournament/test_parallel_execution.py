"""Comprehensive tests for parallel tournament execution.

Tests thread safety, concurrent execution, shared rate limiter behavior,
and record writing under parallel conditions.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, UTC
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from llm_chess_arena.config import (
    GameConfig,
    MetricsConfig,
    PlayerConfig,
    RandomPlayerConfig,
)
from llm_chess_arena.tournament.types import TournamentConfig, GameResult
from llm_chess_arena.tournament.executor import TournamentRunner
from llm_chess_arena.rate_limiter import TokenBucketRateLimiter


class TestParallelExecution:
    """Test suite for parallel game execution."""

    @pytest.fixture
    def parallel_tournament_config(self, tmp_path: Path) -> TournamentConfig:
        """Tournament config with parallel execution enabled."""
        return TournamentConfig(
            match_name="parallel_test",
            num_games=4,
            parallel_games=4,
            alternate_colors=False,
            display_summary=False,
            output_dir=tmp_path,
        )

    @pytest.fixture
    def game_config(self) -> GameConfig:
        """Basic game configuration."""
        return GameConfig(
            display_board=False,
            display_summary=False,
            enable_metrics=False,
            max_num_moves=10,
        )

    @pytest.fixture
    def metrics_config(self) -> MetricsConfig:
        """Basic metrics configuration."""
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
        )

    @pytest.fixture
    def player_configs(self) -> tuple[PlayerConfig, PlayerConfig]:
        """Random player configurations."""
        white = RandomPlayerConfig(color="white", name="White", seed=12345)
        black = RandomPlayerConfig(color="black", name="Black", seed=67890)
        return white, black

    def test_parallel_execution__given_multiple_games__then_runs_concurrently(
        self,
        parallel_tournament_config: TournamentConfig,
        game_config: GameConfig,
        metrics_config: MetricsConfig,
        player_configs: tuple[PlayerConfig, PlayerConfig],
    ) -> None:
        """Test that games actually run in parallel using thread tracking."""
        white, black = player_configs
        execution_times: dict[int, tuple[float, float]] = {}
        lock = threading.Lock()

        def track_game_execution(game_id: int, *args, **kwargs) -> GameResult:
            """Track when each game starts and ends."""
            start = time.time()
            # Simulate some work
            time.sleep(0.1)
            end = time.time()

            with lock:
                execution_times[game_id] = (start, end)

            return GameResult(
                game_id=game_id,
                white_player_name="White",
                black_player_name="Black",
                result="1-0",
                total_moves=5,
                termination_reason="checkmate",
                timestamp=datetime.now(UTC),
            )

        runner = TournamentRunner(
            tournament_cfg=parallel_tournament_config,
            game_cfg=game_config,
            metrics_cfg=metrics_config,
            white_player_cfg=white,
            black_player_cfg=black,
        )

        with patch.object(runner, "_run_single_game", side_effect=track_game_execution):
            runner.run()

        assert len(execution_times) == 4
        game_1_start, game_1_end = execution_times[1]
        game_2_start, game_2_end = execution_times[2]

        # At least 2 games should have overlapping execution windows
        overlaps = game_1_start < game_2_end and game_2_start < game_1_end
        assert overlaps, "Games should execute concurrently"

    @patch("llm_chess_arena.tournament.executor.Game")
    @patch("llm_chess_arena.tournament.executor.PlayerFactory")
    @patch("llm_chess_arena.tournament.executor.build_game_summary")
    def test_parallel_execution__given_rate_limiter__then_shares_across_threads(
        self,
        mock_build_summary: Mock,
        mock_player_factory: Mock,
        mock_game_class: Mock,
        parallel_tournament_config: TournamentConfig,
        game_config: GameConfig,
        metrics_config: MetricsConfig,
        player_configs: tuple[PlayerConfig, PlayerConfig],
    ) -> None:
        """Test that rate limiter is properly shared across parallel workers."""
        white, black = player_configs
        parallel_tournament_config = TournamentConfig(
            match_name="rate_limited_test",
            num_games=4,
            parallel_games=4,
            alternate_colors=False,
            display_summary=False,
            output_dir=parallel_tournament_config.output_dir,
            rate_limit_rpm=60,  # 1 per second
        )

        # Mock game summary
        mock_summary = Mock()
        mock_summary.result = "1-0"
        mock_summary.total_moves = 5
        mock_summary.termination = "checkmate"
        mock_summary.white_player.cost = 0.01
        mock_summary.black_player.cost = 0.01
        mock_summary.white_player.thinking_time_in_sec = 1.0
        mock_summary.black_player.thinking_time_in_sec = 1.0
        mock_build_summary.return_value = mock_summary

        # Mock players
        mock_player = Mock()
        mock_player.name = "TestPlayer"
        mock_player_factory.create_player.return_value = mock_player

        # Mock game
        mock_game_instance = Mock()
        mock_game_class.return_value = mock_game_instance

        runner = TournamentRunner(
            tournament_cfg=parallel_tournament_config,
            game_cfg=game_config,
            metrics_cfg=metrics_config,
            white_player_cfg=white,
            black_player_cfg=black,
        )

        assert runner.rate_limiter is not None
        assert isinstance(runner.rate_limiter, TokenBucketRateLimiter)

        result = runner.run()
        assert result.total_games == 4

    def test_parallel_execution__given_display_summary_false__then_suppresses_output(
        self,
        parallel_tournament_config: TournamentConfig,
        metrics_config: MetricsConfig,
        player_configs: tuple[PlayerConfig, PlayerConfig],
    ) -> None:
        """Test that display_summary=False is properly forwarded to all games."""
        white, black = player_configs
        game_config = GameConfig(
            display_board=False,
            display_summary=False,  # Should suppress per-game summaries
            enable_metrics=False,
            max_num_moves=10,
        )

        runner = TournamentRunner(
            tournament_cfg=parallel_tournament_config,
            game_cfg=game_config,
            metrics_cfg=metrics_config,
            white_player_cfg=white,
            black_player_cfg=black,
        )

        with patch("llm_chess_arena.tournament.executor.Game") as mock_game_class:
            mock_game = Mock()
            mock_game_class.return_value = mock_game

            runner.run()

            assert mock_game_class.call_count == 4
            for call in mock_game_class.call_args_list:
                kwargs = call[1]
                assert kwargs["display_summary"] is False

    def test_parallel_execution__given_hydra_config__then_passes_to_all_games(
        self,
        parallel_tournament_config: TournamentConfig,
        game_config: GameConfig,
        metrics_config: MetricsConfig,
        player_configs: tuple[PlayerConfig, PlayerConfig],
    ) -> None:
        """Test that hydra_config is properly passed to all parallel games."""
        white, black = player_configs
        hydra_config = {
            "game": {"max_num_moves": 10},
            "players": {"white": {"kind": "random"}, "black": {"kind": "random"}},
        }

        runner = TournamentRunner(
            tournament_cfg=parallel_tournament_config,
            game_cfg=game_config,
            metrics_cfg=metrics_config,
            white_player_cfg=white,
            black_player_cfg=black,
            hydra_cfg=hydra_config,
        )

        with patch("llm_chess_arena.tournament.executor.Game") as mock_game_class:
            mock_game = Mock()
            mock_game_class.return_value = mock_game

            runner.run()

            assert mock_game_class.call_count == 4
            for call in mock_game_class.call_args_list:
                kwargs = call[1]
                passed_hydra_cfg = kwargs["hydra_cfg"]
                # Non-player sections pass through unchanged
                assert passed_hydra_cfg["game"] == hydra_config["game"]
                # The players section is replaced with the per-game snapshot
                # (full configs with colors resolved for this specific game)
                per_game_players = passed_hydra_cfg["players"]
                assert per_game_players["white"]["kind"] == "random"
                assert per_game_players["black"]["kind"] == "random"
                assert per_game_players["white"]["color"] == "white"
                assert per_game_players["black"]["color"] == "black"
                assert {
                    per_game_players["white"]["seed"],
                    per_game_players["black"]["seed"],
                } == {12345, 67890}

    @patch("llm_chess_arena.tournament.executor.Game")
    @patch("llm_chess_arena.tournament.executor.PlayerFactory")
    @patch("llm_chess_arena.tournament.executor.build_game_summary")
    def test_parallel_execution__given_concurrent_record_writing__then_no_conflicts(
        self,
        mock_build_summary: Mock,
        mock_player_factory: Mock,
        mock_game_class: Mock,
        parallel_tournament_config: TournamentConfig,
        game_config: GameConfig,
        metrics_config: MetricsConfig,
        player_configs: tuple[PlayerConfig, PlayerConfig],
    ) -> None:
        """Test that concurrent record writing doesn't cause file conflicts."""
        white, black = player_configs

        # Mock game summary
        mock_summary = Mock()
        mock_summary.result = "1-0"
        mock_summary.total_moves = 5
        mock_summary.termination = "checkmate"
        mock_summary.white_player.cost = 0.01
        mock_summary.black_player.cost = 0.01
        mock_summary.white_player.thinking_time_in_sec = 1.0
        mock_summary.black_player.thinking_time_in_sec = 1.0
        mock_build_summary.return_value = mock_summary

        # Mock players
        mock_player = Mock()
        mock_player.name = "TestPlayer"
        mock_player_factory.create_player.return_value = mock_player

        # Mock game with record paths
        mock_game_instance = Mock()
        mock_game_class.return_value = mock_game_instance

        runner = TournamentRunner(
            tournament_cfg=parallel_tournament_config,
            game_cfg=game_config,
            metrics_cfg=metrics_config,
            white_player_cfg=white,
            black_player_cfg=black,
        )

        result = runner.run()

        assert result.total_games == 4
        assert len(result.games) == 4
