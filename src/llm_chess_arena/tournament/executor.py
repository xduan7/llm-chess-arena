"""Tournament executor for running multiple chess games."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, UTC
from typing import TYPE_CHECKING, Any

from loguru import logger

from llm_chess_arena.factory.player_factory import PlayerFactory
from llm_chess_arena.factory.metrics_factory import MetricsFactory
from llm_chess_arena.game import Game
from llm_chess_arena.tournament.types import (
    TournamentConfig,
    GameResult,
    TournamentResult,
)
from llm_chess_arena.tournament.aggregator import aggregate_tournament_results
from llm_chess_arena.utils import build_game_summary, TokenBucketRateLimiter

if TYPE_CHECKING:
    from llm_chess_arena.config import PlayerConfig, GameConfig, MetricsConfig


def _generate_game_schedule(
    tournament_config: TournamentConfig,
    white_player_config: PlayerConfig,
    black_player_config: PlayerConfig,
) -> list[tuple[PlayerConfig, PlayerConfig]]:
    """Generate game schedule with color alternation.

    Args:
        tournament_config: Tournament settings with num_games and alternate_colors.
        white_player_config: Configuration for white player.
        black_player_config: Configuration for black player.

    Returns:
        list: List of (white_config, black_config) tuples for each game.
    """
    schedule = []
    half = tournament_config.num_games // 2

    if tournament_config.alternate_colors:
        # First half: original colors
        for _ in range(half):
            schedule.append((white_player_config, black_player_config))
        # Second half: swapped colors
        for _ in range(tournament_config.num_games - half):
            schedule.append((black_player_config, white_player_config))
    else:
        # All games with same colors
        for _ in range(tournament_config.num_games):
            schedule.append((white_player_config, black_player_config))

    return schedule


class TournamentRunner:
    """Runs multiple games with a single player matchup."""

    def __init__(
        self,
        tournament_config: TournamentConfig,
        game_config: GameConfig,
        metrics_config: MetricsConfig,
        white_player_config: PlayerConfig,
        black_player_config: PlayerConfig,
        hydra_config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize tournament runner.

        Args:
            tournament_config: Tournament settings (num_games, parallel_games, etc.)
            game_config: Game settings (metrics, max_moves, etc.)
            metrics_config: Metrics configuration for Stockfish evaluation
            white_player_config: White player configuration
            black_player_config: Black player configuration
            hydra_config: Optional Hydra configuration dict for reproducibility metadata
        """
        self.tournament_config = tournament_config
        self.game_config = game_config
        self.metrics_config = metrics_config
        self.white_player_config = white_player_config
        self.black_player_config = black_player_config
        self.hydra_config = hydra_config or {}

        # Create rate limiter if configured
        self.rate_limiter = None
        if tournament_config.rate_limit_rpm:
            rpm = tournament_config.rate_limit_rpm
            # INTENTIONAL: Divide rate limit by parallel workers for conservative throttling
            # This is research code - we prioritize simplicity and preventing server floods
            # over optimal throughput. Each worker gets 1/N of the total budget.
            # Example: 60 RPM with 4 workers = 15 RPM effective rate (shared limiter)
            rpm_per_worker = rpm / tournament_config.parallel_games
            self.rate_limiter = TokenBucketRateLimiter(rpm_per_worker)
            logger.info(
                "Rate limiter configured: {} RPM total, {} RPM per worker (conservative)",
                rpm,
                rpm_per_worker,
            )

        # Ensure output directory exists
        self.tournament_config.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Tournament '{}' initialized: {} games, {} parallel",
            tournament_config.match_name,
            tournament_config.num_games,
            tournament_config.parallel_games,
        )

    def run(self) -> TournamentResult:
        """Execute the tournament and return results.

        Returns:
            TournamentResult: Complete tournament results with all games.
        """
        start_time = datetime.now(UTC)

        logger.info("Starting tournament: {}", self.tournament_config.match_name)

        # Generate game schedule with color alternation
        schedule = _generate_game_schedule(
            self.tournament_config,
            self.white_player_config,
            self.black_player_config,
        )

        # Run games
        if self.tournament_config.parallel_games == 1:
            results = self._run_sequential(schedule)
        else:
            results = self._run_parallel(schedule)

        # Extract player names from results
        if not results:
            # No games played - use config defaults
            player1_name = self.white_player_config.name or "Player 1"
            player2_name = self.black_player_config.name or "Player 2"
        else:
            # Get player names from first game
            first_game = results[0]
            player1_name = first_game.white_player_name
            player2_name = first_game.black_player_name

        # Aggregate results
        tournament_result = aggregate_tournament_results(
            match_name=self.tournament_config.match_name,
            results=results,
            start_time=start_time,
            player1_name=player1_name,
            player2_name=player2_name,
        )

        logger.info(
            "Tournament '{}' complete: {} games in {:.1f}s, P1/D/P2: {}/{}/{}, cost: ${:.4f}",
            self.tournament_config.match_name,
            tournament_result.total_games,
            tournament_result.duration_seconds or 0,
            tournament_result.player1_wins,
            tournament_result.draws,
            tournament_result.player2_wins,
            tournament_result.total_cost,
        )

        return tournament_result

    def _run_sequential(
        self, schedule: list[tuple[PlayerConfig, PlayerConfig]]
    ) -> list[GameResult]:
        """Run games sequentially.

        Args:
            schedule: List of (white_config, black_config) tuples.

        Returns:
            list: Game results.
        """
        results = []
        for game_id, (white_cfg, black_cfg) in enumerate(schedule, start=1):
            logger.info(
                "Starting game {}/{} in '{}'",
                game_id,
                len(schedule),
                self.tournament_config.match_name,
            )

            result = self._run_single_game(game_id, white_cfg, black_cfg)
            results.append(result)

        return results

    def _run_parallel(
        self, schedule: list[tuple[PlayerConfig, PlayerConfig]]
    ) -> list[GameResult]:
        """Run games in parallel.

        Args:
            schedule: List of (white_config, black_config) tuples.

        Returns:
            list: Game results (includes failed games marked with error).
        """
        results = []

        with ThreadPoolExecutor(
            max_workers=self.tournament_config.parallel_games
        ) as executor:
            # Submit all games
            future_to_game_info = {
                executor.submit(
                    self._run_single_game,
                    game_id,
                    white_cfg,
                    black_cfg,
                ): (game_id, white_cfg, black_cfg)
                for game_id, (white_cfg, black_cfg) in enumerate(schedule, start=1)
            }

            # Collect results as they complete
            for future in as_completed(future_to_game_info):
                game_id, white_cfg, black_cfg = future_to_game_info[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(
                        "Completed game {}/{}: {}",
                        game_id,
                        len(schedule),
                        result.result,
                    )
                except Exception as e:
                    logger.error("Game {} failed: {}", game_id, e)
                    # Create failed game result instead of crashing tournament
                    failed_result = GameResult(
                        game_id=game_id,
                        white_player_name=white_cfg.name
                        or f"{white_cfg.kind.capitalize()} White",
                        black_player_name=black_cfg.name
                        or f"{black_cfg.kind.capitalize()} Black",
                        result="*",  # Unfinished game
                        total_moves=0,
                        termination_reason=f"Error: {str(e)}",
                        timestamp=datetime.now(UTC),
                    )
                    results.append(failed_result)

        # Sort by game_id to maintain order
        results.sort(key=lambda r: r.game_id)
        return results

    def _run_single_game(
        self,
        game_id: int,
        white_cfg: PlayerConfig,
        black_cfg: PlayerConfig,
    ) -> GameResult:
        """Execute a single chess game and return results.

        Args:
            game_id: Game identifier.
            white_cfg: White player config.
            black_cfg: Black player config.

        Returns:
            GameResult: Results from the completed game.
        """
        # Import here to avoid circular dependency
        from dataclasses import replace

        # Ensure configs have correct colors for this game
        # (Schedule swaps player configs for color alternation, but doesn't update their color field)
        white_cfg = replace(white_cfg, color="white")
        black_cfg = replace(black_cfg, color="black")

        # Create players for this game
        # NOTE: Players are created fresh per game for isolation and simplicity.
        # Optimization: Could pool Stockfish players to reuse processes, but lazy
        # initialization makes this overhead minimal for research use cases.
        white_player = PlayerFactory.create_player(
            white_cfg, rate_limiter=self.rate_limiter
        )
        black_player = PlayerFactory.create_player(
            black_cfg, rate_limiter=self.rate_limiter
        )

        # Set up game recording
        record_dir = (
            self.tournament_config.output_dir
            / self.tournament_config.match_name
            / f"game_{game_id:03d}"
        )
        record_name = "game"

        # Create metrics tracker if enabled
        metrics_tracker = None
        if self.game_config.enable_metrics:
            metrics_tracker = MetricsFactory.create_metrics_tracker(self.metrics_config)

        # Create and run game
        game = Game(
            white_player=white_player,
            black_player=black_player,
            display_board=self.game_config.display_board,
            display_summary=self.game_config.display_summary,
            enable_metrics=self.game_config.enable_metrics,
            metrics_tracker=metrics_tracker,
            record_dir=record_dir,
            record_name=record_name,
            hydra_config=self.hydra_config,
        )

        game.play(max_num_moves=self.game_config.max_num_moves)

        # Build game result
        game_summary = build_game_summary(game)

        game_result = GameResult(
            game_id=game_id,
            white_player_name=white_player.name,
            black_player_name=black_player.name,
            result=game_summary.result,
            total_moves=game_summary.total_moves,
            termination_reason=game_summary.termination,
            white_cost=game_summary.white_player.cost,
            black_cost=game_summary.black_player.cost,
            white_thinking_time=game_summary.white_player.thinking_time_in_seconds,
            black_thinking_time=game_summary.black_player.thinking_time_in_seconds,
            timestamp=datetime.now(UTC),
        )

        # Add metrics if available
        if game.metrics_tracker is not None:
            summary_by_color = game.metrics_tracker.summarize()
            white_summary = summary_by_color.get("white")
            black_summary = summary_by_color.get("black")

            if white_summary is not None:
                game_result.white_centipawn_loss = white_summary.average_centipawn_loss
                # Convert MoveQuality enum keys to strings for storage
                game_result.white_quality_counts = {
                    quality.value: count
                    for quality, count in white_summary.quality_counts.items()
                }
            if black_summary is not None:
                game_result.black_centipawn_loss = black_summary.average_centipawn_loss
                # Convert MoveQuality enum keys to strings for storage
                game_result.black_quality_counts = {
                    quality.value: count
                    for quality, count in black_summary.quality_counts.items()
                }

        # Record paths
        game_result.pgn_path = record_dir / f"{record_name}.pgn"
        game_result.json_path = record_dir / f"{record_name}.json"

        return game_result
