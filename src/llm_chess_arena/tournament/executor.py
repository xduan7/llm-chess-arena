"""Tournament executor for running multiple chess games."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime, UTC
from typing import TYPE_CHECKING, Any

from loguru import logger

from llm_chess_arena.factory import MetricsFactory, PlayerFactory
from llm_chess_arena.game import Game
from llm_chess_arena.tournament.types import (
    TournamentConfig,
    GameResult,
    TournamentResult,
)
from llm_chess_arena.tournament.aggregator import aggregate_tournament_results
from llm_chess_arena.rate_limiter import TokenBucketRateLimiter
from llm_chess_arena.summary import build_game_summary

if TYPE_CHECKING:
    from llm_chess_arena.config import PlayerConfig, GameConfig, MetricsConfig


def _generate_game_schedule(
    tournament_cfg: TournamentConfig,
    white_player_cfg: PlayerConfig,
    black_player_cfg: PlayerConfig,
) -> list[tuple[PlayerConfig, PlayerConfig]]:
    """Generate game schedule with color alternation.

    Args:
        tournament_cfg: Tournament settings with num_games and alternate_colors.
        white_player_cfg: Configuration for white player.
        black_player_cfg: Configuration for black player.

    Returns:
        List of (white_player_cfg, black_player_cfg) tuples for each game.
    """
    schedule = []
    swapped_half = tournament_cfg.num_games // 2

    if tournament_cfg.alternate_colors:
        # Configured colors play first; for odd game counts the extra game
        # keeps the configured assignment (so a 1-game run is never swapped)
        for _ in range(tournament_cfg.num_games - swapped_half):
            schedule.append((white_player_cfg, black_player_cfg))
        for _ in range(swapped_half):
            schedule.append((black_player_cfg, white_player_cfg))
    else:
        for _ in range(tournament_cfg.num_games):
            schedule.append((white_player_cfg, black_player_cfg))

    return schedule


class TournamentRunner:
    """Runs multiple games with a single player matchup."""

    def __init__(
        self,
        tournament_cfg: TournamentConfig,
        game_cfg: GameConfig,
        metrics_cfg: MetricsConfig,
        white_player_cfg: PlayerConfig,
        black_player_cfg: PlayerConfig,
        hydra_cfg: dict[str, Any] | None = None,
    ) -> None:
        """Initialize tournament runner.

        Args:
            tournament_cfg: Tournament settings (num_games, parallel_games, etc.)
            game_cfg: Game settings (metrics, max_moves, etc.)
            metrics_cfg: Metrics configuration for Stockfish evaluation
            white_player_cfg: White player configuration
            black_player_cfg: Black player configuration
            hydra_cfg: Optional Hydra configuration dict for reproducibility metadata
        """
        self.tournament_cfg = tournament_cfg
        self.game_cfg = game_cfg
        self.metrics_cfg = metrics_cfg
        self.white_player_cfg = white_player_cfg
        self.black_player_cfg = black_player_cfg
        self.hydra_cfg = hydra_cfg or {}

        self.rate_limiter: TokenBucketRateLimiter | None = None
        if tournament_cfg.rate_limit_rpm:
            rpm = tournament_cfg.rate_limit_rpm
            # CONSERVATIVE THROTTLING (INTENTIONAL DESIGN):
            # We divide the configured RPM by parallel workers to prevent API floods.
            # This means setting rate_limit_rpm=60 with parallel_games=4 results in
            # an EFFECTIVE rate of 15 RPM total (not 60 RPM). This is intentional:
            #   - Research code prioritizes simplicity over optimal throughput
            #   - Prevents accidental server overload during parallel experiments
            #   - Each worker gets 1/N of the budget with a shared rate limiter
            # DO NOT "fix" this to achieve configured RPM - it's designed conservatively.
            rpm_per_worker = rpm / tournament_cfg.parallel_games
            self.rate_limiter = TokenBucketRateLimiter(rpm_per_worker)
            logger.info(
                "Rate limiter configured: {} RPM total, {} RPM per worker (conservative)",
                rpm,
                rpm_per_worker,
            )

        self.tournament_cfg.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Tournament '{}' initialized: {} games, {} parallel",
            tournament_cfg.match_name,
            tournament_cfg.num_games,
            tournament_cfg.parallel_games,
        )

    def run(self) -> TournamentResult:
        """Execute the tournament and return results.

        Returns:
            Complete tournament results with all games.
        """
        start_time = datetime.now(UTC)

        logger.info("Starting tournament: {}", self.tournament_cfg.match_name)

        schedule = _generate_game_schedule(
            self.tournament_cfg,
            self.white_player_cfg,
            self.black_player_cfg,
        )

        if self.tournament_cfg.parallel_games == 1:
            results = self._run_sequential(schedule)
        else:
            results = self._run_parallel(schedule)

        if not results:
            player1_name = self.white_player_cfg.name or "Player 1"
            player2_name = self.black_player_cfg.name or "Player 2"
        else:
            first_game = results[0]
            player1_name = first_game.white_player_name
            player2_name = first_game.black_player_name

        tournament_result = aggregate_tournament_results(
            match_name=self.tournament_cfg.match_name,
            results=results,
            start_time=start_time,
            player1_name=player1_name,
            player2_name=player2_name,
        )

        logger.info(
            "Tournament '{}' complete: {} games in {:.1f}s, P1/D/P2: {}/{}/{}, cost: ${:.4f}",
            self.tournament_cfg.match_name,
            tournament_result.total_games,
            tournament_result.duration_in_sec or 0,
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
            schedule: List of (white_player_cfg, black_player_cfg) tuples.

        Returns:
            Game results.
        """
        results = []
        for game_id, (white_player_cfg, black_player_cfg) in enumerate(
            schedule, start=1
        ):
            logger.info(
                "Starting game {}/{} in '{}'",
                game_id,
                len(schedule),
                self.tournament_cfg.match_name,
            )

            result = self._run_single_game(game_id, white_player_cfg, black_player_cfg)
            results.append(result)

        return results

    def _run_parallel(
        self, schedule: list[tuple[PlayerConfig, PlayerConfig]]
    ) -> list[GameResult]:
        """Run games in parallel.

        Args:
            schedule: List of (white_player_cfg, black_player_cfg) tuples.

        Returns:
            Game results (includes failed games marked with error).
        """
        results = []

        with ThreadPoolExecutor(
            max_workers=self.tournament_cfg.parallel_games
        ) as executor:
            future_to_game_info = {
                executor.submit(
                    self._run_single_game,
                    game_id,
                    white_player_cfg,
                    black_player_cfg,
                ): (game_id, white_player_cfg, black_player_cfg)
                for game_id, (white_player_cfg, black_player_cfg) in enumerate(
                    schedule, start=1
                )
            }

            for future in as_completed(future_to_game_info):
                game_id, white_player_cfg, black_player_cfg = future_to_game_info[
                    future
                ]
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
                    failed_result = GameResult(
                        game_id=game_id,
                        white_player_name=white_player_cfg.name
                        or f"{white_player_cfg.kind.capitalize()} White",
                        black_player_name=black_player_cfg.name
                        or f"{black_player_cfg.kind.capitalize()} Black",
                        result="*",
                        total_moves=0,
                        termination_reason=f"Error: {str(e)}",
                        timestamp=datetime.now(UTC),
                    )
                    results.append(failed_result)

        results.sort(key=lambda r: r.game_id)
        return results

    def _run_single_game(
        self,
        game_id: int,
        white_player_cfg: PlayerConfig,
        black_player_cfg: PlayerConfig,
    ) -> GameResult:
        """Execute a single chess game and return results.

        Args:
            game_id: Game identifier.
            white_player_cfg: White player configuration.
            black_player_cfg: Black player configuration.

        Returns:
            Results from the completed game.
        """
        # Schedule swaps player configs for color alternation but doesn't update the color field
        white_player_cfg = replace(white_player_cfg, color="white")
        black_player_cfg = replace(black_player_cfg, color="black")

        # Players are created fresh per game for isolation. Stockfish process pooling
        # could reduce overhead, but lazy initialization makes this negligible for research.
        white_player = PlayerFactory.create_player(
            white_player_cfg, rate_limiter=self.rate_limiter
        )
        black_player = PlayerFactory.create_player(
            black_player_cfg, rate_limiter=self.rate_limiter
        )

        record_dir = (
            self.tournament_cfg.output_dir
            / self.tournament_cfg.match_name
            / f"game_{game_id:03d}"
        )
        record_name = "game"

        metrics_tracker = None
        if self.game_cfg.enable_metrics:
            metrics_tracker = MetricsFactory.create_metrics_tracker(self.metrics_cfg)

        # Snapshot the per-game player configs into the stored hydra config:
        # the global config's players section does not reflect color
        # alternation, and resume recreates players from this snapshot.
        game_hydra_cfg = {
            **self.hydra_cfg,
            "players": {
                "white": asdict(white_player_cfg),
                "black": asdict(black_player_cfg),
            },
        }

        game = Game(
            white_player=white_player,
            black_player=black_player,
            display_board=self.game_cfg.display_board,
            display_summary=self.game_cfg.display_summary,
            enable_metrics=self.game_cfg.enable_metrics,
            metrics_tracker=metrics_tracker,
            record_dir=record_dir,
            record_name=record_name,
            hydra_cfg=game_hydra_cfg,
        )

        game.play(max_num_moves=self.game_cfg.max_num_moves)

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
            white_thinking_time_in_sec=game_summary.white_player.thinking_time_in_sec,
            black_thinking_time_in_sec=game_summary.black_player.thinking_time_in_sec,
            timestamp=datetime.now(UTC),
        )

        if game.metrics_tracker is not None:
            summary_by_color = game.metrics_tracker.summarize()
            white_summary = summary_by_color.get("white")
            black_summary = summary_by_color.get("black")

            if white_summary is not None:
                game_result.white_centipawn_loss = white_summary.average_centipawn_loss
                game_result.white_quality_counts = {
                    quality.value: count
                    for quality, count in white_summary.quality_counts.items()
                }
            if black_summary is not None:
                game_result.black_centipawn_loss = black_summary.average_centipawn_loss
                game_result.black_quality_counts = {
                    quality.value: count
                    for quality, count in black_summary.quality_counts.items()
                }

        game_result.pgn_path = record_dir / f"{record_name}.pgn"
        game_result.json_path = record_dir / f"{record_name}.json"

        return game_result
