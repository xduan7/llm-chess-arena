"""Pure aggregation logic for tournament results."""

from __future__ import annotations

from datetime import datetime, UTC
from typing import Any

from loguru import logger

from llm_chess_arena.tournament.types import (
    GameResult,
    TournamentResult,
)


def aggregate_tournament_results(
    match_name: str,
    results: list[GameResult],
    start_time: datetime,
    player1_name: str,
    player2_name: str,
) -> TournamentResult:
    """Aggregate individual game results into tournament summary.

    Args:
        match_name: Tournament match name.
        results: List of game results.
        start_time: Tournament start time.
        player1_name: First player's name.
        player2_name: Second player's name.

    Returns:
        TournamentResult: Aggregated tournament results.
    """
    if player1_name == player2_name:
        logger.warning(
            "Both players are named '{}' - per-player aggregates will merge and "
            "draws will be double-counted. Give the players distinct names "
            "(e.g. via players.white.name) for meaningful statistics.",
            player1_name,
        )

    player_stats: dict[str, dict[str, Any]] = {
        player1_name: {
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "cp_losses": [],
            "thinking_times": [],
            "quality_counts": {},
        },
        player2_name: {
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "cp_losses": [],
            "thinking_times": [],
            "quality_counts": {},
        },
    }

    total_cost = 0.0
    total_moves = 0

    for game in results:
        total_cost += game.white_cost + game.black_cost
        total_moves += game.total_moves

        # Skip failed games (result="*") for win/loss/draw stats
        # Failed games are still counted in total_games and costs
        is_completed_game = game.result in ("1-0", "0-1", "1/2-1/2")

        white_name = game.white_player_name
        if white_name in player_stats:
            if is_completed_game:
                if game.result == "1-0":
                    player_stats[white_name]["wins"] += 1
                elif game.result == "0-1":
                    player_stats[white_name]["losses"] += 1
                else:  # "1/2-1/2"
                    player_stats[white_name]["draws"] += 1

            if game.white_centipawn_loss is not None:
                player_stats[white_name]["cp_losses"].append(game.white_centipawn_loss)
            player_stats[white_name]["thinking_times"].append(
                game.white_thinking_time_in_sec
            )

            for quality, count in game.white_quality_counts.items():
                player_stats[white_name]["quality_counts"][quality] = (
                    player_stats[white_name]["quality_counts"].get(quality, 0) + count
                )

        black_name = game.black_player_name
        if black_name in player_stats:
            if is_completed_game:
                if game.result == "0-1":
                    player_stats[black_name]["wins"] += 1
                elif game.result == "1-0":
                    player_stats[black_name]["losses"] += 1
                else:  # "1/2-1/2"
                    player_stats[black_name]["draws"] += 1

            if game.black_centipawn_loss is not None:
                player_stats[black_name]["cp_losses"].append(game.black_centipawn_loss)
            player_stats[black_name]["thinking_times"].append(
                game.black_thinking_time_in_sec
            )

            for quality, count in game.black_quality_counts.items():
                player_stats[black_name]["quality_counts"][quality] = (
                    player_stats[black_name]["quality_counts"].get(quality, 0) + count
                )

    player1_stats = player_stats[player1_name]
    player2_stats = player_stats[player2_name]

    player1_avg_cp_loss = (
        sum(player1_stats["cp_losses"]) / len(player1_stats["cp_losses"])
        if player1_stats["cp_losses"]
        else None
    )
    player2_avg_cp_loss = (
        sum(player2_stats["cp_losses"]) / len(player2_stats["cp_losses"])
        if player2_stats["cp_losses"]
        else None
    )

    # Calculate thinking time metrics
    player1_total_thinking = sum(player1_stats["thinking_times"])
    player2_total_thinking = sum(player2_stats["thinking_times"])

    player1_avg_thinking_per_game = (
        player1_total_thinking / len(player1_stats["thinking_times"])
        if player1_stats["thinking_times"]
        else 0.0
    )
    player2_avg_thinking_per_game = (
        player2_total_thinking / len(player2_stats["thinking_times"])
        if player2_stats["thinking_times"]
        else 0.0
    )

    # Count total moves made by each player across all games
    player1_total_moves = sum(player1_stats["quality_counts"].values())
    player2_total_moves = sum(player2_stats["quality_counts"].values())

    player1_avg_thinking_per_move = (
        player1_total_thinking / player1_total_moves if player1_total_moves > 0 else 0.0
    )
    player2_avg_thinking_per_move = (
        player2_total_thinking / player2_total_moves if player2_total_moves > 0 else 0.0
    )

    return TournamentResult(
        match_name=match_name,
        player1_name=player1_name,
        player2_name=player2_name,
        start_time=start_time,
        end_time=datetime.now(UTC),
        total_games=len(results),
        player1_wins=player1_stats["wins"],
        player2_wins=player2_stats["wins"],
        draws=player1_stats["draws"],  # Same for both players
        total_cost=total_cost,
        avg_game_length=total_moves / len(results) if results else 0.0,
        player1_avg_centipawn_loss=player1_avg_cp_loss,
        player2_avg_centipawn_loss=player2_avg_cp_loss,
        player1_total_thinking_time_in_sec=player1_total_thinking,
        player2_total_thinking_time_in_sec=player2_total_thinking,
        player1_avg_thinking_time_per_game_in_sec=player1_avg_thinking_per_game,
        player2_avg_thinking_time_per_game_in_sec=player2_avg_thinking_per_game,
        player1_avg_thinking_time_per_move_in_sec=player1_avg_thinking_per_move,
        player2_avg_thinking_time_per_move_in_sec=player2_avg_thinking_per_move,
        player1_quality_counts=player1_stats["quality_counts"],
        player2_quality_counts=player2_stats["quality_counts"],
        games=results,
    )
