"""Tournament results export to JSON and CSV formats."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from loguru import logger

from llm_chess_arena.tournament.types import TournamentResult


class ResultsExporter:
    """Exports tournament results to JSON and CSV formats."""

    @staticmethod
    def export_json(result: TournamentResult, output_path: Path) -> None:
        """Export tournament results to JSON file.

        Args:
            result: Tournament results to export.
            output_path: Path to output JSON file.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info("Exported tournament results to JSON: {}", output_path)

    @staticmethod
    def export_csv(result: TournamentResult, output_path: Path) -> None:
        """Export tournament game results to CSV file.

        Creates a CSV with one row per game for easy pandas analysis.

        Args:
            result: Tournament results to export.
            output_path: Path to output CSV file.
        """
        if not result.games:
            logger.warning("No games to export to CSV")
            return

        # Define CSV schema
        # Note: Individual games still have white/black players (color matters per game)
        fieldnames = [
            "match_name",
            "game_id",
            "white_player",  # Color-specific for this individual game
            "black_player",  # Color-specific for this individual game
            "result",
            "total_moves",
            "white_centipawn_loss",
            "black_centipawn_loss",
            "white_thinking_time",
            "black_thinking_time",
            "white_cost",
            "black_cost",
            # Move quality counts for white player
            "white_best",
            "white_excellent",
            "white_good",
            "white_inaccuracy",
            "white_mistake",
            "white_blunder",
            # Move quality counts for black player
            "black_best",
            "black_excellent",
            "black_good",
            "black_inaccuracy",
            "black_mistake",
            "black_blunder",
            "termination_reason",
            "timestamp",
        ]

        # Write CSV file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for game in result.games:
                row = {
                    "match_name": result.match_name,
                    "game_id": game.game_id,
                    "white_player": game.white_player_name,
                    "black_player": game.black_player_name,
                    "result": game.result,
                    "total_moves": game.total_moves,
                    "white_centipawn_loss": (
                        f"{game.white_centipawn_loss:.2f}"
                        if game.white_centipawn_loss is not None
                        else ""
                    ),
                    "black_centipawn_loss": (
                        f"{game.black_centipawn_loss:.2f}"
                        if game.black_centipawn_loss is not None
                        else ""
                    ),
                    "white_thinking_time": f"{game.white_thinking_time:.2f}",
                    "black_thinking_time": f"{game.black_thinking_time:.2f}",
                    "white_cost": f"{game.white_cost:.6f}",
                    "black_cost": f"{game.black_cost:.6f}",
                    # White player move quality counts
                    "white_best": game.white_quality_counts.get("best", 0),
                    "white_excellent": game.white_quality_counts.get("excellent", 0),
                    "white_good": game.white_quality_counts.get("good", 0),
                    "white_inaccuracy": game.white_quality_counts.get("inaccuracy", 0),
                    "white_mistake": game.white_quality_counts.get("mistake", 0),
                    "white_blunder": game.white_quality_counts.get("blunder", 0),
                    # Black player move quality counts
                    "black_best": game.black_quality_counts.get("best", 0),
                    "black_excellent": game.black_quality_counts.get("excellent", 0),
                    "black_good": game.black_quality_counts.get("good", 0),
                    "black_inaccuracy": game.black_quality_counts.get("inaccuracy", 0),
                    "black_mistake": game.black_quality_counts.get("mistake", 0),
                    "black_blunder": game.black_quality_counts.get("blunder", 0),
                    "termination_reason": game.termination_reason,
                    "timestamp": game.timestamp.isoformat(),
                }
                writer.writerow(row)

        logger.info(
            "Exported {} game results to CSV: {}", len(result.games), output_path
        )

    @staticmethod
    def print_summary(result: TournamentResult) -> None:
        """Print human-readable tournament summary to console.

        Args:
            result: Tournament results to summarize.
        """
        print("\n" + "=" * 80)
        print(f"Tournament: {result.match_name}")
        print("=" * 80)

        print(
            f"Duration: {result.duration_seconds:.1f}s"
            if result.duration_seconds
            else "Duration: N/A"
        )
        print(f"Total Games: {result.total_games}")
        print(f"Total Cost: ${result.total_cost:.4f}")

        print("\n" + "-" * 80)
        print("Results:")
        print("-" * 80)

        print(f"Player 1: {result.player1_name} | Player 2: {result.player2_name}")
        print(f"Games: {result.total_games}")
        print(
            f"Results: P1={result.player1_wins} D={result.draws} P2={result.player2_wins}"
        )
        print(
            f"Win Rates: P1={result.win_rate_player1:.1%} Draw={result.draw_rate:.1%} P2={result.win_rate_player2:.1%}"
        )
        print(f"Avg Game Length: {result.avg_game_length:.1f} moves")

        if result.player1_avg_centipawn_loss is not None:
            print(f"Player 1 Avg CP Loss: {result.player1_avg_centipawn_loss:.1f}")
        if result.player2_avg_centipawn_loss is not None:
            print(f"Player 2 Avg CP Loss: {result.player2_avg_centipawn_loss:.1f}")

        print("\n" + "=" * 80 + "\n")
