"""Game record collection and JSON serialization for research analysis."""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import chess
from loguru import logger

from llm_chess_arena.utils import (
    GameSummary,
    build_game_summary_from_data,
    is_stockfish_available,
)


def iso_timestamp(datetime_object: datetime) -> str:
    """Convert datetime to ISO-8601 with millisecond precision and Z suffix.

    Args:
        datetime_object: DateTime object to convert.

    Returns:
        str: ISO-8601 formatted timestamp (e.g., "2024-01-15T14:30:47.456Z").
    """
    return datetime_object.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


class RecordCollector:
    """Lightweight dict accumulator for game data during gameplay."""

    def __init__(self) -> None:
        """Initialize empty data structure."""
        self.game_record: dict[str, Any] = {
            "moves": [],
            "start_timestamp": None,
            "end_timestamp": None,
            "outcome": None,
            "termination_label_override": None,
        }

    def set_start_timestamp(self, timestamp: str) -> None:
        """Record when the game started.

        Args:
            timestamp: ISO-8601 formatted timestamp.
        """
        self.game_record["start_timestamp"] = timestamp

    def add_move(self, move_data: dict[str, Any]) -> None:
        """Add a move record to the collection.

        Args:
            move_data: Complete move data dict as-is from game loop.
        """
        self.game_record["moves"].append(move_data)

    def update_last_move(self, updates: dict[str, Any]) -> None:
        """Update the most recently added move with additional data.

        Args:
            updates: Dictionary of key-value pairs to add/update in the last move.
        """
        if self.game_record["moves"]:
            self.game_record["moves"][-1].update(updates)

    def set_outcome(self, outcome: chess.Outcome | None) -> None:
        """Store the final game outcome.

        Args:
            outcome: Chess outcome object or None for unfinished games.
        """
        self.game_record["outcome"] = outcome

    def set_termination_label_override(self, override_label: str | None) -> None:
        """Store the termination label override for distinguishing outcome types.

        Args:
            override_label: Override label (e.g., "Resignation", "Illegal Move") or None.
        """
        self.game_record["termination_label_override"] = override_label

    def set_end_timestamp(self, timestamp: str) -> None:
        """Record when the game ended.

        Args:
            timestamp: ISO-8601 formatted timestamp.
        """
        self.game_record["end_timestamp"] = timestamp

    def get_data(self) -> dict[str, Any]:
        """Return accumulated game data.

        Returns:
            Dict containing all collected game data.
        """
        return self.game_record


class RecordWriter:
    """Handles JSON serialization with summary rollup and environment capture."""

    @staticmethod
    def _calculate_summary(
        moves: list[dict[str, Any]],
        outcome: chess.Outcome | None,
        termination_label_override: str | None = None,
        white_player: Any = None,
        black_player: Any = None,
    ) -> dict[str, Any]:
        """Calculate summary statistics from move data using central builder.

        Args:
            moves: List of move dictionaries.
            outcome: Chess game outcome.
            termination_label_override: Override label for termination type.
            white_player: Real white player object (fallback to "White" if None).
            black_player: Real black player object (fallback to "Black" if None).

        Returns:
            Dict containing summary statistics.
        """

        white_player_name = str(white_player) if white_player else "White"
        black_player_name = str(black_player) if black_player else "Black"

        summary = build_game_summary_from_data(
            outcome=outcome,
            move_records=moves,
            white_player_name=white_player_name,
            black_player_name=black_player_name,
            termination_label_override=termination_label_override,
            white_player=white_player,
            black_player=black_player,
        )
        return summary.to_json_dict()

    @staticmethod
    def _capture_environment() -> dict[str, Any]:
        """Capture runtime environment information.

        Returns:
            Dict containing environment details.
        """
        environment_info: dict[str, Any] = {
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": platform.platform(),
        }

        try:
            environment_info["stockfish_available"] = is_stockfish_available()
            # Note: Getting actual version would require running stockfish,
            # which is expensive. We just note it's available.
        except Exception:
            environment_info["stockfish_available"] = False

        return environment_info

    @staticmethod
    def _build_game_outcome_section(
        moves: list[dict[str, Any]],
        outcome: chess.Outcome | None,
        end_timestamp: str,
        termination_label_override: str | None = None,
        game_summary: GameSummary | None = None,
        white_player: Any = None,
        black_player: Any = None,
    ) -> dict[str, Any]:
        """Build the game outcome section.

        Args:
            moves: List of move data.
            outcome: Chess game outcome.
            end_timestamp: When the game ended.
            termination_label_override: Override label for termination type.
            game_summary: Optional pre-computed game summary to avoid redundant calculation.
            white_player: Real white player object with name and metadata.
            black_player: Real black player object with name and metadata.

        Returns:
            Dict containing game outcome details.
        """
        summary = (
            game_summary.to_json_dict()
            if game_summary is not None
            else RecordWriter._calculate_summary(
                moves, outcome, termination_label_override, white_player, black_player
            )
        )

        outcome_section = {
            "result": summary["result"],
            "termination": summary["termination"],
            "total_moves": summary["total_moves"],
            "end_timestamp": end_timestamp,
        }

        if outcome and outcome.winner is not None:
            outcome_section["winner"] = (
                "white" if outcome.winner == chess.WHITE else "black"
            )
        else:
            outcome_section["winner"] = None

        if moves:
            last_move = moves[-1]
            if "position_after" in last_move:
                outcome_section["final_position"] = last_move["position_after"]["fen"]

        return outcome_section

    @staticmethod
    def write(
        collector: RecordCollector,
        hydra_config: dict[str, Any],
        output_path: Path,
        initial_fen: str,
        white_player: Any = None,
        black_player: Any = None,
        game_summary: GameSummary | None = None,
    ) -> None:
        """Write complete game record to JSON file.

        Args:
            collector: RecordCollector with accumulated game data.
            hydra_config: Complete Hydra configuration.
            output_path: Where to write the JSON file.
            initial_fen: The actual starting FEN position of the game.
            white_player: Real white player object with name and metadata.
            black_player: Real black player object with name and metadata.
            game_summary: Optional pre-computed game summary to avoid redundant calculation.
        """
        collected_game_record = collector.get_data()
        moves = collected_game_record["moves"]
        outcome = collected_game_record["outcome"]
        start_timestamp = collected_game_record["start_timestamp"]
        end_timestamp = collected_game_record["end_timestamp"]
        termination_label_override = collected_game_record.get(
            "termination_label_override"
        )

        if game_summary is not None:
            summary_data = game_summary.to_json_dict()
        else:
            logger.warning(
                "GameSummary not provided to RecordWriter - using fallback calculation"
            )
            summary_data = RecordWriter._calculate_summary(
                moves,
                outcome,
                termination_label_override,
                white_player,
                black_player,
            )

        record = {
            "summary": summary_data,
            "environment": RecordWriter._capture_environment(),
            "hydra_config": hydra_config,
            "game_setup": {"initial_fen": initial_fen},
            "moves": moves,
            "game_outcome": RecordWriter._build_game_outcome_section(
                moves,
                outcome,
                end_timestamp,
                termination_label_override,
                game_summary,
                white_player,
                black_player,
            ),
        }

        if start_timestamp:
            record["environment"]["timestamp_start"] = start_timestamp
        if end_timestamp:
            record["environment"]["timestamp_end"] = end_timestamp

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with output_path.open("w", encoding="utf-8") as output_file:
                json.dump(record, output_file, indent=2, ensure_ascii=False)

            logger.info("Saved game record to {}", output_path)

        except Exception as file_write_error:
            logger.warning(
                "Failed to save game record to {}: {}", output_path, file_write_error
            )
