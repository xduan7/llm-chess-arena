"""Game record collection and JSON serialization for research analysis."""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import chess
from loguru import logger


def iso_timestamp(dt: datetime) -> str:
    """Convert datetime to ISO-8601 with millisecond precision and Z suffix.

    Args:
        dt: DateTime object to convert.

    Returns:
        str: ISO-8601 formatted timestamp (e.g., "2024-01-15T14:30:47.456Z").
    """
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


class RecordCollector:
    """Lightweight dict accumulator for game data during gameplay."""

    def __init__(self) -> None:
        """Initialize empty data structure."""
        self.data: Dict[str, Any] = {
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
        self.data["start_timestamp"] = timestamp

    def add_move(self, move_data: Dict[str, Any]) -> None:
        """Add a move record to the collection.

        Args:
            move_data: Complete move data dict as-is from game loop.
        """
        self.data["moves"].append(move_data)

    def update_last_move(self, updates: Dict[str, Any]) -> None:
        """Update the most recently added move with additional data.

        Args:
            updates: Dictionary of key-value pairs to add/update in the last move.
        """
        if self.data["moves"]:
            self.data["moves"][-1].update(updates)

    def set_outcome(self, outcome: chess.Outcome | None) -> None:
        """Store the final game outcome.

        Args:
            outcome: Chess outcome object or None for unfinished games.
        """
        self.data["outcome"] = outcome

    def set_termination_label_override(self, override_label: str | None) -> None:
        """Store the termination label override for distinguishing outcome types.

        Args:
            override_label: Override label (e.g., "Resignation", "Illegal Move") or None.
        """
        self.data["termination_label_override"] = override_label

    def set_end_timestamp(self, timestamp: str) -> None:
        """Record when the game ended.

        Args:
            timestamp: ISO-8601 formatted timestamp.
        """
        self.data["end_timestamp"] = timestamp

    def get_data(self) -> Dict[str, Any]:
        """Return accumulated game data.

        Returns:
            Dict containing all collected game data.
        """
        return self.data


class RecordWriter:
    """Handles JSON serialization with summary rollup and environment capture."""

    @staticmethod
    def _calculate_summary(
        moves: List[Dict[str, Any]],
        outcome: chess.Outcome | None,
        termination_label_override: str | None = None,
    ) -> Dict[str, Any]:
        """Calculate summary statistics from move data.

        Args:
            moves: List of move dictionaries.
            outcome: Chess game outcome.
            termination_label_override: Override label for termination type.

        Returns:
            Dict containing summary statistics.
        """
        result = "1/2-1/2"  # Default to draw
        termination = "unknown"

        if outcome:
            if outcome.winner == chess.WHITE:
                result = "1-0"
            elif outcome.winner == chess.BLACK:
                result = "0-1"
            else:
                result = "1/2-1/2"

            # Use override label if available, otherwise map chess.Termination to string
            if termination_label_override:
                termination = termination_label_override.lower().replace(" ", "_")
            else:
                termination_map = {
                    chess.Termination.CHECKMATE: "checkmate",
                    chess.Termination.STALEMATE: "stalemate",
                    chess.Termination.INSUFFICIENT_MATERIAL: "insufficient_material",
                    chess.Termination.SEVENTYFIVE_MOVES: "75_moves",
                    chess.Termination.FIVEFOLD_REPETITION: "fivefold_repetition",
                    chess.Termination.FIFTY_MOVES: "50_moves",
                    chess.Termination.THREEFOLD_REPETITION: "threefold_repetition",
                    chess.Termination.VARIANT_WIN: "variant_win",
                    chess.Termination.VARIANT_LOSS: "variant_loss",  # Fallback for unspecified variant loss
                    chess.Termination.VARIANT_DRAW: "max_moves",  # Used for max moves
                }
                termination = termination_map.get(outcome.termination, "unknown")

        summary: Dict[str, Any] = {
            "result": result,
            "termination": termination,
            "total_moves": len(moves),
            "players": {},
        }

        for color in ["white", "black"]:
            player_moves = [m for m in moves if m.get("player") == color]

            thinking_times = [
                m.get("thinking_time_seconds", 0)
                for m in player_moves
                if m.get("thinking_time_seconds")
            ]
            total_thinking_time = sum(thinking_times) if thinking_times else 0

            player_summary: Dict[str, Any] = {}

            if total_thinking_time > 0:
                player_summary["thinking_time_seconds"] = round(total_thinking_time, 1)

            llm_moves = [m for m in player_moves if m.get("llm_decision_process")]
            if llm_moves:
                total_api_calls = 0
                total_retry_attempts = 0
                total_prompt_tokens = 0
                total_completion_tokens = 0

                for move in llm_moves:
                    llm_process = move.get("llm_decision_process", {})
                    api_calls = llm_process.get("api_calls", [])

                    total_api_calls += len(api_calls)

                    # Count retries (attempts beyond the first)
                    if len(api_calls) > 1:
                        total_retry_attempts += len(api_calls) - 1

                    for call in api_calls:
                        response = call.get("response")
                        if response and "usage" in response:
                            usage = response["usage"]
                            total_prompt_tokens += usage.get("prompt_tokens", 0)
                            total_completion_tokens += usage.get("completion_tokens", 0)

                if total_api_calls > 0:
                    player_summary["api_calls"] = total_api_calls
                if total_retry_attempts > 0:
                    player_summary["retry_attempts"] = total_retry_attempts
                if total_prompt_tokens > 0:
                    player_summary["tokens_prompt"] = total_prompt_tokens
                if total_completion_tokens > 0:
                    player_summary["tokens_completion"] = total_completion_tokens

            # Only add player section if there's data
            if player_summary:
                summary["players"][color] = player_summary

        return summary

    @staticmethod
    def _capture_environment() -> Dict[str, Any]:
        """Capture runtime environment information.

        Returns:
            Dict containing environment details.
        """
        env: Dict[str, Any] = {
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": platform.platform(),
        }

        try:
            import shutil

            if shutil.which("stockfish"):
                env["stockfish_available"] = True
                # Note: Getting actual version would require running stockfish,
                # which is expensive. We just note it's available.
            else:
                env["stockfish_available"] = False
        except Exception:
            env["stockfish_available"] = False

        return env

    @staticmethod
    def _build_game_outcome_section(
        moves: List[Dict[str, Any]],
        outcome: chess.Outcome | None,
        end_timestamp: str,
        termination_label_override: str | None = None,
    ) -> Dict[str, Any]:
        """Build the game outcome section.

        Args:
            moves: List of move data.
            outcome: Chess game outcome.
            end_timestamp: When the game ended.
            termination_label_override: Override label for termination type.

        Returns:
            Dict containing game outcome details.
        """
        summary = RecordWriter._calculate_summary(
            moves, outcome, termination_label_override
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
        hydra_config: Dict[str, Any],
        output_path: Path,
        initial_fen: str,
    ) -> None:
        """Write complete game record to JSON file.

        Args:
            collector: RecordCollector with accumulated game data.
            hydra_config: Complete Hydra configuration.
            output_path: Where to write the JSON file.
            initial_fen: The actual starting FEN position of the game.
        """
        data = collector.get_data()
        moves = data["moves"]
        outcome = data["outcome"]
        start_timestamp = data["start_timestamp"]
        end_timestamp = data["end_timestamp"]
        termination_label_override = data.get("termination_label_override")

        record = {
            "summary": RecordWriter._calculate_summary(
                moves, outcome, termination_label_override
            ),
            "environment": RecordWriter._capture_environment(),
            "hydra_config": hydra_config,
            "game_setup": {"initial_fen": initial_fen},
            "moves": moves,
            "game_outcome": RecordWriter._build_game_outcome_section(
                moves, outcome, end_timestamp, termination_label_override
            ),
        }

        if start_timestamp:
            record["environment"]["timestamp_start"] = start_timestamp
        if end_timestamp:
            record["environment"]["timestamp_end"] = end_timestamp

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with output_path.open("w", encoding="utf-8") as f:
                json.dump(record, f, indent=2, ensure_ascii=False)

            logger.info("Saved game record to {}", output_path)

        except Exception as exc:
            logger.warning("Failed to save game record to {}: {}", output_path, exc)
