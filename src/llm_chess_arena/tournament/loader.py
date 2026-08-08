"""Tournament state loading: game results for aggregation, resumability checks."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from llm_chess_arena.exceptions import InvalidGameRecordError
from llm_chess_arena.record import load_game_record, validate_record_resumable
from llm_chess_arena.tournament.types import GameResult

GAME_DIR_PATTERN = re.compile(r"game_(\d+)")


class TournamentLoader:
    """Load tournament state from disk for resume operations."""

    @staticmethod
    def load_results(results_path: Path) -> dict[str, Any]:
        """Load and parse results.json file.

        Args:
            results_path: Path to results.json file.

        Returns:
            Parsed results dictionary.

        Raises:
            FileNotFoundError: If results file doesn't exist.
            InvalidGameRecordError: If JSON is malformed.
        """
        if not results_path.exists():
            raise FileNotFoundError(f"Results file not found: {results_path}")

        try:
            with results_path.open("r", encoding="utf-8") as f:
                result: dict[str, Any] = json.load(f)
                return result
        except json.JSONDecodeError as e:
            raise InvalidGameRecordError(
                f"Invalid JSON in results file {results_path}: {e}"
            ) from e

    @staticmethod
    def find_all_game_dirs(tournament_dir: Path) -> list[Path]:
        """Find all game_NNN directories in tournament.

        Args:
            tournament_dir: Path to tournament directory.

        Returns:
            List of game directory paths, sorted by game number.
        """
        if not tournament_dir.is_dir():
            return []

        game_dirs = []
        for item in tournament_dir.iterdir():
            if item.is_dir() and GAME_DIR_PATTERN.match(item.name):
                game_dirs.append(item)

        # Sort by game number
        def sort_key(p: Path) -> int:
            match = GAME_DIR_PATTERN.search(p.name)
            return int(match.group(1)) if match else -1

        game_dirs.sort(key=sort_key)
        return game_dirs

    @staticmethod
    def find_resumable_games(tournament_dir: Path) -> list[tuple[int, Path]]:
        """Find games that need to be resumed.

        Criteria for resumable games:
        - Has termination_metadata.resumable=true
        - Result is still "Unfinished" (a completed resume rewrites the result)
        - Has valid JSON structure and player configs for recreation

        Args:
            tournament_dir: Path to tournament directory.

        Returns:
            List of (game_id, game_json_path) tuples for resumable games.
        """
        resumable_games = []
        game_dirs = TournamentLoader.find_all_game_dirs(tournament_dir)

        for game_dir in game_dirs:
            # Extract game ID from directory name (e.g., "game_001" -> 1)
            match = re.search(r"game_(\d+)", game_dir.name)
            if not match:
                continue
            game_id = int(match.group(1))

            game_json = game_dir / "game.json"
            can_resume, reason = TournamentLoader.validate_game_resumable(game_json)

            if can_resume:
                resumable_games.append((game_id, game_json))
            elif reason:
                # Log why game is not resumable (for debugging)
                logger.debug(f"Game {game_id} not resumable: {reason}")

        return resumable_games

    @staticmethod
    def validate_game_resumable(game_json_path: Path) -> tuple[bool, str]:
        """Validate whether a single game can be resumed.

        Checks:
        - File exists and is valid JSON
        - Has termination_metadata.resumable=true
        - game_outcome.result is still "Unfinished" (a game that was resumed
          and interrupted again stays resumable; a completed one does not)
        - Has hydra_config.players.white/black (needed for player recreation)

        Args:
            game_json_path: Path to game.json file.

        Returns:
            Tuple of (can_resume: bool, reason_if_not: str).
            If can_resume is True, reason will be empty string.
        """
        if not game_json_path.exists():
            return False, "file does not exist"

        try:
            data = load_game_record(game_json_path)
        except InvalidGameRecordError as e:
            return False, f"malformed JSON: {e}"
        except Exception as e:
            return False, f"failed to read file: {e}"

        return validate_record_resumable(data)


def load_game_result(json_path: Path, game_id: int) -> GameResult:
    """Load a GameResult from a game.json file written by RecordWriter.

    Player names, costs, and thinking times come from the summary section;
    centipawn losses and quality counts are derived from the per-move
    ``stockfish_evaluation`` entries.

    Args:
        json_path: Path to game.json file.
        game_id: Game identifier (from directory name).

    Returns:
        Reconstructed GameResult.

    Raises:
        InvalidGameRecordError: If JSON is malformed or missing required fields.
    """
    try:
        data = load_game_record(json_path)
    except FileNotFoundError:
        raise InvalidGameRecordError(f"Game record not found: {json_path}") from None

    # Validate required top-level sections
    required_sections = ["game_outcome", "summary"]
    for section in required_sections:
        if section not in data:
            raise InvalidGameRecordError(
                f"Missing required section '{section}' in {json_path}"
            )

    game_outcome = data["game_outcome"]
    summary = data["summary"]

    # Extract required fields
    try:
        result = game_outcome["result"]
        total_moves = game_outcome["total_moves"]
        termination_reason = game_outcome["termination"]
    except KeyError as e:
        raise InvalidGameRecordError(
            f"Missing required field {e} in {json_path}"
        ) from e

    white_player_name = summary.get("white_player")
    black_player_name = summary.get("black_player")
    if not white_player_name or not black_player_name:
        logger.warning(
            f"Game record {json_path} lacks player names in summary - "
            "falling back to generic labels; aggregation by name will skip it"
        )
        white_player_name = white_player_name or "White"
        black_player_name = black_player_name or "Black"

    moves = data.get("moves", [])
    (
        white_centipawn_loss,
        white_quality_counts,
        white_thinking_from_moves,
    ) = _player_move_stats(moves, "white")
    (
        black_centipawn_loss,
        black_quality_counts,
        black_thinking_from_moves,
    ) = _player_move_stats(moves, "black")

    players_summary = summary.get("players") or {}
    white_summary = players_summary.get("white") or {}
    black_summary = players_summary.get("black") or {}

    white_cost = white_summary.get("cost", 0.0)
    black_cost = black_summary.get("cost", 0.0)
    white_thinking_time = (
        white_summary.get("thinking_time_in_sec") or white_thinking_from_moves
    )
    black_thinking_time = (
        black_summary.get("thinking_time_in_sec") or black_thinking_from_moves
    )

    # Parse timestamp (strip 'Z' suffix for fromisoformat compatibility)
    timestamp_str = game_outcome.get("end_timestamp")
    if timestamp_str:
        try:
            # Remove 'Z' suffix and replace with '+00:00' for UTC
            if timestamp_str.endswith("Z"):
                timestamp_str = timestamp_str[:-1] + "+00:00"
            timestamp = datetime.fromisoformat(timestamp_str)
        except (ValueError, TypeError):
            timestamp = datetime.now(timezone.utc)
    else:
        timestamp = datetime.now(timezone.utc)

    # Detect if game was resumed
    was_resumed = "resumption_metadata" in data
    original_termination = None
    if was_resumed:
        resumption_meta = data.get("resumption_metadata", {})
        if isinstance(resumption_meta, dict):
            original_term_meta = resumption_meta.get("original_termination")
            if isinstance(original_term_meta, dict):
                original_termination = original_term_meta.get("error_type")

    # Construct PGN path if exists
    pgn_path = None
    if json_path.parent.is_dir():
        potential_pgn = json_path.parent / "game.pgn"
        if potential_pgn.exists():
            pgn_path = potential_pgn

    return GameResult(
        game_id=game_id,
        white_player_name=white_player_name,
        black_player_name=black_player_name,
        result=result,
        total_moves=total_moves,
        termination_reason=termination_reason,
        white_centipawn_loss=white_centipawn_loss,
        black_centipawn_loss=black_centipawn_loss,
        white_thinking_time_in_sec=white_thinking_time,
        black_thinking_time_in_sec=black_thinking_time,
        white_cost=white_cost,
        black_cost=black_cost,
        white_quality_counts=white_quality_counts,
        black_quality_counts=black_quality_counts,
        timestamp=timestamp,
        pgn_path=pgn_path,
        json_path=json_path,
        was_resumed=was_resumed,
        original_termination_reason=original_termination,
    )


def _player_move_stats(
    moves: list[dict[str, Any]], color: str
) -> tuple[float | None, dict[str, int], float]:
    """Derive per-player metrics from the record's move entries.

    Args:
        moves: Move dictionaries from the game record.
        color: "white" or "black".

    Returns:
        Tuple of (average centipawn loss or None, quality counts,
        total thinking time in seconds).
    """
    centipawn_losses: list[float] = []
    quality_counts: dict[str, int] = {}
    thinking_time_in_sec = 0.0

    for move in moves:
        if move.get("player") != color:
            continue

        thinking_time_in_sec += move.get("thinking_time_in_sec") or 0.0

        evaluation = move.get("stockfish_evaluation")
        if not isinstance(evaluation, dict):
            continue

        quality = evaluation.get("quality")
        if quality:
            quality_counts[quality] = quality_counts.get(quality, 0) + 1

        centipawn_loss = evaluation.get("centipawn_loss")
        if centipawn_loss is not None:
            centipawn_losses.append(float(centipawn_loss))

    average_centipawn_loss = (
        sum(centipawn_losses) / len(centipawn_losses) if centipawn_losses else None
    )
    return average_centipawn_loss, quality_counts, thinking_time_in_sec
