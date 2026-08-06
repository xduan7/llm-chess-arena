"""Simplified type definitions for tournament execution and results tracking."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from llm_chess_arena.exceptions import InvalidGameRecordError


@dataclass(slots=True)
class TournamentConfig:
    """Tournament configuration - runs multiple games with one player matchup."""

    match_name: str
    num_games: int
    parallel_games: int
    alternate_colors: bool
    display_summary: bool
    output_dir: Path
    rate_limit_rpm: int | None = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.num_games <= 0:
            raise ValueError(f"num_games must be positive, got {self.num_games}")
        if self.parallel_games < 1:
            raise ValueError(f"parallel_games must be >= 1, got {self.parallel_games}")
        if self.rate_limit_rpm is not None and self.rate_limit_rpm <= 0:
            raise ValueError(
                f"rate_limit_rpm must be positive when specified, got {self.rate_limit_rpm}"
            )

        # Warn about odd number of games with color alternation (not perfectly balanced)
        if self.alternate_colors and self.num_games > 1 and self.num_games % 2 != 0:
            logger.warning(
                f"Tournament has odd number of games ({self.num_games}) with alternate_colors=True. "
                f"Color distribution will be imbalanced."
            )

        if isinstance(self.output_dir, str):
            object.__setattr__(self, "output_dir", Path(self.output_dir))


@dataclass(slots=True)
class GameResult:
    """Result of a single game within a tournament."""

    game_id: int
    white_player_name: str
    black_player_name: str
    result: str  # "1-0", "0-1", "1/2-1/2", or "Unfinished"
    total_moves: int
    termination_reason: str
    white_centipawn_loss: float | None = None
    black_centipawn_loss: float | None = None
    white_thinking_time_in_sec: float = 0.0
    black_thinking_time_in_sec: float = 0.0
    white_cost: float = 0.0
    black_cost: float = 0.0
    white_quality_counts: dict[str, int] = field(default_factory=dict)
    black_quality_counts: dict[str, int] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    pgn_path: Path | None = None
    json_path: Path | None = None

    # Resume tracking
    was_resumed: bool = False
    original_termination_reason: str | None = None

    @staticmethod
    def load_from_game_json(json_path: Path, game_id: int) -> GameResult:
        """Load GameResult from a game.json file written by RecordWriter.

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
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise InvalidGameRecordError(f"Invalid JSON in {json_path}: {e}") from e
        except FileNotFoundError:
            raise InvalidGameRecordError(f"Game record not found: {json_path}")

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
        ) = GameResult._player_move_stats(moves, "white")
        (
            black_centipawn_loss,
            black_quality_counts,
            black_thinking_from_moves,
        ) = GameResult._player_move_stats(moves, "black")

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

    @staticmethod
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


@dataclass(slots=True)
class TournamentResult:
    """Aggregated results for a tournament."""

    match_name: str
    player1_name: str  # First player (no fixed color)
    player2_name: str  # Second player (no fixed color)
    start_time: datetime
    end_time: datetime | None = None

    # Win/Loss/Draw counts
    total_games: int = 0
    player1_wins: int = 0  # Player 1's wins (regardless of color)
    player2_wins: int = 0  # Player 2's wins (regardless of color)
    draws: int = 0

    # Aggregate metrics
    total_cost: float = 0.0
    avg_game_length: float = 0.0
    player1_avg_centipawn_loss: float | None = None
    player2_avg_centipawn_loss: float | None = None
    player1_total_thinking_time_in_sec: float = 0.0
    player2_total_thinking_time_in_sec: float = 0.0
    player1_avg_thinking_time_per_game_in_sec: float = 0.0  # For backward compatibility
    player2_avg_thinking_time_per_game_in_sec: float = 0.0  # For backward compatibility
    player1_avg_thinking_time_per_move_in_sec: float = 0.0
    player2_avg_thinking_time_per_move_in_sec: float = 0.0
    player1_quality_counts: dict[str, int] = field(default_factory=dict)
    player2_quality_counts: dict[str, int] = field(default_factory=dict)

    # Individual game results
    games: list[GameResult] = field(default_factory=list)

    # Resume tracking
    resumed_games: list[int] = field(default_factory=list)
    resume_timestamp: datetime | None = None

    @property
    def duration_in_sec(self) -> float | None:
        """Tournament duration in seconds."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time).total_seconds()

    @property
    def win_rate_player1(self) -> float:
        """Win rate for Player 1."""
        if self.total_games == 0:
            return 0.0
        return self.player1_wins / self.total_games

    @property
    def win_rate_player2(self) -> float:
        """Win rate for Player 2."""
        if self.total_games == 0:
            return 0.0
        return self.player2_wins / self.total_games

    @property
    def draw_rate(self) -> float:
        """Draw rate."""
        if self.total_games == 0:
            return 0.0
        return self.draws / self.total_games

    @property
    def player1_quality_percentages(self) -> dict[str, float]:
        """Calculate move quality percentages for Player 1."""
        total = sum(self.player1_quality_counts.values())
        if total == 0:
            return {}
        return {
            quality: (count / total) * 100
            for quality, count in self.player1_quality_counts.items()
        }

    @property
    def player2_quality_percentages(self) -> dict[str, float]:
        """Calculate move quality percentages for Player 2."""
        total = sum(self.player2_quality_counts.values())
        if total == 0:
            return {}
        return {
            quality: (count / total) * 100
            for quality, count in self.player2_quality_counts.items()
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        base: dict[str, Any] = {
            "match_name": self.match_name,
            "player1": self.player1_name,
            "player2": self.player2_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_in_sec": self.duration_in_sec,
            "total_games": self.total_games,
            "results": {
                "player1_wins": self.player1_wins,
                "player2_wins": self.player2_wins,
                "draws": self.draws,
            },
            "win_rates": {
                "player1": self.win_rate_player1,
                "player2": self.win_rate_player2,
                "draw": self.draw_rate,
            },
            "total_cost": self.total_cost,
            "avg_game_length": self.avg_game_length,
            "avg_metrics": {
                "player1_centipawn_loss": self.player1_avg_centipawn_loss,
                "player2_centipawn_loss": self.player2_avg_centipawn_loss,
                "player1_thinking_time": self.player1_avg_thinking_time_per_game_in_sec,
                "player2_thinking_time": self.player2_avg_thinking_time_per_game_in_sec,
            },
            "thinking_time": {
                "player1_total": self.player1_total_thinking_time_in_sec,
                "player2_total": self.player2_total_thinking_time_in_sec,
                "player1_avg_per_game": self.player1_avg_thinking_time_per_game_in_sec,
                "player2_avg_per_game": self.player2_avg_thinking_time_per_game_in_sec,
                "player1_avg_per_move": self.player1_avg_thinking_time_per_move_in_sec,
                "player2_avg_per_move": self.player2_avg_thinking_time_per_move_in_sec,
            },
            "move_quality": {
                "player1_counts": self.player1_quality_counts,
                "player2_counts": self.player2_quality_counts,
                "player1_percentages": self.player1_quality_percentages,
                "player2_percentages": self.player2_quality_percentages,
            },
        }

        if self.resumed_games:
            base["resume_info"] = {
                "resumed_game_ids": self.resumed_games,
                "resume_timestamp": (
                    self.resume_timestamp.isoformat() if self.resume_timestamp else None
                ),
            }

        return base
