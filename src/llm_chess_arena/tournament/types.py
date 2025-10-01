"""Simplified type definitions for tournament execution and results tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger


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

        # Ensure output_dir is a Path
        if isinstance(self.output_dir, str):
            object.__setattr__(self, "output_dir", Path(self.output_dir))


@dataclass(slots=True)
class GameResult:
    """Result of a single game within a tournament."""

    game_id: int
    white_player_name: str
    black_player_name: str
    result: str  # "1-0", "0-1", "1/2-1/2"
    total_moves: int
    termination_reason: str
    white_centipawn_loss: float | None = None
    black_centipawn_loss: float | None = None
    white_thinking_time: float = 0.0
    black_thinking_time: float = 0.0
    white_cost: float = 0.0
    black_cost: float = 0.0
    white_quality_counts: dict[str, int] = field(default_factory=dict)
    black_quality_counts: dict[str, int] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now())

    # File paths
    pgn_path: Path | None = None
    json_path: Path | None = None


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
    player1_avg_thinking_time: float = 0.0
    player2_avg_thinking_time: float = 0.0
    player1_quality_counts: dict[str, int] = field(default_factory=dict)
    player2_quality_counts: dict[str, int] = field(default_factory=dict)

    # Individual game results
    games: list[GameResult] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float | None:
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
        return {
            "match_name": self.match_name,
            "player1": self.player1_name,
            "player2": self.player2_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
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
                "player1_thinking_time": self.player1_avg_thinking_time,
                "player2_thinking_time": self.player2_avg_thinking_time,
            },
            "move_quality": {
                "player1_counts": self.player1_quality_counts,
                "player2_counts": self.player2_quality_counts,
                "player1_percentages": self.player1_quality_percentages,
                "player2_percentages": self.player2_quality_percentages,
            },
        }
