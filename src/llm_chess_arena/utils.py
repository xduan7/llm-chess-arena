"""Utility helpers for chess move serialization, validation, and system utilities."""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Final, Any, Protocol

import chess
import chess.engine
from loguru import logger

from llm_chess_arena.core.policies import move_validation
from llm_chess_arena.exceptions import (
    IllegalMoveError,
    InvalidMoveError,
    AmbiguousMoveError,
)


def get_legal_moves_in_uci(board: chess.Board) -> list[str]:
    """Get all legal moves in UCI format from the current board state.

    Args:
        board: Current chess board state.

    Returns:
        list[str]: Legal moves in UCI notation (e.g., ["e2e4", "g1f3"]).
    """
    return [move.uci() for move in board.legal_moves]


def get_move_history_in_uci(board: chess.Board) -> list[str]:
    """Get the move history in UCI format from the current board state.

    Args:
        board: Current chess board state with move history.

    Returns:
        list[str]: Moves in UCI notation (e.g., ["e2e4", "e7e5", "g1f3"]).
    """
    return [move.uci() for move in board.move_stack]


@move_validation
def parse_attempted_move_to_uci(attempted_move: str, board_in_fen: str) -> str:
    """Parse a move string to UCI format, trying UCI first then SAN.

    Args:
        attempted_move: Move text in UCI (e2e4) or SAN (Nf3, O-O).
        board_in_fen: FEN string representing the position.

    Returns:
        str: Move in UCI format (e.g., "e2e4").

    Raises:
        InvalidMoveError: If notation is syntactically invalid.
        AmbiguousMoveError: If SAN is ambiguous in this position.
        IllegalMoveError: If move is not legal in this position.
    """
    board = chess.Board(fen=board_in_fen)

    attempted_move_normalized = normalize_castling_notation(attempted_move)

    try:
        move = chess.Move.from_uci(attempted_move_normalized)
        if move not in board.legal_moves:
            raise IllegalMoveError(
                f"Illegal move in current position: '{attempted_move}'"
            )
        return str(move.uci())
    except ValueError:
        try:
            move = board.parse_san(attempted_move_normalized)
            return str(move.uci())
        except chess.AmbiguousMoveError as ambiguous_move_error:
            raise AmbiguousMoveError(
                f"Ambiguous SAN move: '{attempted_move}'"
            ) from ambiguous_move_error
        except chess.InvalidMoveError as invalid_move_error:
            raise InvalidMoveError(
                f"Invalid move notation: '{attempted_move}'"
            ) from invalid_move_error
        except chess.IllegalMoveError as illegal_move_error:
            raise IllegalMoveError(
                f"Illegal move in current position: '{attempted_move}'"
            ) from illegal_move_error


# Common platform-specific locations checked after PATH lookup.
# Cache for Stockfish availability check to avoid repeated filesystem calls
_stockfish_availability_cache: bool | None = None

COMMON_STOCKFISH_PATHS: tuple[str, ...] = (
    "/usr/local/bin/stockfish",
    "/usr/bin/stockfish",
    "/opt/homebrew/bin/stockfish",
    "C:/Program Files/Stockfish/stockfish.exe",
    "C:/Program Files (x86)/Stockfish/stockfish.exe",
)


def find_stockfish_binary(explicit_path: str | None = None) -> str:
    """Resolve a usable Stockfish executable path.

    Args:
        explicit_path: Optional user-supplied path to the Stockfish binary.

    Returns:
        str: Absolute path to the executable.

    Raises:
        FileNotFoundError: If no executable binary can be located.
    """
    if explicit_path:
        candidate_path = Path(explicit_path)
        if not candidate_path.exists():
            raise FileNotFoundError(f"Stockfish binary not found at: {candidate_path}")
        if not os.access(str(candidate_path), os.X_OK):
            raise FileNotFoundError(
                "Stockfish binary exists but is not executable at: "
                f"{candidate_path}\n"
                f"Try: chmod +x {candidate_path}"
            )
        return str(candidate_path.resolve())

    env_path_str = os.getenv("STOCKFISH_BINARY_PATH")
    if env_path_str:
        env_path = Path(env_path_str)
        if not env_path.exists():
            logger.warning(
                "Environment variable STOCKFISH_BINARY_PATH set to {} but file does not exist",
                env_path,
            )
        elif not os.access(str(env_path), os.X_OK):
            logger.warning(
                "Stockfish binary from STOCKFISH_BINARY_PATH exists but is not executable: {}",
                env_path,
            )
        else:
            logger.debug(
                "Found Stockfish binary from STOCKFISH_BINARY_PATH: {}",
                env_path,
            )
            return str(env_path.resolve())

    system_path = shutil.which("stockfish")
    if system_path:
        logger.debug("Found Stockfish binary in PATH: {}", system_path)
        return system_path

    for potential_path in COMMON_STOCKFISH_PATHS:
        candidate_path = Path(potential_path)
        if candidate_path.exists() and os.access(str(candidate_path), os.X_OK):
            logger.debug("Found Stockfish binary in common path: {}", candidate_path)
            return str(candidate_path.resolve())

    raise FileNotFoundError(
        "Stockfish not found. Please install it or provide the binary path.\n"
        "You can either:\n"
        "  1. Set STOCKFISH_BINARY_PATH in your .env file\n"
        "  2. Pass binary_path parameter when creating StockfishPlayer\n"
        "  3. Install Stockfish:\n"
        "     macOS: brew install stockfish\n"
        "     Ubuntu/Debian: apt-get install stockfish\n"
        "     Windows: Download from https://stockfishchess.org/download/"
    )


def is_stockfish_available() -> bool:
    """Check if Stockfish is available on the system.

    This function caches the result to avoid repeated filesystem calls.
    Uses the same logic as find_stockfish_binary but returns a boolean
    instead of raising exceptions.

    Returns:
        bool: True if Stockfish is available, False otherwise.
    """
    global _stockfish_availability_cache

    if _stockfish_availability_cache is not None:
        return _stockfish_availability_cache

    try:
        find_stockfish_binary()
        _stockfish_availability_cache = True
        return True
    except FileNotFoundError:
        _stockfish_availability_cache = False
        return False


def initialize_stockfish_engine(
    binary_path: str, engine_options: dict[str, Any] | None = None
) -> chess.engine.SimpleEngine:
    """Initialize and configure a Stockfish engine instance.

    Provides shared initialization logic for both StockfishPlayer and
    StockfishMetricsEvaluator to eliminate code duplication.

    Args:
        binary_path: Path to the Stockfish executable.
        engine_options: Optional UCI engine configuration options.

    Returns:
        chess.engine.SimpleEngine: Configured Stockfish engine instance.

    Raises:
        Exception: If engine initialization or configuration fails.
    """
    stockfish_engine = chess.engine.SimpleEngine.popen_uci(binary_path)
    try:
        if engine_options:
            stockfish_engine.configure(engine_options)
        return stockfish_engine
    except Exception:
        stockfish_engine.quit()
        raise


TERMINATION_LABELS: Final[dict[chess.Termination, str]] = {
    chess.Termination.CHECKMATE: "Checkmate",
    chess.Termination.STALEMATE: "Stalemate",
    chess.Termination.INSUFFICIENT_MATERIAL: "Insufficient material",
    chess.Termination.SEVENTYFIVE_MOVES: "75-move rule",
    chess.Termination.FIVEFOLD_REPETITION: "Fivefold repetition",
    chess.Termination.THREEFOLD_REPETITION: "Threefold repetition",
    chess.Termination.FIFTY_MOVES: "50-move rule",
    chess.Termination.VARIANT_WIN: "Variant-specific win",
    chess.Termination.VARIANT_LOSS: "Variant-specific loss",
    chess.Termination.VARIANT_DRAW: "Variant-specific draw",
}


@dataclass(slots=True)
class PlayerSummary:
    """Summary statistics for a single player."""

    name: str
    color: str  # "white" or "black"
    thinking_time_in_sec: float = 0.0
    api_calls: int = 0
    retry_count: int = 0
    tokens_prompt: int = 0
    tokens_completion: int = 0
    tokens_total: int = 0
    total_decisions: int = 0
    voting_ties: int = 0
    network_errors: int = 0
    average_latency_in_ms: float = 0.0
    average_response_length: float = 0.0
    cost: float = 0.0

    @property
    def avg_thinking_time_per_move_in_sec(self) -> float:
        """Calculate average thinking time per move."""
        if self.total_decisions == 0:
            return 0.0
        return self.thinking_time_in_sec / self.total_decisions


@dataclass(slots=True)
class GameSummary:
    """Comprehensive game summary with outcome and player statistics."""

    # Basic outcome info
    result: str  # "1-0", "0-1", "1/2-1/2"
    termination: str  # "checkmate", "stalemate", etc.
    total_moves: int
    winner_name: str | None
    winner_color: chess.Color | None

    # Player summaries
    white_player: PlayerSummary
    black_player: PlayerSummary

    # Optional overrides
    termination_label_override: str | None = None
    termination_note: str | None = None

    def to_cli_lines(self) -> list[str]:
        """Generate CLI display lines."""
        outcome_summary = build_game_outcome_summary(
            outcome=self._to_chess_outcome(),
            white_player_name=self.white_player.name,
            black_player_name=self.black_player.name,
            total_moves=self.total_moves,
            termination_label_override=self.termination_label_override,
            termination_note=self.termination_note,
        )

        lines = [
            outcome_summary.outcome_line,
            outcome_summary.termination_line,
            outcome_summary.total_moves_line,
        ]

        if outcome_summary.winner_line:
            lines.insert(2, outcome_summary.winner_line)

        return lines

    def to_json_dict(self) -> dict[str, Any]:
        """Generate JSON export dictionary."""
        winner_color_string = None
        if self.winner_color is not None:
            if hasattr(self.winner_color, "name"):
                winner_color_string = self.winner_color.name.lower()
            elif self.winner_color == chess.WHITE or self.winner_color is True:
                winner_color_string = "white"
            elif self.winner_color == chess.BLACK or self.winner_color is False:
                winner_color_string = "black"
            else:
                # Last resort fallback
                winner_color_string = str(self.winner_color).lower()

        game_export_data: dict[str, Any] = {
            "result": self.result,
            "termination": self.termination,
            "total_moves": self.total_moves,
            "winner": self.winner_name,
            "winner_color": winner_color_string,
            "players": {},
        }

        for player_summary in [self.white_player, self.black_player]:
            player_export_data: dict[str, Any] = {}

            if player_summary.thinking_time_in_sec > 0:
                player_export_data["thinking_time_in_sec"] = round(
                    player_summary.thinking_time_in_sec, 1
                )
            if player_summary.api_calls > 0:
                player_export_data["api_calls"] = player_summary.api_calls
            if player_summary.retry_count > 0:
                player_export_data["retry_count"] = player_summary.retry_count
            if player_summary.tokens_prompt > 0:
                player_export_data["tokens_prompt"] = player_summary.tokens_prompt
            if player_summary.tokens_completion > 0:
                player_export_data["tokens_completion"] = (
                    player_summary.tokens_completion
                )

            if player_export_data:
                game_export_data["players"][player_summary.color] = player_export_data

        return game_export_data

    def log_usage_summary(self) -> None:
        """Log usage summary for each player."""
        for player_summary in [self.white_player, self.black_player]:
            if player_summary.tokens_prompt > 0 or player_summary.tokens_completion > 0:
                total_tokens = (
                    player_summary.tokens_prompt + player_summary.tokens_completion
                )
                logger.info(
                    "{} token usage: {} prompt tokens, {} completion tokens, {} total",
                    player_summary.name,
                    player_summary.tokens_prompt,
                    player_summary.tokens_completion,
                    total_tokens,
                )

                if player_summary.cost > 0:
                    logger.info(
                        "{} cost: ${:.6f}",
                        player_summary.name,
                        player_summary.cost,
                    )

                if player_summary.total_decisions > 0:
                    logger.info(
                        "{} made {} moves using {} API calls with {} retries",
                        player_summary.name,
                        player_summary.total_decisions,
                        player_summary.api_calls,
                        player_summary.retry_count,
                    )

                    if player_summary.thinking_time_in_sec > 0:
                        logger.info(
                            "{} thinking time: {:.2f}s total, {:.3f}s per move",
                            player_summary.name,
                            player_summary.thinking_time_in_sec,
                            player_summary.avg_thinking_time_per_move_in_sec,
                        )

                    if player_summary.voting_ties > 0:
                        logger.info(
                            "{} had {} voting ties",
                            player_summary.name,
                            player_summary.voting_ties,
                        )

                    if player_summary.network_errors > 0:
                        logger.info(
                            "{} had {} network errors",
                            player_summary.name,
                            player_summary.network_errors,
                        )

                    if player_summary.average_latency_in_ms > 0:
                        logger.info(
                            "{} average response time: {:.0f}ms",
                            player_summary.name,
                            player_summary.average_latency_in_ms,
                        )

                    if player_summary.average_response_length > 0:
                        logger.info(
                            "{} average response length: {:.0f} characters",
                            player_summary.name,
                            player_summary.average_response_length,
                        )

    def _to_chess_outcome(self) -> chess.Outcome | None:
        """Convert summary data back to chess.Outcome for compatibility."""
        if self.result == "1/2-1/2":
            winner = None
        elif self.result == "1-0":
            winner = chess.WHITE
        elif self.result == "0-1":
            winner = chess.BLACK
        else:
            return None

        termination_map = {
            "checkmate": chess.Termination.CHECKMATE,
            "stalemate": chess.Termination.STALEMATE,
            "insufficient_material": chess.Termination.INSUFFICIENT_MATERIAL,
            "75_moves": chess.Termination.SEVENTYFIVE_MOVES,
            "fivefold_repetition": chess.Termination.FIVEFOLD_REPETITION,
            "50_moves": chess.Termination.FIFTY_MOVES,
            "threefold_repetition": chess.Termination.THREEFOLD_REPETITION,
            "variant_win": chess.Termination.VARIANT_WIN,
            "variant_loss": chess.Termination.VARIANT_LOSS,
            "max_num_moves": chess.Termination.VARIANT_DRAW,
        }

        termination = termination_map.get(
            self.termination, chess.Termination.VARIANT_DRAW
        )
        return chess.Outcome(termination=termination, winner=winner)

    def to_game_outcome_summary(self) -> GameOutcomeSummary:
        """Convert to legacy GameOutcomeSummary format for backward compatibility."""
        return build_game_outcome_summary(
            outcome=self._to_chess_outcome(),
            white_player_name=self.white_player.name,
            black_player_name=self.black_player.name,
            total_moves=self.total_moves,
            termination_label_override=self.termination_label_override,
            termination_note=self.termination_note,
        )


def _extract_game_outcome_info(
    outcome: chess.Outcome | None,
    white_player_name: str,
    black_player_name: str,
) -> tuple[str, str, str | None, chess.Color | None]:
    """Extract result, termination, winner info from game outcome.

    Returns:
        Tuple of (result, termination, winner_name, winner_color)
    """
    result = "1/2-1/2"  # Default to draw
    termination = "unknown"
    winner_name = None
    winner_color = None

    if outcome:
        if outcome.winner == chess.WHITE:
            result = "1-0"
            winner_color = chess.WHITE
            winner_name = white_player_name
        elif outcome.winner == chess.BLACK:
            result = "0-1"
            winner_color = chess.BLACK
            winner_name = black_player_name
        else:
            result = "1/2-1/2"

        termination_map = {
            chess.Termination.CHECKMATE: "checkmate",
            chess.Termination.STALEMATE: "stalemate",
            chess.Termination.INSUFFICIENT_MATERIAL: "insufficient_material",
            chess.Termination.SEVENTYFIVE_MOVES: "75_moves",
            chess.Termination.FIVEFOLD_REPETITION: "fivefold_repetition",
            chess.Termination.FIFTY_MOVES: "50_moves",
            chess.Termination.THREEFOLD_REPETITION: "threefold_repetition",
            chess.Termination.VARIANT_WIN: "variant_win",
            chess.Termination.VARIANT_LOSS: "variant_loss",
            chess.Termination.VARIANT_DRAW: "max_num_moves",
        }
        termination = termination_map.get(outcome.termination, "unknown")

    return result, termination, winner_name, winner_color


def _process_moves_data(
    move_records: list[dict[str, Any]],
    white_player_summary: PlayerSummary,
    black_player_summary: PlayerSummary,
) -> None:
    """Process move data to extract player statistics."""
    for move_record in move_records:
        current_player_summary = (
            white_player_summary
            if move_record.get("player") == "white"
            else black_player_summary
        )

        if move_record.get("thinking_time_in_sec"):
            current_player_summary.thinking_time_in_sec += move_record[
                "thinking_time_in_sec"
            ]

        llm_decision_process = move_record.get("llm_decision_process")
        if llm_decision_process:
            api_call_records = llm_decision_process.get("api_calls", [])
            current_player_summary.api_calls += len(api_call_records)

            if len(api_call_records) > 1:
                current_player_summary.retry_count += len(api_call_records) - 1

            for api_call_record in api_call_records:
                api_call_response = api_call_record.get("response")
                if api_call_response and "usage" in api_call_response:
                    response_usage = api_call_response["usage"]
                    current_player_summary.tokens_prompt += response_usage.get(
                        "prompt_tokens", 0
                    )
                    current_player_summary.tokens_completion += response_usage.get(
                        "completion_tokens", 0
                    )


def _extract_player_metrics(
    player: Any,
    player_summary: PlayerSummary,
) -> None:
    """Extract LLM metrics from player object."""
    usage_totals_method = getattr(player, "get_usage_totals", None)
    if callable(usage_totals_method):
        try:
            usage_totals = usage_totals_method()
            if usage_totals:
                if hasattr(usage_totals, "cost") and usage_totals.cost > 0:
                    player_summary.cost = usage_totals.cost
                # Override with more accurate usage if available
                if (
                    hasattr(usage_totals, "prompt_tokens")
                    and usage_totals.prompt_tokens > 0
                ):
                    player_summary.tokens_prompt = usage_totals.prompt_tokens
                if (
                    hasattr(usage_totals, "completion_tokens")
                    and usage_totals.completion_tokens > 0
                ):
                    player_summary.tokens_completion = usage_totals.completion_tokens
                if (
                    hasattr(usage_totals, "total_tokens")
                    and usage_totals.total_tokens > 0
                ):
                    player_summary.tokens_total = usage_totals.total_tokens
        except Exception:
            pass

    # Extended performance metrics (voting_ties, latencies, etc.) are now calculated
    # directly from game move records when needed.


def build_game_summary(
    game: Any,
    move_records: list[dict[str, Any]] | None = None,
) -> GameSummary:
    """Build comprehensive game summary from game object and optional move data.

    Args:
        game: Game object with outcome, players, and board state.
        move_records: Optional list of move dictionaries with LLM metadata.

    Returns:
        GameSummary: Comprehensive summary for CLI, JSON, and logging.
    """
    result, termination, winner_name, winner_color = _extract_game_outcome_info(
        game.outcome,
        str(game.white_player),
        str(game.black_player),
    )
    total_moves = len(game.board.move_stack)

    white_player_summary = PlayerSummary(name=str(game.white_player), color="white")
    black_player_summary = PlayerSummary(name=str(game.black_player), color="black")

    if move_records:
        _process_moves_data(
            move_records,
            white_player_summary,
            black_player_summary,
        )

    _extract_player_metrics(game.white_player, white_player_summary)
    _extract_player_metrics(game.black_player, black_player_summary)

    return GameSummary(
        result=result,
        termination=termination,
        total_moves=total_moves,
        winner_name=winner_name,
        winner_color=winner_color,
        white_player=white_player_summary,
        black_player=black_player_summary,
        termination_label_override=getattr(game, "_termination_label_override", None),
        termination_note=getattr(game, "_termination_note", None),
    )


def build_game_summary_from_data(
    outcome: chess.Outcome | None,
    move_records: list[dict[str, Any]],
    white_player_name: str = "White",
    black_player_name: str = "Black",
    termination_label_override: str | None = None,
    termination_note: str | None = None,
    white_player: Any = None,
    black_player: Any = None,
) -> GameSummary:
    """Build comprehensive game summary directly from raw data without requiring a game object.

    Args:
        outcome: Chess game outcome with winner and termination info.
        move_records: List of move dictionaries with metadata.
        white_player_name: Name of the white player.
        black_player_name: Name of the black player.
        termination_label_override: Override label for termination type.
        termination_note: Additional termination details.
        white_player: Optional player object for LLM usage extraction.
        black_player: Optional player object for LLM usage extraction.

    Returns:
        GameSummary: Comprehensive summary for CLI, JSON, and logging.
    """
    result, termination, winner_name, winner_color = _extract_game_outcome_info(
        outcome, white_player_name, black_player_name
    )
    total_moves = len(move_records)

    white_player_summary = PlayerSummary(name=white_player_name, color="white")
    black_player_summary = PlayerSummary(name=black_player_name, color="black")

    _process_moves_data(
        move_records,
        white_player_summary,
        black_player_summary,
    )

    if white_player is not None:
        _extract_player_metrics(white_player, white_player_summary)
    if black_player is not None:
        _extract_player_metrics(black_player, black_player_summary)

    return GameSummary(
        result=result,
        termination=termination,
        total_moves=total_moves,
        winner_name=winner_name,
        winner_color=winner_color,
        white_player=white_player_summary,
        black_player=black_player_summary,
        termination_label_override=termination_label_override,
        termination_note=termination_note,
    )


def normalize_castling_notation(move_text: str) -> str:
    """Normalize castling notation to standard format.

    Args:
        move_text: Move text that may contain castling notation.

    Returns:
        str: Move text with normalized castling notation.
    """
    move_normalized = move_text.strip()

    if move_normalized.lower() in ["o-o", "0-0"]:
        return "O-O"
    elif move_normalized.lower() in ["o-o-o", "0-0-0"]:
        return "O-O-O"

    return move_normalized


@dataclass(slots=True)
class GameOutcomeSummary:
    """Structured outcome data for post-game displays and logging."""

    outcome_line: str
    termination_line: str
    total_moves_line: str
    winner_line: str | None
    winner_name: str | None
    winner_color: chess.Color | None


def humanize_termination(termination: chess.Termination | None) -> str:
    """Convert a python-chess termination enum to a readable label.

    Args:
        termination: Termination enum from python-chess, or ``None`` if the game
            is in progress.

    Returns:
        str: Human-friendly description of the termination state.
    """

    if termination is None:
        return "Game in progress"

    result = TERMINATION_LABELS.get(termination)
    if result is not None:
        return result
    # Explicit type annotation to help mypy
    termination_name: str = termination.name
    return termination_name.replace("_", " ").title()


def build_game_outcome_summary(
    outcome: chess.Outcome | None,
    white_player_name: str,
    black_player_name: str,
    total_moves: int,
    *,
    termination_label_override: str | None = None,
    termination_note: str | None = None,
) -> GameOutcomeSummary:
    """Create human-friendly summary strings for a finished game.

    Args:
        outcome: python-chess outcome information, or ``None`` if unavailable.
        white_player_name: Display name for the white player.
        black_player_name: Display name for the black player.
        total_moves: Number of moves played in the game.

    Returns:
        GameOutcomeSummary: Structured summary fields for logs and rendering.
    """

    if outcome is None:
        return GameOutcomeSummary(
            outcome_line="Outcome: Game did not finish",
            termination_line="Termination: Unknown",
            total_moves_line=f"Total moves: {total_moves}",
            winner_line=None,
            winner_name=None,
            winner_color=None,
        )

    winner_color = outcome.winner
    termination_label = termination_label_override or humanize_termination(
        outcome.termination
    )
    termination_line = f"Termination: {termination_label}"
    if termination_note:
        termination_line = f"{termination_line} ({termination_note})"

    is_draw = winner_color is None
    if is_draw:
        outcome_line = "Outcome: Draw"
        winner_name: str | None = None
        winner_line = None
    else:
        winner_name = (
            white_player_name if winner_color == chess.WHITE else black_player_name
        )
        color_label = "White" if winner_color == chess.WHITE else "Black"
        outcome_line = f"Outcome: {winner_name} ({color_label}) wins"
        winner_line = None

    total_moves_line = f"Total moves: {total_moves}"

    return GameOutcomeSummary(
        outcome_line=outcome_line,
        termination_line=termination_line,
        total_moves_line=total_moves_line,
        winner_line=winner_line,
        winner_name=winner_name,
        winner_color=winner_color,
    )


class RateLimiter(Protocol):
    """Protocol for rate limiting implementations."""

    def acquire_permit(self, provider_name: str, timeout_in_sec: float) -> bool:
        """Try to acquire a permit.

        Args:
            provider_name: API provider name (ignored in simple implementation).
            timeout_in_sec: Maximum time to wait for a permit.

        Returns:
            bool: True if permit acquired, False if timeout.
        """
        ...

    def report_rate_limit_error(
        self, provider_name: str, _retry_after: float | None
    ) -> None:
        """Report a rate limit error (ignored in simple implementation).

        Args:
            provider_name: API provider that returned the rate limit error.
            _retry_after: Optional retry-after value from API (in seconds).
        """
        ...


class TokenBucketRateLimiter:
    """Simple global rate limiter enforcing requests per minute (RPM).

    Dead simple:
    - Enforces GLOBAL limit on API CALLS per minute
    - rate_limit_rpm: 60 = max 60 API requests per minute total
    - Blocks until request slot available or timeout
    - Thread-safe
    """

    def __init__(self, requests_per_minute: float) -> None:
        """Initialize rate limiter.

        Args:
            requests_per_minute: Max API CALLS per minute (global limit).
        """
        self.rpm = requests_per_minute
        self.requests_per_second = requests_per_minute / 60.0

        # Track available request slots
        self.available_requests = 1.0
        self.last_update = time.time()
        self.lock = Lock()

    def acquire_permit(self, provider_name: str, timeout_in_sec: float) -> bool:
        """Try to acquire permission for one API call.

        Args:
            provider_name: Ignored (global limit).
            timeout_in_sec: Max wait time.

        Returns:
            bool: True if permit acquired, False if timeout.
        """
        deadline = time.time() + timeout_in_sec

        while time.time() < deadline:
            with self.lock:
                now = time.time()

                # Refill request slots based on time elapsed
                elapsed = now - self.last_update
                self.available_requests = min(
                    1.0, self.available_requests + elapsed * self.requests_per_second
                )
                self.last_update = now

                if self.available_requests >= 1.0:
                    # Have a slot - use it
                    self.available_requests -= 1.0
                    return True

                # Calculate wait time until we have a slot
                wait_in_sec = (1.0 - self.available_requests) / self.requests_per_second
                sleep_time = min(wait_in_sec, 0.1)

            # Sleep outside lock
            if time.time() + sleep_time > deadline:
                return False
            time.sleep(sleep_time)

        return False

    def report_rate_limit_error(
        self, provider_name: str, _retry_after: float | None
    ) -> None:
        """Ignored - if you hit limits, configure lower RPM."""
        pass
