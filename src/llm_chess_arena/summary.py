"""Game summary domain model shared by records, rendering, and logging."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Any

import chess
from loguru import logger

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


@dataclass(slots=True)
class GameOutcomeSummary:
    """Structured outcome data for post-game displays and logging."""

    outcome_line: str
    termination_line: str
    total_moves_line: str


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
        )

    winner_color = outcome.winner
    termination_label = termination_label_override or humanize_termination(
        outcome.termination
    )
    termination_line = f"Termination: {termination_label}"
    if termination_note:
        termination_line = f"{termination_line} ({termination_note})"

    if winner_color is None:
        outcome_line = "Outcome: Draw"
    else:
        winner_name = (
            white_player_name if winner_color == chess.WHITE else black_player_name
        )
        color_label = "White" if winner_color == chess.WHITE else "Black"
        outcome_line = f"Outcome: {winner_name} ({color_label}) wins"

    return GameOutcomeSummary(
        outcome_line=outcome_line,
        termination_line=termination_line,
        total_moves_line=f"Total moves: {total_moves}",
    )


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
    result: str  # "1-0", "0-1", "1/2-1/2", or "Unfinished" for interrupted games
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
            "white_player": self.white_player.name,
            "black_player": self.black_player.name,
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
            if player_summary.cost > 0:
                player_export_data["cost"] = player_summary.cost

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


def _extract_game_outcome_info(
    outcome: chess.Outcome | None,
    white_player_name: str,
    black_player_name: str,
) -> tuple[str, str, str | None, chess.Color | None]:
    """Extract result, termination, winner info from game outcome.

    Returns:
        Tuple of (result, termination, winner_name, winner_color)
    """
    # No outcome means the game was interrupted (e.g., network error) - it must
    # not be reported as a draw, and "Unfinished" marks the record as resumable.
    result = "Unfinished"
    termination = "unfinished"
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
    # Plain names (not str(player), which appends the color suffix) so summary
    # names line up with tournament aggregation keyed by player name
    white_player_name = getattr(game.white_player, "name", str(game.white_player))
    black_player_name = getattr(game.black_player, "name", str(game.black_player))

    result, termination, winner_name, winner_color = _extract_game_outcome_info(
        game.outcome,
        white_player_name,
        black_player_name,
    )
    total_moves = len(game.board.move_stack)

    white_player_summary = PlayerSummary(name=white_player_name, color="white")
    black_player_summary = PlayerSummary(name=black_player_name, color="black")

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
