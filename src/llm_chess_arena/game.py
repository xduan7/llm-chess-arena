"""Core game loop coordinating chess players and board state.

Module Organization:
- Game Class: Central orchestrator for chess matches
  * Initialization: Player setup, metrics configuration, recording setup
  * Game Loop: Move execution, validation, board updates
  * Move Handling: Player decision processing, error management
  * Recording: PGN and JSON output, move data collection
  * Metrics: Stockfish evaluation, win probability tracking
  * Cleanup: Resource management and summary reporting
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping
import time

import chess
import chess.pgn
from loguru import logger

from llm_chess_arena.exceptions import (
    AmbiguousMoveError,
    IllegalMoveError,
    InvalidMoveError,
    LLMPermanentError,
)
from llm_chess_arena.player.base_player import BasePlayer
from llm_chess_arena.renderer import display_board_with_context, display_game_summary
from llm_chess_arena.types import PlayerDecision
from llm_chess_arena.metrics import MOVE_QUALITY_ORDER, MetricsTracker, MoveQuality
from llm_chess_arena.moves import parse_attempted_move_to_uci
from llm_chess_arena.summary import (
    GameSummary,
    build_game_outcome_summary,
    build_game_summary,
)
from llm_chess_arena.record import RecordCollector, RecordWriter, iso_timestamp


class Game:
    """Orchestrates a chess game between two players."""

    def __init__(
        self,
        white_player: BasePlayer,
        black_player: BasePlayer,
        display_board: bool,
        display_summary: bool,
        enable_metrics: bool,
        metrics_tracker: MetricsTracker | None = None,
        record_dir: str | Path | None = None,
        record_name: str | None = None,
        hydra_cfg: dict[str, Any] | None = None,
    ) -> None:
        """Initialize a chess game.

        Args:
            white_player: Player controlling white pieces.
            black_player: Player controlling black pieces.
            display_board: Whether to display the board after each move.
            display_summary: Whether to display game summary at end.
            enable_metrics: Whether to compute move quality metrics.
            metrics_tracker: Optional preconfigured metrics tracker.
            record_dir: Optional directory path for writing game records when
                the game completes. When ``None``, no records are written.
            record_name: Optional custom name for record files. When ``None``,
                uses timestamp-based naming.
            hydra_cfg: Optional Hydra configuration dict for game records.

        Raises:
            ValueError: If players have incorrect colors assigned.
        """
        if white_player.color != "white":
            raise ValueError(f"White player has wrong color: {white_player.color}")
        if black_player.color != "black":
            raise ValueError(f"Black player has wrong color: {black_player.color}")

        self.white_player = white_player
        self.black_player = black_player
        self.board = chess.Board()
        self.display_board = display_board
        self.display_summary = display_summary
        self.metrics_tracker = (
            metrics_tracker
            if metrics_tracker is not None
            else (
                MetricsTracker.from_stockfish(require_stockfish=enable_metrics)
                if enable_metrics
                else None
            )
        )
        self._rendered_metrics_summary = False
        self._record_dir = (
            Path(record_dir).expanduser() if record_dir is not None else None
        )
        self._record_name = record_name
        self._start_timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self._termination_label_override: str | None = None
        self._termination_note: str | None = None
        self._termination_metadata: dict[str, Any] | None = None
        self._hydra_cfg = hydra_cfg or {}

        self._initial_fen: str = self.board.fen()

        self._record_collector = RecordCollector() if record_dir is not None else None

        self._white_thinking_time_in_sec = 0.0
        self._black_thinking_time_in_sec = 0.0

        self._current_white_win_probability: float | None = None

        metrics_enabled = bool(
            self.metrics_tracker is not None and self.metrics_tracker.enabled
        )

        if enable_metrics and not metrics_enabled:
            logger.info(
                "Game initialized without metrics (Stockfish unavailable): {} vs {}",
                white_player,
                black_player,
            )
        else:
            logger.info("Game initialized: {} vs {}", white_player, black_player)
        self._outcome: chess.Outcome | None = None
        self._resumed_from: Path | None = None
        self._original_termination_metadata: dict[str, Any] | None = None

    def _restore_resumed_state(
        self,
        *,
        board: chess.Board,
        initial_fen: str,
        moves: list[dict[str, Any]],
        start_timestamp: str | None,
        resumed_from: Path,
        original_termination_metadata: dict[str, Any] | None,
    ) -> None:
        """Install replayed state when resuming from a saved record.

        Called by ``factory.resume_game_from_file`` after constructing the
        Game; keeps all mutation of game internals inside the class.

        Args:
            board: Board with the recorded moves already applied.
            initial_fen: FEN the original game started from.
            moves: Move dictionaries from the saved record.
            start_timestamp: Original start timestamp to preserve, if any.
            resumed_from: Path of the record this game was resumed from.
            original_termination_metadata: Termination metadata of the
                interrupted run, kept for resumption bookkeeping.
        """
        self.board = board
        self._initial_fen = initial_fen

        if self._record_collector is not None:
            self._record_collector.load_existing_moves(moves)
            # Preserve original start timestamp
            if start_timestamp:
                self._record_collector.set_start_timestamp(start_timestamp)

        # Restore cumulative thinking time so resumed summaries stay accurate
        for move_record in moves:
            recorded_thinking_time = move_record.get("thinking_time_in_sec")
            if not recorded_thinking_time:
                continue
            if move_record.get("player") == "white":
                self._white_thinking_time_in_sec += recorded_thinking_time
            else:
                self._black_thinking_time_in_sec += recorded_thinking_time

        self._resumed_from = resumed_from
        self._original_termination_metadata = original_termination_metadata

    @property
    def current_player(self) -> BasePlayer:
        """Get the player whose turn it is to move.

        Returns:
            BasePlayer: Currently active player instance.
        """
        return (
            self.white_player if self.board.turn == chess.WHITE else self.black_player
        )

    @property
    def outcome(self) -> chess.Outcome | None:
        """Get the outcome of the game if finished.

        Returns:
            chess.Outcome | None: Outcome object when game is over, else None.
        """
        if self._outcome is None and self.board.is_game_over():
            self._outcome = self.board.outcome()
        return self._outcome

    @property
    def finished(self) -> bool:
        """Check if the game is finished.

        Returns:
            bool: True if the game is over, False otherwise.
        """
        return self.outcome is not None

    @property
    def winner(self) -> BasePlayer | None:
        """Get the winner of the game if finished.

        Returns:
            BasePlayer | None: Winning player, or None for draw / ongoing games.
        """
        if not self.finished or self.outcome is None:
            return None

        winner_color = self.outcome.winner
        if winner_color is None:
            return None

        color_to_player = {
            chess.WHITE: self.white_player,
            chess.BLACK: self.black_player,
        }
        return color_to_player[winner_color]

    @property
    def initial_fen(self) -> str:
        """Get the initial FEN position when the game was created.

        Returns:
            str: FEN string of the starting position.
        """
        return self._initial_fen

    def make_move(self) -> None:
        """Execute a single move in the game.

        Raises:
            InvalidMoveError: If decision has invalid action or missing move.
            Exception: Any exception from player() or from_uci() is propagated.
        """
        start_time = time.time()

        # Copy prevents players from mutating game state
        current_player = self.current_player
        decision = current_player(board=self.board.copy())
        decision_artifacts = None
        get_decision_artifacts = getattr(
            current_player, "get_last_decision_artifacts", None
        )
        if callable(get_decision_artifacts):
            decision_artifacts = get_decision_artifacts()

        # Prefer player's self-reported thinking time over wall-clock measurement to avoid double-counting
        wall_clock_move_time_in_sec = time.time() - start_time
        player_reported_time_in_sec = getattr(decision, "thinking_time_in_sec", None)

        recorded_move_time_in_sec = (
            player_reported_time_in_sec
            if player_reported_time_in_sec is not None
            else wall_clock_move_time_in_sec
        )

        if current_player.color == "white":
            self._white_thinking_time_in_sec += recorded_move_time_in_sec
        else:
            self._black_thinking_time_in_sec += recorded_move_time_in_sec

        if decision.action == "resign":
            self._record_move_if_configured(decision, decision_artifacts)
            self._handle_resignation(decision)
            return
        elif decision.action == "move":
            move_in_uci = self._record_move_if_configured(decision, decision_artifacts)
            self._handle_move(decision, move_in_uci, decision_artifacts)
        else:
            raise InvalidMoveError(f"Unsupported action: {decision.action}")

    def _handle_resignation(self, decision: PlayerDecision) -> None:
        """Handle player resignation."""
        # Note: chess library doesn't have RESIGNATION termination
        # Using VARIANT_LOSS for termination when a player resigns
        self._outcome = chess.Outcome(
            termination=chess.Termination.VARIANT_LOSS,  # Non-standard loss by resignation
            winner=(
                chess.BLACK if self.current_player.color == "white" else chess.WHITE
            ),
        )
        logger.info("{} resigns", self.current_player)

        self._termination_label_override = "Resignation"
        self._termination_note = None

        # Resignations are non-resumable and intentional
        resignation_reason = getattr(decision, "reason", "Player chose to resign")
        self._termination_metadata = {
            "resumable": False,
            "error_type": "Resignation",
            "player_color": self.current_player.color,
            "error_message": resignation_reason,
            "halfmove_index": len(self.board.move_stack),
            "fullmove_number": self.board.fullmove_number,
            "fen": self.board.fen(),
            "intentional": True,
        }

    def _handle_move(
        self,
        decision: PlayerDecision,
        cached_move_in_uci: str | None,
        decision_artifacts: Any | None,
    ) -> None:
        """Validate the player's move and apply it to the board.

        Args:
            decision: Move decision returned by the active player.

        Raises:
            InvalidMoveError: Missing or malformed move text.
            IllegalMoveError: Move fails legality checks for the position.
            AmbiguousMoveError: Move text is ambiguous within the position.
        """
        if decision.attempted_move is None:
            raise InvalidMoveError("Move action requires attempted_move")

        active_player = self.current_player
        board_before_move = self.board.copy()

        move_in_uci = (
            cached_move_in_uci
            if cached_move_in_uci is not None
            else parse_attempted_move_to_uci(decision.attempted_move, self.board.fen())
        )

        move = chess.Move.from_uci(move_in_uci)
        # Board.push() does not validate, so re-check even player-normalized moves
        if move not in self.board.legal_moves:
            raise IllegalMoveError(f"Illegal move in current position: '{move_in_uci}'")
        move_number = (len(self.board.move_stack) // 2) + 1
        logger.info("Move {}: {} plays {}", move_number, active_player, move_in_uci)
        self.board.push(move)

        move_quality: MoveQuality | None = None
        if self.metrics_tracker is not None:
            try:
                move_metrics = self.metrics_tracker.record_move(
                    board_before_move,
                    move,
                    player_name=active_player.name,
                )
                if move_metrics is not None:
                    move_quality = move_metrics.quality

                    if (
                        hasattr(move_metrics, "actual_centipawns")
                        and move_metrics.actual_centipawns is not None
                    ):
                        centipawn_evaluation = move_metrics.actual_centipawns
                        if self.board.turn == chess.WHITE:
                            centipawn_evaluation = -centipawn_evaluation

                        # This is a rough approximation - Stockfish WDL would be more accurate
                        self._current_white_win_probability = 0.5 + 0.5 * (
                            centipawn_evaluation / 100.0
                        ) / (1.0 + abs(centipawn_evaluation / 100.0))
                        self._current_white_win_probability = max(
                            0.0, min(1.0, self._current_white_win_probability)
                        )

                    if self._record_collector is not None:
                        stockfish_evaluation: dict[str, Any] = {
                            "stockfish_evaluation": {
                                "quality": move_quality.value,
                            }
                        }
                        if (
                            hasattr(move_metrics, "centipawn_loss")
                            and move_metrics.centipawn_loss is not None
                        ):
                            stockfish_evaluation["stockfish_evaluation"][
                                "centipawn_loss"
                            ] = move_metrics.centipawn_loss
                        if (
                            hasattr(move_metrics, "best_move_in_uci")
                            and move_metrics.best_move_in_uci is not None
                        ):
                            stockfish_evaluation["stockfish_evaluation"][
                                "best_move_in_uci"
                            ] = move_metrics.best_move_in_uci
                        if (
                            hasattr(move_metrics, "best_move_hit")
                            and move_metrics.best_move_hit is not None
                        ):
                            stockfish_evaluation["stockfish_evaluation"][
                                "best_move_hit"
                            ] = move_metrics.best_move_hit

                        self._record_collector.update_last_move(stockfish_evaluation)

            except Exception as metrics_error:  # pragma: no cover
                logger.warning(
                    "Failed to record metrics for move {}: {}",
                    move_in_uci,
                    metrics_error,
                )

    def _get_move_qualities_for_display(self) -> list[MoveQuality | None] | None:
        """Derive move qualities from MetricsTracker for display purposes.

        Returns:
            list[MoveQuality | None] | None: List of move qualities or None if metrics disabled.
        """
        if self.metrics_tracker is None:
            return None

        return self.metrics_tracker.get_ordered_move_qualities(self.board.move_stack)

    def play(self, max_num_moves: int | None = None) -> None:
        """Run the game until completion, illegal move, or max moves reached.

        Args:
            max_num_moves: Maximum number of moves (half-moves) before stopping.
                          None means play until a game outcome is reached.

        Note:
            Illegal moves cause the offending player to forfeit.
            Other exceptions are logged and re-raised.
        """
        logger.info(
            "Starting game: {} (White) vs {} (Black)",
            self.white_player.name,
            self.black_player.name,
        )

        if (
            self._record_collector is not None
            and not self._record_collector.has_start_timestamp()
        ):
            # Guard keeps the original start timestamp intact on resumed games
            self._record_collector.set_start_timestamp(iso_timestamp(datetime.now(UTC)))

        self._termination_metadata = None

        self._reset_llm_usage_counters()
        try:
            half_move_count = 0
            while not self.finished:
                if max_num_moves is not None and half_move_count >= max_num_moves:
                    logger.info("Stopping: Maximum moves ({}) reached", max_num_moves)
                    self._outcome = chess.Outcome(
                        termination=chess.Termination.VARIANT_DRAW,
                        winner=None,
                    )
                    break

                try:
                    self.make_move()
                    half_move_count += 1

                    if self.display_board:
                        current_move = (
                            self.board.peek() if self.board.move_stack else None
                        )
                        try:
                            display_board_with_context(
                                self.board,
                                current_player=self.current_player.name,
                                last_move=current_move,
                                white_player=str(self.white_player),
                                black_player=str(self.black_player),
                                move_qualities=self._get_move_qualities_for_display(),
                                white_thinking_time_in_sec=self._white_thinking_time_in_sec,
                                black_thinking_time_in_sec=self._black_thinking_time_in_sec,
                                white_win_probability=self._current_white_win_probability,
                            )
                        except Exception as render_error:
                            # Don't let display errors terminate the game
                            logger.warning("Board rendering failed: {}", render_error)
                except (TimeoutError, ConnectionError) as network_error:
                    # Network errors are resumable - don't set outcome
                    error_type = (
                        "timeout"
                        if isinstance(network_error, TimeoutError)
                        else "connection"
                    )
                    self._termination_label_override = f"Network {error_type} error"
                    self._termination_metadata = {
                        "resumable": True,
                        "error_type": network_error.__class__.__name__,
                        "player_color": self.current_player.color,
                        "error_message": self._sanitize_error_message(
                            str(network_error)
                        ),
                        "halfmove_index": len(self.board.move_stack),
                        "fullmove_number": self.board.fullmove_number,
                        "fen": self.board.fen(),
                    }
                    logger.warning(
                        "Game interrupted by network {} error at move {}: {} - game can be resumed",
                        error_type,
                        self.board.fullmove_number,
                        network_error,
                    )
                    # Don't set _outcome - keep it None so game appears unfinished
                    break
                except LLMPermanentError as llm_error:
                    # LLM permanent errors (auth, invalid request, etc.) are non-resumable
                    logger.warning(
                        "Game over due to permanent LLM error by {}: {}",
                        self.current_player,
                        llm_error,
                    )
                    self._outcome = chess.Outcome(
                        termination=chess.Termination.VARIANT_LOSS,
                        winner=(
                            chess.BLACK
                            if self.current_player.color == "white"
                            else chess.WHITE
                        ),
                    )
                    self._termination_label_override = "LLM permanent error"
                    self._termination_metadata = {
                        "resumable": False,
                        "error_type": llm_error.__class__.__name__,
                        "player_color": self.current_player.color,
                        "error_message": self._sanitize_error_message(str(llm_error)),
                        "halfmove_index": len(self.board.move_stack),
                        "fullmove_number": self.board.fullmove_number,
                        "fen": self.board.fen(),
                    }
                    break
                except (
                    IllegalMoveError,
                    InvalidMoveError,
                    AmbiguousMoveError,
                ) as move_error:
                    logger.warning(
                        "Game over due to {} by {}: {}",
                        move_error.__class__.__name__,
                        self.current_player,
                        move_error,
                    )
                    self._outcome = chess.Outcome(
                        termination=chess.Termination.VARIANT_LOSS,
                        winner=(
                            chess.BLACK
                            if self.current_player.color == "white"
                            else chess.WHITE
                        ),
                    )
                    self._termination_label_override = (
                        f"{move_error.__class__.__name__.replace('Error', ' error')}"
                    )
                    self._termination_metadata = {
                        "resumable": False,
                        "error_type": move_error.__class__.__name__,
                        "player_color": self.current_player.color,
                        "error_message": self._sanitize_error_message(str(move_error)),
                        "halfmove_index": len(self.board.move_stack),
                        "fullmove_number": self.board.fullmove_number,
                        "fen": self.board.fen(),
                    }
                    break
                except Exception as unexpected_error:
                    logger.exception(
                        "Unexpected error during player move by {}: {}",
                        self.current_player,
                        unexpected_error,
                    )
                    raise

            if self.outcome:
                logger.info("Game finished after {} moves", len(self.board.move_stack))
                if self.winner:
                    logger.info("Winner: {}", self.winner)
                else:
                    logger.info("Game ended in a draw")
        finally:
            if self._record_collector is not None:
                self._record_collector.set_end_timestamp(
                    iso_timestamp(datetime.now(UTC))
                )
                self._record_collector.set_outcome(self.outcome)
                self._record_collector.set_termination_label_override(
                    self._termination_label_override
                )
                self._record_collector.set_termination_metadata(
                    self._termination_metadata
                )

            game_summary = build_game_summary(self)
            self._log_llm_usage_summary(game_summary)
            self._save_history_if_configured(game_summary)
            if self.metrics_tracker is not None:
                self._log_metrics_summary()
            self._cleanup_players()

    def _record_move_if_configured(
        self, decision: PlayerDecision, decision_artifacts: Any | None
    ) -> str | None:
        """Record move data to the record collector if active.

        Args:
            decision: The player decision about to be executed.
        """
        if self._record_collector is None:
            return None

        move_number = len(self.board.move_stack) + 1
        active_player = self.current_player
        position_before_fen = self.board.fen()
        position_after_fen = None
        normalized_uci: str | None = None
        move_in_uci: str | None = None
        if decision.action == "move" and decision.attempted_move:
            # Apply the move temporarily to get the position after
            try:
                normalized_uci = getattr(decision_artifacts, "normalized_uci", None)
                move_in_uci = (
                    normalized_uci
                    if normalized_uci is not None
                    else parse_attempted_move_to_uci(
                        decision.attempted_move, self.board.fen()
                    )
                )

                move = chess.Move.from_uci(move_in_uci)
                if move in self.board.legal_moves:
                    board_copy = self.board.copy()
                    board_copy.push(move)
                    position_after_fen = board_copy.fen()
            except (
                ValueError,
                chess.InvalidMoveError,
                InvalidMoveError,
                IllegalMoveError,
                AmbiguousMoveError,
            ):
                pass

        move_data: dict[str, Any] = {
            "move_number": move_number,
            "player": active_player.color,
            "timestamp": iso_timestamp(datetime.now(UTC)),
            "position_before": {
                "fen": position_before_fen,
            },
            "final_decision": {
                "action": decision.action,
            },
        }

        if position_after_fen is not None:
            move_data["position_after"] = {
                "fen": position_after_fen,
            }

        if decision.action == "move" and decision.attempted_move:
            # Store normalized UCI for reliable game resume
            # Use move_in_uci if parsing succeeded, otherwise fall back to raw text
            uci_for_record = (
                move_in_uci if move_in_uci is not None else decision.attempted_move
            )
            move_data["final_decision"]["attempted_move_in_uci"] = uci_for_record
        elif decision.action == "resign":
            move_data["final_decision"]["resignation_reason"] = getattr(
                decision, "reason", "Player resigned"
            )

        decision_process = getattr(decision_artifacts, "decision_process", None)
        if decision_process is not None:
            move_data["llm_decision_process"] = decision_process

        if (
            hasattr(decision, "thinking_time_in_sec")
            and decision.thinking_time_in_sec is not None
        ):
            move_data["thinking_time_in_sec"] = decision.thinking_time_in_sec

        self._record_collector.add_move(move_data)
        return normalized_uci

    def _cleanup_players(self) -> None:
        """Clean up player resources."""
        if hasattr(self.white_player, "close"):
            try:
                self.white_player.close()
            except Exception as close_error:
                logger.warning("Error closing white player: {}", close_error)

        if hasattr(self.black_player, "close"):
            try:
                self.black_player.close()
            except Exception as close_error:
                logger.warning("Error closing black player: {}", close_error)

        if self.metrics_tracker is not None:
            try:
                self.metrics_tracker.close()
            except Exception as close_error:  # pragma: no cover
                logger.warning("Error closing metrics tracker: {}", close_error)

    def _log_metrics_summary(self) -> None:
        """Display aggregated metrics for each player after the game."""
        if self.metrics_tracker is None:
            return

        summary_by_player_color = self.metrics_tracker.summarize()
        white_summary = summary_by_player_color.get("white")
        black_summary = summary_by_player_color.get("black")

        outcome_summary = build_game_outcome_summary(
            outcome=self.outcome,
            white_player_name=self.white_player.name,
            black_player_name=self.black_player.name,
            total_moves=len(self.board.move_stack),
            termination_label_override=self._termination_label_override,
            termination_note=self._termination_note,
        )

        if self.display_summary:
            rendered = display_game_summary(
                white_player=str(self.white_player),
                black_player=str(self.black_player),
                white_summary=white_summary,
                black_summary=black_summary,
                outcome_summary=outcome_summary,
            )
            self._rendered_metrics_summary = rendered
        else:
            self._rendered_metrics_summary = False

        for player_color, player_metrics_summary in summary_by_player_color.items():
            if player_metrics_summary.moves_evaluated == 0:
                continue

            player = self.white_player if player_color == "white" else self.black_player

            avg_loss = (
                f"{player_metrics_summary.average_centipawn_loss:.1f}"
                if player_metrics_summary.average_centipawn_loss is not None
                else "N/A"
            )
            hit_rate = (
                f"{player_metrics_summary.best_move_hit_rate:.3f}"
                if player_metrics_summary.best_move_hit_rate is not None
                else "N/A"
            )

            logger.debug(
                "Move analysis for {}: average loss {} centipawns, found best move {}% of time, move qualities: {}",
                str(player),
                avg_loss,
                hit_rate,
                self._format_quality_summary(player_metrics_summary.quality_counts),
            )

    def _result_string(self) -> str:
        """PGN result derived from the recorded outcome, "*" while unfinished."""
        outcome = self.outcome
        if outcome is None:
            return "*"
        if outcome.winner is None:
            return "1/2-1/2"
        return "1-0" if outcome.winner == chess.WHITE else "0-1"

    @staticmethod
    def _format_quality_summary(quality_counts: Mapping[MoveQuality, int]) -> str:
        """Format move quality distribution for logging."""
        parts: list[str] = []
        for quality in MOVE_QUALITY_ORDER:
            count = quality_counts.get(quality, 0)
            if count:
                parts.append(f"{quality.value}:{count}")
        return ", ".join(parts) if parts else "none"

    @staticmethod
    def _sanitize_error_message(error_msg: str) -> str:
        """Sanitize error messages to avoid leaking sensitive information.

        Args:
            error_msg: Raw error message that may contain API tokens or keys.

        Returns:
            str: Sanitized error message with tokens removed and length limited.
        """
        # Remove common token patterns (API keys often contain these patterns)
        # Pattern: sk-... or Bearer ... or api_key=...
        patterns = [
            r"sk-[a-zA-Z0-9]{20,}",  # OpenAI-style keys
            r"Bearer\s+[a-zA-Z0-9_\-\.]{20,}",  # Bearer tokens
            r"api[_-]?key[=:]\s*['\"]?[a-zA-Z0-9_\-\.]{20,}['\"]?",  # API key parameters
            r"token[=:]\s*['\"]?[a-zA-Z0-9_\-\.]{20,}['\"]?",  # Token parameters
        ]

        for pattern in patterns:
            error_msg = re.sub(pattern, "[REDACTED]", error_msg, flags=re.IGNORECASE)

        # Truncate only after redaction so a key never straddles the cut point
        max_length = 500
        if len(error_msg) > max_length:
            error_msg = error_msg[:max_length] + "... (truncated)"

        return error_msg

    def _log_llm_usage_summary(self, game_summary: GameSummary | None = None) -> None:
        """Log cumulative LiteLLM usage for each player using central summary builder."""
        resolved_game_summary = (
            game_summary if game_summary is not None else build_game_summary(self)
        )
        resolved_game_summary.log_usage_summary()

    def _reset_llm_usage_counters(self) -> None:
        """Reset usage counters on players that support it before a game."""

        players = (self.white_player, self.black_player)

        for player in players:
            reset_usage = getattr(player, "reset_usage", None)
            if not callable(reset_usage):
                continue

            try:
                reset_usage()
            except Exception as reset_error:  # pragma: no cover
                logger.debug(
                    "Could not reset token usage counters for {}: {}",
                    player,
                    reset_error,
                )

    def _save_history_if_configured(
        self, game_summary: GameSummary | None = None
    ) -> None:
        """Persist the PGN history and JSON record when configuration requests it.

        Records are written for finished games and for interrupted games whose
        termination metadata marks them resumable, so a later resume can pick
        the game up from disk.
        """

        if self._record_dir is None:
            return
        interrupted_but_resumable = bool(
            self._termination_metadata is not None
            and self._termination_metadata.get("resumable")
        )
        if not self.finished and not interrupted_but_resumable:
            return

        filename = self._record_name if self._record_name else self._start_timestamp
        pgn_path = self._record_dir / f"{filename}.pgn"
        json_path = self._record_dir / f"{filename}.json"

        self._record_dir.mkdir(parents=True, exist_ok=True)

        try:
            board_copy = self.board.copy(stack=True)
            pgn_game = chess.pgn.Game.from_board(board_copy)
            pgn_game.headers["Event"] = "LLM Chess Arena"
            pgn_game.headers["Date"] = datetime.now(UTC).strftime("%Y.%m.%d")
            pgn_game.headers["White"] = str(self.white_player)
            pgn_game.headers["Black"] = str(self.black_player)
            # board.result() only knows board-derived endings; resignations and
            # forfeits live in self._outcome, so derive the header from that.
            pgn_game.headers["Result"] = self._result_string()

            exporter = chess.pgn.StringExporter(
                headers=True,
                variations=False,
                comments=False,
            )
            pgn_path.write_text(pgn_game.accept(exporter), encoding="utf-8")
            logger.info("Saved PGN history to {}", pgn_path)
        except Exception as pgn_save_error:  # pragma: no cover
            logger.warning(
                "Could not save chess game history to {}: {}", pgn_path, pgn_save_error
            )

        if self._record_collector is not None:
            try:
                RecordWriter.write(
                    self._record_collector,
                    self._hydra_cfg,
                    json_path,
                    self._initial_fen,
                    self.white_player,
                    self.black_player,
                    game_summary,
                    resumed_from=self._resumed_from,
                    original_termination_metadata=self._original_termination_metadata,
                )
            except Exception as json_save_error:  # pragma: no cover
                logger.warning(
                    "Failed to save game record to {}: {}", json_path, json_save_error
                )
