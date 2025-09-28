"""Core game loop coordinating chess players and board state."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Mapping
import time

import chess
import chess.pgn
from loguru import logger

from llm_chess_arena.exceptions import (
    AmbiguousMoveError,
    IllegalMoveError,
    InvalidMoveError,
)
from llm_chess_arena.player.base_player import BasePlayer
from llm_chess_arena.renderer import display_board_with_context, display_game_summary
from llm_chess_arena.types import PlayerDecision
from llm_chess_arena.metrics import MOVE_QUALITY_ORDER, MetricsTracker, MoveQuality
from llm_chess_arena.utils import (
    build_game_outcome_summary,
    parse_attempted_move_to_uci,
)
from llm_chess_arena.record import RecordCollector, RecordWriter, iso_timestamp


class Game:
    """Orchestrates a chess game between two players."""

    def __init__(
        self,
        white_player: BasePlayer,
        black_player: BasePlayer,
        display_board: bool = False,
        enable_metrics: bool = True,
        metrics_tracker: MetricsTracker | None = None,
        record_dir: str | Path | None = None,
        record_name: str | None = None,
        hydra_config: Dict[str, Any] | None = None,
    ) -> None:
        """Initialize a chess game.

        Args:
            white_player: Player controlling white pieces.
            black_player: Player controlling black pieces.
            display_board: Whether to display the board after each move.
            enable_metrics: Whether to compute move quality metrics.
            metrics_tracker: Optional preconfigured metrics tracker.
            record_dir: Optional directory path for writing game records when
                the game completes. When ``None``, no records are written.
            record_name: Optional custom name for record files. When ``None``,
                uses timestamp-based naming.
            hydra_config: Optional Hydra configuration dict for game records.

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
        self.metrics_tracker = (
            metrics_tracker
            if metrics_tracker is not None
            else (MetricsTracker.from_stockfish() if enable_metrics else None)
        )
        self._move_qualities: list[MoveQuality | None] = []
        self._rendered_metrics_summary = False
        self._record_dir = (
            Path(record_dir).expanduser() if record_dir is not None else None
        )
        self._record_name = record_name
        self._start_timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self._termination_label_override: str | None = None
        self._termination_note: str | None = None
        self._hydra_config = hydra_config or {}

        # Capture the initial FEN for game records
        self._initial_fen = self.board.fen()

        self._record_collector = RecordCollector() if record_dir is not None else None

        # Track thinking time for each player
        self._white_thinking_time = 0.0
        self._black_thinking_time = 0.0

        # Track current win probability for display
        self._current_win_probability: float | None = None

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

    def make_move(self) -> None:
        """Execute a single move in the game.

        Raises:
            InvalidMoveError: If decision has invalid action or missing move.
            Exception: Any exception from player() or from_uci() is propagated.
        """
        # Track thinking time
        start_time = time.time()

        # Copy prevents players from mutating game state
        decision = self.current_player(board=self.board.copy())

        # Calculate and accumulate thinking time
        thinking_time = time.time() - start_time
        if self.current_player.color == "white":
            self._white_thinking_time += thinking_time
        else:
            self._black_thinking_time += thinking_time

        # Also use thinking time from decision if available
        if (
            hasattr(decision, "thinking_time_seconds")
            and decision.thinking_time_seconds is not None
        ):
            if self.current_player.color == "white":
                self._white_thinking_time += decision.thinking_time_seconds
            else:
                self._black_thinking_time += decision.thinking_time_seconds

        if decision.action == "resign":
            # Record resignation before handling it (and before early return)
            self._record_move_if_configured(decision)
            self._handle_resignation(decision)
            return
        elif decision.action == "move":
            # Record move data before applying it (to capture correct player and move number)
            self._record_move_if_configured(decision)
            self._handle_move(decision)
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

    def _handle_move(self, decision: PlayerDecision) -> None:
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

        player = self.current_player
        board_before_move = self.board.copy()

        uci_move = parse_attempted_move_to_uci(
            decision.attempted_move, self.board.fen()
        )

        move = chess.Move.from_uci(uci_move)
        move_number = (len(self.board.move_stack) // 2) + 1
        turn_indicator = "." if player.color == "white" else "..."
        logger.info(
            "Move {}{} {} plays {}", move_number, turn_indicator, player.name, uci_move
        )
        self.board.push(move)

        move_quality: MoveQuality | None = None
        if self.metrics_tracker is not None:
            try:
                metrics = self.metrics_tracker.record_move(
                    board_before_move,
                    move,
                )
                if metrics is not None:
                    move_quality = metrics.quality

                    # Update win probability from metrics (from white's perspective)
                    if (
                        hasattr(metrics, "actual_centipawns")
                        and metrics.actual_centipawns is not None
                    ):
                        # Convert centipawns to approximate win probability
                        cp = metrics.actual_centipawns
                        # Use tanh function to convert centipawns to win probability
                        # This is a rough approximation - Stockfish WDL would be more accurate
                        self._current_win_probability = 0.5 + 0.5 * (cp / 100.0) / (
                            1.0 + abs(cp / 100.0)
                        )
                        # Clamp to [0, 1]
                        self._current_win_probability = max(
                            0.0, min(1.0, self._current_win_probability)
                        )

                    # Add stockfish evaluation to the last recorded move if collector is active
                    if self._record_collector is not None:
                        stockfish_eval: Dict[str, Any] = {
                            "stockfish_evaluation": {
                                "quality": move_quality.value,
                            }
                        }
                        # Add additional metrics if available
                        if (
                            hasattr(metrics, "centipawn_loss")
                            and metrics.centipawn_loss is not None
                        ):
                            stockfish_eval["stockfish_evaluation"][
                                "centipawn_loss"
                            ] = metrics.centipawn_loss
                        if (
                            hasattr(metrics, "best_move_uci")
                            and metrics.best_move_uci is not None
                        ):
                            stockfish_eval["stockfish_evaluation"][
                                "best_move_uci"
                            ] = metrics.best_move_uci
                        if (
                            hasattr(metrics, "best_move_hit")
                            and metrics.best_move_hit is not None
                        ):
                            stockfish_eval["stockfish_evaluation"][
                                "best_move_hit"
                            ] = metrics.best_move_hit

                        self._record_collector.update_last_move(stockfish_eval)

            except Exception as exc:  # pragma: no cover - safeguards metrics path
                logger.warning(
                    "Failed to record metrics for move {}: {}", uci_move, exc
                )
        self._move_qualities.append(move_quality)

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

        # Record game start timestamp
        if self._record_collector is not None:
            self._record_collector.set_start_timestamp(iso_timestamp(datetime.now(UTC)))

        self._reset_llm_usage_counters()
        try:
            num_moves = 0
            while not self.finished:
                if max_num_moves is not None and num_moves >= max_num_moves:
                    logger.info("Stopping: Maximum moves ({}) reached", max_num_moves)
                    self._outcome = chess.Outcome(
                        termination=chess.Termination.VARIANT_DRAW,  # Draw by max moves
                        winner=None,
                    )
                    break

                try:
                    self.make_move()
                    num_moves += 1

                    if self.display_board:
                        current_move = (
                            self.board.peek() if self.board.move_stack else None
                        )
                        display_board_with_context(
                            self.board,
                            current_player=self.current_player.name,
                            last_move=current_move,
                            white_player=str(self.white_player),
                            black_player=str(self.black_player),
                            move_qualities=self._move_qualities,
                            white_thinking_time=self._white_thinking_time,
                            black_thinking_time=self._black_thinking_time,
                            white_win_probability=self._current_win_probability,
                        )
                except (
                    IllegalMoveError,
                    InvalidMoveError,
                    AmbiguousMoveError,
                ) as e:
                    logger.warning(
                        "Game over due to {} by {}: {}",
                        e.__class__.__name__,
                        self.current_player,
                        e,
                    )
                    self._outcome = chess.Outcome(
                        termination=chess.Termination.VARIANT_LOSS,  # Loss due to illegal/invalid move
                        winner=(
                            chess.BLACK
                            if self.current_player.color == "white"
                            else chess.WHITE
                        ),
                    )
                    break
                except Exception as e:
                    logger.exception(
                        "Unexpected error during player move by {}: {}",
                        self.current_player,
                        e,
                    )
                    raise

            if self.outcome:
                logger.info("Game finished after {} moves", len(self.board.move_stack))
                if self.winner:
                    logger.info("Winner: {}", self.winner)
                else:
                    logger.info("Game ended in a draw")
        finally:
            # Set end timestamp and save record if configured
            if self._record_collector is not None:
                self._record_collector.set_end_timestamp(
                    iso_timestamp(datetime.now(UTC))
                )
                self._record_collector.set_outcome(self.outcome)
                self._record_collector.set_termination_label_override(
                    self._termination_label_override
                )

            self._log_llm_usage_summary()
            self._save_history_if_configured()
            if self.metrics_tracker is not None:
                self._log_metrics_summary()
            self._cleanup_players()

    def _record_move_if_configured(self, decision: PlayerDecision) -> None:
        """Record move data to the record collector if active.

        Args:
            decision: The player decision about to be executed.
        """
        if self._record_collector is None:
            return

        # Calculate move number as ply count (half-moves: 1, 2, 3, ...)
        # Since we're recording BEFORE the move is applied, use move stack length + 1
        move_number = len(self.board.move_stack) + 1

        # Current player is the one making the move (since we record before applying)
        player = self.current_player

        # Current board state (before the move)
        position_before_fen = self.board.fen()

        # For resignations, we won't have position_after since move isn't applied
        position_after_fen = None
        if decision.action == "move" and decision.attempted_move:
            # Apply the move temporarily to get the position after
            try:
                move = chess.Move.from_uci(decision.attempted_move)
                if move in self.board.legal_moves:
                    board_copy = self.board.copy()
                    board_copy.push(move)
                    position_after_fen = board_copy.fen()
            except (ValueError, chess.InvalidMoveError):
                # Invalid move, position_after will remain None
                pass

        move_data: Dict[str, Any] = {
            "move_number": move_number,
            "player": player.color,
            "timestamp": iso_timestamp(datetime.now(UTC)),
            "position_before": {
                "fen": position_before_fen,
            },
            "final_decision": {
                "action": decision.action,
            },
        }

        # Add position_after only for valid moves
        if position_after_fen is not None:
            move_data["position_after"] = {
                "fen": position_after_fen,
            }

        # Add move-specific data
        if decision.action == "move" and decision.attempted_move:
            move_data["final_decision"]["move_uci"] = decision.attempted_move
        elif decision.action == "resign":
            move_data["final_decision"]["resignation_reason"] = getattr(
                decision, "reason", "Player resigned"
            )

        # Add LLM decision process if available
        if hasattr(decision, "llm_decision_process"):
            move_data["llm_decision_process"] = decision.llm_decision_process

        # Add thinking time if available
        if (
            hasattr(decision, "thinking_time_seconds")
            and decision.thinking_time_seconds is not None
        ):
            move_data["thinking_time_seconds"] = decision.thinking_time_seconds

        # Add stockfish evaluation if available and this was a move
        # Note: Since we're recording before the move is applied, we can't access
        # the stockfish evaluation yet. This would need to be added after move evaluation.

        self._record_collector.add_move(move_data)

    def _cleanup_players(self) -> None:
        """Clean up player resources."""
        # Player adapters expose ``close`` optionally; guard each call accordingly.
        if hasattr(self.white_player, "close"):
            try:
                self.white_player.close()
            except Exception as e:
                logger.warning("Error closing white player: {}", e)

        if hasattr(self.black_player, "close"):
            try:
                self.black_player.close()
            except Exception as e:
                logger.warning("Error closing black player: {}", e)

        if self.metrics_tracker is not None:
            try:
                self.metrics_tracker.close()
            except Exception as e:  # pragma: no cover - defensive cleanup
                logger.warning("Error closing metrics tracker: {}", e)

    def _log_metrics_summary(self) -> None:
        """Display aggregated metrics for each player after the game."""
        if self.metrics_tracker is None:
            return

        summaries = self.metrics_tracker.summarize()
        white_summary = summaries.get("white")
        black_summary = summaries.get("black")

        outcome_summary = build_game_outcome_summary(
            outcome=self.outcome,
            white_player_name=str(self.white_player),
            black_player_name=str(self.black_player),
            total_moves=len(self.board.move_stack),
            termination_label_override=self._termination_label_override,
            termination_note=self._termination_note,
        )

        rendered = display_game_summary(
            white_player=str(self.white_player),
            black_player=str(self.black_player),
            white_summary=white_summary,
            black_summary=black_summary,
            outcome_summary=outcome_summary,
        )
        self._rendered_metrics_summary = rendered

        # Still log for debugging/records
        for color, summary in summaries.items():
            if summary.moves_evaluated == 0:
                continue

            player = self.white_player if color == "white" else self.black_player

            avg_loss = (
                f"{summary.average_centipawn_loss:.1f}"
                if summary.average_centipawn_loss is not None
                else "N/A"
            )
            hit_rate = (
                f"{summary.best_move_hit_rate:.3f}"
                if summary.best_move_hit_rate is not None
                else "N/A"
            )

            logger.debug(
                "Metrics for {}: avg_centipawn_loss={}, best_move_hit_rate={}, qualities={}",
                str(player),
                avg_loss,
                hit_rate,
                self._format_quality_summary(summary.quality_counts),
            )

    @staticmethod
    def _format_quality_summary(quality_counts: Mapping[MoveQuality, int]) -> str:
        """Format move quality distribution for logging."""
        parts: list[str] = []
        for quality in MOVE_QUALITY_ORDER:
            count = quality_counts.get(quality, 0)
            if count:
                parts.append(f"{quality.value}:{count}")
        return ", ".join(parts) if parts else "none"

    def _log_llm_usage_summary(self) -> None:
        """Log cumulative LiteLLM usage for each player if available."""

        players = (self.white_player, self.black_player)

        for player in players:
            get_usage = getattr(player, "get_usage_totals", None)
            if not callable(get_usage):
                continue

            try:
                usage = get_usage()
            except Exception as exc:  # pragma: no cover - guard optional hook
                logger.debug("Failed to retrieve usage totals for {}: {}", player, exc)
                continue

            if usage is None:
                continue

            logger.info(
                "LLM player {} total usage: prompt_tokens={}, completion_tokens={}, "
                "total_tokens={}, cost=${:.6f}",
                player,
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.total_tokens,
                usage.cost,
            )

    def _reset_llm_usage_counters(self) -> None:
        """Reset usage counters on players that support it before a game."""

        players = (self.white_player, self.black_player)

        for player in players:
            reset_usage = getattr(player, "reset_usage", None)
            if not callable(reset_usage):
                continue

            try:
                reset_usage()
            except Exception as exc:  # pragma: no cover - defensive hook
                logger.debug("Failed to reset usage for {}: {}", player, exc)

    def _save_history_if_configured(self) -> None:
        """Persist the PGN history and JSON record when configuration requests it."""

        if self._record_dir is None:
            return
        if not self.finished:
            return

        # Generate filename
        filename = self._record_name if self._record_name else self._start_timestamp
        pgn_path = self._record_dir / f"{filename}.pgn"
        json_path = self._record_dir / f"{filename}.json"

        # Ensure directory exists
        self._record_dir.mkdir(parents=True, exist_ok=True)

        # Save PGN history
        try:
            board_copy = self.board.copy(stack=True)
            pgn_game = chess.pgn.Game.from_board(board_copy)
            pgn_game.headers["Event"] = "LLM Chess Arena"
            pgn_game.headers["Date"] = datetime.now(UTC).strftime("%Y.%m.%d")
            pgn_game.headers["White"] = str(self.white_player)
            pgn_game.headers["Black"] = str(self.black_player)
            pgn_game.headers["Result"] = board_copy.result(claim_draw=True)

            exporter = chess.pgn.StringExporter(
                headers=True,
                variations=False,
                comments=False,
            )
            pgn_path.write_text(pgn_game.accept(exporter), encoding="utf-8")
            logger.info("Saved PGN history to {}", pgn_path)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Failed to save PGN history to {}: {}", pgn_path, exc)

        # Save JSON record if collector is active
        if self._record_collector is not None:
            try:
                RecordWriter.write(
                    self._record_collector,
                    self._hydra_config,
                    json_path,
                    self._initial_fen,
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Failed to save game record to {}: {}", json_path, exc)
