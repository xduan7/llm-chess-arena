"""Core game loop coordinating chess players and board state."""

from __future__ import annotations

from typing import Any, Mapping

import chess
from loguru import logger

from llm_chess_arena.exceptions import (
    AmbiguousMoveError,
    IllegalMoveError,
    InvalidMoveError,
)
from llm_chess_arena.player.base_player import BasePlayer
from llm_chess_arena.renderer import display_board_with_context
from llm_chess_arena.types import PlayerDecision
from llm_chess_arena.metrics import MOVE_QUALITY_ORDER, MetricsTracker, MoveQuality
from llm_chess_arena.utils import parse_attempted_move_to_uci


class Game:
    """Orchestrates a chess game between two players."""

    def __init__(
        self,
        white_player: BasePlayer,
        black_player: BasePlayer,
        display_board: bool = False,
        enable_metrics: bool = True,
        metrics_tracker: MetricsTracker | None = None,
    ) -> None:
        """Initialize a chess game.

        Args:
            white_player: Player controlling white pieces.
            black_player: Player controlling black pieces.
            display_board: Whether to display the board after each move.
            enable_metrics: Whether to compute move quality metrics.
            metrics_tracker: Optional preconfigured metrics tracker.

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

        winner_color = self.outcome.winner  # outcome is never None if game is over
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
        # Copy prevents players from mutating game state
        decision = self.current_player(board=self.board.copy())

        if decision.action == "resign":
            self._handle_resignation()
            return
        elif decision.action == "move":
            self._handle_move(decision)
        else:
            raise InvalidMoveError(f"Unsupported action: {decision.action}")

    def _handle_resignation(self) -> None:
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
        logger.debug("{} plays: {}", player, uci_move)
        self.board.push(move)

        if self.metrics_tracker is not None:
            try:
                self.metrics_tracker.record_move(
                    board_before_move,
                    move,
                    player_name=player.name,
                )
            except Exception as exc:  # pragma: no cover - safeguards metrics path
                logger.warning(
                    "Failed to record metrics for move {}: {}", uci_move, exc
                )

    def play(self, max_num_moves: int | None = None) -> None:
        """Run the game until completion, illegal move, or max moves reached.

        Args:
            max_num_moves: Maximum number of moves (half-moves) before stopping.
                          None means play until a game outcome is reached.

        Note:
            Illegal moves cause the offending player to forfeit.
            Other exceptions are logged and re-raised.
        """
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

                    # Display board after move if requested
                    if self.display_board:
                        current_move = (
                            self.board.peek() if self.board.move_stack else None
                        )
                        display_board_with_context(
                            self.board,
                            current_player=self.current_player.name,
                            move_count=self.board.fullmove_number,
                            last_move=current_move,
                            white_player=self.white_player.name,
                            black_player=self.black_player.name,
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
            if self.metrics_tracker is not None:
                self._log_metrics_summary()
            # Clean up Stockfish subprocess and LLM connections
            self._cleanup_players()

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
        """Log aggregated metrics for each player after the game."""
        if self.metrics_tracker is None:
            return

        summaries = self.metrics_tracker.summarize()
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

            logger.info(
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

    def __enter__(self) -> Game:
        """Context manager entry.

        Returns:
            Game: Self reference for use in with statements.
        """
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - ensures cleanup.

        Args:
            exc_type: Exception type if an exception occurred.
            exc_val: Exception value if an exception occurred.
            exc_tb: Exception traceback if an exception occurred.
        """
        self._cleanup_players()
