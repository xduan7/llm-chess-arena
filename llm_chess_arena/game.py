import chess
from loguru import logger
from typing import Any

from llm_chess_arena.player.base_player import BasePlayer
from llm_chess_arena.board_display import display_board_with_context
from llm_chess_arena.exceptions import (
    InvalidMoveError,
    IllegalMoveError,
    AmbiguousMoveError,
)
from llm_chess_arena.types import PlayerDecision


class Game:
    """Orchestrates a chess game between two players."""

    def __init__(
        self,
        white_player: BasePlayer,
        black_player: BasePlayer,
        display_board: bool = False,
    ) -> None:
        """Initialize a chess game.

        Args:
            white_player: Player controlling white pieces.
            black_player: Player controlling black pieces.
            display_board: Whether to display the board after each move.

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
        logger.info(f"Game initialized: {white_player} vs {black_player}")
        self._outcome = None

    @property
    def current_player(self) -> BasePlayer:
        """Get the player whose turn it is to move.

        Returns:
            The current player (white or black).
        """
        return (
            self.white_player if self.board.turn == chess.WHITE else self.black_player
        )

    @property
    def outcome(self) -> chess.Outcome | None:
        """Get the outcome of the game if finished.

        Returns:
            The outcome object if the game is over, else None.
        """
        if self._outcome is None and self.board.is_game_over():
            self._outcome = self.board.outcome()
        return self._outcome

    @property
    def finished(self) -> bool:
        """Check if the game is finished.

        Returns:
            True if the game is over, False otherwise.
        """
        return self.outcome is not None

    @property
    def winner(self) -> BasePlayer | None:
        """Get the winner of the game if finished.

        Returns:
            The winning player, or None if it's a draw or game is not over.
        """
        if not self.finished or self.outcome is None:
            return None

        winner = self.outcome.winner  # outcome is never None if game is over
        if winner == chess.WHITE:
            return self.white_player
        elif winner == chess.BLACK:
            return self.black_player
        else:
            return None

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
        logger.info(f"{self.current_player} resigns")

    def _handle_move(self, decision: PlayerDecision) -> None:
        """Validate and apply a move to the board.

        Args:
            decision: Player's decision containing the move.

        Raises:
            InvalidMoveError: If move format is invalid or move is missing.
            IllegalMoveError: If move is illegal in current position.
        """
        if decision.attempted_move is None:
            raise InvalidMoveError("Move action requires attempted_move")

        from llm_chess_arena.utils import parse_attempted_move_to_uci

        uci_move = parse_attempted_move_to_uci(
            decision.attempted_move, self.board.fen()
        )

        move = chess.Move.from_uci(uci_move)
        logger.debug(f"{self.current_player} plays: {uci_move}")
        self.board.push(move)

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
                    logger.info(f"Stopping: Maximum moves ({max_num_moves}) reached")
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
                            move_count=(len(self.board.move_stack) + 1) // 2,
                            last_move=current_move,
                        )
                except (
                    IllegalMoveError,
                    InvalidMoveError,
                    AmbiguousMoveError,
                ) as e:
                    logger.warning(
                        f"Game over due to {e.__class__.__name__} by {self.current_player}: {e}"
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
                        f"Unexpected error during player move by {self.current_player}: {e}"
                    )
                    raise

            if self.outcome:
                logger.info(f"Game finished after {len(self.board.move_stack)} moves")
                logger.info(
                    f"Winner: {self.winner}" if self.winner else "Game ended in a draw"
                )
        finally:
            # Clean up Stockfish subprocess and LLM connections
            self._cleanup_players()

    def _cleanup_players(self) -> None:
        """Clean up player resources."""
        if hasattr(self.white_player, "close"):
            try:
                self.white_player.close()
            except Exception as e:
                logger.warning(f"Error closing white player: {e}")

        if hasattr(self.black_player, "close"):
            try:
                self.black_player.close()
            except Exception as e:
                logger.warning(f"Error closing black player: {e}")

    def __enter__(self) -> "Game":
        """Context manager entry.

        Returns:
            Self for use in with statement.
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
