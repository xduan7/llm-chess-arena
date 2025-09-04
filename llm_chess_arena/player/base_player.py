from abc import ABC, abstractmethod

import chess

from llm_chess_arena.types import PlayerDecisionContext, PlayerDecision, Color
from llm_chess_arena.utils import get_legal_moves_in_uci, get_move_history_in_uci


class BasePlayer(ABC):
    """Abstract base class for all chess players."""

    def __init__(self, name: str, color: Color) -> None:
        """Initialize player with name and color.

        Args:
            name: Display name.
            color: 'white' or 'black'.
        """
        self.name = name
        self.color = color

    def __call__(self, board: chess.Board) -> PlayerDecision:
        """Get decision for current position using template method pattern.

        This orchestrates the decision process: first extracting a standardized
        context from the board, then calling the abstract _make_decision method
        that subclasses implement.

        Args:
            board: Current board state.

        Returns:
            Player's decision.
        """
        context = self._extract_context(board)
        decision = self._make_decision(context)
        return decision

    def _extract_context(self, board: chess.Board) -> PlayerDecisionContext:
        """Extract decision context from board.

        Subclasses can override to add custom fields.

        Args:
            board: Current board state.

        Returns:
            Context with FEN, legal moves, and history.
        """
        context = PlayerDecisionContext(
            board_in_fen=board.fen(),
            player_color=self.color,
            legal_moves_in_uci=get_legal_moves_in_uci(board),
            move_history_in_uci=get_move_history_in_uci(board),
        )
        return context

    @abstractmethod
    def _make_decision(self, context: PlayerDecisionContext) -> PlayerDecision:
        """Make decision based on context.

        Args:
            context: Game context.

        Returns:
            Player's decision.
        """
        pass

    def close(self) -> None:
        """Clean up resources if needed."""
        pass

    def get_move(self, board: chess.Board, **kwargs) -> chess.Move:
        """DEPRECATED: Temporary backward-compatibility wrapper.

        This method exists only to make tests pass during migration.
        Will be removed once all tests are updated to use __call__().

        Args:
            board: Current board state.
            **kwargs: Ignored for compatibility.

        Returns:
            chess.Move object.
        """
        import warnings

        warnings.warn(
            "get_move() is deprecated, use player(board) instead",
            DeprecationWarning,
            stacklevel=2,
        )
        decision = self(board)
        if decision.action == "move":
            return chess.Move.from_uci(decision.attempted_move)
        else:
            # For non-move actions like resign, return None or raise
            raise RuntimeError(f"Player decided to {decision.action}, not move")

    def __str__(self) -> str:
        """Format as 'Name (W)' or 'Name (B)'."""
        return f"{self.name} ({self.color[0].upper()})"
