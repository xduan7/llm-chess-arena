"""Abstract base class for chess-playing agents."""

from abc import ABC, abstractmethod

import chess

from llm_chess_arena.types import PlayerDecisionContext, PlayerDecision, Color
from llm_chess_arena.utils import get_legal_moves_in_uci, get_move_history_in_uci


class BasePlayer(ABC):
    """Abstract base class that standardizes player decision flow."""

    def __init__(self, name: str, color: Color) -> None:
        """Initialize shared player metadata.

        Args:
            name: Display name used in logs and UI elements.
            color: Chess side this player controls.
        """
        self.name = name
        self.color = color

    def __call__(self, board: chess.Board) -> PlayerDecision:
        """Compute the player's next move using the template method pattern.

        The method extracts a normalized decision context from ``board`` before
        invoking the subclass-specific ``_make_decision`` implementation.

        Args:
            board: Current game state.

        Returns:
            PlayerDecision: Structured decision describing the requested move.
        """
        context = self._extract_context(board)
        decision = self._make_decision(context)
        return decision

    def _extract_context(self, board: chess.Board) -> PlayerDecisionContext:
        """Build the canonical decision context from ``board``."""
        return PlayerDecisionContext(
            board_in_fen=board.fen(),
            player_color=self.color,
            legal_moves_in_uci=get_legal_moves_in_uci(board),
            move_history_in_uci=get_move_history_in_uci(board),
        )

    @abstractmethod
    def _make_decision(self, context: PlayerDecisionContext) -> PlayerDecision:
        """Return a decision derived from the normalized ``context``.

        Args:
            context: Structured snapshot of the current game state.

        Returns:
            PlayerDecision: Decision model describing the desired action.
        """
        raise NotImplementedError

    def close(self) -> None:
        """Release resources for subclasses that manage external state."""
        return None

    def __str__(self) -> str:
        """Return the player name suffixed by its color initial."""
        return f"{self.name} ({self.color[0].upper()})"
