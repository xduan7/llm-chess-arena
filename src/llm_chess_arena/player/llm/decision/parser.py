"""Move validation helper for LLM decisions."""

from __future__ import annotations

from llm_chess_arena.exceptions import InvalidMoveError
from llm_chess_arena.types import PlayerDecision
from llm_chess_arena.utils import parse_attempted_move_to_uci


class MoveParser:
    """Validate and normalize moves returned by the LLM handler."""

    def validate_and_normalize(
        self, decision: PlayerDecision, board_in_fen: str
    ) -> PlayerDecision:
        """Validate and normalize a player decision for execution.

        Args:
            decision: Player decision to validate and normalize.
            board_in_fen: Board position in FEN format for move validation.

        Returns:
            PlayerDecision: Normalized decision with attempted_move in UCI format
            for move actions, or the original decision for resign actions.

        Raises:
            NotImplementedError: If the decision action is not supported.
            InvalidMoveError: If move decision is missing attempted_move text.
        """
        if decision.action == "resign":
            return decision
        if decision.action != "move":
            raise NotImplementedError(
                f"LLM currently only supports 'move' and 'resign' actions, got '{decision.action}'"
            )
        if decision.attempted_move is None:
            raise InvalidMoveError("LLM move decision missing attempted_move text")

        move_uci = parse_attempted_move_to_uci(decision.attempted_move, board_in_fen)
        return decision.model_copy(update={"attempted_move": move_uci})
