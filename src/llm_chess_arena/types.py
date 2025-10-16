"""Shared Pydantic models and literals for the chess arena."""

from typing import Literal

from typing_extensions import Self
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator

PlayerColor = Literal["white", "black"]
# Currently supported actions
PlayerAction = Literal["move", "resign"]


class PlayerDecisionContext(BaseModel):
    """Context for player decision-making, extracted from board state.

    Allows dynamic field addition for player-specific metadata.
    """

    board_in_fen: str
    player_color: PlayerColor
    legal_moves_in_uci: list[str]
    move_history_in_uci: list[str] = Field(default_factory=list)
    time_remaining_in_sec: float | None = None

    # Dynamic fields allow LLMs/engines to pass custom metadata without schema changes
    model_config = ConfigDict(extra="allow")

    @field_validator("legal_moves_in_uci", mode="after")
    def validate_legal_moves_exist(cls, legal_moves: list[str]) -> list[str]:
        """Ensure legal moves exist - empty list indicates undetected game termination.

        Args:
            legal_moves: Legal moves in UCI format.

        Returns:
            list[str]: Validated legal moves.

        Raises:
            ValueError: If empty (game should detect termination before requesting moves).
        """
        if not legal_moves:
            raise ValueError(
                "`legal_moves_in_uci` cannot be empty. "
                "The game should handle checkmate/stalemate before asking for moves."
            )
        return legal_moves


class PlayerDecision(BaseModel):
    """Player's decision after evaluating context.

    Move required only for action='move', otherwise must be None.

    Common dynamically-added attributes (via extra="allow"):
        thinking_time_in_sec (float): Player-reported thinking time
        response (str): Raw LLM response text (for retry prompts)
        reason (str): Reason for resignation or error
    """

    action: PlayerAction
    attempted_move: str | None = None

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="after")
    def validate_decision_consistency(self) -> Self:
        """Ensure attempted_move presence matches action type.

        Returns:
            Validated instance.

        Raises:
            ValueError: If move presence inconsistent with action.
        """
        if self.action == "move" and not self.attempted_move:
            raise ValueError("`attempted_move` is required when action='move'")
        if self.action != "move" and self.attempted_move is not None:
            raise ValueError("`attempted_move` must be None unless action='move'")
        return self
