"""Focused tests for decision DTO validation logic."""

import pytest
from pydantic import ValidationError

from llm_chess_arena.types import PlayerDecision, PlayerDecisionContext


class TestPlayerDecisionContext:
    """Validation scenarios for PlayerDecisionContext."""

    def test_legal_moves_required(self) -> None:
        """The context must include at least one legal move."""
        with pytest.raises(ValidationError) as exc_info:
            PlayerDecisionContext(
                board_in_fen="8/8/8/8/8/8/8/8 w - - 0 1",
                player_color="white",
                legal_moves_in_uci=[],
            )

        assert "cannot be empty" in str(exc_info.value)

    def test_extra_fields_are_preserved(self) -> None:
        """Custom metadata should pass through for downstream consumers."""
        context = PlayerDecisionContext(
            board_in_fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            player_color="white",
            legal_moves_in_uci=["e2e4"],
            evaluation_score=0.42,
        )

        assert context.evaluation_score == 0.42


class TestPlayerDecision:
    """Validation scenarios for PlayerDecision."""

    def test_move_action_requires_attempted_move(self) -> None:
        """Moves must include the text the player attempted."""
        with pytest.raises(ValidationError) as exc_info:
            PlayerDecision(action="move", attempted_move=None)

        assert "required when action='move'" in str(exc_info.value)

    def test_move_action_rejects_empty_text(self) -> None:
        """Empty strings should be rejected like missing moves."""
        with pytest.raises(ValidationError) as exc_info:
            PlayerDecision(action="move", attempted_move="")

        assert "required when action='move'" in str(exc_info.value)

    def test_resign_action_disallows_move_text(self) -> None:
        """Resign decisions cannot carry leftover move text."""
        with pytest.raises(ValidationError) as exc_info:
            PlayerDecision(action="resign", attempted_move="e2e4")

        assert "must be None unless action='move'" in str(exc_info.value)

    def test_extra_fields_are_preserved(self) -> None:
        """Extra metadata should remain available on the decision model."""
        decision = PlayerDecision(
            action="move",
            attempted_move="e2e4",
            confidence=0.9,
        )

        assert decision.confidence == 0.9
