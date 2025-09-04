"""Unit tests for DTO types (PlayerDecisionContext and PlayerDecision)."""

import pytest
from pydantic import ValidationError

from llm_chess_arena.types import PlayerDecisionContext, PlayerDecision


class TestPlayerDecisionContext:
    """Tests for PlayerDecisionContext DTO."""

    def test_valid_context_creation(self):
        """Test creating a valid context with all required fields."""
        context = PlayerDecisionContext(
            board_in_fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            player_color="white",
            legal_moves_in_uci=["e2e4", "d2d4", "g1f3"],
            move_history_in_uci=[],
        )

        assert (
            context.board_in_fen
            == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        )
        assert context.player_color == "white"
        assert len(context.legal_moves_in_uci) == 3
        assert context.move_history_in_uci == []
        assert context.time_remaining_in_seconds is None

    def test_context_with_optional_fields(self):
        """Test context with optional time_remaining field."""
        context = PlayerDecisionContext(
            board_in_fen="8/8/8/8/8/8/8/8 w - - 0 1",
            player_color="black",
            legal_moves_in_uci=["a7a6"],
            move_history_in_uci=["e2e4", "e7e5"],
            time_remaining_in_seconds=300.5,
        )

        assert context.time_remaining_in_seconds == 300.5
        assert context.move_history_in_uci == ["e2e4", "e7e5"]

    def test_context_with_extra_fields(self):
        """Test that extra fields are allowed for extensibility."""
        context = PlayerDecisionContext(
            board_in_fen="8/8/8/8/8/8/8/8 w - - 0 1",
            player_color="white",
            legal_moves_in_uci=["a2a3"],
            custom_field="custom_value",
            engine_evaluation=0.5,
        )

        assert context.custom_field == "custom_value"
        assert context.engine_evaluation == 0.5

    def test_empty_legal_moves_raises_error(self):
        """Test that empty legal_moves_in_uci raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            PlayerDecisionContext(
                board_in_fen="8/8/8/8/8/8/8/8 w - - 0 1",
                player_color="white",
                legal_moves_in_uci=[],  # Empty list should raise error
                move_history_in_uci=[],
            )

        errors = exc_info.value.errors()
        assert any("`legal_moves_in_uci` cannot be empty" in str(e) for e in errors)

    def test_invalid_player_color(self):
        """Test that invalid player color raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            PlayerDecisionContext(
                board_in_fen="8/8/8/8/8/8/8/8 w - - 0 1",
                player_color="red",  # Invalid color
                legal_moves_in_uci=["a2a3"],
                move_history_in_uci=[],
            )

        errors = exc_info.value.errors()
        assert any("Input should be 'white' or 'black'" in str(e) for e in errors)

    def test_missing_required_fields(self):
        """Test that missing required fields raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            PlayerDecisionContext(
                board_in_fen="8/8/8/8/8/8/8/8 w - - 0 1",
                # Missing player_color and legal_moves_in_uci
            )

        errors = exc_info.value.errors()
        assert len(errors) >= 2
        field_names = [e["loc"][0] for e in errors]
        assert "player_color" in field_names
        assert "legal_moves_in_uci" in field_names


class TestPlayerDecision:
    """Tests for PlayerDecision DTO."""

    def test_valid_move_decision(self):
        """Test creating a valid move decision."""
        decision = PlayerDecision(action="move", attempted_move="e2e4")

        assert decision.action == "move"
        assert decision.attempted_move == "e2e4"

    def test_valid_resign_decision(self):
        """Test creating a valid resign decision."""
        decision = PlayerDecision(action="resign", attempted_move=None)

        assert decision.action == "resign"
        assert decision.attempted_move is None

    def test_decision_with_extra_fields(self):
        """Test that extra fields are allowed for extensibility."""
        decision = PlayerDecision(
            action="move",
            attempted_move="g1f3",
            confidence_score=0.95,
            explanation="Developing the knight",
        )

        assert decision.confidence_score == 0.95
        assert decision.explanation == "Developing the knight"

    def test_move_action_without_attempted_move_raises_error(self):
        """Test that action='move' without attempted_move raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            PlayerDecision(
                action="move",
                attempted_move=None,  # Should not be None for move action
            )

        errors = exc_info.value.errors()
        assert any(
            "`attempted_move` is required when action='move'" in str(e) for e in errors
        )

    def test_resign_action_with_attempted_move_raises_error(self):
        """Test that action='resign' with attempted_move raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            PlayerDecision(
                action="resign",
                attempted_move="e2e4",  # Should be None for resign action
            )

        errors = exc_info.value.errors()
        assert any(
            "`attempted_move` must be None unless action='move'" in str(e)
            for e in errors
        )

    def test_invalid_action_type(self):
        """Test that invalid action type raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            PlayerDecision(
                action="draw",  # Not a valid action yet
                attempted_move=None,
            )

        errors = exc_info.value.errors()
        assert any("Input should be 'move' or 'resign'" in str(e) for e in errors)

    def test_empty_string_attempted_move_with_move_action(self):
        """Test that empty string for attempted_move with action='move' raises error."""
        with pytest.raises(ValidationError) as exc_info:
            PlayerDecision(
                action="move",
                attempted_move="",  # Empty string should fail validation
            )

        errors = exc_info.value.errors()
        assert any(
            "`attempted_move` is required when action='move'" in str(e) for e in errors
        )


class TestDTOIntegration:
    """Tests for DTO integration scenarios."""

    def test_context_to_decision_flow(self):
        """Test typical flow from context to decision."""
        # Create context
        context = PlayerDecisionContext(
            board_in_fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            player_color="white",
            legal_moves_in_uci=["e2e4", "d2d4", "g1f3"],
            move_history_in_uci=[],
        )

        # Simulate player decision based on context
        decision = PlayerDecision(
            action="move",
            attempted_move=context.legal_moves_in_uci[0],  # Pick first legal move
        )

        assert decision.attempted_move in context.legal_moves_in_uci

    def test_serialization_deserialization(self):
        """Test that DTOs can be serialized and deserialized."""
        context = PlayerDecisionContext(
            board_in_fen="8/8/8/8/8/8/8/8 w - - 0 1",
            player_color="white",
            legal_moves_in_uci=["a2a3"],
            custom_field="test",
        )

        # Serialize to dict
        context_dict = context.model_dump()
        assert context_dict["custom_field"] == "test"

        # Deserialize back
        context_restored = PlayerDecisionContext(**context_dict)
        assert context_restored.custom_field == "test"
        assert context_restored == context

    def test_json_serialization(self):
        """Test JSON serialization of DTOs."""
        decision = PlayerDecision(
            action="move",
            attempted_move="e2e4",
            metadata={"engine": "stockfish", "depth": 10},
        )

        # Serialize to JSON
        json_str = decision.model_dump_json()
        assert "e2e4" in json_str
        assert "stockfish" in json_str

        # Deserialize from JSON
        decision_restored = PlayerDecision.model_validate_json(json_str)
        assert decision_restored.attempted_move == "e2e4"
        assert decision_restored.metadata["engine"] == "stockfish"
