"""Tests for the move parser component."""

from __future__ import annotations

import chess
import pytest

from llm_chess_arena.exceptions import (
    AmbiguousMoveError,
    IllegalMoveError,
    InvalidMoveError,
)
from llm_chess_arena.player.llm.decision import MoveParser
from llm_chess_arena.types import PlayerDecision


@pytest.fixture()
def parser() -> MoveParser:
    return MoveParser()


def test_validate_and_normalize_succeeds(parser: MoveParser) -> None:
    """Legal UCI moves should pass through unchanged."""
    decision = PlayerDecision(action="move", attempted_move="e2e4")
    normalized = parser.validate_and_normalize(decision, chess.STARTING_FEN)
    assert normalized.attempted_move == "e2e4"


def test_resign_decision_passthrough(parser: MoveParser) -> None:
    """Resign decisions should be returned without modification."""
    decision = PlayerDecision(action="resign")
    assert parser.validate_and_normalize(decision, chess.STARTING_FEN) is decision


def test_missing_move_raises(parser: MoveParser) -> None:
    """Missing attempted move should trigger InvalidMoveError."""
    decision = PlayerDecision.model_construct(action="move", attempted_move=None)
    with pytest.raises(InvalidMoveError):
        parser.validate_and_normalize(decision, chess.STARTING_FEN)


def test_illegal_move_raises(parser: MoveParser) -> None:
    """Illegal UCI moves must raise IllegalMoveError."""
    decision = PlayerDecision(action="move", attempted_move="a1a8")
    with pytest.raises(IllegalMoveError):
        parser.validate_and_normalize(decision, chess.STARTING_FEN)


def test_ambiguous_move_raises(parser: MoveParser) -> None:
    """Ambiguous SAN should produce AmbiguousMoveError."""
    ambiguous_fen = "r1bqkbnr/pppppppp/8/8/2N5/5N2/PPPPPPPP/R1BQKB1R w KQkq - 0 1"
    decision = PlayerDecision(action="move", attempted_move="Ne5")
    with pytest.raises(AmbiguousMoveError):
        parser.validate_and_normalize(decision, ambiguous_fen)
