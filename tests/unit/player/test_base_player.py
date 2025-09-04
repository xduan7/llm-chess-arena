"""Unit tests for BasePlayer abstract class and its concrete methods."""

import pytest
import chess

from llm_chess_arena.player.base_player import BasePlayer
from llm_chess_arena.types import PlayerDecision, PlayerDecisionContext


class ConcretePlayer(BasePlayer):
    """Minimal concrete implementation for testing BasePlayer."""

    def _make_decision(self, context: PlayerDecisionContext) -> PlayerDecision:
        """Simple implementation that returns first legal move."""
        if context.legal_moves_in_uci:
            return PlayerDecision(
                action="move", attempted_move=context.legal_moves_in_uci[0]
            )
        return PlayerDecision(action="resign")


class TestBasePlayerExtractContext:
    """Tests for BasePlayer._extract_context method."""

    def test_extract_context__given_initial_board__when_called__then_populates_correctly(
        self,
    ):
        """Test context extraction from initial chess position."""
        player = ConcretePlayer(name="Test", color="white")
        board = chess.Board()

        context = player._extract_context(board)

        # Verify all fields are populated correctly
        assert context.board_in_fen == board.fen()
        assert context.player_color == "white"
        assert len(context.legal_moves_in_uci) == 20  # Initial white moves
        assert "e2e4" in context.legal_moves_in_uci
        assert "g1f3" in context.legal_moves_in_uci
        assert context.move_history_in_uci == []
        assert context.time_remaining_in_seconds is None

    def test_extract_context__given_board_with_history__when_called__then_includes_moves(
        self,
    ):
        """Test that move history is correctly extracted."""
        player = ConcretePlayer(name="Test", color="black")
        board = chess.Board()

        # Make some moves
        board.push_san("e4")
        board.push_san("e5")
        board.push_san("Nf3")

        context = player._extract_context(board)

        assert context.player_color == "black"
        assert context.move_history_in_uci == ["e2e4", "e7e5", "g1f3"]
        assert len(context.legal_moves_in_uci) > 0

    def test_extract_context__given_checkmate_position__when_called__then_raises_validation_error(
        self,
    ):
        """Test that context extraction raises error when player is checkmated."""
        player = ConcretePlayer(name="Test", color="black")
        # Fool's mate position (black is checkmated)
        board = chess.Board(
            "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2"
        )
        board.push_san("Qh4#")

        # Should raise validation error as legal_moves_in_uci cannot be empty
        with pytest.raises(ValueError, match="legal_moves_in_uci.*cannot be empty"):
            player._extract_context(board)

    def test_extract_context__given_stalemate_position__when_called__then_raises_validation_error(
        self,
    ):
        """Test that context extraction raises error in stalemate position."""
        player = ConcretePlayer(name="Test", color="black")
        # Stalemate position
        board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")

        # Should raise validation error as legal_moves_in_uci cannot be empty
        with pytest.raises(ValueError, match="legal_moves_in_uci.*cannot be empty"):
            player._extract_context(board)

    def test_extract_context__given_promotion_moves__when_called__then_includes_all_promotions(
        self,
    ):
        """Test that pawn promotion moves are included in legal moves."""
        player = ConcretePlayer(name="Test", color="white")
        # White pawn ready to promote
        board = chess.Board("8/P7/8/8/8/8/8/8 w - - 0 1")

        context = player._extract_context(board)

        # Should include all promotion options
        assert "a7a8q" in context.legal_moves_in_uci  # Queen
        assert "a7a8r" in context.legal_moves_in_uci  # Rook
        assert "a7a8b" in context.legal_moves_in_uci  # Bishop
        assert "a7a8n" in context.legal_moves_in_uci  # Knight

    def test_extract_context__given_en_passant__when_available__then_included_in_moves(
        self,
    ):
        """Test that en passant captures are included in legal moves."""
        player = ConcretePlayer(name="Test", color="white")
        # Position where en passant is possible
        board = chess.Board(
            "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"
        )

        context = player._extract_context(board)

        # En passant capture should be available
        assert "e5f6" in context.legal_moves_in_uci

    def test_extract_context__given_castling_rights__when_available__then_included(
        self,
    ):
        """Test that castling moves are included when available."""
        player = ConcretePlayer(name="Test", color="white")
        # Position with castling available
        board = chess.Board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")

        context = player._extract_context(board)

        # Both castling moves should be available
        assert "e1g1" in context.legal_moves_in_uci  # Kingside
        assert "e1c1" in context.legal_moves_in_uci  # Queenside

    def test_extract_context__given_complex_position__when_called__then_correct_legal_moves(
        self,
    ):
        """Test context extraction from a complex middlegame position."""
        player = ConcretePlayer(name="Test", color="white")
        # Complex position from actual game
        board = chess.Board(
            "r1bqk2r/pp1nbppp/2p1pn2/3p4/2PP4/2N1PN2/PP2BPPP/R1BQKR2 w Qkq - 0 8"
        )

        context = player._extract_context(board)

        # Verify context is populated
        assert context.board_in_fen == board.fen()
        assert len(context.legal_moves_in_uci) > 20  # Many moves available
        assert context.player_color == "white"


class TestBasePlayerCallMethod:
    """Tests for BasePlayer.__call__ method."""

    def test_call__given_valid_board__when_invoked__then_returns_decision(self):
        """Test that calling player as function returns PlayerDecision."""
        player = ConcretePlayer(name="Test", color="white")
        board = chess.Board()

        decision = player(board)

        assert isinstance(decision, PlayerDecision)
        assert decision.action == "move"
        # Our ConcretePlayer returns the first legal move
        legal_moves = list(board.legal_moves)
        assert decision.attempted_move == legal_moves[0].uci()

    def test_call__given_checkmate__when_invoked__then_raises_error(self):
        """Test that calling player when checkmated raises error."""
        player = ConcretePlayer(name="Test", color="black")
        # Black is checkmated
        board = chess.Board(
            "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 0 3"
        )

        # Should raise due to empty legal moves validation
        with pytest.raises(ValueError, match="legal_moves_in_uci.*cannot be empty"):
            player(board)

    def test_call__preserves_board_state__when_invoked__then_board_unchanged(self):
        """Test that calling player doesn't modify the board state."""
        player = ConcretePlayer(name="Test", color="white")
        board = chess.Board()
        initial_fen = board.fen()

        player(board)

        # Board should be unchanged
        assert board.fen() == initial_fen
        assert len(board.move_stack) == 0


class TestBasePlayerColorValidation:
    """Tests for player color validation."""

    def test_init__given_valid_colors__when_created__then_accepts(self):
        """Test that valid colors are accepted."""
        white_player = ConcretePlayer(name="Test", color="white")
        assert white_player.color == "white"

        black_player = ConcretePlayer(name="Test", color="black")
        assert black_player.color == "black"

    def test_init__given_invalid_color__when_created__then_stores_but_may_fail_later(
        self,
    ):
        """Test that invalid color is stored but may fail in type validation."""
        # The base class doesn't validate color, it just stores it
        # Validation happens when creating PlayerDecisionContext
        player = ConcretePlayer(name="Test", color="red")  # type: ignore
        assert player.color == "red"

        # But it will fail when trying to extract context
        board = chess.Board()
        with pytest.raises(ValueError):
            player._extract_context(board)
