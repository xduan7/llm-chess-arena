"""Integration tests for chess edge cases and special rules."""

import chess

from llm_chess_arena.game import Game
from llm_chess_arena.player.random_player import RandomPlayer


class TestChessEdgeCases:
    """Test special chess rules and edge cases in full game context."""

    def test_en_passant_capture(self):
        """Test that en passant captures work correctly in a game."""
        white = RandomPlayer(color="white", seed=42)
        black = RandomPlayer(color="black", seed=43)
        game = Game(white, black)

        # Set up position for en passant
        # White pawn on e5, black plays d7-d5
        game.board = chess.Board(
            "rnbqkb1r/ppp1pppp/5n2/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3"
        )

        # Now white can capture en passant
        assert "e5d6" in [m.uci() for m in game.board.legal_moves]

        # Make the en passant capture
        game.board.push_uci("e5d6")

        # Verify the captured pawn is removed
        assert game.board.piece_at(chess.D5) is None  # Black pawn removed
        assert game.board.piece_at(chess.D6) is not None  # White pawn on d6

    def test_castling_kingside(self):
        """Test kingside castling mechanics."""
        white = RandomPlayer(color="white")
        black = RandomPlayer(color="black")
        game = Game(white, black)

        # Clear path for kingside castling
        game.board = chess.Board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")

        # Verify castling is legal
        assert "e1g1" in [m.uci() for m in game.board.legal_moves]

        # Perform castling
        game.board.push_uci("e1g1")

        # Verify final positions
        assert game.board.piece_at(chess.G1).piece_type == chess.KING
        assert game.board.piece_at(chess.F1).piece_type == chess.ROOK
        assert game.board.piece_at(chess.E1) is None
        assert game.board.piece_at(chess.H1) is None

    def test_castling_queenside(self):
        """Test queenside castling mechanics."""
        white = RandomPlayer(color="white")
        black = RandomPlayer(color="black")
        game = Game(white, black)

        # Clear path for queenside castling
        game.board = chess.Board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")

        # Verify castling is legal
        assert "e1c1" in [m.uci() for m in game.board.legal_moves]

        # Perform castling
        game.board.push_uci("e1c1")

        # Verify final positions
        assert game.board.piece_at(chess.C1).piece_type == chess.KING
        assert game.board.piece_at(chess.D1).piece_type == chess.ROOK
        assert game.board.piece_at(chess.E1) is None
        assert game.board.piece_at(chess.A1) is None

    def test_pawn_promotion(self):
        """Test pawn promotion to different pieces."""
        white = RandomPlayer(color="white")
        black = RandomPlayer(color="black")
        game = Game(white, black)

        # White pawn ready to promote
        game.board = chess.Board("8/P7/8/8/8/8/8/8 w - - 0 1")

        # All promotion options should be legal
        legal_moves = [m.uci() for m in game.board.legal_moves]
        assert "a7a8q" in legal_moves  # Queen
        assert "a7a8r" in legal_moves  # Rook
        assert "a7a8b" in legal_moves  # Bishop
        assert "a7a8n" in legal_moves  # Knight

        # Promote to queen
        game.board.push_uci("a7a8q")
        assert game.board.piece_at(chess.A8).piece_type == chess.QUEEN

    def test_stalemate_detection(self):
        """Test that stalemate is properly detected."""
        white = RandomPlayer(color="white")
        black = RandomPlayer(color="black")
        game = Game(white, black)

        # Classic stalemate position
        game.board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")

        # Verify it's stalemate
        assert game.board.is_stalemate()
        assert game.board.is_game_over()
        assert not game.board.is_checkmate()
        assert game.board.result() == "1/2-1/2"

    def test_insufficient_material_draw(self):
        """Test draw by insufficient material."""
        white = RandomPlayer(color="white")
        black = RandomPlayer(color="black")
        game = Game(white, black)

        # King vs King
        game.board = chess.Board("8/8/8/4k3/8/8/8/4K3 w - - 0 1")
        assert game.board.is_insufficient_material()
        assert game.board.is_game_over()

        # King and Bishop vs King
        game.board = chess.Board("8/8/8/4k3/8/8/8/4KB2 w - - 0 1")
        assert game.board.is_insufficient_material()

        # King and Knight vs King
        game.board = chess.Board("8/8/8/4k3/8/8/8/4KN2 w - - 0 1")
        assert game.board.is_insufficient_material()

        # King and two Knights vs King is NOT insufficient material in python-chess
        # (technically can mate but can't force it)
        game.board = chess.Board("8/8/8/4k3/8/8/8/2N1KN2 w - - 0 1")
        # python-chess considers this sufficient material
        assert not game.board.is_insufficient_material()

    def test_threefold_repetition(self):
        """Test threefold repetition detection."""
        white = RandomPlayer(color="white")
        black = RandomPlayer(color="black")
        game = Game(white, black)

        # Create a position that will repeat
        moves = [
            "Nf3",
            "Nf6",  # Develop knights
            "Ng1",
            "Ng8",  # Move back
            "Nf3",
            "Nf6",  # Repeat position 1
            "Ng1",
            "Ng8",  # Move back again
            "Nf3",
            "Nf6",  # Repeat position 2 (threefold)
        ]

        for move in moves:
            game.board.push_san(move)

        # Should be able to claim draw
        assert game.board.can_claim_threefold_repetition()

    def test_fifty_move_rule(self):
        """Test fifty-move rule detection."""
        white = RandomPlayer(color="white")
        black = RandomPlayer(color="black")
        game = Game(white, black)

        # Create a position with high halfmove clock (approaching 50-move rule)
        game.board = chess.Board("8/8/8/3k4/3K4/8/8/8 w - - 99 50")

        # Make one more move to reach 100 half-moves (50 full moves)
        game.board.push_san("Kd3")

        # Should be able to claim draw after 50 moves without pawn move or capture
        assert game.board.halfmove_clock == 100
        assert game.board.can_claim_fifty_moves()


class TestComplexPositions:
    """Test complex game positions and scenarios."""

    def test_pinned_piece_cannot_move(self):
        """Test that pinned pieces have restricted movement."""
        white = RandomPlayer(color="white")
        black = RandomPlayer(color="black")
        game = Game(white, black)

        # Knight pinned by bishop
        game.board = chess.Board(
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 4"
        )
        game.board.push_san("Be7")  # Move black bishop
        game.board.push_san("Bxf7+")  # White bishop checks, pinning knight

        # Black knight on f6 is pinned and can't move
        legal_moves = list(game.board.legal_moves)
        knight_moves = [
            m
            for m in legal_moves
            if game.board.piece_at(m.from_square)
            and game.board.piece_at(m.from_square).piece_type == chess.KNIGHT
        ]

        # The f6 knight should have no legal moves (it's pinned)
        f6_knight_moves = [m for m in knight_moves if m.from_square == chess.F6]
        assert len(f6_knight_moves) == 0

    def test_discovered_check(self):
        """Test discovered check scenario."""
        white = RandomPlayer(color="white")
        black = RandomPlayer(color="black")
        game = Game(white, black)

        # Set up discovered check position
        # White bishop on a1, white knight blocking, black king on h8
        game.board = chess.Board("7k/8/8/8/8/8/1N6/B7 w - - 0 1")

        # Move knight to discover check
        game.board.push_san("Nd3")

        # Black king should be in check from bishop
        assert game.board.is_check()

        # Black must respond to check
        legal_moves = list(game.board.legal_moves)
        # All legal moves should address the check
        for move in legal_moves:
            test_board = game.board.copy()
            test_board.push(move)
            assert not test_board.is_check()
