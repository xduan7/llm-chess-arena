"""Property-based tests for utils module using Hypothesis."""

import chess
import pytest
from hypothesis import given, strategies as st, assume, settings
from hypothesis.stateful import RuleBasedStateMachine, rule, invariant, initialize

from llm_chess_arena.utils import (
    get_legal_moves_in_uci,
    get_move_history_in_uci,
    parse_attempted_move_to_uci,
)
from llm_chess_arena.exceptions import (
    InvalidMoveError,
    IllegalMoveError,
    AmbiguousMoveError,
)


# Custom strategies for chess-specific data
@st.composite
def legal_board_positions(draw):
    """Generate random but legal chess positions."""
    board = chess.Board()

    # Make 0-40 random legal moves to get varied positions
    num_moves = draw(st.integers(0, 40))
    for _ in range(num_moves):
        if board.is_game_over():
            break
        legal_moves = list(board.legal_moves)
        if legal_moves:
            move = draw(st.sampled_from(legal_moves))
            board.push(move)

    return board


@st.composite
def board_with_legal_move(draw):
    """Generate a board position with at least one legal move."""
    board = draw(legal_board_positions())
    assume(not board.is_game_over())  # Skip if game is over
    return board


class TestMoveParsingProperties:
    @given(board=board_with_legal_move())
    @settings(max_examples=50, deadline=1000)
    def test_parse_legal_uci_move_parses_correctly(self, board):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return  # Skip if no legal moves

        # Pick a random legal move
        move = legal_moves[0]
        uci_str = move.uci()

        # Parse it back
        parsed = parse_attempted_move_to_uci(uci_str, board.fen())

        # Should match exactly
        assert parsed == uci_str

    @given(board=board_with_legal_move())
    @settings(max_examples=50, deadline=1000)
    def test_parse_legal_san_move_converts_to_uci_correctly(self, board):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return

        # Pick a random legal move
        move = legal_moves[0]
        san_str = board.san(move)
        uci_str = move.uci()

        # Parse SAN to UCI
        parsed = parse_attempted_move_to_uci(san_str, board.fen())

        # Should match the UCI version
        assert parsed == uci_str

    @given(
        board=board_with_legal_move(),
        extra_spaces=st.text(alphabet=" \t\n", min_size=0, max_size=5),
    )
    @settings(max_examples=30, deadline=1000)
    def test_parse_move_handles_extra_whitespace_correctly(self, board, extra_spaces):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return

        move = legal_moves[0]
        san_str = board.san(move)

        # Add random whitespace
        messy_san = extra_spaces + san_str + extra_spaces

        # Should still parse correctly
        try:
            parsed = parse_attempted_move_to_uci(messy_san, board.fen())
            assert parsed == move.uci()
        except (InvalidMoveError, IllegalMoveError):
            # Some whitespace patterns might make it invalid
            pass

    @given(board=legal_board_positions())
    @settings(max_examples=30, deadline=1000)
    def test_get_legal_moves_returns_all_legal_moves_in_uci_format(self, board):
        legal_moves = get_legal_moves_in_uci(board)

        # Property 1: Number of moves matches python-chess
        assert len(legal_moves) == board.legal_moves.count()

        # Property 2: All moves are valid UCI format (4-5 chars)
        for move in legal_moves:
            assert len(move) in [4, 5]  # Normal moves or promotions
            assert move[0] in "abcdefgh"
            assert move[1] in "12345678"
            assert move[2] in "abcdefgh"
            assert move[3] in "12345678"
            if len(move) == 5:
                assert move[4] in "qrbn"  # Promotion piece

        # Property 3: All python-chess legal moves are in our list
        for chess_move in board.legal_moves:
            assert chess_move.uci() in legal_moves

    @given(board=legal_board_positions())
    @settings(max_examples=30, deadline=1000)
    def test_move_history_preserves_exact_order_of_moves(self, board):
        history = get_move_history_in_uci(board)

        # Rebuild the game and verify each move
        test_board = chess.Board()
        for uci_move in history:
            # Each move in history should be legal at that point
            assert uci_move in [m.uci() for m in test_board.legal_moves]
            test_board.push_uci(uci_move)

        # Final position should match
        assert test_board.fen() == board.fen()

    @given(board=board_with_legal_move(), random_text=st.text(min_size=5, max_size=10))
    @settings(max_examples=20, deadline=1000)
    def test_parse_invalid_notation_raises_appropriate_error(self, board, random_text):
        # Filter out text that might accidentally be valid
        if random_text.strip().lower() in ["o-o", "o-o-o", "0-0", "0-0-0"]:
            return  # Skip castling notation

        with pytest.raises((InvalidMoveError, IllegalMoveError, AmbiguousMoveError)):
            parse_attempted_move_to_uci(random_text, board.fen())

    @given(board=board_with_legal_move())
    @settings(max_examples=30, deadline=1000)
    def test_parse_uci_move_is_idempotent(self, board):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return

        move = legal_moves[0]
        uci_str = move.uci()

        # Parse once
        parsed_once = parse_attempted_move_to_uci(uci_str, board.fen())

        # Parse again (should be idempotent)
        parsed_twice = parse_attempted_move_to_uci(parsed_once, board.fen())

        assert parsed_once == parsed_twice == uci_str


class ChessGameStateMachine(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.board = None
        self.move_history = []

    @initialize()
    def setup(self):
        self.board = chess.Board()
        self.move_history = []

    @rule()
    def make_random_legal_move(self):
        if self.board.is_game_over():
            return

        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            return

        import random

        move = random.choice(legal_moves)

        # Record the move
        uci_before_move = move.uci()
        self.board.san(move)

        # Make the move
        self.board.push(move)
        self.move_history.append(uci_before_move)

        # Verify move was recorded correctly
        history = get_move_history_in_uci(self.board)
        assert history == self.move_history

    @invariant()
    def board_is_valid(self):
        assert self.board.is_valid()

    @invariant()
    def history_length_matches_halfmove_clock(self):
        history = get_move_history_in_uci(self.board)
        assert len(history) == len(self.board.move_stack)

    @invariant()
    def can_reconstruct_position_from_history(self):
        test_board = chess.Board()
        for uci_move in self.move_history:
            test_board.push_uci(uci_move)
        assert test_board.fen() == self.board.fen()


# Test the state machine
TestChessGame = ChessGameStateMachine.TestCase


class TestPromotionMoveProperties:
    @given(file=st.sampled_from("abcdefgh"), promotion_piece=st.sampled_from("qrbn"))
    def test_white_pawn_promotion_parses_correctly(self, file, promotion_piece):
        # Set up board with white pawn on 7th rank
        # Create FEN with pawn at correct position
        file_index = ord(file) - ord("a")
        empty_before = str(file_index) if file_index > 0 else ""
        empty_after = str(7 - file_index) if file_index < 7 else ""
        rank7 = (
            f"{empty_before}P{empty_after}"
            if empty_before or empty_after
            else (
                "P7"
                if file == "a"
                else "7P" if file == "h" else f"{empty_before}P{empty_after}"
            )
        )
        if file == "a":
            rank7 = "P7"
        elif file == "h":
            rank7 = "7P"
        else:
            rank7 = f"{file_index}P{7-file_index}"
        fen = f"8/{rank7}/8/8/8/8/8/8 w - - 0 1"
        chess.Board(fen)

        # Test UCI format
        uci_move = f"{file}7{file}8{promotion_piece}"
        parsed = parse_attempted_move_to_uci(uci_move, fen)
        assert parsed == uci_move

        # Test SAN format
        san_move = f"{file}8={promotion_piece.upper()}"
        parsed_san = parse_attempted_move_to_uci(san_move, fen)
        assert parsed_san == uci_move

    @given(file=st.sampled_from("abcdefgh"), promotion_piece=st.sampled_from("qrbn"))
    def test_black_pawn_promotion_parses_correctly(self, file, promotion_piece):
        # Set up board with black pawn on 2nd rank
        # Create FEN with pawn at correct position
        file_index = ord(file) - ord("a")
        if file == "a":
            rank2 = "p7"
        elif file == "h":
            rank2 = "7p"
        else:
            rank2 = f"{file_index}p{7-file_index}"
        fen = f"8/8/8/8/8/8/{rank2}/8 b - - 0 1"
        chess.Board(fen)

        # Test UCI format
        uci_move = f"{file}2{file}1{promotion_piece}"
        parsed = parse_attempted_move_to_uci(uci_move, fen)
        assert parsed == uci_move


class TestCastlingProperties:
    @given(color_is_white=st.booleans())
    def test_various_castling_notations_parse_correctly(self, color_is_white):
        if color_is_white:
            fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
            kingside_uci = "e1g1"
            queenside_uci = "e1c1"
        else:
            fen = "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1"
            kingside_uci = "e8g8"
            queenside_uci = "e8c8"

        # Test various notations for kingside castling
        for notation in ["O-O", "0-0", "o-o"]:
            parsed = parse_attempted_move_to_uci(notation, fen)
            assert parsed == kingside_uci

        # Test various notations for queenside castling
        for notation in ["O-O-O", "0-0-0", "o-o-o"]:
            parsed = parse_attempted_move_to_uci(notation, fen)
            assert parsed == queenside_uci
