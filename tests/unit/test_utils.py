"""Unit tests for chess utility helper functions."""

import chess
import pytest

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


class TestGetLegalMovesInUCI:
    """Unit tests for generating legal UCI move lists."""

    def test_starting_position__returns_exactly_20_legal_moves(self):
        board = chess.Board()
        moves = get_legal_moves_in_uci(board)
        assert len(moves) == 20
        assert "e2e4" in moves
        assert "g1f3" in moves

    def test_checkmate_position__returns_empty_move_list(self):
        fools_mate_position = (
            "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
        )
        board = chess.Board(fools_mate_position)
        moves = get_legal_moves_in_uci(board)
        assert moves == []

    def test_all_moves__formatted_as_valid_uci_notation(self):
        board = chess.Board()
        moves = get_legal_moves_in_uci(board)
        for move in moves:
            assert len(move) in [4, 5]  # 4 for normal, 5 for promotion
            assert move[0] in "abcdefgh"
            assert move[1] in "12345678"
            assert move[2] in "abcdefgh"
            assert move[3] in "12345678"

    def test_en_passant_capture__included_when_available(self):
        position_with_en_passant_opportunity = (
            "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"
        )
        board = chess.Board(position_with_en_passant_opportunity)
        moves = get_legal_moves_in_uci(board)
        assert "e5f6" in moves

    def test_castling_moves__included_when_both_sides_available(self):
        position_with_castling_rights = (
            "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"
        )
        board = chess.Board(position_with_castling_rights)
        moves = get_legal_moves_in_uci(board)
        assert "e1g1" in moves  # Kingside
        assert "e1c1" in moves  # Queenside


class TestGetMoveHistoryInUCI:
    """Unit tests ensuring move history is captured accurately."""

    def test_new_board__returns_empty_move_history(self):
        board = chess.Board()
        history = get_move_history_in_uci(board)
        assert history == []

    def test_single_move__returns_one_element_history(self):
        board = chess.Board()
        board.push_san("e4")
        history = get_move_history_in_uci(board)
        assert history == ["e2e4"]

    def test_multiple_moves__returns_complete_history_in_order(self):
        board = chess.Board()
        board.push_san("e4")
        board.push_san("e5")
        board.push_san("Nf3")
        history = get_move_history_in_uci(board)
        assert history == ["e2e4", "e7e5", "g1f3"]

    def test_sicilian_defense_opening__preserves_chronological_move_order(self):
        board = chess.Board()
        sicilian_defense_moves = ["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4"]
        for move in sicilian_defense_moves:
            board.push_san(move)

        history = get_move_history_in_uci(board)
        assert len(history) == 7
        assert history[0] == "e2e4"
        assert history[-1] == "f3d4"

    def test_castling_move__recorded_as_king_movement_in_uci(self):
        position_allowing_castling = (
            "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"
        )
        board = chess.Board(position_allowing_castling)
        board.push_san("O-O")
        history = get_move_history_in_uci(board)
        assert history == ["e1g1"]


class TestParseAttemptedMoveToUCI:
    """Unit tests for parsing attempted moves into UCI."""

    def test_valid_uci_move__returns_unchanged(self):
        starting_position_fen = (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        )
        result = parse_attempted_move_to_uci("e2e4", starting_position_fen)
        assert result == "e2e4"

    def test_valid_san_moves__converted_to_uci_format(self):
        starting_position_fen = (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        )

        pawn_move_result = parse_attempted_move_to_uci("e4", starting_position_fen)
        assert pawn_move_result == "e2e4"

        knight_move_result = parse_attempted_move_to_uci("Nf3", starting_position_fen)
        assert knight_move_result == "g1f3"

    def test_castling_notation__converts_to_king_move_in_uci(self):
        position_with_castling = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"

        kingside_castling = parse_attempted_move_to_uci("O-O", position_with_castling)
        assert kingside_castling == "e1g1"

        queenside_castling = parse_attempted_move_to_uci(
            "O-O-O", position_with_castling
        )
        assert queenside_castling == "e1c1"

    def test_en_passant_capture__parsed_correctly_from_san(self):
        position_with_en_passant = (
            "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"
        )
        result = parse_attempted_move_to_uci("exf6", position_with_en_passant)
        assert result == "e5f6"

    def test_pawn_promotion__converts_san_to_lowercase_uci(self):
        promotion_position = "8/P7/8/8/8/8/8/8 w - - 0 1"

        san_promotion_result = parse_attempted_move_to_uci("a8=Q", promotion_position)
        assert san_promotion_result == "a7a8q"

        uci_promotion_result = parse_attempted_move_to_uci("a7a8q", promotion_position)
        assert uci_promotion_result == "a7a8q"

    def test_invalid_notation__raises_invalid_move_error_with_descriptive_message(self):
        starting_position = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

        with pytest.raises(InvalidMoveError) as exc_info:
            parse_attempted_move_to_uci("Z9", starting_position)
        assert "Invalid move notation" in str(exc_info.value)

        with pytest.raises(InvalidMoveError) as exc_info:
            parse_attempted_move_to_uci("xyz", starting_position)
        assert "Invalid move notation" in str(exc_info.value)

    def test_illegal_uci_move__raises_illegal_move_error(self):
        starting_position = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

        with pytest.raises(IllegalMoveError) as exc_info:
            parse_attempted_move_to_uci(
                "e2e5", starting_position
            )  # Pawn can't jump 3 squares
        assert "Illegal move" in str(exc_info.value)

    def test_illegal_san_move__raises_illegal_move_error(self):
        starting_position = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

        with pytest.raises(IllegalMoveError) as exc_info:
            parse_attempted_move_to_uci(
                "Nd4", starting_position
            )  # Knight can't reach d4 from start
        assert "Illegal move" in str(exc_info.value)

    def test_ambiguous_san__raises_ambiguous_move_error(self):
        two_knights_same_target_position = "8/8/8/3N1N2/8/8/8/8 w - - 0 1"

        with pytest.raises(AmbiguousMoveError) as exc_info:
            parse_attempted_move_to_uci("Ne3", two_knights_same_target_position)
        assert "Ambiguous" in str(exc_info.value)

    def test_disambiguated_san__resolves_correctly_by_file(self):
        two_knights_position = "8/8/8/3N1N2/8/8/8/8 w - - 0 1"

        d_file_knight_move = parse_attempted_move_to_uci("Nde3", two_knights_position)
        assert d_file_knight_move == "d5e3"

        f_file_knight_move = parse_attempted_move_to_uci("Nfe3", two_knights_position)
        assert f_file_knight_move == "f5e3"


class TestUtilsEdgeCases:
    """Edge-case scenarios covering move generation utilities."""

    def test_stalemate_position__returns_empty_legal_moves_list(self):
        stalemate_position = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"
        board = chess.Board(stalemate_position)
        moves = get_legal_moves_in_uci(board)
        assert moves == []

    def test_pawn_promotion__generates_all_four_piece_options(self):
        pawn_on_seventh_rank = "8/P7/8/8/8/8/8/8 w - - 0 1"
        board = chess.Board(pawn_on_seventh_rank)
        moves = get_legal_moves_in_uci(board)

        assert "a7a8q" in moves  # Queen
        assert "a7a8r" in moves  # Rook
        assert "a7a8b" in moves  # Bishop
        assert "a7a8n" in moves  # Knight

    def test_parse_attempted_move__does_not_mutate_original_board_state(self):
        starting_position = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

        parse_attempted_move_to_uci("e4", starting_position)
        parse_attempted_move_to_uci("Nf3", starting_position)
        parse_attempted_move_to_uci("d4", starting_position)

        board_from_original_fen = chess.Board(starting_position)
        moves = get_legal_moves_in_uci(board_from_original_fen)
        assert len(moves) == 20
