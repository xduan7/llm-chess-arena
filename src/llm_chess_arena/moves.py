"""Chess move parsing, normalization, and serialization helpers."""

from __future__ import annotations

import chess

from llm_chess_arena.core.policies import move_validation
from llm_chess_arena.exceptions import (
    IllegalMoveError,
    InvalidMoveError,
    AmbiguousMoveError,
)


def get_legal_moves_in_uci(board: chess.Board) -> list[str]:
    """Get all legal moves in UCI format from the current board state.

    Args:
        board: Current chess board state.

    Returns:
        list[str]: Legal moves in UCI notation (e.g., ["e2e4", "g1f3"]).
    """
    return [move.uci() for move in board.legal_moves]


def get_move_history_in_uci(board: chess.Board) -> list[str]:
    """Get the move history in UCI format from the current board state.

    Args:
        board: Current chess board state with move history.

    Returns:
        list[str]: Moves in UCI notation (e.g., ["e2e4", "e7e5", "g1f3"]).
    """
    return [move.uci() for move in board.move_stack]


def normalize_castling_notation(move_text: str) -> str:
    """Normalize castling notation to standard format.

    Args:
        move_text: Move text that may contain castling notation.

    Returns:
        str: Move text with normalized castling notation.
    """
    move_normalized = move_text.strip()

    if move_normalized.lower() in ["o-o", "0-0"]:
        return "O-O"
    elif move_normalized.lower() in ["o-o-o", "0-0-0"]:
        return "O-O-O"

    return move_normalized


@move_validation
def parse_attempted_move_to_uci(attempted_move: str, board_in_fen: str) -> str:
    """Parse a move string to UCI format, trying UCI first then SAN.

    Args:
        attempted_move: Move text in UCI (e2e4) or SAN (Nf3, O-O).
        board_in_fen: FEN string representing the position.

    Returns:
        str: Move in UCI format (e.g., "e2e4").

    Raises:
        InvalidMoveError: If notation is syntactically invalid.
        AmbiguousMoveError: If SAN is ambiguous in this position.
        IllegalMoveError: If move is not legal in this position.
    """
    board = chess.Board(fen=board_in_fen)

    attempted_move_normalized = normalize_castling_notation(attempted_move)

    try:
        move = chess.Move.from_uci(attempted_move_normalized)
        if move not in board.legal_moves:
            raise IllegalMoveError(
                f"Illegal move in current position: '{attempted_move}'"
            )
        return str(move.uci())
    except ValueError:
        try:
            move = board.parse_san(attempted_move_normalized)
            return str(move.uci())
        except chess.AmbiguousMoveError as ambiguous_move_error:
            raise AmbiguousMoveError(
                f"Ambiguous SAN move: '{attempted_move}'"
            ) from ambiguous_move_error
        except chess.InvalidMoveError as invalid_move_error:
            raise InvalidMoveError(
                f"Invalid move notation: '{attempted_move}'"
            ) from invalid_move_error
        except chess.IllegalMoveError as illegal_move_error:
            raise IllegalMoveError(
                f"Illegal move in current position: '{attempted_move}'"
            ) from illegal_move_error
