import chess

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
        List of legal moves in UCI notation (e.g., ["e2e4", "g1f3"]).
    """
    return [move.uci() for move in board.legal_moves]


def get_move_history_in_uci(board: chess.Board) -> list[str]:
    """Get the move history in UCI format from the current board state.

    Args:
        board: Current chess board state with move history.

    Returns:
        List of moves in UCI notation (e.g., ["e2e4", "e7e5", "g1f3"]).
    """
    return [move.uci() for move in board.move_stack]


def parse_attempted_move_to_uci(attempted_move: str, board_in_fen: str) -> str:
    """Parse a move string to UCI format, trying UCI first then SAN.

    Args:
        attempted_move: Move text in UCI (e2e4) or SAN (Nf3, O-O).
        board_in_fen: FEN string representing the position.

    Returns:
        Move in UCI format (e.g., "e2e4").

    Raises:
        InvalidMoveError: If notation is syntactically invalid.
        AmbiguousMoveError: If SAN is ambiguous in this position.
        IllegalMoveError: If move is not legal in this position.
    """
    # Fresh board from FEN avoids mutating caller state
    board = chess.Board(fen=board_in_fen)

    # Normalize castling notation to uppercase (handle o-o, O-O, 0-0 variants)
    move_normalized = attempted_move.strip()
    if move_normalized.lower() in ["o-o", "0-0"]:
        move_normalized = "O-O"
    elif move_normalized.lower() in ["o-o-o", "0-0-0"]:
        move_normalized = "O-O-O"
    else:
        move_normalized = attempted_move

    try:
        move = chess.Move.from_uci(move_normalized)
        if move not in board.legal_moves:
            raise IllegalMoveError(
                f"Illegal move in current position: '{attempted_move}'"
            )
        return move.uci()
    except ValueError:
        try:
            move = board.parse_san(move_normalized)
            return move.uci()
        except chess.AmbiguousMoveError as e:
            raise AmbiguousMoveError(f"Ambiguous SAN move: '{attempted_move}'") from e
        except chess.InvalidMoveError as e:
            raise InvalidMoveError(f"Invalid move notation: '{attempted_move}'") from e
        except chess.IllegalMoveError as e:
            raise IllegalMoveError(
                f"Illegal move in current position: '{attempted_move}'"
            ) from e
