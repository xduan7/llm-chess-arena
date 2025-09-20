"""Utility helpers for chess move serialization, validation, and system utilities."""

import os
import shutil
from pathlib import Path

import chess
from loguru import logger

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
    # Fresh board from FEN avoids mutating caller state
    board = chess.Board(fen=board_in_fen)

    # Normalize castling notation to uppercase (handle o-o, O-O, 0-0 variants)
    move_normalized = attempted_move.strip()
    if move_normalized.lower() in ["o-o", "0-0"]:
        move_normalized = "O-O"
    elif move_normalized.lower() in ["o-o-o", "0-0-0"]:
        move_normalized = "O-O-O"

    try:
        move = chess.Move.from_uci(move_normalized)
        if move not in board.legal_moves:
            raise IllegalMoveError(
                f"Illegal move in current position: '{attempted_move}'"
            )
        move_uci = str(move.uci())
        return move_uci
    except ValueError:
        try:
            move = board.parse_san(move_normalized)
            move_uci = str(move.uci())
            return move_uci
        except chess.AmbiguousMoveError as e:
            raise AmbiguousMoveError(f"Ambiguous SAN move: '{attempted_move}'") from e
        except chess.InvalidMoveError as e:
            raise InvalidMoveError(f"Invalid move notation: '{attempted_move}'") from e
        except chess.IllegalMoveError as e:
            raise IllegalMoveError(
                f"Illegal move in current position: '{attempted_move}'"
            ) from e


# Common platform-specific locations checked after PATH lookup.
COMMON_STOCKFISH_PATHS: tuple[str, ...] = (
    "/usr/local/bin/stockfish",
    "/usr/bin/stockfish",
    "/opt/homebrew/bin/stockfish",
    "C:/Program Files/Stockfish/stockfish.exe",
    "C:/Program Files (x86)/Stockfish/stockfish.exe",
)


def find_stockfish_binary(explicit_path: str | None = None) -> str:
    """Resolve a usable Stockfish executable path.

    Args:
        explicit_path: Optional user-supplied path to the Stockfish binary.

    Returns:
        str: Absolute path to the executable.

    Raises:
        FileNotFoundError: If no executable binary can be located.
    """
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"Stockfish binary not found at: {path}")
        if not os.access(str(path), os.X_OK):
            raise FileNotFoundError(
                f"Stockfish binary exists but is not executable at: {path}\n"
                f"Try: chmod +x {path}"
            )
        return str(path.resolve())

    env_path = os.getenv("STOCKFISH_BINARY_PATH")
    if env_path:
        env_binary = Path(env_path)
        if not env_binary.exists():
            logger.warning(
                "Environment variable STOCKFISH_BINARY_PATH set to {} but file does not exist",
                env_binary,
            )
        elif not os.access(str(env_binary), os.X_OK):
            logger.warning(
                "Stockfish binary from STOCKFISH_BINARY_PATH exists but is not executable: {}",
                env_binary,
            )
        else:
            logger.debug(
                "Found Stockfish binary from STOCKFISH_BINARY_PATH: {}", env_binary
            )
            return str(env_binary.resolve())

    system_path = shutil.which("stockfish")
    if system_path:
        logger.debug("Found Stockfish binary in PATH: {}", system_path)
        return system_path

    for candidate in COMMON_STOCKFISH_PATHS:
        path = Path(candidate)
        if path.exists() and os.access(str(path), os.X_OK):
            logger.debug("Found Stockfish binary in common path: {}", path)
            return str(path.resolve())

    raise FileNotFoundError(
        "Stockfish not found. Please install it or provide the binary path.\n"
        "You can either:\n"
        "  1. Set STOCKFISH_BINARY_PATH in your .env file\n"
        "  2. Pass binary_path parameter when creating StockfishPlayer\n"
        "  3. Install Stockfish:\n"
        "     macOS: brew install stockfish\n"
        "     Ubuntu/Debian: apt-get install stockfish\n"
        "     Windows: Download from https://stockfishchess.org/download/"
    )
