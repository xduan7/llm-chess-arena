"""Stockfish binary discovery and engine initialization."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import chess.engine
from loguru import logger

# Common platform-specific locations checked after PATH lookup.
COMMON_STOCKFISH_PATHS: tuple[str, ...] = (
    "/usr/local/bin/stockfish",
    "/usr/bin/stockfish",
    "/opt/homebrew/bin/stockfish",
    "C:/Program Files/Stockfish/stockfish.exe",
    "C:/Program Files (x86)/Stockfish/stockfish.exe",
)

# Cache for Stockfish availability check to avoid repeated filesystem calls
_stockfish_availability_cache: bool | None = None


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
        candidate_path = Path(explicit_path)
        if not candidate_path.exists():
            raise FileNotFoundError(f"Stockfish binary not found at: {candidate_path}")
        if not os.access(str(candidate_path), os.X_OK):
            raise FileNotFoundError(
                "Stockfish binary exists but is not executable at: "
                f"{candidate_path}\n"
                f"Try: chmod +x {candidate_path}"
            )
        return str(candidate_path.resolve())

    env_path_str = os.getenv("STOCKFISH_BINARY_PATH")
    if env_path_str:
        env_path = Path(env_path_str)
        if not env_path.exists():
            logger.warning(
                "Environment variable STOCKFISH_BINARY_PATH set to {} but file does not exist",
                env_path,
            )
        elif not os.access(str(env_path), os.X_OK):
            logger.warning(
                "Stockfish binary from STOCKFISH_BINARY_PATH exists but is not executable: {}",
                env_path,
            )
        else:
            logger.debug(
                "Found Stockfish binary from STOCKFISH_BINARY_PATH: {}",
                env_path,
            )
            return str(env_path.resolve())

    system_path = shutil.which("stockfish")
    if system_path:
        logger.debug("Found Stockfish binary in PATH: {}", system_path)
        return system_path

    for potential_path in COMMON_STOCKFISH_PATHS:
        candidate_path = Path(potential_path)
        if candidate_path.exists() and os.access(str(candidate_path), os.X_OK):
            logger.debug("Found Stockfish binary in common path: {}", candidate_path)
            return str(candidate_path.resolve())

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


def is_stockfish_available() -> bool:
    """Check if Stockfish is available on the system.

    This function caches the result to avoid repeated filesystem calls.
    Uses the same logic as find_stockfish_binary but returns a boolean
    instead of raising exceptions.

    Returns:
        bool: True if Stockfish is available, False otherwise.
    """
    global _stockfish_availability_cache

    if _stockfish_availability_cache is not None:
        return _stockfish_availability_cache

    try:
        find_stockfish_binary()
        _stockfish_availability_cache = True
        return True
    except FileNotFoundError:
        _stockfish_availability_cache = False
        return False


def initialize_stockfish_engine(
    binary_path: str, engine_options: dict[str, Any] | None = None
) -> chess.engine.SimpleEngine:
    """Initialize and configure a Stockfish engine instance.

    Provides shared initialization logic for both StockfishPlayer and
    StockfishMetricsEvaluator to eliminate code duplication.

    Args:
        binary_path: Path to the Stockfish executable.
        engine_options: Optional UCI engine configuration options.

    Returns:
        chess.engine.SimpleEngine: Configured Stockfish engine instance.

    Raises:
        Exception: If engine initialization or configuration fails.
    """
    stockfish_engine = chess.engine.SimpleEngine.popen_uci(binary_path)
    try:
        if engine_options:
            stockfish_engine.configure(engine_options)
        return stockfish_engine
    except Exception:
        stockfish_engine.quit()
        raise
