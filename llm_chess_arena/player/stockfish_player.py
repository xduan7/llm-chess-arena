import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import chess
import chess.engine
from loguru import logger

from llm_chess_arena.player.base_player import BasePlayer
from llm_chess_arena.types import Color, PlayerDecisionContext, PlayerDecision

# Default depth prevents infinite analysis when limits not specified
DEFAULT_ENGINE_LIMITS = {"depth": 10}


class StockfishPlayer(BasePlayer):
    """Chess player powered by Stockfish engine.

    Uses lazy initialization: engine subprocess starts only on first move request.
    This prevents hanging processes if game init fails after player creation.

    Note:
        Call close() explicitly for clean shutdown, or use try/finally.
        If program crashes after engine starts, subprocess may linger requiring manual kill.
    """

    def __init__(
        self,
        *,
        name: str = "Stockfish",
        color: Color,
        binary_path: Optional[str] = None,
        engine_limits: Optional[Dict[str, Any]] = None,
        engine_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize Stockfish player configuration.

        Note: The engine subprocess is not started until the first move is
        requested (lazy initialization). This avoids hanging processes if
        Game initialization fails after player creation.

        Args:
            name: Display name.
            color: 'white' or 'black'.
            binary_path: Explicit path or None to auto-detect.
            engine_limits: Search constraints (depth, time, nodes).
            engine_options: UCI configuration (threads, skill level).

        Raises:
            FileNotFoundError: If binary not found during path resolution.
        """
        super().__init__(name, color)

        self.engine: Optional[chess.engine.SimpleEngine] = None
        self.binary_path = self._find_stockfish_binary(binary_path)
        self.engine_limits = engine_limits or DEFAULT_ENGINE_LIMITS
        self.engine_options = engine_options or {}

        logger.debug(
            f"StockfishPlayer configured with limits={self.engine_limits} (engine not started yet)"
        )

    @staticmethod
    def _find_stockfish_binary(explicit_path: Optional[str] = None) -> str:
        """Locate Stockfish binary.

        Search order: explicit path, env var, PATH, common locations.

        Args:
            explicit_path: Explicit path or None for auto-detection.

        Returns:
            Resolved path to executable.

        Raises:
            FileNotFoundError: If not found anywhere.
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
            path = Path(env_path)
            if not path.exists():
                logger.warning(
                    f"Environment variable STOCKFISH_BINARY_PATH set to"
                    f" {path}, but file does not exist."
                )
            elif not os.access(str(path), os.X_OK):
                logger.warning(
                    f"Stockfish binary from STOCKFISH_BINARY_PATH exists but"
                    f" is not executable: {path}"
                )
            else:
                logger.debug(
                    f"Found Stockfish binary from STOCKFISH_BINARY_PATH: {path}"
                )
                return str(path.resolve())

        system_path = shutil.which("stockfish")
        if system_path:
            logger.debug(f"Found Stockfish binary in PATH: {system_path}")
            return system_path

        common_paths = [
            "/usr/local/bin/stockfish",
            "/usr/bin/stockfish",
            "/opt/homebrew/bin/stockfish",
            "C:\\Program Files\\Stockfish\\stockfish.exe",
            "C:\\Program Files (x86)\\Stockfish\\stockfish.exe",
        ]
        for common_path in common_paths:
            path = Path(common_path)
            if path.exists() and os.access(str(path), os.X_OK):
                logger.debug(f"Found Stockfish binary in common path: {path}")
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

    def _start_engine(self) -> None:
        """Start the Stockfish engine subprocess (lazy initialization).

        Raises:
            RuntimeError: If engine initialization fails.
        """
        if self.engine is not None:
            return

        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(self.binary_path)
            self.engine.configure(self.engine_options)
            logger.info(f"Stockfish engine started with limits={self.engine_limits}")
        except Exception as e:
            if self.engine:
                try:
                    self.engine.quit()
                except Exception:
                    pass
                self.engine = None
            raise RuntimeError(f"Failed to initialize Stockfish engine: {e}") from e

    def _make_decision(self, context: PlayerDecisionContext) -> PlayerDecision:
        """Query Stockfish for best move.

        Args:
            context: Game context with FEN string.

        Returns:
            Decision with engine's best move.

        Raises:
            RuntimeError: If engine fails or cannot be started.
        """
        if self.engine is None:
            self._start_engine()

        try:
            # Fresh board from FEN respects DTO pattern and avoids state mutation
            board = chess.Board(context.board_in_fen)

            limit = chess.engine.Limit(**self.engine_limits)
            result = self.engine.play(board, limit)

            if result.move is None:
                raise chess.engine.EngineError(
                    "Stockfish returned None instead of a move"
                )

            return PlayerDecision(action="move", attempted_move=result.move.uci())

        except chess.engine.EngineError as e:
            raise RuntimeError(f"Stockfish failed to generate move: {e}") from e

    def close(self) -> None:
        """Gracefully terminate engine process.

        IMPORTANT: Always call this method or use with a try-finally block
        to prevent hanging Stockfish processes. The engine subprocess will
        continue running even after Python exceptions if not properly closed.
        """
        if self.engine is not None:
            try:
                self.engine.quit()
                logger.debug("Stockfish engine closed successfully")
            except Exception as e:
                logger.error(f"Error closing Stockfish engine: {e}")
            finally:
                self.engine = None
