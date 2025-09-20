"""Stockfish-backed chess player implementation."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Mapping

import chess
import chess.engine
from loguru import logger

from llm_chess_arena.player.base_player import BasePlayer
from llm_chess_arena.types import Color, PlayerDecisionContext, PlayerDecision

# Default depth prevents infinite analysis when limits not specified
DEFAULT_ENGINE_LIMITS: dict[str, Any] = {"depth": 10}


class StockfishPlayer(BasePlayer):
    """Chess player powered by a lazily-initialized Stockfish engine.

    The engine subprocess is started only when a decision is requested to avoid
    spawning lingering processes if player construction fails. Always call
    close() to terminate the engine once the player is no longer needed.
    """

    def __init__(
        self,
        *,
        name: str = "Stockfish",
        color: Color,
        binary_path: str | None = None,
        engine_limits: Mapping[str, Any] | None = None,
        engine_options: Mapping[str, Any] | None = None,
    ) -> None:
        """Configure the Stockfish engine interface.

        Args:
            name: Display name shown in logs and summaries.
            color: Chess side this player controls.
            binary_path: Explicit path or None to auto-detect.
            engine_limits: Search constraints such as depth or time.
            engine_options: UCI configuration such as threads or skill level.

        Raises:
            FileNotFoundError: If Stockfish cannot be located.
        """
        super().__init__(name, color)

        self.engine: chess.engine.SimpleEngine | None = None
        self.binary_path = self._find_stockfish_binary(binary_path)
        self.engine_limits = (
            dict(engine_limits) if engine_limits else DEFAULT_ENGINE_LIMITS.copy()
        )
        self.engine_options = dict(engine_options) if engine_options else {}

        logger.debug(
            "StockfishPlayer configured with limits={} (engine not started yet)",
            self.engine_limits,
        )

    @staticmethod
    def _find_stockfish_binary(explicit_path: str | None = None) -> str:
        """Resolve the Stockfish binary path.

        Args:
            explicit_path: Optional user-supplied binary path.

        Returns:
            str: Absolute path to the Stockfish executable.

        Raises:
            FileNotFoundError: If no executable is discovered.
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

        # Allow environment customization so deployments can pin a managed binary.
        env_path = os.getenv("STOCKFISH_BINARY_PATH")
        if env_path:
            path = Path(env_path)
            if not path.exists():
                logger.warning(
                    "Environment variable STOCKFISH_BINARY_PATH set to {} but file does not exist",
                    path,
                )
            elif not os.access(str(path), os.X_OK):
                logger.warning(
                    "Stockfish binary from STOCKFISH_BINARY_PATH exists but is not executable: {}",
                    path,
                )
            else:
                logger.debug(
                    "Found Stockfish binary from STOCKFISH_BINARY_PATH: {}", path
                )
                return str(path.resolve())

        system_path = shutil.which("stockfish")
        if system_path:
            logger.debug("Found Stockfish binary in PATH: {}", system_path)
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

    def _start_engine(self) -> None:
        """Start the Stockfish engine subprocess on first demand."""
        if self.engine is not None:
            return

        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(self.binary_path)
            self.engine.configure(self.engine_options)
            logger.info("Stockfish engine started with limits={}", self.engine_limits)
        except Exception as e:
            if self.engine:
                try:
                    self.engine.quit()
                except Exception:
                    pass
                self.engine = None
            raise RuntimeError(
                "Failed to initialize Stockfish engine: {}".format(e)
            ) from e

    def _make_decision(self, context: PlayerDecisionContext) -> PlayerDecision:
        """Query Stockfish for the strongest move and wrap the response.

        Args:
            context: Decision context describing the current board state.

        Returns:
            PlayerDecision: Move decision emitted by Stockfish.

        Raises:
            RuntimeError: If Stockfish fails to return a move.
        """
        if self.engine is None:
            self._start_engine()

        engine = self.engine
        if engine is None:
            raise RuntimeError("Stockfish engine failed to start")

        try:
            # Fresh board from FEN respects DTO pattern and avoids state mutation
            board = chess.Board(context.board_in_fen)

            limit = chess.engine.Limit(**self.engine_limits)
            result = engine.play(board, limit)

            if result.move is None:
                raise chess.engine.EngineError(
                    "Stockfish returned None instead of a move"
                )

            return PlayerDecision(action="move", attempted_move=result.move.uci())

        except chess.engine.EngineError as e:
            raise RuntimeError("Stockfish failed to generate move: {}".format(e)) from e

    def close(self) -> None:
        """Terminate the Stockfish subprocess if it was started.

        Always invoke this method (or wrap the player in try/finally)
        to prevent orphaned engine processes after exceptions.
        """
        if self.engine is not None:
            try:
                self.engine.quit()
                logger.debug("Stockfish engine closed successfully")
            except Exception as e:
                logger.error("Error closing Stockfish engine: {}", e)
            finally:
                self.engine = None
