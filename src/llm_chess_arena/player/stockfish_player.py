"""Stockfish-backed chess player implementation."""

from __future__ import annotations

from typing import Any, Mapping

import chess
import chess.engine
from loguru import logger

from llm_chess_arena.player.base_player import BasePlayer
from llm_chess_arena.utils import find_stockfish_binary
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
        self.binary_path = find_stockfish_binary(binary_path)
        self.engine_limits = (
            dict(engine_limits) if engine_limits else DEFAULT_ENGINE_LIMITS.copy()
        )
        self.engine_options = dict(engine_options) if engine_options else {}

        logger.debug(
            "StockfishPlayer configured with limits={} (engine not started yet)",
            self.engine_limits,
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
        """Query Stockfish for the strongest move and wrap the response."""
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
