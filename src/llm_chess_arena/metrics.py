"""Utilities for computing and aggregating chess evaluation metrics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Protocol

import chess
import chess.engine
from loguru import logger

from llm_chess_arena.player.stockfish_player import StockfishPlayer
from llm_chess_arena.types import Color

MATE_SCORE = 100_000
ZERO_LOSS_EPSILON = 1e-6
CP_LOSS_THRESHOLD_EXCELLENT = 50
CP_LOSS_THRESHOLD_GOOD = 100
CP_LOSS_THRESHOLD_INACCURACY = 200
CP_LOSS_THRESHOLD_MISTAKE = 300


class MoveQuality(Enum):
    """Discrete categorization of move quality based on engine evaluation."""

    BEST = "best"
    EXCELLENT = "excellent"
    GOOD = "good"
    INACCURACY = "inaccuracy"
    MISTAKE = "mistake"
    BLUNDER = "blunder"


MOVE_QUALITY_ORDER: tuple[MoveQuality, ...] = (
    MoveQuality.BEST,
    MoveQuality.EXCELLENT,
    MoveQuality.GOOD,
    MoveQuality.INACCURACY,
    MoveQuality.MISTAKE,
    MoveQuality.BLUNDER,
)


def classify_move_quality(
    *,
    best_move_hit: bool,
    centipawn_loss: float,
    best_move_is_mate: bool,
    played_move_is_mate: bool,
) -> MoveQuality:
    """Categorize move quality based on engine evaluation results.

    Args:
        best_move_hit: Whether the player matched the engine's top move.
        centipawn_loss: Non-negative difference between the best move and played move.
        best_move_is_mate: True if the engine's recommended move gives a mating line.
        played_move_is_mate: True if the played move still yields a mating line.

    Returns:
        MoveQuality: Discrete quality label for the move.
    """
    if best_move_is_mate and not played_move_is_mate:
        return MoveQuality.BLUNDER

    if best_move_hit or centipawn_loss <= ZERO_LOSS_EPSILON:
        return MoveQuality.BEST

    loss = centipawn_loss
    if loss < CP_LOSS_THRESHOLD_EXCELLENT:
        return MoveQuality.EXCELLENT
    if loss < CP_LOSS_THRESHOLD_GOOD:
        return MoveQuality.GOOD
    if loss < CP_LOSS_THRESHOLD_INACCURACY:
        return MoveQuality.INACCURACY
    if loss < CP_LOSS_THRESHOLD_MISTAKE:
        return MoveQuality.MISTAKE
    return MoveQuality.BLUNDER


@dataclass(frozen=True)
class MoveMetrics:
    """Evaluation metrics for a single move."""

    player_color: Color
    move_uci: str
    best_move_uci: str
    centipawn_loss: float
    win_probability_delta: float
    best_move_hit: bool
    quality: MoveQuality
    best_move_centipawns: float | None = None
    actual_centipawns: float | None = None


@dataclass(frozen=True)
class MetricsSummary:
    """Aggregated metrics for a player's moves."""

    moves_evaluated: int
    average_centipawn_loss: float | None
    best_move_hit_rate: float | None
    quality_counts: Mapping[MoveQuality, int]


class MoveMetricsEvaluator(Protocol):
    """Protocol describing move evaluation implementations."""

    def evaluate_move(self, board: chess.Board, move: chess.Move) -> MoveMetrics:
        """Compute quality metrics for a move.

        Args:
            board: Position from which the move originates.
            move: Candidate move under evaluation.

        Returns:
            MoveMetrics: Evaluation results for the move.
        """

    def close(self) -> None:
        """Release evaluator resources."""
        ...


class StockfishMetricsEvaluator:
    """Compute move quality metrics using a Stockfish analysis engine."""

    def __init__(
        self,
        *,
        depth: int = 10,
        binary_path: str | None = None,
        engine_options: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize the Stockfish-backed evaluator.

        Args:
            depth: Search depth for analysis.
            binary_path: Optional explicit Stockfish binary path.
            engine_options: Optional UCI engine options.
        """
        self.depth = depth
        self.binary_path = StockfishPlayer._find_stockfish_binary(
            binary_path
        )  # noqa: SLF001
        self.engine_options = dict(engine_options or {})
        self._engine: chess.engine.SimpleEngine | None = None
        self._wdl_model: chess.engine.WdlModel = "sf"

    def evaluate_move(self, board: chess.Board, move: chess.Move) -> MoveMetrics:
        """Evaluate ``move`` and compare it with the engine-recommended alternative.

        Args:
            board: Position from which the move should be evaluated.
            move: Candidate move produced by the player under evaluation.

        Returns:
            MoveMetrics: Computed quality metrics for the candidate move.

        Raises:
            RuntimeError: If Stockfish fails to return an evaluation.
        """
        engine = self._ensure_engine()

        board_for_engine = board.copy(stack=False)
        player_color: Color = (
            "white" if board_for_engine.turn == chess.WHITE else "black"
        )
        player_turn = chess.WHITE if player_color == "white" else chess.BLACK

        limit = chess.engine.Limit(depth=self.depth)

        best_move_result = engine.play(board_for_engine, limit)
        best_move = best_move_result.move
        if best_move is None:
            raise RuntimeError("Stockfish did not return a best move during evaluation")

        best_metrics = self._evaluate_resulting_position(
            board_for_engine, best_move, player_turn
        )
        actual_metrics = self._evaluate_resulting_position(
            board_for_engine, move, player_turn
        )

        centipawn_loss = max(0.0, best_metrics.centipawns - actual_metrics.centipawns)
        win_probability_delta = (
            best_metrics.win_probability - actual_metrics.win_probability
        )
        best_move_hit = move == best_move

        return MoveMetrics(
            player_color=player_color,
            move_uci=move.uci(),
            best_move_uci=best_move.uci(),
            centipawn_loss=centipawn_loss,
            win_probability_delta=win_probability_delta,
            best_move_hit=best_move_hit,
            quality=classify_move_quality(
                best_move_hit=best_move_hit,
                centipawn_loss=centipawn_loss,
                best_move_is_mate=best_metrics.is_mate,
                played_move_is_mate=actual_metrics.is_mate,
            ),
            best_move_centipawns=best_metrics.centipawns,
            actual_centipawns=actual_metrics.centipawns,
        )

    def close(self) -> None:
        """Shut down the Stockfish engine if it was started."""
        if self._engine is None:
            return
        try:
            self._engine.quit()
        except Exception as exc:  # pragma: no cover - defensive cleanup
            logger.warning("Error while closing Stockfish metrics engine: {}", exc)
        finally:
            self._engine = None

    def _ensure_engine(self) -> chess.engine.SimpleEngine:
        """Ensure the Stockfish engine process is running and configured."""
        if self._engine is not None:
            return self._engine

        engine = chess.engine.SimpleEngine.popen_uci(self.binary_path)
        try:
            if self.engine_options:
                engine.configure(self.engine_options)
        except Exception:
            engine.quit()
            raise

        self._engine = engine
        return engine

    def _evaluate_resulting_position(
        self,
        board: chess.Board,
        move: chess.Move,
        player_turn: chess.Color,
    ) -> "_PositionEvaluation":
        """Evaluate the board after applying ``move`` from ``board``."""
        engine = self._ensure_engine()
        limit = chess.engine.Limit(depth=self.depth)

        resulting_board = board.copy(stack=False)
        resulting_board.push(move)

        info = engine.analyse(resulting_board, limit)
        score = info.get("score")
        if score is None:
            raise RuntimeError("Stockfish analysis did not include a score field")

        pov_score = score.pov(player_turn)
        centipawns = float(pov_score.score(mate_score=MATE_SCORE))

        wdl = pov_score.wdl(model=self._wdl_model)
        win_probability = wdl.expectation()

        return _PositionEvaluation(
            centipawns=centipawns,
            win_probability=win_probability,
            is_mate=pov_score.is_mate(),
        )


@dataclass(frozen=True)
class _PositionEvaluation:
    centipawns: float
    win_probability: float
    is_mate: bool


class MetricsTracker:
    """Orchestrates evaluation and aggregation of move metrics."""

    def __init__(self, evaluator: MoveMetricsEvaluator | None) -> None:
        """Create a tracker that optionally evaluates moves using ``evaluator``.

        Args:
            evaluator: Move metrics evaluator or ``None`` to disable evaluation.
        """
        self._evaluator = evaluator
        self._metrics_by_color: dict[Color, list[MoveMetrics]] = {
            "white": [],
            "black": [],
        }
        self._metrics_disabled_logged = False

    @property
    def enabled(self) -> bool:
        """Return whether metrics evaluation is available."""
        return self._evaluator is not None

    @classmethod
    def from_stockfish(
        cls,
        *,
        depth: int = 10,
        binary_path: str | None = None,
        engine_options: Mapping[str, Any] | None = None,
    ) -> "MetricsTracker":
        """Construct a tracker backed by a Stockfish-powered evaluator.

        Args:
            depth: Search depth used for Stockfish analysis.
            binary_path: Optional explicit path to the Stockfish executable.
            engine_options: Optional UCI options passed to Stockfish.

        Returns:
            MetricsTracker: Tracker instance that evaluates moves with Stockfish
            when the engine is available; otherwise metrics collection is disabled.
        """
        try:
            evaluator: MoveMetricsEvaluator | None = StockfishMetricsEvaluator(
                depth=depth,
                binary_path=binary_path,
                engine_options=engine_options,
            )
        except (FileNotFoundError, chess.engine.EngineError, OSError) as exc:
            logger.warning(
                "Stockfish unavailable - metrics evaluation disabled. "
                "Set STOCKFISH_BINARY_PATH or install Stockfish to enable metrics."
            )
            logger.debug("Stockfish initialization failure details: {}", exc)
            evaluator = None
        return cls(evaluator)

    def record_move(
        self,
        board_before_move: chess.Board,
        move: chess.Move,
        *,
        player_name: str,
    ) -> MoveMetrics | None:
        """Evaluate ``move`` and record the resulting metrics.

        Args:
            board_before_move: Position prior to applying ``move``.
            move: Move executed by the player.
            player_name: Name of the player whose move is being recorded.

        Returns:
            MoveMetrics | None: Metrics for the move when evaluation succeeds;
            ``None`` if metrics are disabled or evaluation fails.
        """
        if self._evaluator is None:
            if not self._metrics_disabled_logged:
                logger.debug("Metrics evaluator unavailable - skipping move metrics")
                self._metrics_disabled_logged = True
            return None

        try:
            metrics = self._evaluator.evaluate_move(board_before_move, move)
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("Disabling metrics after evaluator error: {}", exc)
            if self._evaluator is not None:
                try:
                    self._evaluator.close()
                except Exception as close_exc:  # pragma: no cover
                    logger.debug(
                        "Error while closing evaluator after failure: {}",
                        close_exc,
                    )
            self._evaluator = None
            self._metrics_disabled_logged = True
            return None

        self._metrics_by_color[metrics.player_color].append(metrics)

        logger.debug(
            "Metrics for {} -> {}: centipawn_loss={:.1f}, win_prob_delta={:.3f}, best_move_hit={}, quality={}",
            player_name,
            metrics.move_uci,
            metrics.centipawn_loss,
            metrics.win_probability_delta,
            metrics.best_move_hit,
            metrics.quality.value,
        )
        return metrics

    def summarize(self) -> dict[Color, MetricsSummary]:
        """Aggregate metrics for each player color.

        Returns:
            dict[Color, MetricsSummary]: Summary metrics keyed by player color.
        """
        summaries: dict[Color, MetricsSummary] = {}
        for color, metrics_list in self._metrics_by_color.items():
            moves_evaluated = len(metrics_list)
            if moves_evaluated == 0:
                summaries[color] = MetricsSummary(
                    moves_evaluated=0,
                    average_centipawn_loss=None,
                    best_move_hit_rate=None,
                    quality_counts={},
                )
                continue

            average_centipawn_loss = (
                sum(m.centipawn_loss for m in metrics_list) / moves_evaluated
            )
            best_move_hits = sum(1 for m in metrics_list if m.best_move_hit)
            best_move_hit_rate = best_move_hits / moves_evaluated
            quality_counts = Counter(m.quality for m in metrics_list)
            summaries[color] = MetricsSummary(
                moves_evaluated=moves_evaluated,
                average_centipawn_loss=average_centipawn_loss,
                best_move_hit_rate=best_move_hit_rate,
                quality_counts=dict(quality_counts),
            )
        return summaries

    def close(self) -> None:
        """Close the underlying evaluator if present."""
        if self._evaluator is None:
            return
        self._evaluator.close()
        self._evaluator = None


__all__ = [
    "MOVE_QUALITY_ORDER",
    "MetricsTracker",
    "MoveMetrics",
    "MetricsSummary",
    "MoveQuality",
    "StockfishMetricsEvaluator",
    "classify_move_quality",
]
