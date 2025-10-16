"""Utilities for computing and aggregating chess evaluation metrics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Protocol

import chess
import chess.engine
from loguru import logger

from llm_chess_arena.core.policies import metrics_operation
from llm_chess_arena.utils import find_stockfish_binary, initialize_stockfish_engine
from llm_chess_arena.types import PlayerColor

MATE_SCORE = 100_000
ZERO_LOSS_EPSILON = 1e-6
CP_LOSS_THRESHOLD_EXCELLENT = 50
CP_LOSS_THRESHOLD_GOOD = 100
CP_LOSS_THRESHOLD_INACCURACY = 200
CP_LOSS_THRESHOLD_MISTAKE = 300


@dataclass(frozen=True)
class MoveQualityThresholds:
    """Centipawn thresholds that define move quality buckets."""

    excellent: float = CP_LOSS_THRESHOLD_EXCELLENT
    good: float = CP_LOSS_THRESHOLD_GOOD
    inaccuracy: float = CP_LOSS_THRESHOLD_INACCURACY
    mistake: float = CP_LOSS_THRESHOLD_MISTAKE


DEFAULT_MOVE_QUALITY_THRESHOLDS = MoveQualityThresholds()


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
    thresholds: MoveQualityThresholds = DEFAULT_MOVE_QUALITY_THRESHOLDS,
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

    if centipawn_loss < thresholds.excellent:
        return MoveQuality.EXCELLENT
    if centipawn_loss < thresholds.good:
        return MoveQuality.GOOD
    if centipawn_loss < thresholds.inaccuracy:
        return MoveQuality.INACCURACY
    if centipawn_loss < thresholds.mistake:
        return MoveQuality.MISTAKE
    return MoveQuality.BLUNDER


@dataclass(frozen=True)
class MoveMetrics:
    """Evaluation metrics for a single move."""

    player_color: PlayerColor
    move_in_uci: str
    best_move_in_uci: str
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
        thresholds: MoveQualityThresholds | None = None,
        max_centipawn_loss: int | None = None,
    ) -> None:
        """Initialize the Stockfish-backed evaluator.

        Args:
            depth: Search depth for analysis.
            binary_path: Optional explicit Stockfish binary path.
            engine_options: Stockfish UCI options.
            thresholds: Move quality thresholds.
            max_centipawn_loss: Cap for centipawn loss per move to prevent mate positions from skewing ACPL.
        """
        self.depth = depth
        self.binary_path = find_stockfish_binary(binary_path)
        self.engine_options = dict(engine_options or {})
        self._engine: chess.engine.SimpleEngine | None = None
        self._win_draw_loss_model: chess.engine.WdlModel = "sf"
        self._thresholds = thresholds or DEFAULT_MOVE_QUALITY_THRESHOLDS
        self._max_centipawn_loss = max_centipawn_loss

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
        player_color: PlayerColor = (
            "white" if board_for_engine.turn == chess.WHITE else "black"
        )
        player_turn_color = chess.WHITE if player_color == "white" else chess.BLACK

        search_limit = chess.engine.Limit(depth=self.depth)

        best_move_result = engine.play(board_for_engine, search_limit)
        best_move = best_move_result.move
        if best_move is None:
            raise RuntimeError("Stockfish did not return a best move during evaluation")

        best_move_evaluation = self._evaluate_resulting_position(
            board_for_engine, best_move, player_turn_color
        )
        played_move_evaluation = self._evaluate_resulting_position(
            board_for_engine, move, player_turn_color
        )

        centipawn_loss = max(
            0.0,
            best_move_evaluation.centipawns - played_move_evaluation.centipawns,
        )

        # Apply cap if configured (prevents mate positions from skewing ACPL)
        if self._max_centipawn_loss is not None:
            centipawn_loss = min(centipawn_loss, float(self._max_centipawn_loss))

        win_probability_delta = (
            best_move_evaluation.win_probability
            - played_move_evaluation.win_probability
        )
        best_move_hit = move == best_move

        return MoveMetrics(
            player_color=player_color,
            move_in_uci=move.uci(),
            best_move_in_uci=best_move.uci(),
            centipawn_loss=centipawn_loss,
            win_probability_delta=win_probability_delta,
            best_move_hit=best_move_hit,
            quality=classify_move_quality(
                best_move_hit=best_move_hit,
                centipawn_loss=centipawn_loss,
                best_move_is_mate=best_move_evaluation.is_mate,
                played_move_is_mate=played_move_evaluation.is_mate,
                thresholds=self._thresholds,
            ),
            best_move_centipawns=best_move_evaluation.centipawns,
            actual_centipawns=played_move_evaluation.centipawns,
        )

    def close(self) -> None:
        """Shut down the Stockfish engine if it was started."""
        if self._engine is None:
            return
        try:
            self._engine.quit()
        except Exception as engine_close_error:  # pragma: no cover
            logger.warning(
                "Error while closing Stockfish metrics engine: {}", engine_close_error
            )
        finally:
            self._engine = None

    def _ensure_engine(self) -> chess.engine.SimpleEngine:
        """Ensure the Stockfish engine process is running and configured."""
        if self._engine is not None:
            return self._engine

        self._engine = initialize_stockfish_engine(
            self.binary_path, self.engine_options
        )
        return self._engine

    def _evaluate_resulting_position(
        self,
        board: chess.Board,
        move: chess.Move,
        player_turn_color: chess.Color,
    ) -> "_PositionEvaluation":
        """Evaluate the board after applying ``move`` from ``board``."""
        engine = self._ensure_engine()
        search_limit = chess.engine.Limit(depth=self.depth)

        resulting_board = board.copy(stack=False)
        resulting_board.push(move)

        analysis_result = engine.analyse(resulting_board, search_limit)
        evaluation_score = analysis_result.get("score")
        if evaluation_score is None:
            raise RuntimeError("Stockfish analysis did not include a score field")

        player_perspective_score = evaluation_score.pov(player_turn_color)
        centipawns = float(player_perspective_score.score(mate_score=MATE_SCORE))

        win_draw_loss_distribution = player_perspective_score.wdl(
            model=self._win_draw_loss_model
        )
        win_probability = win_draw_loss_distribution.expectation()

        return _PositionEvaluation(
            centipawns=centipawns,
            win_probability=win_probability,
            is_mate=player_perspective_score.is_mate(),
        )


@dataclass(frozen=True)
class _PositionEvaluation:
    """Lightweight container for engine evaluation results."""

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
        self._metrics_by_player_color: dict[PlayerColor, list[MoveMetrics]] = {
            "white": [],
            "black": [],
        }
        self._metrics_disabled_notice_logged = False

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
        thresholds: MoveQualityThresholds | None = None,
        max_centipawn_loss: int | None = None,
        require_stockfish: bool = True,
    ) -> "MetricsTracker":
        """Construct a tracker backed by a Stockfish-powered evaluator.

        Args:
            depth: Search depth used for Stockfish analysis.
            binary_path: Optional explicit path to the Stockfish executable.
            engine_options: Optional UCI options passed to Stockfish.
            thresholds: Optional override for move quality thresholds.
            max_centipawn_loss: Cap for centipawn loss per move (prevents mate scores from skewing ACPL).
            require_stockfish: If True (default), raises an exception when Stockfish is unavailable.
                If False, returns a disabled tracker that logs warnings.

        Returns:
            MetricsTracker: Tracker instance with Stockfish evaluator, or disabled tracker
            if require_stockfish=False and Stockfish is unavailable.

        Raises:
            RuntimeError: If require_stockfish=True and Stockfish cannot be initialized.
        """
        try:
            evaluator: MoveMetricsEvaluator | None = StockfishMetricsEvaluator(
                depth=depth,
                binary_path=binary_path,
                engine_options=engine_options,
                thresholds=thresholds,
                max_centipawn_loss=max_centipawn_loss,
            )
        except (
            FileNotFoundError,
            chess.engine.EngineError,
            OSError,
        ) as stockfish_initialization_error:
            if require_stockfish:
                raise RuntimeError(
                    "Stockfish not found but metrics are required. "
                    "Either install Stockfish or disable metrics. "
                    "Install: brew install stockfish (macOS) or apt install stockfish (Ubuntu)"
                ) from stockfish_initialization_error

            logger.warning(
                "Stockfish unavailable - metrics evaluation disabled. "
                "Set STOCKFISH_BINARY_PATH or install Stockfish to enable metrics."
            )
            logger.info(
                "Install Stockfish: brew install stockfish (macOS) or apt install stockfish (Ubuntu)"
            )
            logger.debug(
                "Stockfish initialization failure details: {}",
                stockfish_initialization_error,
            )
            evaluator = None
        return cls(evaluator)

    @metrics_operation
    def record_move(
        self,
        board_before_move: chess.Board,
        move: chess.Move,
        player_name: str | None = None,
    ) -> MoveMetrics | None:
        """Evaluate ``move`` and record the resulting metrics.

        Args:
            board_before_move: Position prior to applying ``move``.
            move: Move executed by the player.
            player_name: Optional display name for logging context.

        Returns:
            MoveMetrics | None: Metrics for the move when evaluation succeeds;
            ``None`` if metrics are disabled or evaluation fails.
        """
        if self._evaluator is None:
            if not self._metrics_disabled_notice_logged:
                logger.debug("Metrics evaluator unavailable - skipping move metrics")
                self._metrics_disabled_notice_logged = True
            return None

        try:
            move_metrics = self._evaluator.evaluate_move(board_before_move, move)
        except Exception as evaluation_error:  # pragma: no cover
            logger.warning(
                "Disabling metrics after evaluator error: {}", evaluation_error
            )
            if self._evaluator is not None:
                try:
                    self._evaluator.close()
                except Exception as close_error:  # pragma: no cover
                    logger.debug(
                        "Error while closing evaluator after failure: {}",
                        close_error,
                    )
            self._evaluator = None
            self._metrics_disabled_notice_logged = True
            return None

        self._metrics_by_player_color[move_metrics.player_color].append(move_metrics)

        player_label = player_name or move_metrics.player_color
        best_move_text = (
            "matched engine's best move"
            if move_metrics.best_move_hit
            else "did not match best move"
        )
        logger.debug(
            "Move evaluation for {} playing {}: {:.1f} centipawn loss, {:.1%} win probability change, {}, quality: {}",
            player_label,
            move_metrics.move_in_uci,
            move_metrics.centipawn_loss,
            move_metrics.win_probability_delta,
            best_move_text,
            move_metrics.quality.value,
        )
        return move_metrics

    def summarize(self) -> dict[PlayerColor, MetricsSummary]:
        """Aggregate metrics for each player color.

        Returns:
            dict[PlayerColor, MetricsSummary]: Summary metrics keyed by player color.
        """
        summary_by_player_color: dict[PlayerColor, MetricsSummary] = {}
        for player_color, player_metrics in self._metrics_by_player_color.items():
            moves_evaluated = len(player_metrics)
            if moves_evaluated == 0:
                summary_by_player_color[player_color] = MetricsSummary(
                    moves_evaluated=0,
                    average_centipawn_loss=None,
                    best_move_hit_rate=None,
                    quality_counts={},
                )
                continue

            average_centipawn_loss = (
                sum(metric.centipawn_loss for metric in player_metrics)
                / moves_evaluated
            )
            best_move_hits = sum(1 for metric in player_metrics if metric.best_move_hit)
            best_move_hit_rate = best_move_hits / moves_evaluated
            quality_counts = Counter(metric.quality for metric in player_metrics)
            summary_by_player_color[player_color] = MetricsSummary(
                moves_evaluated=moves_evaluated,
                average_centipawn_loss=average_centipawn_loss,
                best_move_hit_rate=best_move_hit_rate,
                quality_counts=dict(quality_counts),
            )
        return summary_by_player_color

    def get_ordered_move_qualities(
        self, move_stack: list[Any]
    ) -> list[MoveQuality | None]:
        """Return move qualities ordered to match the game's move sequence.

        Args:
            move_stack: The game's move stack to determine ordering.

        Returns:
            list[MoveQuality | None]: Move qualities in chronological order,
            or None for moves without metrics.
        """
        white_metrics = self._metrics_by_player_color["white"]
        black_metrics = self._metrics_by_player_color["black"]

        move_qualities: list[MoveQuality | None] = []
        white_metric_index = black_metric_index = 0

        for move_index, move in enumerate(move_stack):
            move_color = "white" if move_index % 2 == 0 else "black"

            if move_color == "white" and white_metric_index < len(white_metrics):
                player_move_metrics = white_metrics[white_metric_index]
                move_qualities.append(
                    player_move_metrics.quality if player_move_metrics else None
                )
                white_metric_index += 1
            elif move_color == "black" and black_metric_index < len(black_metrics):
                player_move_metrics = black_metrics[black_metric_index]
                move_qualities.append(
                    player_move_metrics.quality if player_move_metrics else None
                )
                black_metric_index += 1
            else:
                # No metrics available for this move
                move_qualities.append(None)

        return move_qualities

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
