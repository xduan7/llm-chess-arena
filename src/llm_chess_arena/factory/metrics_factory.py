"""Factory for metrics tracker construction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Any

from llm_chess_arena.core.policies import config_operation
from llm_chess_arena.metrics import MetricsTracker, MoveQualityThresholds

if TYPE_CHECKING:  # pragma: no cover
    from llm_chess_arena.config import MetricsConfig


class MetricsFactory:
    """Create metrics trackers based on configuration."""

    @staticmethod
    @config_operation
    def create_metrics_tracker(
        metrics_cfg: "MetricsConfig",
    ) -> MetricsTracker:
        """Create a Stockfish-based metrics tracker from configuration.

        Args:
            metrics_cfg: Metrics configuration containing Stockfish settings and quality thresholds.

        Returns:
            MetricsTracker: Configured metrics tracker for move evaluation.
        """
        engine_options: Mapping[str, Any] | None = None
        if metrics_cfg.stockfish_engine_options:
            engine_options = dict(metrics_cfg.stockfish_engine_options)

        thresholds_cfg = metrics_cfg.quality_thresholds
        thresholds = MoveQualityThresholds(
            excellent=thresholds_cfg.excellent,
            good=thresholds_cfg.good,
            inaccuracy=thresholds_cfg.inaccuracy,
            mistake=thresholds_cfg.mistake,
        )

        return MetricsTracker.from_stockfish(
            depth=metrics_cfg.stockfish_depth,
            binary_path=metrics_cfg.stockfish_binary_path,
            engine_options=engine_options,
            thresholds=thresholds,
            max_centipawn_loss=metrics_cfg.max_centipawn_loss_per_move,
        )
