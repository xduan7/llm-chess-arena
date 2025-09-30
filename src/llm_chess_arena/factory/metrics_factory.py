"""Factory for metrics tracker construction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Any

from llm_chess_arena.core.policies import config_operation
from llm_chess_arena.metrics import MetricsTracker, MoveQualityThresholds

if TYPE_CHECKING:  # pragma: no cover - typing only
    from llm_chess_arena.config import MetricsConfig


class MetricsFactory:
    """Create metrics trackers based on configuration."""

    @staticmethod
    @config_operation
    def create_metrics_tracker(
        metrics_config: "MetricsConfig",
    ) -> MetricsTracker:
        """Create a Stockfish-based metrics tracker from configuration.

        Args:
            metrics_config: Metrics configuration containing Stockfish settings and quality thresholds.

        Returns:
            MetricsTracker: Configured metrics tracker for move evaluation.
        """
        engine_options: Mapping[str, Any] | None = None
        if metrics_config.stockfish_engine_options:
            engine_options = dict(metrics_config.stockfish_engine_options)

        thresholds_config = metrics_config.quality_thresholds
        thresholds = MoveQualityThresholds(
            excellent=thresholds_config.excellent,
            good=thresholds_config.good,
            inaccuracy=thresholds_config.inaccuracy,
            mistake=thresholds_config.mistake,
        )

        return MetricsTracker.from_stockfish(
            depth=metrics_config.stockfish_depth,
            binary_path=metrics_config.stockfish_binary_path,
            engine_options=engine_options,
            thresholds=thresholds,
        )
