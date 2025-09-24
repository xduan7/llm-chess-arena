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
    def create_metrics_tracker(config: "MetricsConfig") -> MetricsTracker:
        """Create a Stockfish-based metrics tracker from configuration.

        Args:
            config: Metrics configuration containing Stockfish settings and quality thresholds.

        Returns:
            MetricsTracker: Configured metrics tracker for move evaluation.
        """
        engine_options: Mapping[str, Any] | None = None
        if config.stockfish_engine_options:
            engine_options = dict(config.stockfish_engine_options)

        thresholds_config = config.quality_thresholds
        thresholds = MoveQualityThresholds(
            excellent=thresholds_config.excellent,
            good=thresholds_config.good,
            inaccuracy=thresholds_config.inaccuracy,
            mistake=thresholds_config.mistake,
        )

        return MetricsTracker.from_stockfish(
            depth=config.stockfish_depth,
            binary_path=config.stockfish_binary_path,
            engine_options=engine_options,
            thresholds=thresholds,
        )
