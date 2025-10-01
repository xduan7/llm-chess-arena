"""Unit tests for metrics factory."""

from __future__ import annotations

from llm_chess_arena.config import MetricsConfig, MoveQualityThresholdsConfig
from llm_chess_arena.factory.metrics_factory import MetricsFactory
from llm_chess_arena.metrics import MetricsTracker


def test_create_metrics_tracker() -> None:
    """Verify factory produces a tracker instance from configuration."""
    config = MetricsConfig(
        max_centipawn_loss_per_move=1000,
        stockfish_depth=8,
        stockfish_binary_path=None,
        stockfish_engine_options={"Threads": 1},
        quality_thresholds=MoveQualityThresholdsConfig(
            excellent=25,
            good=75,
            inaccuracy=150,
            mistake=300,
        ),
    )

    tracker = MetricsFactory.create_metrics_tracker(config)

    assert isinstance(tracker, MetricsTracker)
    assert tracker.enabled in {True, False}
