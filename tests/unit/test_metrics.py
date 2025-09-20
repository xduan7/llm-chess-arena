"""Unit tests for metrics evaluation and aggregation utilities."""

import chess

from llm_chess_arena.metrics import (
    MoveMetrics,
    MetricsTracker,
    MoveQuality,
    classify_move_quality,
)


class DummyEvaluator:
    """Simple evaluator returning predefined metrics objects."""

    def __init__(self, outputs):
        self.outputs = outputs
        self.index = 0
        self.closed = False

    def evaluate_move(
        self, board: chess.Board, move: chess.Move
    ) -> MoveMetrics:  # noqa: D401
        metrics = self.outputs[self.index]
        self.index += 1
        return metrics

    def close(self) -> None:  # noqa: D401
        self.closed = True


def test_metrics_tracker_records_and_summarizes_moves():
    """Tracker aggregates centipawn loss and best-move hits per color."""
    metrics_sequence = [
        MoveMetrics(
            player_color="white",
            move_uci="e2e4",
            best_move_uci="e2e4",
            centipawn_loss=5.0,
            win_probability_delta=0.02,
            best_move_hit=False,
            quality=MoveQuality.EXCELLENT,
        ),
        MoveMetrics(
            player_color="black",
            move_uci="e7e5",
            best_move_uci="e7e5",
            centipawn_loss=0.0,
            win_probability_delta=0.0,
            best_move_hit=True,
            quality=MoveQuality.BEST,
        ),
        MoveMetrics(
            player_color="white",
            move_uci="g1f3",
            best_move_uci="g1f3",
            centipawn_loss=0.0,
            win_probability_delta=0.0,
            best_move_hit=True,
            quality=MoveQuality.BEST,
        ),
    ]

    dummy_evaluator = DummyEvaluator(metrics_sequence)
    tracker = MetricsTracker(dummy_evaluator)

    board = chess.Board()
    moves = ["e2e4", "e7e5", "g1f3"]
    player_names = ["White", "Black", "White"]

    for move_uci, player_name in zip(moves, player_names, strict=True):
        move = chess.Move.from_uci(move_uci)
        board_before = board.copy()
        board.push(move)
        tracker.record_move(board_before, move, player_name=player_name)

    summary = tracker.summarize()

    assert summary["white"].moves_evaluated == 2
    assert summary["black"].moves_evaluated == 1
    assert summary["white"].average_centipawn_loss == 2.5
    assert summary["white"].best_move_hit_rate == 0.5
    assert summary["white"].quality_counts[MoveQuality.EXCELLENT] == 1
    assert summary["white"].quality_counts[MoveQuality.BEST] == 1
    assert summary["black"].average_centipawn_loss == 0.0
    assert summary["black"].best_move_hit_rate == 1.0
    assert summary["black"].quality_counts[MoveQuality.BEST] == 1

    tracker.close()
    assert dummy_evaluator.closed


def test_metrics_tracker_handles_disabled_evaluator():
    """When evaluator is missing, tracker skips metrics safely."""
    tracker = MetricsTracker(evaluator=None)
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    board_before = board.copy()
    board.push(move)

    result = tracker.record_move(board_before, move, player_name="White")
    assert result is None

    summary = tracker.summarize()
    assert summary["white"].moves_evaluated == 0
    assert summary["white"].average_centipawn_loss is None
    assert summary["white"].best_move_hit_rate is None
    assert summary["white"].quality_counts == {}

    tracker.close()


class FailingEvaluator:
    """Evaluator that always raises to simulate engine failure."""

    def __init__(self) -> None:
        self.closed = False

    def evaluate_move(
        self, board: chess.Board, move: chess.Move
    ) -> MoveMetrics:  # noqa: D401
        raise RuntimeError("engine failure")

    def close(self) -> None:  # noqa: D401
        self.closed = True


def test_metrics_tracker_disables_after_evaluator_error():
    """Tracker should disable metrics after evaluator errors."""
    failing = FailingEvaluator()
    tracker = MetricsTracker(failing)
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")

    board_before = board.copy()
    board.push(move)

    result = tracker.record_move(board_before, move, player_name="White")

    assert result is None
    assert not tracker.enabled
    assert failing.closed

    tracker.close()


def test_classify_move_quality_respects_zero_loss() -> None:
    """Zero centipawn loss should still count as a best-quality move."""
    result = classify_move_quality(
        best_move_hit=False,
        centipawn_loss=0.0,
        best_move_is_mate=False,
        played_move_is_mate=False,
    )
    assert result == MoveQuality.BEST


def test_classify_move_quality_missed_mate_is_blunder() -> None:
    """Missing a forced mate should be treated as a blunder regardless of loss size."""
    result = classify_move_quality(
        best_move_hit=False,
        centipawn_loss=5.0,
        best_move_is_mate=True,
        played_move_is_mate=False,
    )
    assert result == MoveQuality.BLUNDER


def test_classify_move_quality_threshold_boundaries() -> None:
    """Boundary centipawn losses fall into the documented categories."""
    assert (
        classify_move_quality(
            best_move_hit=False,
            centipawn_loss=49.9,
            best_move_is_mate=False,
            played_move_is_mate=False,
        )
        == MoveQuality.EXCELLENT
    )
    assert (
        classify_move_quality(
            best_move_hit=False,
            centipawn_loss=50.0,
            best_move_is_mate=False,
            played_move_is_mate=False,
        )
        == MoveQuality.GOOD
    )
    assert (
        classify_move_quality(
            best_move_hit=False,
            centipawn_loss=100.0,
            best_move_is_mate=False,
            played_move_is_mate=False,
        )
        == MoveQuality.INACCURACY
    )
    assert (
        classify_move_quality(
            best_move_hit=False,
            centipawn_loss=200.0,
            best_move_is_mate=False,
            played_move_is_mate=False,
        )
        == MoveQuality.MISTAKE
    )
    assert (
        classify_move_quality(
            best_move_hit=False,
            centipawn_loss=300.0,
            best_move_is_mate=False,
            played_move_is_mate=False,
        )
        == MoveQuality.BLUNDER
    )
