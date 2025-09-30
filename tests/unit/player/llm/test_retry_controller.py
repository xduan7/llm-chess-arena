"""Tests for the retry controller component."""

from __future__ import annotations

from llm_chess_arena.player.llm.decision import RetryAttempt, RetryController


def test_iter_attempts_yields_expected_count() -> None:
    """Iterating should produce retry attempts including initial try."""
    controller = RetryController(max_retries=2)
    attempts = list(controller.iter_attempts())

    assert attempts == [
        RetryAttempt(attempt_number=1, max_attempts=3),
        RetryAttempt(attempt_number=2, max_attempts=3),
        RetryAttempt(attempt_number=3, max_attempts=3),
    ]


def test_create_resignation_logs() -> None:
    """Generating a resignation should produce a resign decision."""
    controller = RetryController(max_retries=0)
    decision = controller.create_resignation()

    assert decision.action == "resign"
    assert decision.attempted_move is None
