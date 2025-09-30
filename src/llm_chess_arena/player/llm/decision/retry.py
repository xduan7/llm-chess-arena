"""Retry budget helper for LLM move generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from llm_chess_arena.types import PlayerDecision


@dataclass(frozen=True)
class RetryAttempt:
    """Metadata describing a single retry attempt."""

    attempt_number: int
    max_attempts: int

    @property
    def is_final_attempt(self) -> bool:
        """Return ``True`` when this attempt equals the maximum allowed."""
        return self.attempt_number == self.max_attempts


class RetryController:
    """Track retry attempts and generate resignation decisions."""

    def __init__(self, max_retries: int) -> None:
        """Initialize retry controller with maximum retry limit.

        Args:
            max_retries: Maximum number of retries allowed before resignation.
                        0 means one attempt with no retries, 3 means up to 4 total attempts.

        Raises:
            ValueError: If max_retries is negative.
        """
        if max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {max_retries}")

        self.max_attempts = max_retries + 1

    def iter_attempts(self) -> Iterator[RetryAttempt]:
        """Generate retry attempts up to the configured maximum.

        Yields:
            RetryAttempt: Metadata for each retry attempt.
        """
        for attempt_number in range(1, self.max_attempts + 1):
            yield RetryAttempt(
                attempt_number=attempt_number, max_attempts=self.max_attempts
            )

    def create_resignation(self) -> PlayerDecision:
        """Create a resignation decision after exhausting retry attempts.

        Returns:
            PlayerDecision: Resignation decision.
        """
        return PlayerDecision(action="resign")
