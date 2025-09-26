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
        """
        self.max_retries = max_retries
        self._max_attempts = max_retries + 1
        self.attempts_used: int = 0

    def iter_attempts(self) -> Iterator[RetryAttempt]:
        """Generate retry attempts up to the configured maximum.

        Yields:
            RetryAttempt: Metadata for each retry attempt.
        """
        for attempt_number in range(1, self._max_attempts + 1):
            yield RetryAttempt(
                attempt_number=attempt_number, max_attempts=self._max_attempts
            )

    def mark_attempt(self, attempt_number: int) -> None:
        """Record that an attempt has been used.

        Args:
            attempt_number: The attempt number that was executed.
        """
        self.attempts_used = attempt_number

    def create_resignation(self, reason: str) -> PlayerDecision:
        """Create a resignation decision after exhausting retry attempts.

        Args:
            reason: Description of why resignation was necessary.

        Returns:
            PlayerDecision: Resignation decision with logging.
        """
        return PlayerDecision(action="resign")
