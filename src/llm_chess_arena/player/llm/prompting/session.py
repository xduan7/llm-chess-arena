"""Prompt session helper maintaining retry context for LLM queries."""

from __future__ import annotations

from dataclasses import dataclass

from .handlers import BaseLLMMoveHandler
from llm_chess_arena.types import PlayerDecisionContext


@dataclass
class PromptSession:
    """Generate initial and retry prompts while preserving context."""

    handler: BaseLLMMoveHandler
    context: PlayerDecisionContext
    _current_prompt: str | None = None

    def ensure_initial_prompt(self) -> str:
        """Generate and return the initial prompt for the session.

        Returns:
            str: The initial prompt generated from the context, cached for reuse.
        """
        if self._current_prompt is None:
            self._current_prompt = self.handler.get_prompt(**self.context.model_dump())
        return self._current_prompt

    def build_retry_prompt(
        self,
        *,
        exception_name: str,
        last_response: str | None,
        last_attempted_move: str | None,
    ) -> str:
        """Generate a retry prompt incorporating failure context.

        Args:
            exception_name: Name of the exception that triggered the retry.
            last_response: The previous LLM response that failed to parse.
            last_attempted_move: The move that was attempted from the response.

        Returns:
            str: Retry prompt with context about the previous failure.
        """
        base_prompt = self._current_prompt or self.ensure_initial_prompt()
        self._current_prompt = self.handler.get_retry_prompt(
            exception_name=exception_name,
            last_prompt=base_prompt,
            last_response=last_response,
            last_attempted_move=last_attempted_move,
            **self.context.model_dump(),
        )
        return self._current_prompt

    @property
    def current_prompt(self) -> str | None:
        """Get the most recent prompt used for the session.

        Returns:
            str | None: Current prompt text, or None if no prompt has been generated.
        """
        return self._current_prompt
