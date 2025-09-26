"""LLM-backed player that orchestrates prompting, voting, and retries."""

from __future__ import annotations

from loguru import logger

from llm_chess_arena.exceptions import (
    AmbiguousMoveError,
    IllegalMoveError,
    InvalidMoveError,
    LLMPermanentError,
    LLMEmptyResponseError,
)
from llm_chess_arena.player.base_player import BasePlayer
from llm_chess_arena.player.llm.connector import LLMConnector, UsageRecord
from llm_chess_arena.player.llm.prompting import BaseLLMMoveHandler
from llm_chess_arena.player.llm.decision import MoveParser
from llm_chess_arena.player.llm.prompting import PromptSession
from llm_chess_arena.player.llm.decision import RetryController
from llm_chess_arena.player.llm.decision import VoteAggregator
from llm_chess_arena.types import Color, PlayerDecision, PlayerDecisionContext


class LLMPlayer(BasePlayer):
    """Chess player that queries an LLM and applies optional majority voting."""

    def __init__(
        self,
        *,
        name: str | None = None,
        color: Color,
        connector: LLMConnector,
        handler: BaseLLMMoveHandler,
        max_move_retries: int = 3,
        num_votes: int = 1,
    ) -> None:
        """Initialize an LLM-backed chess player.

        Args:
            name: Display name for the player. Defaults to connector.model.
            color: Chess side this player controls.
            connector: LLM connector for API communication.
            handler: Handler for parsing and formatting LLM responses.
            max_move_retries: Maximum retries for invalid moves before resignation.
            num_votes: Number of LLM responses to aggregate via majority voting.

        Raises:
            ValueError: If num_votes is less than 1.
        """
        if num_votes < 1:
            raise ValueError(f"`num_votes` must be >= 1, got {num_votes}")

        super().__init__(name or connector.model, color)
        self.connector = connector
        self.handler = handler
        self.max_move_retries = max_move_retries
        self.num_votes = num_votes

        self._vote_aggregator = VoteAggregator(handler)
        self._move_parser = MoveParser()
        self._retry_controller = RetryController(max_move_retries)

        self.last_move_attempts: int = 0
        self.last_move_decision: PlayerDecision | None = None

    def _make_decision(self, context: PlayerDecisionContext) -> PlayerDecision:
        """Produce a move decision after coordinating prompts, voting, and retries.

        Args:
            context: Normalized snapshot of the current board state.

        Returns:
            PlayerDecision: Validated decision ready for execution.

        Raises:
            LLMPermanentError: When the provider reports an unrecoverable API error.
        """
        logger.info(
            "LLM player {} starting move decision for position with {} legal moves",
            self.name,
            len(context.legal_moves_in_uci),
        )

        prompt_session = PromptSession(self.handler, context)
        decision: PlayerDecision | None = None

        # RetryController handles prompt-level retries when parsing fails.
        # Network failures are surfaced immediately so we do not stack retry loops.
        for attempt in self._retry_controller.iter_attempts():
            self._retry_controller.mark_attempt(attempt.attempt_number)
            self.last_move_attempts = attempt.attempt_number

            if attempt.attempt_number > 1:
                logger.info(
                    "LLM player {} retry attempt {}/{} after previous failure",
                    self.name,
                    attempt.attempt_number,
                    attempt.max_attempts,
                )
            else:
                logger.debug(
                    "LLM player {} initial move attempt for position {}",
                    self.name,
                    context.board_in_fen,
                )

            prompt = (
                prompt_session.current_prompt or prompt_session.ensure_initial_prompt()
            )

            try:
                responses = self.connector.query(prompt, n=self.num_votes)
                logger.debug(
                    "LLM player {} received {} responses (avg length: {} chars)",
                    self.name,
                    len(responses),
                    (
                        sum(len(r) for r in responses) // len(responses)
                        if responses
                        else 0
                    ),
                )
                decision = self._vote_aggregator.aggregate_responses(responses)
                self._log_last_call_usage()
                logger.debug(
                    "LLM returned decision: action={}, move={}",
                    decision.action,
                    decision.attempted_move if decision.action == "move" else "N/A",
                )

                validated_decision = self._move_parser.validate_and_normalize(
                    decision, context.board_in_fen
                )
                self.last_move_decision = validated_decision

                logger.info(
                    "LLM player {} successfully generated valid move {} after {} attempt(s)",
                    self,
                    validated_decision.attempted_move,
                    attempt.attempt_number,
                )
                return validated_decision

            except LLMPermanentError:
                logger.error("Permanent LLM error - terminating move attempt")
                raise
            except (TimeoutError, ConnectionError) as exc:
                # Network failures mean the connector already exhausted its network retry budget.
                # Resign immediately - no move retries needed for network issues.
                logger.warning("{} resigned due to network failure: {}", self, str(exc))
                resignation = self._retry_controller.create_resignation()
                self.last_move_decision = resignation
                return resignation
            except (
                InvalidMoveError,
                IllegalMoveError,
                AmbiguousMoveError,
                LLMEmptyResponseError,
            ) as exc:
                # Handle empty response differently (no decision object exists)
                if isinstance(exc, LLMEmptyResponseError):
                    retry_status = (
                        f"Retrying move attempt (empty response: {exc})"
                        if not attempt.is_final_attempt
                        else f"No retries left after empty response: {exc}"
                    )
                    logger.warning(
                        "LLM player {} attempt {} failed with empty response: {}. {}",
                        self,
                        attempt.attempt_number,
                        exc,
                        retry_status,
                    )

                    if not attempt.is_final_attempt:
                        # No decision to add to retry prompt for empty responses
                        continue
                else:
                    # Handle invalid move errors (decision object exists)
                    invalid_move = decision.attempted_move if decision else "unknown"
                    retry_status = (
                        "Retrying with prior response and invalid move context"
                        if not attempt.is_final_attempt
                        else "No retries left"
                    )
                    logger.warning(
                        "LLM player {} attempt {} failed with {}: {}. Invalid move: '{}'. {}",
                        self,
                        attempt.attempt_number,
                        exc.__class__.__name__,
                        exc,
                        invalid_move,
                        retry_status,
                    )

                    if not attempt.is_final_attempt and decision is not None:
                        prompt_session.build_retry_prompt(
                            exception_name=exc.__class__.__name__,
                            last_response=getattr(decision, "response", None),
                            last_attempted_move=decision.attempted_move,
                        )
                        logger.debug(
                            "Generated retry prompt with error context for {}",
                            exc.__class__.__name__,
                        )
                        self.last_move_decision = decision
                        continue
                break
            except NotImplementedError:
                logger.error(
                    "LLM player {} returned unsupported action '{}', resigning.",
                    self,
                    decision.action if decision else "unknown",
                )
                resignation = PlayerDecision(action="resign")
                self.last_move_decision = resignation
                return resignation

        resignation = self._retry_controller.create_resignation()
        self.last_move_decision = resignation
        return resignation

    def close(self) -> None:
        """Release LLM connector resources."""
        if hasattr(self.connector, "close"):
            try:
                self.connector.close()
                logger.debug("LLM connector closed successfully")
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Error closing LLM connector: {}", exc)

    def get_usage_totals(self) -> UsageRecord:
        """Get cumulative LLM usage statistics for this player.

        Returns:
            UsageRecord: Total token usage and cost across all requests.
        """
        return self.connector.get_total_usage()

    def reset_usage(self) -> None:
        """Reset usage counters and move tracking to initial state."""
        reset_hook = getattr(self.connector, "reset_usage", None)
        if callable(reset_hook):
            reset_hook()
        self.last_move_attempts = 0
        self.last_move_decision = None

    def _log_last_call_usage(self) -> None:
        """Emit debug information about the most recent connector usage."""
        usage = self.connector.get_last_usage()
        if usage is None:
            logger.debug(
                "LLM player {} reported no usage metrics for the last call", self
            )
            return

        votes_suffix = f" ({self.num_votes} votes)" if self.num_votes > 1 else ""
        logger.info(
            "LLM player {} usage this call{}: prompt_tokens={}, completion_tokens={}, total_tokens={}, cost=${:.6f}",
            self,
            votes_suffix,
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.total_tokens,
            usage.cost,
        )
