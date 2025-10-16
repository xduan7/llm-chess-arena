"""LLM-backed player that orchestrates prompting, voting, and retries."""

from __future__ import annotations

from datetime import datetime, UTC
from typing import Any
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
from llm_chess_arena.player.llm.prompting import PromptSession
from llm_chess_arena.player.llm.decision import RetryController
from llm_chess_arena.player.llm.decision import VoteAggregator
from llm_chess_arena.player.llm.types import (
    DecisionArtifacts,
    VoteMetadata,
)
from llm_chess_arena.utils import parse_attempted_move_to_uci
from llm_chess_arena.record import iso_timestamp
from llm_chess_arena.types import PlayerColor, PlayerDecision, PlayerDecisionContext


class LLMPlayer(BasePlayer):
    """Chess player that queries an LLM and applies optional majority voting."""

    def __init__(
        self,
        *,
        name: str | None = None,
        color: PlayerColor,
        connector: LLMConnector,
        handler: BaseLLMMoveHandler,
        max_move_retries: int,
        num_votes: int,
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
        self._retry_controller = RetryController(max_move_retries)

        # Public fields for external inspection (used by tests and metrics)
        self.last_move_attempts: int = 0  # Number of attempts for the most recent move
        self.last_move_decision: PlayerDecision | None = (
            None  # Most recent decision made
        )

        # Internal tracking for aggregated metrics calculation
        self._last_decision_artifacts: DecisionArtifacts | None = None

    def _make_decision(self, context: PlayerDecisionContext) -> PlayerDecision:
        """Produce a move decision after coordinating prompts, voting, and retries."""

        logger.info(
            "LLM player {} starting move decision for position with {} legal moves",
            self.name,
            len(context.legal_moves_in_uci),
        )

        prompt_session = PromptSession(self.handler, context)
        self._last_decision_artifacts = None
        final_decision_process: dict[str, Any] | None = None
        candidate_decision: PlayerDecision | None = None
        cumulative_thinking_time_in_sec = 0.0

        for retry_attempt in self._retry_controller.iter_attempts():
            self.last_move_attempts = retry_attempt.attempt_number

            if retry_attempt.attempt_number > 1:
                logger.info(
                    "LLM player {} retry attempt {}/{} after previous failure",
                    self.name,
                    retry_attempt.attempt_number,
                    retry_attempt.max_attempts,
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

            llm_decision_process: dict[str, Any] = {
                "api_calls": [],
                "network_errors": [],
                "move_errors": [],
                "voting_process": None,
            }

            call_start_time = datetime.now(UTC)
            api_call_record: dict[str, Any] = {
                "attempt": retry_attempt.attempt_number,
                "timestamp": iso_timestamp(call_start_time),
                "request": {
                    "model": getattr(self.connector, "model", "unknown"),
                    "temperature": getattr(self.connector, "temperature", None),
                    "max_num_tokens": getattr(self.connector, "max_num_tokens", None),
                    "n": self.num_votes,
                    "messages": [{"role": "user", "content": prompt}],
                },
            }
            llm_decision_process["api_calls"].append(api_call_record)

            try:
                responses = self.connector.query(prompt, n=self.num_votes)

                call_end_time = datetime.now(UTC)
                latency_in_ms = int(
                    (call_end_time - call_start_time).total_seconds() * 1000
                )
                cumulative_thinking_time_in_sec += latency_in_ms / 1000.0
                api_call_record["response"] = {
                    "timestamp": iso_timestamp(call_end_time),
                    "latency_in_ms": latency_in_ms,
                    "choices": [
                        {
                            "index": response_index,
                            "content": response,
                        }
                        for response_index, response in enumerate(responses)
                    ],
                }

                usage_metrics = self.connector.get_last_usage()
                if usage_metrics:
                    api_call_record["response"]["usage"] = {
                        "prompt_tokens": usage_metrics.prompt_tokens,
                        "completion_tokens": usage_metrics.completion_tokens,
                        "total_tokens": usage_metrics.total_tokens,
                        "cost": usage_metrics.cost,
                    }

                logger.debug(
                    "LLM player {} received {} responses (avg length: {:.1f} chars)",
                    self.name,
                    len(responses),
                    (
                        sum(len(response) for response in responses) / len(responses)
                        if responses
                        else 0
                    ),
                )

                vote_result = self._vote_aggregator.aggregate_responses(responses)
                candidate_decision = vote_result.decision
                vote_metadata = vote_result.metadata

                if vote_metadata is not None:
                    llm_decision_process["voting_process"] = (
                        self._vote_metadata_to_dict(vote_metadata)
                    )

                self._log_last_call_usage()
                logger.debug(
                    "LLM returned decision: action={}, move={}",
                    candidate_decision.action,
                    (
                        candidate_decision.attempted_move
                        if candidate_decision.action == "move"
                        else "N/A"
                    ),
                )

                # Validate and normalize the move decision
                if candidate_decision.action == "resign":
                    normalized_decision = candidate_decision.model_copy(
                        update={"thinking_time_in_sec": cumulative_thinking_time_in_sec}
                    )
                    normalized_uci = None
                elif candidate_decision.action == "move":
                    if candidate_decision.attempted_move is None:
                        raise InvalidMoveError(
                            "LLM move decision missing attempted_move text"
                        )
                    normalized_uci = parse_attempted_move_to_uci(
                        candidate_decision.attempted_move, context.board_in_fen
                    )
                    normalized_decision = candidate_decision.model_copy(
                        update={
                            "attempted_move": normalized_uci,
                            "thinking_time_in_sec": cumulative_thinking_time_in_sec,
                        }
                    )
                else:
                    raise NotImplementedError(
                        "LLM currently only supports 'move' and 'resign' actions, "
                        f"got '{candidate_decision.action}'"
                    )

                self.last_move_decision = normalized_decision
                self._last_decision_artifacts = DecisionArtifacts(
                    normalized_uci=normalized_uci,
                    vote_metadata=vote_metadata,
                    decision_process=llm_decision_process,
                )

                logger.info(
                    "LLM player {} successfully generated valid move {} after {} attempt(s)",
                    self.name,
                    normalized_decision.attempted_move,
                    retry_attempt.attempt_number,
                )
                return normalized_decision

            except LLMPermanentError:
                logger.error("Permanent LLM error - terminating move attempt")
                raise
            except (TimeoutError, ConnectionError) as network_error:
                llm_decision_process["network_errors"].append(
                    {
                        "attempt": retry_attempt.attempt_number,
                        "error_code": network_error.__class__.__name__.upper().replace(
                            "ERROR", ""
                        ),
                        "error_message": str(network_error),
                    }
                )
                logger.warning(
                    "{} resigned due to network failure: {}",
                    self.name,
                    str(network_error),
                )
                resignation_decision = self._retry_controller.create_resignation()
                resignation_decision = resignation_decision.model_copy(
                    update={"thinking_time_in_sec": cumulative_thinking_time_in_sec}
                )
                self.last_move_decision = resignation_decision
                self._last_decision_artifacts = DecisionArtifacts(
                    normalized_uci=None,
                    vote_metadata=None,
                    decision_process=llm_decision_process,
                )
                return resignation_decision
            except (
                InvalidMoveError,
                IllegalMoveError,
                AmbiguousMoveError,
                LLMEmptyResponseError,
            ) as move_error:
                if isinstance(move_error, LLMEmptyResponseError):
                    retry_status = (
                        f"Retrying move attempt (empty response: {move_error})"
                        if not retry_attempt.is_final_attempt
                        else f"No retries left after empty response: {move_error}"
                    )
                    logger.warning(
                        "LLM player {} attempt {} failed with empty response: {}. {}",
                        self.name,
                        retry_attempt.attempt_number,
                        move_error,
                        retry_status,
                    )

                    if not retry_attempt.is_final_attempt:
                        continue
                else:
                    attempted_move = (
                        candidate_decision.attempted_move
                        if candidate_decision is not None
                        else "unknown"
                    )
                    llm_decision_process["move_errors"].append(
                        {
                            "attempt": retry_attempt.attempt_number,
                            "attempted_move_in_uci": attempted_move,
                            "error_type": move_error.__class__.__name__,
                            "error_message": str(move_error),
                        }
                    )

                    retry_status = (
                        "Retrying with prior response and invalid move context"
                        if not retry_attempt.is_final_attempt
                        else "No retries left"
                    )
                    logger.warning(
                        "LLM player {} attempt {} failed with {}: {}. Invalid move: '{}'. {}",
                        self.name,
                        retry_attempt.attempt_number,
                        move_error.__class__.__name__,
                        move_error,
                        attempted_move,
                        retry_status,
                    )

                    if (
                        not retry_attempt.is_final_attempt
                        and candidate_decision is not None
                    ):
                        prompt_session.build_retry_prompt(
                            exception_name=move_error.__class__.__name__,
                            last_response=getattr(candidate_decision, "response", None),
                            last_attempted_move=candidate_decision.attempted_move,
                        )
                        logger.debug(
                            "Generated retry prompt with error context for {}",
                            move_error.__class__.__name__,
                        )
                        self.last_move_decision = candidate_decision
                        continue

                final_decision_process = llm_decision_process
                break
            except NotImplementedError:
                logger.error(
                    "LLM player {} returned unsupported action '{}', resigning.",
                    self.name,
                    (
                        candidate_decision.action
                        if candidate_decision is not None
                        else "unknown"
                    ),
                )
                resignation_decision = PlayerDecision(
                    action="resign",
                    thinking_time_in_sec=cumulative_thinking_time_in_sec,
                )
                self.last_move_decision = resignation_decision
                self._last_decision_artifacts = DecisionArtifacts(
                    normalized_uci=None,
                    vote_metadata=None,
                    decision_process=llm_decision_process,
                )
                return resignation_decision
        else:
            final_decision_process = {
                "api_calls": [],
                "network_errors": [],
                "move_errors": [],
                "voting_process": None,
            }

        resignation_decision = self._retry_controller.create_resignation()
        resignation_decision = resignation_decision.model_copy(
            update={"thinking_time_in_sec": cumulative_thinking_time_in_sec}
        )
        self.last_move_decision = resignation_decision
        self._last_decision_artifacts = DecisionArtifacts(
            normalized_uci=None,
            vote_metadata=None,
            decision_process=final_decision_process,
        )
        return resignation_decision

    def close(self) -> None:
        """Release LLM connector resources."""
        if hasattr(self.connector, "close"):
            try:
                self.connector.close()
                logger.debug("LLM connector closed successfully")
            except Exception as connector_close_error:  # pragma: no cover
                logger.warning("Error closing LLM connector: {}", connector_close_error)

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
        self._last_decision_artifacts = None

    def _log_last_call_usage(self) -> None:
        """Emit debug information about the most recent connector usage."""
        usage_metrics = self.connector.get_last_usage()
        if usage_metrics is None:
            logger.debug(
                "LLM player {} reported no usage metrics for the last call", self.name
            )
            return

        votes_suffix = f" ({self.num_votes} votes)" if self.num_votes > 1 else ""
        logger.info(
            "LLM player {} usage this call{}: prompt_tokens={}, completion_tokens={}, total_tokens={}, cost=${:.6f}",
            self.name,
            votes_suffix,
            usage_metrics.prompt_tokens,
            usage_metrics.completion_tokens,
            usage_metrics.total_tokens,
            usage_metrics.cost,
        )

    def get_last_decision_artifacts(self) -> DecisionArtifacts | None:
        """Return artifacts captured during the most recent decision."""

        return self._last_decision_artifacts

    @staticmethod
    def _vote_metadata_to_dict(metadata: VoteMetadata) -> dict[str, Any]:
        """Convert vote metadata into a dictionary for record serialization."""

        vote_tally = {
            f"{count.action}:{count.attempted_move_in_uci or ''}": count.count
            for count in metadata.tallies
        }

        winner = (
            metadata.winning_move
            if metadata.winning_action == "move"
            else metadata.winning_action
        )

        return {
            "parsed_responses": metadata.parsed_responses,
            "vote_tally": vote_tally,
            "winner": winner,
            "tie_broken": metadata.tie_broken,
        }

    # LLM performance metrics are now calculated directly from game move records.
