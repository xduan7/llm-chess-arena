"""Majority voting helper for combining multiple LLM responses."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Literal

from loguru import logger

from llm_chess_arena.exceptions import ParseMoveError, LLMEmptyResponseError
from ..prompting.handlers import BaseLLMMoveHandler
from llm_chess_arena.types import PlayerDecision
from ..types import VoteAggregation, VoteCount, VoteMetadata


class VoteAggregator:
    """Aggregate multiple LLM responses into a single decision."""

    def __init__(self, handler: BaseLLMMoveHandler) -> None:
        """Initialize the aggregator with a move handler.

        Args:
            handler: Handler for parsing individual LLM responses.
        """
        self._handler = handler

    def aggregate_responses(self, responses: Iterable[str]) -> VoteAggregation:
        """Aggregate multiple LLM responses using majority voting.

        Args:
            responses: Collection of LLM response texts to aggregate.

        Returns:
            PlayerDecision: The decision selected by majority vote, or a debug
            decision if all responses fail to parse.

        Raises:
            LLMEmptyResponseError: If no responses are provided.
        """
        response_texts = list(responses)
        if not response_texts:
            raise LLMEmptyResponseError("No responses received from LLM provider")

        decisions = self._parse_responses(response_texts)
        if not decisions:
            return self._build_debug_decision(response_texts)

        decision_tuples = [
            (decision.action, decision.attempted_move) for decision in decisions
        ]
        vote_counts = Counter(decision_tuples)
        most_voted_tuple, vote_count = vote_counts.most_common(1)[0]

        ties = [
            decision_tuple
            for decision_tuple, count in vote_counts.items()
            if count == vote_count
        ]
        if len(decisions) > 1:  # Only log voting details when there are multiple votes
            if len(ties) > 1:
                logger.info(
                    "Vote tie between {} options with {} votes each. Selected: {}",
                    len(ties),
                    vote_count,
                    (
                        most_voted_tuple[1]
                        if most_voted_tuple[0] == "move"
                        else most_voted_tuple[0]
                    ),
                )
            else:
                logger.debug(
                    "Majority voting: {}/{} votes for move '{}'",
                    vote_count,
                    len(decisions),
                    (
                        most_voted_tuple[1]
                        if most_voted_tuple[0] == "move"
                        else most_voted_tuple[0]
                    ),
                )

        vote_metadata = self._build_vote_metadata(
            decisions=decisions,
            vote_counts=vote_counts,
            winning_tuple=most_voted_tuple,
            tie_broken=len(ties) > 1,
        )

        for decision in decisions:
            if (decision.action, decision.attempted_move) == most_voted_tuple:
                return VoteAggregation(decision=decision, metadata=vote_metadata)

        # Defensive fallback - should never happen because tuple came from decisions.
        fallback_decision = decisions[0]
        return VoteAggregation(decision=fallback_decision, metadata=vote_metadata)

    def _parse_responses(self, responses: list[str]) -> list[PlayerDecision]:
        """Return parsed decisions while logging failures for context."""
        decisions: list[PlayerDecision] = []
        total_response_count = len(responses)
        for response_num, response in enumerate(responses, start=1):
            try:
                decision = self._handler.parse_decision_from_response(response)
            except ParseMoveError as parse_error:
                logger.warning(
                    "Response {} of {}: Failed to parse move - {}",
                    response_num,
                    total_response_count,
                    parse_error,
                )
                continue

            if decision is not None:
                decisions.append(decision)
                logger.debug(
                    "Response {} of {}: Successfully parsed move '{}'",
                    response_num,
                    total_response_count,
                    decision.attempted_move,
                )

        logger.debug(
            "Voting results: {} valid moves from {} total responses",
            len(decisions),
            total_response_count,
        )

        return decisions

    def _build_vote_metadata(
        self,
        *,
        decisions: list[PlayerDecision],
        vote_counts: Counter[tuple[Literal["move", "resign"], str | None]],
        winning_tuple: tuple[str, str | None],
        tie_broken: bool,
    ) -> VoteMetadata:
        parsed_responses = [
            {
                "action": decision.action,
                "attempted_move_in_uci": decision.attempted_move,
            }
            for decision in decisions
        ]

        tallies = [
            VoteCount(action=action, attempted_move_in_uci=move, count=count)
            for (action, move), count in vote_counts.items()
        ]

        winning_action, winning_move = winning_tuple

        return VoteMetadata(
            winning_action=winning_action,
            winning_move=winning_move,
            tallies=tallies,
            parsed_responses=parsed_responses,
            tie_broken=tie_broken,
        )

    def _build_debug_decision(self, responses: list[str]) -> VoteAggregation:
        """Create a synthetic decision containing all raw responses."""
        logger.error(
            "All {} LLM responses failed to parse - logging all responses for debugging",
            len(responses),
        )
        for response_index, response in enumerate(responses, start=1):
            logger.error(
                "Response {} of {} failed: {!r}",
                response_index,
                len(responses),
                response,
            )

        combined_responses = "\n".join(
            [
                f"--- Response {response_index + 1}/{len(responses)} ---\n{response_text}"
                for response_index, response_text in enumerate(responses)
            ]
        )
        decision = PlayerDecision(
            action="resign",
            attempted_move=None,
            reason=f"All {len(responses)} LLM responses failed to parse",
            response=combined_responses,
        )
        return VoteAggregation(decision=decision, metadata=None)
