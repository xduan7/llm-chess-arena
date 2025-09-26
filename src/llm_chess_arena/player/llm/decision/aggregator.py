"""Majority voting helper for combining multiple LLM responses."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from loguru import logger

from llm_chess_arena.exceptions import ParseMoveError
from ..prompting.handlers import BaseLLMMoveHandler
from llm_chess_arena.types import PlayerDecision


class VoteAggregator:
    """Aggregate multiple LLM responses into a single decision."""

    def __init__(self, handler: BaseLLMMoveHandler) -> None:
        """Initialize the aggregator with a move handler.

        Args:
            handler: Handler for parsing individual LLM responses.
        """
        self._handler = handler

    def aggregate_responses(self, responses: Iterable[str]) -> PlayerDecision:
        """Aggregate multiple LLM responses using majority voting.

        Args:
            responses: Collection of LLM response texts to aggregate.

        Returns:
            PlayerDecision: The decision selected by majority vote, or a debug
            decision if all responses fail to parse.

        Raises:
            ConnectionError: If no responses are provided.
        """
        response_list = list(responses)
        if not response_list:
            raise ConnectionError("No responses received from LLM provider")

        decisions = self._parse_responses(response_list)
        if not decisions:
            return self._build_debug_decision(response_list)

        decision_tuples = [(d.action, d.attempted_move) for d in decisions]
        vote_counts = Counter(decision_tuples)
        most_voted_tuple, vote_count = vote_counts.most_common(1)[0]

        ties = [item for item, count in vote_counts.items() if count == vote_count]
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

        for decision in decisions:
            if (decision.action, decision.attempted_move) == most_voted_tuple:
                return decision

        # Defensive fallback - should never happen because tuple came from decisions.
        return decisions[0]

    def _parse_responses(self, responses: list[str]) -> list[PlayerDecision]:
        """Return parsed decisions while logging failures for context."""
        decisions: list[PlayerDecision] = []
        total = len(responses)
        for idx, response in enumerate(responses, start=1):
            try:
                decision = self._handler.parse_decision_from_response(response)
            except ParseMoveError as exc:
                logger.warning("Vote {}/{}: {}", idx, total, exc)
                continue

            if decision is not None:
                decisions.append(decision)
                logger.debug(
                    "Vote {}/{}: Parsed move '{}' from response",
                    idx,
                    total,
                    decision.attempted_move,
                )

        logger.debug(
            "Successfully parsed {}/{} responses for voting",
            len(decisions),
            total,
        )

        return decisions

    def _build_debug_decision(self, responses: list[str]) -> PlayerDecision:
        """Create a synthetic decision containing all raw responses."""
        logger.error(
            "All {} LLM response(s) failed to parse - logging all responses for debugging",
            len(responses),
        )
        for index, response in enumerate(responses, start=1):
            logger.error("Failed response {}/{}: {!r}", index, len(responses), response)

        combined = "\n".join(
            [
                f"--- Response {i + 1}/{len(responses)} ---\n{resp}"
                for i, resp in enumerate(responses)
            ]
        )
        return PlayerDecision(action="move", attempted_move="???", response=combined)
