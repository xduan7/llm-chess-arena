"""Tests for the vote aggregator component."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from llm_chess_arena.exceptions import ParseMoveError
from llm_chess_arena.player.llm.decision import VoteAggregator
from llm_chess_arena.types import PlayerDecision


@pytest.fixture()
def handler() -> Mock:
    handler = Mock()
    return handler


@pytest.fixture()
def aggregator(handler: Mock) -> VoteAggregator:
    return VoteAggregator(handler)


def test_majority_vote_simple(aggregator: VoteAggregator, handler: Mock) -> None:
    """Majority vote should return the most common parsed decision."""
    responses = ["move e4", "move e4", "move d4"]

    handler.parse_decision_from_response.side_effect = [
        PlayerDecision(action="move", attempted_move="e4"),
        PlayerDecision(action="move", attempted_move="e4"),
        PlayerDecision(action="move", attempted_move="d4"),
    ]

    decision = aggregator.aggregate_responses(responses)

    assert decision.attempted_move == "e4"


def test_tie_breaking_picks_first(aggregator: VoteAggregator, handler: Mock) -> None:
    """Ties should resolve in favor of the first winning decision."""
    responses = ["move e4", "move d4", "move e4", "move d4"]

    handler.parse_decision_from_response.side_effect = [
        PlayerDecision(action="move", attempted_move="e4"),
        PlayerDecision(action="move", attempted_move="d4"),
        PlayerDecision(action="move", attempted_move="e4"),
        PlayerDecision(action="move", attempted_move="d4"),
    ]

    decision = aggregator.aggregate_responses(responses)

    assert decision.attempted_move == "e4"


def test_parse_failures_skipped(aggregator: VoteAggregator, handler: Mock) -> None:
    """Parsing failures should be ignored while counting votes."""
    responses = ["garbage", "move e4", "more garbage"]

    def side_effect(response: str) -> PlayerDecision:
        if "garbage" in response:
            raise ParseMoveError("Could not parse")
        return PlayerDecision(action="move", attempted_move="e4")

    handler.parse_decision_from_response.side_effect = side_effect

    decision = aggregator.aggregate_responses(responses)

    assert decision.attempted_move == "e4"


def test_all_responses_fail_returns_debug(
    aggregator: VoteAggregator, handler: Mock
) -> None:
    """If every response fails, fall back to debug decision."""
    handler.parse_decision_from_response.side_effect = ParseMoveError("bad")

    decision = aggregator.aggregate_responses(["fail1", "fail2"])

    assert decision.attempted_move == "???"
    assert "Response 1/2" in decision.response


def test_empty_responses_raise_connection_error(aggregator: VoteAggregator) -> None:
    """Empty response lists should raise a connection error."""
    with pytest.raises(ConnectionError, match="No responses"):
        aggregator.aggregate_responses([])
