"""Typed helper structures for LLM decision flow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llm_chess_arena.types import PlayerDecision


@dataclass(slots=True)
class VoteCount:
    """Count of votes for a particular action/move combination."""

    action: str
    attempted_move_in_uci: str | None
    count: int


@dataclass(slots=True)
class VoteMetadata:
    """Metadata describing how the vote aggregation resolved a decision."""

    winning_action: str
    winning_move: str | None
    tallies: list[VoteCount]
    parsed_responses: list[dict[str, str | None]]
    tie_broken: bool


@dataclass(slots=True)
class VoteAggregation:
    """Result of aggregating multiple model responses."""

    decision: PlayerDecision
    metadata: VoteMetadata | None

    @property
    def action(self) -> str:
        """Delegate to underlying decision's action."""
        return self.decision.action

    @property
    def attempted_move(self) -> str | None:
        """Delegate to underlying decision's attempted_move."""
        return self.decision.attempted_move


@dataclass(slots=True)
class DecisionArtifacts:
    """Auxiliary artifacts produced while generating an LLM move."""

    normalized_uci: str | None
    vote_metadata: VoteMetadata | None
    decision_process: dict[str, Any]
