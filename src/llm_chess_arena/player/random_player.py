"""Random baseline player that samples uniformly from legal moves."""

import random

from llm_chess_arena.player.base_player import BasePlayer
from llm_chess_arena.types import Color, PlayerDecisionContext, PlayerDecision


class RandomPlayer(BasePlayer):
    """Selects legal moves uniformly at random for baseline comparisons."""

    def __init__(
        self,
        *,
        name: str = "Random Player",
        color: Color,
        seed: int | None = None,
    ) -> None:
        """Configure a random-move chess player.

        Args:
            name: Human-readable identifier shown in logs and UIs.
            color: Chess side controlled by the player.
            seed: Optional RNG seed for reproducible move sequences.
        """
        super().__init__(name, color)
        self.seed = seed
        self.rng = random.Random(seed)

    def _make_decision(self, context: PlayerDecisionContext) -> PlayerDecision:
        """Return a uniformly sampled legal move."""
        selected_move = self.rng.choice(context.legal_moves_in_uci)
        return PlayerDecision(action="move", attempted_move=selected_move)
