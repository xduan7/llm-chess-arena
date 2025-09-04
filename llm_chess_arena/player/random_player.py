import random

from llm_chess_arena.player.base_player import BasePlayer
from llm_chess_arena.types import Color, PlayerDecisionContext, PlayerDecision


class RandomPlayer(BasePlayer):
    """Chess player that selects moves randomly.

    Used for testing and as baseline benchmark.
    """

    def __init__(
        self,
        *,
        name: str = "Random Player",
        color: Color,
        seed: None | int = None,
    ) -> None:
        """Initialize with optional seed for reproducibility.

        Args:
            name: Display name.
            color: 'white' or 'black'.
            seed: RNG seed for reproducible games.
        """
        super().__init__(name, color)
        self.seed = seed
        self.rng = random.Random(seed)

    def _make_decision(self, context: PlayerDecisionContext) -> PlayerDecision:
        """Select random legal move.

        Args:
            context: Game context with legal moves.

        Returns:
            Decision with randomly selected move.
        """
        selected_move = self.rng.choice(context.legal_moves_in_uci)
        return PlayerDecision(action="move", attempted_move=selected_move)
