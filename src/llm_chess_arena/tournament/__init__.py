"""Tournament system for running multiple chess games with statistical analysis."""

from llm_chess_arena.tournament.types import (
    TournamentConfig,
    GameResult,
    TournamentResult,
)
from llm_chess_arena.tournament.executor import TournamentRunner
from llm_chess_arena.tournament.export import ResultsExporter
from llm_chess_arena.tournament.aggregator import aggregate_tournament_results

__all__ = [
    "TournamentConfig",
    "GameResult",
    "TournamentResult",
    "TournamentRunner",
    "ResultsExporter",
    "aggregate_tournament_results",
]
