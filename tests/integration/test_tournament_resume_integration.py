"""End-to-end tournament resume tests.

Unlike the unit tests, nothing here fabricates record files: games run through
the real Game/RecordWriter pipeline, get interrupted by genuine (simulated)
network errors, and are resumed from the records actually written to disk.
This exercises the full chain: error propagation -> unfinished-record
persistence -> resumability detection -> player recreation -> aggregate
regeneration.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from llm_chess_arena.game import Game
from llm_chess_arena.player.base_player import BasePlayer
from llm_chess_arena.player.llm import GameArenaLLMMoveHandler, LLMPlayer
from llm_chess_arena.player.random_player import RandomPlayer
from llm_chess_arena.tournament.aggregator import aggregate_tournament_results
from llm_chess_arena.tournament.export import ResultsExporter
from llm_chess_arena.tournament.loader import TournamentLoader, load_game_result
from llm_chess_arena.tournament.resume import TournamentResumer
from llm_chess_arena.types import PlayerColor, PlayerDecision, PlayerDecisionContext
from tests.fixtures.mock_llm_connector import MockLLMConnector

MAX_NUM_MOVES = 120


class FlakyNetworkPlayer(BasePlayer):
    """Plays deterministic legal moves, then simulates a network outage.

    Mimics an LLM player whose connector exhausted its retries: the raised
    ConnectionError must propagate so the game is saved as resumable.
    """

    def __init__(
        self, *, name: str, color: PlayerColor, moves_before_failure: int
    ) -> None:
        super().__init__(name, color)
        self.moves_before_failure = moves_before_failure
        self.moves_made = 0

    def _make_decision(self, context: PlayerDecisionContext) -> PlayerDecision:
        if self.moves_made >= self.moves_before_failure:
            raise ConnectionError("Simulated network outage after connector retries")
        self.moves_made += 1
        return PlayerDecision(
            action="move", attempted_move=sorted(context.legal_moves_in_uci)[0]
        )


class FlakyMockConnector(MockLLMConnector):
    """Mock connector that succeeds a fixed number of times, then errors."""

    def __init__(self, fail_after_queries: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.fail_after_queries = fail_after_queries

    def query(self, *args, **kwargs):
        if self.query_count >= self.fail_after_queries:
            raise ConnectionError("Simulated API outage after connector retries")
        return super().query(*args, **kwargs)


def _tournament_hydra_cfg() -> dict:
    """Per-game hydra config as the tournament executor would store it."""
    return {
        "game": {
            "display_board": False,
            "display_summary": False,
            "enable_metrics": False,
            "max_num_moves": MAX_NUM_MOVES,
        },
        "players": {
            "white": {"kind": "random", "name": "P1", "color": "white", "seed": 1},
            "black": {"kind": "random", "name": "P2", "color": "black", "seed": 2},
        },
    }


def _run_game(game_dir: Path, white, black) -> Game:
    """Run one real game with recording enabled, tolerating interruption."""
    game = Game(
        white_player=white,
        black_player=black,
        display_board=False,
        display_summary=False,
        enable_metrics=False,
        record_dir=game_dir,
        record_name="game",
        hydra_cfg=_tournament_hydra_cfg(),
    )
    game.play(max_num_moves=MAX_NUM_MOVES)
    return game


@pytest.fixture
def interrupted_tournament(tmp_path: Path) -> Path:
    """Build a two-game tournament where game 1 was genuinely interrupted.

    - game_001: white hits a network error after 3 moves -> resumable record
    - game_002: completes normally (max-move draw at worst)
    - results.json: aggregated from the records on disk, as the CLI would
    """
    tournament = tmp_path / "match"

    interrupted_game = _run_game(
        tournament / "game_001",
        FlakyNetworkPlayer(name="P1", color="white", moves_before_failure=3),
        RandomPlayer(name="P2", color="black", seed=2),
    )
    assert not interrupted_game.finished  # Sanity: the outage really interrupted it

    completed_game = _run_game(
        tournament / "game_002",
        RandomPlayer(name="P1", color="white", seed=1),
        RandomPlayer(name="P2", color="black", seed=2),
    )
    assert completed_game.finished

    game_results = [
        load_game_result(tournament / "game_001" / "game.json", 1),
        load_game_result(tournament / "game_002" / "game.json", 2),
    ]
    tournament_result = aggregate_tournament_results(
        match_name="match",
        results=game_results,
        start_time=datetime.now(timezone.utc),
        player1_name="P1",
        player2_name="P2",
    )
    ResultsExporter.export_json(tournament_result, tournament / "results.json")
    ResultsExporter.export_csv(tournament_result, tournament / "results.csv")

    return tournament


class TestInterruptedRecordOnDisk:
    """The record written for an interrupted game must be resumable as-is."""

    def test_interrupted_game_writes_resumable_record(
        self, interrupted_tournament: Path
    ) -> None:
        game_json_path = interrupted_tournament / "game_001" / "game.json"
        assert game_json_path.exists()

        record = json.loads(game_json_path.read_text())
        assert record["game_outcome"]["result"] == "Unfinished"
        assert record["summary"]["result"] == "Unfinished"
        assert record["termination_metadata"]["resumable"] is True
        assert record["termination_metadata"]["error_type"] == "ConnectionError"
        assert record["hydra_config"]["players"]["white"]["kind"] == "random"
        # Moves made before the outage are preserved for replay
        assert len(record["moves"]) >= 5

    def test_interrupted_game_not_counted_as_draw(
        self, interrupted_tournament: Path
    ) -> None:
        """An unfinished game must not pollute win/draw statistics."""
        results = json.loads((interrupted_tournament / "results.json").read_text())

        assert results["total_games"] == 2
        counted_games = (
            results["results"]["player1_wins"]
            + results["results"]["player2_wins"]
            + results["results"]["draws"]
        )
        # Only the completed game contributes a W/D/L outcome
        assert counted_games == 1

    def test_loader_detects_exactly_the_interrupted_game(
        self, interrupted_tournament: Path
    ) -> None:
        resumable = TournamentLoader.find_resumable_games(interrupted_tournament)

        assert [game_id for game_id, _ in resumable] == [1]


class TestEndToEndResume:
    """Resume the interrupted game for real and verify the whole chain."""

    def test_resume_completes_interrupted_game(
        self, interrupted_tournament: Path
    ) -> None:
        game_json_path = interrupted_tournament / "game_001" / "game.json"
        moves_before_resume = len(json.loads(game_json_path.read_text())["moves"])

        result = TournamentResumer(interrupted_tournament).resume()

        assert result.resumed_games == [1]
        assert result.total_games == 2

        # The rewritten record is complete and marked as resumed
        record = json.loads(game_json_path.read_text())
        assert record["game_outcome"]["result"] in ("1-0", "0-1", "1/2-1/2")
        assert len(record["moves"]) > moves_before_resume
        resumption = record["resumption_metadata"]
        assert resumption["original_termination"]["error_type"] == "ConnectionError"

        # The pristine interrupted record is archived
        archive = json.loads(
            (interrupted_tournament / "game_001" / "game.json.original").read_text()
        )
        assert archive["game_outcome"]["result"] == "Unfinished"

        # Canonical results were regenerated: both games now carry outcomes
        results = json.loads((interrupted_tournament / "results.json").read_text())
        counted_games = (
            results["results"]["player1_wins"]
            + results["results"]["player2_wins"]
            + results["results"]["draws"]
        )
        assert results["total_games"] == 2
        assert counted_games == 2
        assert results["resume_info"]["resumed_game_ids"] == [1]
        assert (interrupted_tournament / "results.json.original").exists()

        # Lock was released
        assert not (interrupted_tournament / ".resume.lock").exists()

    def test_second_resume_is_a_noop(self, interrupted_tournament: Path) -> None:
        TournamentResumer(interrupted_tournament).resume()

        second_result = TournamentResumer(interrupted_tournament).resume()

        assert second_result.resumed_games == []
        assert second_result.total_games == 2


class TestLLMPlayerNetworkErrorEndToEnd:
    """A real LLMPlayer outage must produce a resumable record too."""

    def test_llm_network_error_produces_resumable_record(self, tmp_path: Path) -> None:
        connector = FlakyMockConnector(
            fail_after_queries=1,
            responses=["Final Answer: e4"],
        )
        llm_player = LLMPlayer(
            name="LLM",
            color="white",
            connector=connector,
            handler=GameArenaLLMMoveHandler(),
            max_move_retries=1,
            num_votes=1,
        )

        game_dir = tmp_path / "llm_game"
        game = _run_game(
            game_dir, llm_player, RandomPlayer(name="P2", color="black", seed=2)
        )

        assert not game.finished

        record = json.loads((game_dir / "game.json").read_text())
        assert record["game_outcome"]["result"] == "Unfinished"
        assert record["termination_metadata"]["resumable"] is True
        assert record["termination_metadata"]["error_type"] == "ConnectionError"
        assert record["termination_metadata"]["player_color"] == "white"
        # Both pre-outage moves were recorded, including the LLM decision trail
        assert len(record["moves"]) == 2
        assert record["moves"][0]["llm_decision_process"]["api_calls"]
