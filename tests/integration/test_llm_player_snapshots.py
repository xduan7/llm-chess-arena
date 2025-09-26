"""Integration snapshots guarding current LLMPlayer behavior."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from llm_chess_arena.player.llm import (
    GameArenaLLMMoveHandler,
    LLMConnector,
    LLMPlayer,
)
from llm_chess_arena.types import PlayerDecisionContext
from llm_chess_arena.config import load_app_config, run_game_from_config


class TestLLMPlayerBehaviorSnapshots:
    """Capture representative LLMPlayer flows before refactoring."""

    @pytest.fixture()
    def context(self) -> PlayerDecisionContext:
        """Provide a baseline decision context for integration snapshots."""
        return PlayerDecisionContext(
            board_in_fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            player_color="white",
            legal_moves_in_uci=["e2e4", "d2d4", "g1f3", "b1c3"],
            move_history_in_uci=[],
        )

    def test_single_move_decision_snapshot(
        self, context: PlayerDecisionContext
    ) -> None:
        """Basic flow: single sample produces a legal move."""
        connector = Mock(spec=LLMConnector)
        connector.model = "test-model"
        connector.query.return_value = [
            "I need to develop my pieces. Final Answer: Nf3"
        ]
        connector.get_last_usage.return_value = None
        connector.get_total_usage.return_value = Mock(
            prompt_tokens=10, completion_tokens=5, total_tokens=15, cost=0.001
        )

        player = LLMPlayer(
            color="white",
            connector=connector,
            handler=GameArenaLLMMoveHandler(),
            max_move_retries=2,
            num_votes=1,
        )

        decision = player._make_decision(context)

        assert decision.action == "move"
        assert decision.attempted_move == "g1f3"
        assert player.last_move_attempts == 1
        assert connector.query.call_count == 1
        _, kwargs = connector.query.call_args
        assert kwargs.get("n") == 1

    def test_majority_voting_snapshot(self, context: PlayerDecisionContext) -> None:
        """Majority voting should respect first-occurrence tie-breaking."""
        connector = Mock(spec=LLMConnector)
        connector.model = "test-model"
        connector.query.return_value = [
            "Let me think... Final Answer: e4",
            "Best move is clearly Final Answer: e4",
            "I'll play Final Answer: d4",
        ]
        connector.get_last_usage.return_value = None
        connector.get_total_usage.return_value = Mock(
            prompt_tokens=30, completion_tokens=15, total_tokens=45, cost=0.003
        )

        player = LLMPlayer(
            color="white",
            connector=connector,
            handler=GameArenaLLMMoveHandler(),
            num_votes=3,
        )

        decision = player._make_decision(context)

        assert decision.action == "move"
        assert decision.attempted_move == "e2e4"
        connector.query.assert_called_once()
        _, kwargs = connector.query.call_args
        assert kwargs.get("n") == 3

    def test_retry_behavior_snapshot(self, context: PlayerDecisionContext) -> None:
        """Invalid moves trigger retry prompt with context."""
        connector = Mock(spec=LLMConnector)
        connector.model = "test-model"
        connector.query.side_effect = [
            ["Final Answer: Zz9"],
            ["After considering the position: Final Answer: e4"],
        ]
        connector.get_last_usage.return_value = None
        connector.get_total_usage.return_value = Mock(
            prompt_tokens=20, completion_tokens=10, total_tokens=30, cost=0.002
        )

        player = LLMPlayer(
            color="white",
            connector=connector,
            handler=GameArenaLLMMoveHandler(),
            max_move_retries=2,
        )

        decision = player._make_decision(context)

        assert decision.action == "move"
        assert decision.attempted_move == "e2e4"
        assert player.last_move_attempts == 2
        assert connector.query.call_count == 2

        first_prompt = connector.query.call_args_list[0].args[0]
        second_prompt = connector.query.call_args_list[1].args[0]
        assert first_prompt != second_prompt
        assert "previously suggested move" in second_prompt


def test_hydra_config_loading_smoke_test() -> None:
    """Key Hydra configs should load with common overrides."""
    configs_to_test = [
        ("config", []),
        ("config", ["players@players.white=random"]),
        ("config", ["players@players.white=stockfish/elo_1600"]),
        (
            "config",
            [
                "players@players.white=llm/default",
                "players.white.connector.model=gpt-4o-mini",
            ],
        ),
    ]

    for config_name, overrides in configs_to_test:
        app_config = load_app_config(config_name, overrides)
        assert app_config.players.white is not None
        assert app_config.players.black is not None


def test_end_to_end_game_with_metrics() -> None:
    """Full game run should complete whether metrics are enabled or not."""
    app_config = load_app_config(
        "config",
        [
            "game.max_num_moves=10",
            "game.display_board=false",
            "game.enable_metrics=true",
        ],
    )

    game = run_game_from_config(app_config)
    assert game.board.move_stack  # At least one move played
    assert len(game.board.move_stack) <= 10

    app_config_no_metrics = load_app_config(
        "config",
        [
            "game.max_num_moves=10",
            "game.display_board=false",
            "game.enable_metrics=false",
        ],
    )

    game_no_metrics = run_game_from_config(app_config_no_metrics)
    assert game_no_metrics.board.move_stack
    assert len(game_no_metrics.board.move_stack) <= 10
