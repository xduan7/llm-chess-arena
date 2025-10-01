"""Unit tests for the player factory."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_chess_arena.config import (
    LLMConnectorConfig,
    LLMHandlerConfig,
    LLMPlayerConfig,
    RandomPlayerConfig,
    StockfishPlayerConfig,
)
from llm_chess_arena.factory.player_factory import PlayerFactory
from llm_chess_arena.player.llm import LLMPlayer
from llm_chess_arena.player.random_player import RandomPlayer
from llm_chess_arena.player.stockfish_player import StockfishPlayer


class TestPlayerFactory:
    """Validate that PlayerFactory instantiates supported player types."""

    def test_create_random_player(self) -> None:
        """Random configuration should yield a RandomPlayer instance."""
        config = RandomPlayerConfig(color="white", name="Test Random", seed=42)
        player = PlayerFactory.create_player(config)

        assert isinstance(player, RandomPlayer)
        assert player.color == "white"
        assert player.name == "Test Random"
        assert player.seed == 42

    def test_create_stockfish_player(self) -> None:
        """Stockfish configuration should return a StockfishPlayer."""
        config = StockfishPlayerConfig(
            color="black",
            name="Test Stockfish",
            engine_limits={"depth": 5},
            engine_options={"Threads": 2},
        )

        player = PlayerFactory.create_player(config)

        assert isinstance(player, StockfishPlayer)
        assert player.color == "black"
        assert player.name == "Test Stockfish"
        assert player.engine_limits == {"depth": 5}
        assert player.engine_options == {"Threads": 2}

    def test_create_llm_player(self) -> None:
        """LLM configuration should construct an LLMPlayer."""
        connector_config = LLMConnectorConfig(
            model="gpt-4",
            temperature=0.5,
            request_timeout_in_seconds=60.0,
            max_api_request_retries=3,
        )
        handler_config = LLMHandlerConfig(kind="game_arena")
        config = LLMPlayerConfig(
            color="white",
            name="Test LLM",
            connector=connector_config,
            handler=handler_config,
            max_move_retries=5,
            num_votes=3,
        )

        player = PlayerFactory.create_player(config)

        assert isinstance(player, LLMPlayer)
        assert player.color == "white"
        assert player.name == "Test LLM"
        assert player.max_move_retries == 5
        assert player.num_votes == 3

    def test_llm_player_missing_connector(self) -> None:
        """Missing connector must raise a validation error."""
        config = LLMPlayerConfig(
            color="white",
            connector=None,  # type: ignore[arg-type]
            handler=LLMHandlerConfig(kind="game_arena"),
        )

        with pytest.raises(ValueError, match="requires connector"):
            PlayerFactory.create_player(config)

    def test_unsupported_player_kind(self) -> None:
        """Unsupported kinds should raise ValueError."""
        dummy_config = SimpleNamespace(kind="unsupported", color="white")

        with pytest.raises(ValueError, match="Unsupported player kind"):
            PlayerFactory.create_player(dummy_config)  # type: ignore[arg-type]
