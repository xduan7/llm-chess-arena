"""Construction helpers for configured chess players."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from llm_chess_arena.core.policies import config_operation
from llm_chess_arena.player.base_player import BasePlayer
from llm_chess_arena.player.random_player import RandomPlayer
from llm_chess_arena.player.stockfish_player import StockfishPlayer
from llm_chess_arena.player.llm import (
    GameArenaLLMMoveHandler,
    LLMConnector,
    LLMPlayer,
)

if TYPE_CHECKING:  # pragma: no cover - import for typing only
    from llm_chess_arena.config import (
        LLMConnectorConfig,
        LLMHandlerConfig,
        LLMPlayerConfig,
        PlayerConfig,
        RandomPlayerConfig,
        StockfishPlayerConfig,
    )


class PlayerFactory:
    """Factory for instantiating player objects from configuration."""

    @staticmethod
    @config_operation
    def create_player(config: "PlayerConfig") -> BasePlayer:
        """Create a player implementation from its configuration dataclass.

        Args:
            config: Player configuration containing kind and type-specific settings.

        Returns:
            BasePlayer: Configured player instance ready for gameplay.

        Raises:
            ValueError: If the player kind is unsupported or configuration is invalid.
        """
        kind = getattr(config, "kind", None)
        if kind == "random":
            return PlayerFactory._create_random_player(
                cast("RandomPlayerConfig", config)
            )
        if kind == "stockfish":
            return PlayerFactory._create_stockfish_player(
                cast("StockfishPlayerConfig", config)
            )
        if kind == "llm":
            return PlayerFactory._create_llm_player(cast("LLMPlayerConfig", config))
        raise ValueError(f"Unsupported player kind: {kind}")

    @staticmethod
    def _create_random_player(config: "RandomPlayerConfig") -> RandomPlayer:
        """Create random player from configuration."""
        name = config.name or f"Random {config.color.capitalize()}"
        return RandomPlayer(name=name, color=config.color, seed=config.seed)

    @staticmethod
    def _create_stockfish_player(config: "StockfishPlayerConfig") -> StockfishPlayer:
        """Create Stockfish player from configuration."""
        name = config.name or "Stockfish"
        limits = dict(config.engine_limits) if config.engine_limits else None
        options = dict(config.engine_options) if config.engine_options else None
        return StockfishPlayer(
            name=name,
            color=config.color,
            binary_path=config.binary_path,
            engine_limits=limits,
            engine_options=options,
        )

    @staticmethod
    def _create_llm_player(config: "LLMPlayerConfig") -> LLMPlayer:
        """Create LLM player from configuration."""
        connector_cfg = config.connector
        if connector_cfg is None:
            raise ValueError("LLM player configuration requires connector settings")

        connector = PlayerFactory._create_llm_connector(connector_cfg)
        handler = PlayerFactory._create_llm_handler(config.handler)
        name = config.name or connector_cfg.model

        return LLMPlayer(
            name=name,
            color=config.color,
            connector=connector,
            handler=handler,
            max_move_retries=config.max_move_retries,
            num_votes=config.num_votes,
        )

    @staticmethod
    def _create_llm_connector(config: "LLMConnectorConfig") -> LLMConnector:
        """Create LLM connector from configuration."""
        return LLMConnector(
            model=config.model,
            temperature=config.temperature,
            max_tokens=(
                int(config.max_tokens) if config.max_tokens is not None else None
            ),
            timeout=config.timeout,
            max_retries=config.max_retries,
            provider=config.provider,
            api_base=config.api_base,
        )

    @staticmethod
    def _create_llm_handler(
        config: "LLMHandlerConfig" | None,
    ) -> GameArenaLLMMoveHandler:
        """Create LLM move handler from configuration."""
        kind = getattr(config, "kind", "game_arena")
        if kind == "game_arena":
            return GameArenaLLMMoveHandler()
        raise ValueError(f"Unsupported LLM handler kind: {kind}")
