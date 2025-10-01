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
    def create_player(
        player_config: "PlayerConfig", rate_limiter: object | None = None
    ) -> BasePlayer:
        """Create a player implementation from its configuration dataclass.

        Args:
            player_config: Player configuration containing kind and type-specific settings.
            rate_limiter: Optional tournament rate limiter for API throttling (LLM players only).

        Returns:
            BasePlayer: Configured player instance ready for gameplay.

        Raises:
            ValueError: If the player kind is unsupported or configuration is invalid.
        """
        kind = getattr(player_config, "kind", None)
        if kind == "random":
            return PlayerFactory._create_random_player(
                cast("RandomPlayerConfig", player_config)
            )
        if kind == "stockfish":
            return PlayerFactory._create_stockfish_player(
                cast("StockfishPlayerConfig", player_config)
            )
        if kind == "llm":
            return PlayerFactory._create_llm_player(
                cast("LLMPlayerConfig", player_config), rate_limiter=rate_limiter
            )
        raise ValueError(f"Unsupported player kind: {kind}")

    @staticmethod
    def _create_random_player(
        random_player_config: "RandomPlayerConfig",
    ) -> RandomPlayer:
        """Create random player from configuration."""
        name = (
            random_player_config.name
            or f"Random {random_player_config.color.capitalize()}"
        )
        return RandomPlayer(
            name=name,
            color=random_player_config.color,
            seed=random_player_config.seed,
        )

    @staticmethod
    def _create_stockfish_player(
        stockfish_player_config: "StockfishPlayerConfig",
    ) -> StockfishPlayer:
        """Create Stockfish player from configuration."""
        name = stockfish_player_config.name or "Stockfish"
        engine_limits = (
            dict(stockfish_player_config.engine_limits)
            if stockfish_player_config.engine_limits
            else None
        )
        engine_options = (
            dict(stockfish_player_config.engine_options)
            if stockfish_player_config.engine_options
            else None
        )
        return StockfishPlayer(
            name=name,
            color=stockfish_player_config.color,
            binary_path=stockfish_player_config.binary_path,
            engine_limits=engine_limits,
            engine_options=engine_options,
        )

    @staticmethod
    def _create_llm_player(
        llm_player_config: "LLMPlayerConfig", rate_limiter: object | None = None
    ) -> LLMPlayer:
        """Create LLM player from configuration.

        Args:
            llm_player_config: LLM player configuration.
            rate_limiter: Optional tournament rate limiter for API throttling.

        Returns:
            LLMPlayer: Configured LLM player instance.
        """
        connector_config = llm_player_config.connector
        if connector_config is None:
            raise ValueError("LLM player configuration requires connector settings")

        connector = PlayerFactory._create_llm_connector(
            connector_config, rate_limiter=rate_limiter
        )
        handler = PlayerFactory._create_llm_handler(llm_player_config.handler)
        name = llm_player_config.name or connector_config.model

        if llm_player_config.max_move_retries is None:
            raise ValueError(
                "max_move_retries must be set by config normalization - check config pipeline"
            )
        if llm_player_config.num_votes is None:
            raise ValueError(
                "num_votes must be set by config normalization - check config pipeline"
            )

        return LLMPlayer(
            name=name,
            color=llm_player_config.color,
            connector=connector,
            handler=handler,
            max_move_retries=llm_player_config.max_move_retries,
            num_votes=llm_player_config.num_votes,
        )

    @staticmethod
    def _create_llm_connector(
        connector_config: "LLMConnectorConfig",
        rate_limiter: object | None = None,
    ) -> LLMConnector:
        """Create LLM connector from configuration.

        Args:
            connector_config: Connector configuration.
            rate_limiter: Optional tournament rate limiter for API throttling.

        Returns:
            LLMConnector: Configured connector instance.
        """
        if connector_config.model is None:
            raise ValueError("LLM connector requires a model to be specified")

        max_num_tokens = None
        if connector_config.max_num_tokens is not None:
            if (
                isinstance(connector_config.max_num_tokens, float)
                and connector_config.max_num_tokens < 1
            ):
                raise ValueError(
                    f"Fractional max_num_tokens ({connector_config.max_num_tokens}) cannot be converted to integer - "
                    f"this indicates a configuration error where token normalization failed"
                )
            max_num_tokens = int(connector_config.max_num_tokens)

        return LLMConnector(
            model=connector_config.model,
            temperature=connector_config.temperature,
            max_num_tokens=max_num_tokens,
            request_timeout_in_seconds=connector_config.request_timeout_in_seconds,
            max_api_request_retries=connector_config.max_api_request_retries,
            provider=connector_config.provider,
            api_base=connector_config.api_base,
            rate_limiter=rate_limiter,
        )

    @staticmethod
    def _create_llm_handler(
        handler_config: "LLMHandlerConfig" | None,
    ) -> GameArenaLLMMoveHandler:
        """Create LLM move handler from configuration."""
        kind = getattr(handler_config, "kind", "game_arena")
        if kind == "game_arena":
            return GameArenaLLMMoveHandler()
        raise ValueError(f"Unsupported LLM handler kind: {kind}")
