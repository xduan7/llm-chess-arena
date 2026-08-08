"""Factories that build configured chess arena components."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, cast

from llm_chess_arena.policies import config_operation
from llm_chess_arena.metrics import MetricsTracker, MoveQualityThresholds
from llm_chess_arena.player.base_player import BasePlayer
from llm_chess_arena.player.random_player import RandomPlayer
from llm_chess_arena.player.stockfish_player import StockfishPlayer
from llm_chess_arena.player.llm import (
    GameArenaLLMMoveHandler,
    LLMConnector,
    LLMPlayer,
)

if TYPE_CHECKING:  # pragma: no cover
    from llm_chess_arena.config import (
        LLMConnectorConfig,
        LLMHandlerConfig,
        LLMPlayerConfig,
        MetricsConfig,
        PlayerConfig,
        RandomPlayerConfig,
        StockfishPlayerConfig,
    )
    from llm_chess_arena.rate_limiter import TokenBucketRateLimiter


class PlayerFactory:
    """Factory for instantiating player objects from configuration."""

    @staticmethod
    @config_operation
    def create_player(
        player_cfg: "PlayerConfig", rate_limiter: "TokenBucketRateLimiter | None" = None
    ) -> BasePlayer:
        """Create a player implementation from its configuration dataclass.

        Args:
            player_cfg: Player configuration containing kind and type-specific settings.
            rate_limiter: Optional tournament rate limiter for API throttling (LLM players only).

        Returns:
            BasePlayer: Configured player instance ready for gameplay.

        Raises:
            ValueError: If the player kind is unsupported or configuration is invalid.
        """
        kind = getattr(player_cfg, "kind", None)
        if kind == "random":
            return PlayerFactory._create_random_player(
                cast("RandomPlayerConfig", player_cfg)
            )
        if kind == "stockfish":
            return PlayerFactory._create_stockfish_player(
                cast("StockfishPlayerConfig", player_cfg)
            )
        if kind == "llm":
            return PlayerFactory._create_llm_player(
                cast("LLMPlayerConfig", player_cfg), rate_limiter=rate_limiter
            )
        raise ValueError(f"Unsupported player kind: {kind}")

    @staticmethod
    def _create_random_player(
        random_player_cfg: "RandomPlayerConfig",
    ) -> RandomPlayer:
        """Create random player from configuration."""
        if random_player_cfg.color is None:
            raise ValueError("Random player configuration must have a color assigned")
        name = (
            random_player_cfg.name or f"Random {random_player_cfg.color.capitalize()}"
        )
        return RandomPlayer(
            name=name,
            color=random_player_cfg.color,
            seed=random_player_cfg.seed,
        )

    @staticmethod
    def _create_stockfish_player(
        stockfish_player_cfg: "StockfishPlayerConfig",
    ) -> StockfishPlayer:
        """Create Stockfish player from configuration."""
        if stockfish_player_cfg.color is None:
            raise ValueError(
                "Stockfish player configuration must have a color assigned"
            )
        name = stockfish_player_cfg.name or "Stockfish"
        engine_limits = (
            dict(stockfish_player_cfg.engine_limits)
            if stockfish_player_cfg.engine_limits
            else None
        )
        engine_options = (
            dict(stockfish_player_cfg.engine_options)
            if stockfish_player_cfg.engine_options
            else None
        )
        return StockfishPlayer(
            name=name,
            color=stockfish_player_cfg.color,
            binary_path=stockfish_player_cfg.binary_path,
            engine_limits=engine_limits,
            engine_options=engine_options,
        )

    @staticmethod
    def _create_llm_player(
        llm_player_cfg: "LLMPlayerConfig",
        rate_limiter: "TokenBucketRateLimiter | None" = None,
    ) -> LLMPlayer:
        """Create LLM player from configuration.

        Args:
            llm_player_cfg: LLM player configuration.
            rate_limiter: Optional tournament rate limiter for API throttling.

        Returns:
            LLMPlayer: Configured LLM player instance.
        """
        if llm_player_cfg.color is None:
            raise ValueError("LLM player configuration must have a color assigned")

        connector_cfg = llm_player_cfg.connector
        if connector_cfg is None:
            raise ValueError("LLM player configuration requires connector settings")

        connector = PlayerFactory._create_llm_connector(
            connector_cfg, rate_limiter=rate_limiter
        )
        handler = PlayerFactory._create_llm_handler(llm_player_cfg.handler)
        name = llm_player_cfg.name or connector_cfg.model

        assert llm_player_cfg.max_move_retries is not None
        assert llm_player_cfg.num_votes is not None

        return LLMPlayer(
            name=name,
            color=llm_player_cfg.color,
            connector=connector,
            handler=handler,
            max_move_retries=llm_player_cfg.max_move_retries,
            num_votes=llm_player_cfg.num_votes,
        )

    @staticmethod
    def _create_llm_connector(
        connector_cfg: "LLMConnectorConfig",
        rate_limiter: "TokenBucketRateLimiter | None" = None,
    ) -> LLMConnector:
        """Create LLM connector from configuration.

        Args:
            connector_cfg: Connector configuration.
            rate_limiter: Optional tournament rate limiter for API throttling.

        Returns:
            LLMConnector: Configured connector instance.
        """
        if connector_cfg.model is None:
            raise ValueError("LLM connector requires a model to be specified")

        max_num_tokens = None
        if connector_cfg.max_num_tokens is not None:
            if (
                isinstance(connector_cfg.max_num_tokens, float)
                and connector_cfg.max_num_tokens < 1
            ):
                raise ValueError(
                    f"Fractional max_num_tokens ({connector_cfg.max_num_tokens}) cannot be converted to integer - "
                    f"this indicates a configuration error where token normalization failed"
                )
            max_num_tokens = int(connector_cfg.max_num_tokens)

        return LLMConnector(
            model=connector_cfg.model,
            temperature=connector_cfg.temperature,
            max_num_tokens=max_num_tokens,
            request_timeout_in_seconds=connector_cfg.request_timeout_in_seconds,
            max_api_request_retries=connector_cfg.max_api_request_retries,
            provider=connector_cfg.provider,
            api_base=connector_cfg.api_base,
            rate_limiter=rate_limiter,
        )

    @staticmethod
    def _create_llm_handler(
        handler_cfg: "LLMHandlerConfig" | None,
    ) -> GameArenaLLMMoveHandler:
        """Create LLM move handler from configuration."""
        kind = getattr(handler_cfg, "kind", "game_arena")
        if kind == "game_arena":
            return GameArenaLLMMoveHandler()
        raise ValueError(f"Unsupported LLM handler kind: {kind}")


class MetricsFactory:
    """Create metrics trackers based on configuration."""

    @staticmethod
    @config_operation
    def create_metrics_tracker(
        metrics_cfg: "MetricsConfig",
    ) -> MetricsTracker:
        """Create a Stockfish-based metrics tracker from configuration.

        Args:
            metrics_cfg: Metrics configuration containing Stockfish settings and quality thresholds.

        Returns:
            MetricsTracker: Configured metrics tracker for move evaluation.
        """
        engine_options: Mapping[str, Any] | None = None
        if metrics_cfg.stockfish_engine_options:
            engine_options = dict(metrics_cfg.stockfish_engine_options)

        thresholds_cfg = metrics_cfg.quality_thresholds
        thresholds = MoveQualityThresholds(
            excellent=thresholds_cfg.excellent,
            good=thresholds_cfg.good,
            inaccuracy=thresholds_cfg.inaccuracy,
            mistake=thresholds_cfg.mistake,
        )

        return MetricsTracker.from_stockfish(
            depth=metrics_cfg.stockfish_depth,
            binary_path=metrics_cfg.stockfish_binary_path,
            engine_options=engine_options,
            thresholds=thresholds,
            max_centipawn_loss=metrics_cfg.max_centipawn_loss_per_move,
        )
