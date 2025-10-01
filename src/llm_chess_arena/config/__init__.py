"""Public configuration API for the Chess Arena project."""

from __future__ import annotations

from llm_chess_arena.config.schema import (
    AppConfig,
    EnvConfig,
    GameConfig,
    LLMConnectorConfig,
    LLMHandlerConfig,
    LLMPlayerConfig,
    MetricsConfig,
    MoveQualityThresholdsConfig,
    PlayerConfig,
    PlayerConfigBase,
    PlayersConfig,
    RandomPlayerConfig,
    StockfishPlayerConfig,
    normalize_llm_player_config,
    parse_player_config,
    parse_players_config,
    resolve_model_limit,
)
from llm_chess_arena.config.loader import (
    app_config_from_dictconfig,
    apply_env_config,
    configure_logging,
    load_app_config,
    load_env,
)

__all__ = [
    "AppConfig",
    "EnvConfig",
    "GameConfig",
    "LLMConnectorConfig",
    "LLMHandlerConfig",
    "LLMPlayerConfig",
    "MetricsConfig",
    "MoveQualityThresholdsConfig",
    "PlayerConfig",
    "PlayerConfigBase",
    "PlayersConfig",
    "RandomPlayerConfig",
    "StockfishPlayerConfig",
    "app_config_from_dictconfig",
    "apply_env_config",
    "configure_logging",
    "load_app_config",
    "load_env",
    "normalize_llm_player_config",
    "parse_player_config",
    "parse_players_config",
    "resolve_model_limit",
]
