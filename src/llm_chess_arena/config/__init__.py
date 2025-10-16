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
    normalize_llm_player_cfg,
    parse_player_cfg,
    parse_players_cfg,
    resolve_model_limit,
)
from llm_chess_arena.config.loader import (
    app_cfg_from_dictconfig,
    apply_env_cfg,
    configure_logging,
    load_app_cfg,
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
    "app_cfg_from_dictconfig",
    "apply_env_cfg",
    "configure_logging",
    "load_app_cfg",
    "load_env",
    "normalize_llm_player_cfg",
    "parse_player_cfg",
    "parse_players_cfg",
    "resolve_model_limit",
]
