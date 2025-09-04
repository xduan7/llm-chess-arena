"""Configuration module for loading environment variables."""

import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from loguru import logger

# Track whether environment has been loaded
_ENV_LOADED = False


def load_env(filename: str | None = None, override: bool = False) -> Path | None:
    """Load environment variables from .env file.

    Once loaded, subsequent calls are skipped unless override=True.
    Tests should use override=True to reload different configs.

    Args:
        filename: Optional .env filename. Defaults to ENV_FILE env var or '.env'.
        override: Whether to override existing environment variables.

    Returns:
        Path to the .env file that was loaded, or None if not found.
    """
    global _ENV_LOADED

    if _ENV_LOADED and not override:
        return None

    env_file = filename or os.environ.get("ENV_FILE", ".env")
    dotenv_path = find_dotenv(env_file, usecwd=True)

    if dotenv_path:
        load_dotenv(dotenv_path, override=override)
        _ENV_LOADED = True
        logger.debug(f"Loaded environment from: {dotenv_path}")
        return Path(dotenv_path)
    else:
        logger.debug(f"No .env file found: {env_file}")
        return None
