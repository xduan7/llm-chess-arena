"""Model capability registry: output-token limits and Argo model allowlist.

Curated tables take precedence over LiteLLM's vendor registry where the
serving platform imposes its own constraints (e.g., Argo caps Claude output
at 21k tokens). Extend these tables when the Argo proxy adds models.
"""

from __future__ import annotations

import os
from typing import Any, Mapping

import litellm
from loguru import logger


_MODEL_METADATA_CACHE: dict[str, Mapping[str, Any]] = {}


def _load_default_max_num_tokens_ratio() -> float:
    """Load default token completion ratio from environment variable."""
    env_ratio_str = os.getenv("LLM_DEFAULT_MAX_NUM_TOKENS_RATIO")
    if env_ratio_str is None:
        return 0.8
    try:
        parsed_ratio = float(env_ratio_str)
    except ValueError:
        return 0.8
    return parsed_ratio if 0 < parsed_ratio <= 1 else 0.8


DEFAULT_MAX_NUM_TOKENS_RATIO = _load_default_max_num_tokens_ratio()


BASE_MODEL_OUTPUT_TOKEN_LIMITS: dict[str, int] = {
    "gpt-4.1": 16_384,
    "gpt-4.1-mini": 16_384,
    "gpt-4.1-nano": 16_384,
    "gpt-4o": 16_384,
    "gpt-4o-mini": 16_384,
    "gpt-5": 128_000,
    "gpt-5-mini": 128_000,
    "gpt-5-nano": 128_000,
    # gpt-5.1+ limits assume the family's 128k output cap - verify when
    # vendor documentation for these releases is available
    "gpt-5.1": 128_000,
    "gpt-5.2": 128_000,
    "gpt-5.4": 128_000,
    "gpt-5.4-mini": 128_000,
    "gpt-5.4-nano": 128_000,
    "gpt-5.5": 128_000,
    "gpt-5.6-sol": 128_000,
    "gpt-5.6-luna": 128_000,
    "gpt-5.6-terra": 128_000,
    "gpt-o3": 100_000,
    "o3": 100_000,
    "gpt-o3-mini": 100_000,
    "o3-mini": 100_000,
    "gpt-o4-mini": 65_536,
    "o4-mini": 65_536,
    "claude-4.1-opus": 32_000,
    "claude-opus-4.1": 32_000,
    "claude-4-opus": 32_000,
    "claude-opus-4": 32_000,
    "claude-4.5-sonnet": 64_000,
    "claude-sonnet-4.5": 64_000,
    "claude-4-sonnet": 64_000,
    "claude-sonnet-4": 64_000,
    "claude-3.7-sonnet": 128_000,
    "claude-sonnet-3.7": 128_000,
    "claude-3.5-sonnet-v2": 8_000,
    "claude-sonnet-3.5-v2": 8_000,
    "gemini-2.5-pro": 65_536,
    "gemini-2.5-flash": 65_536,
    # gemini-3.x limits assume the family's 64k output cap - verify when
    # vendor documentation for these releases is available
    "gemini-3.5-flash": 65_536,
    "gemini-3.1-flash-lite": 65_536,
    # Claude 4.5+ tiers: 64k output like Sonnet 4.5 (Argo overrides cap
    # these at 21k regardless); newer-than-4.5 values are assumptions
    "claude-4.5-opus": 64_000,
    "claude-opus-4.5": 64_000,
    "claude-4.6-opus": 64_000,
    "claude-opus-4.6": 64_000,
    "claude-4.7-opus": 64_000,
    "claude-opus-4.7": 64_000,
    "claude-4.8-opus": 64_000,
    "claude-opus-4.8": 64_000,
    "claude-5-opus": 64_000,
    "claude-opus-5": 64_000,
    "claude-4.6-sonnet": 64_000,
    "claude-sonnet-4.6": 64_000,
    "claude-5-sonnet": 64_000,
    "claude-sonnet-5": 64_000,
    "claude-4.5-haiku": 64_000,
    "claude-haiku-4.5": 64_000,
    # Argonne-internal test models: conservative floor until limits are known
    "laguna-s-2.1-[test]": 8_192,
    "laguna-xs-2.1-[test]": 8_192,
    "gemma-4-31b-[test]": 8_192,
}


# Argo-specific token limit overrides
# Argo platform imposes additional constraints beyond vendor limits
ARGO_MODEL_OUTPUT_TOKEN_OVERRIDES: dict[str, int] = {
    # Claude models: Argo requires streaming for >21,000 tokens
    # Official limits: Opus 32K, Sonnet 4.5/4 64K, Sonnet 3.7 128K
    "claude-4.1-opus": 21_000,
    "claude-opus-4.1": 21_000,
    "claude-4-opus": 21_000,
    "claude-opus-4": 21_000,
    "claude-4.5-sonnet": 21_000,
    "claude-sonnet-4.5": 21_000,
    "claude-4-sonnet": 21_000,
    "claude-sonnet-4": 21_000,
    "claude-3.7-sonnet": 21_000,
    "claude-sonnet-3.7": 21_000,
    "claude-3.5-sonnet-v2": 8_000,
    "claude-sonnet-3.5-v2": 8_000,
    "claude-4.5-opus": 21_000,
    "claude-opus-4.5": 21_000,
    "claude-4.6-opus": 21_000,
    "claude-opus-4.6": 21_000,
    "claude-4.7-opus": 21_000,
    "claude-opus-4.7": 21_000,
    "claude-4.8-opus": 21_000,
    "claude-opus-4.8": 21_000,
    "claude-5-opus": 21_000,
    "claude-opus-5": 21_000,
    "claude-4.6-sonnet": 21_000,
    "claude-sonnet-4.6": 21_000,
    "claude-5-sonnet": 21_000,
    "claude-sonnet-5": 21_000,
    "claude-4.5-haiku": 21_000,
    "claude-haiku-4.5": 21_000,
}


ARGO_MODEL_CANONICAL_NAMES: dict[str, str] = {
    "argo:gpt-3.5-turbo": "gpt-3.5-turbo",
    "argo:gpt-3.5-turbo-16k": "gpt-3.5-turbo-16k",
    "argo:gpt-4": "gpt-4",
    "argo:gpt-4-32k": "gpt-4-32k",
    "argo:gpt-4-turbo": "gpt-4-turbo",
    "argo:gpt-4o": "gpt-4o",
    "argo:gpt-o1-preview": "gpt-o1-preview",
    "argo:o1-preview": "o1-preview",
    "argo:gpt-4o-latest": "gpt-4o-latest",
    "argo:gpt-o1-mini": "gpt-o1-mini",
    "argo:o1-mini": "o1-mini",
    "argo:gpt-o3-mini": "gpt-o3-mini",
    "argo:o3-mini": "o3-mini",
    "argo:gpt-o1": "gpt-o1",
    "argo:o1": "o1",
    "argo:gpt-o3": "gpt-o3",
    "argo:o3": "o3",
    "argo:gpt-o4-mini": "gpt-o4-mini",
    "argo:o4-mini": "o4-mini",
    "argo:gpt-4.1": "gpt-4.1",
    "argo:gpt-4.1-mini": "gpt-4.1-mini",
    "argo:gpt-4.1-nano": "gpt-4.1-nano",
    "argo:gpt-5": "gpt-5",
    "argo:gpt-5-mini": "gpt-5-mini",
    "argo:gpt-5-nano": "gpt-5-nano",
    "argo:gpt-5.1": "gpt-5.1",
    "argo:gpt-5.2": "gpt-5.2",
    "argo:gpt-5.4": "gpt-5.4",
    "argo:gpt-5.4-mini": "gpt-5.4-mini",
    "argo:gpt-5.4-nano": "gpt-5.4-nano",
    "argo:gpt-5.5": "gpt-5.5",
    "argo:gpt-5.6-sol": "gpt-5.6-sol",
    "argo:gpt-5.6-luna": "gpt-5.6-luna",
    "argo:gpt-5.6-terra": "gpt-5.6-terra",
    "argo:gemini-2.5-pro": "gemini-2.5-pro",
    "argo:gemini-2.5-flash": "gemini-2.5-flash",
    "argo:gemini-3.5-flash": "gemini-3.5-flash",
    "argo:gemini-3.1-flash-lite": "gemini-3.1-flash-lite",
    "argo:claude-4.1-opus": "claude-4.1-opus",
    "argo:claude-opus-4.1": "claude-opus-4.1",
    "argo:claude-4-opus": "claude-4-opus",
    "argo:claude-opus-4": "claude-opus-4",
    "argo:claude-4.5-sonnet": "claude-4.5-sonnet",
    "argo:claude-sonnet-4.5": "claude-sonnet-4.5",
    "argo:claude-4-sonnet": "claude-4-sonnet",
    "argo:claude-sonnet-4": "claude-sonnet-4",
    "argo:claude-3.7-sonnet": "claude-3.7-sonnet",
    "argo:claude-sonnet-3.7": "claude-sonnet-3.7",
    "argo:claude-3.5-sonnet-v2": "claude-3.5-sonnet-v2",
    "argo:claude-sonnet-3.5-v2": "claude-sonnet-3.5-v2",
    "argo:claude-4.5-opus": "claude-4.5-opus",
    "argo:claude-opus-4.5": "claude-opus-4.5",
    "argo:claude-4.6-opus": "claude-4.6-opus",
    "argo:claude-opus-4.6": "claude-opus-4.6",
    "argo:claude-4.7-opus": "claude-4.7-opus",
    "argo:claude-opus-4.7": "claude-opus-4.7",
    "argo:claude-4.8-opus": "claude-4.8-opus",
    "argo:claude-opus-4.8": "claude-opus-4.8",
    "argo:claude-5-opus": "claude-5-opus",
    "argo:claude-opus-5": "claude-opus-5",
    "argo:claude-4.6-sonnet": "claude-4.6-sonnet",
    "argo:claude-sonnet-4.6": "claude-sonnet-4.6",
    "argo:claude-5-sonnet": "claude-5-sonnet",
    "argo:claude-sonnet-5": "claude-sonnet-5",
    "argo:claude-4.5-haiku": "claude-4.5-haiku",
    "argo:claude-haiku-4.5": "claude-haiku-4.5",
    "argo:laguna-s-2.1-[test]": "laguna-s-2.1-[test]",
    "argo:laguna-xs-2.1-[test]": "laguna-xs-2.1-[test]",
    "argo:gemma-4-31b-[test]": "gemma-4-31b-[test]",
    "argo:text-embedding-ada-002": "text-embedding-ada-002",
    "argo:text-embedding-3-small": "text-embedding-3-small",
    "argo:text-embedding-3-large": "text-embedding-3-large",
}


MODEL_OUTPUT_TOKEN_LIMITS: dict[str, int] = {
    **BASE_MODEL_OUTPUT_TOKEN_LIMITS,
    **{
        argo_model: (
            ARGO_MODEL_OUTPUT_TOKEN_OVERRIDES.get(canonical_model_name)
            or BASE_MODEL_OUTPUT_TOKEN_LIMITS[canonical_model_name]
        )
        for argo_model, canonical_model_name in ARGO_MODEL_CANONICAL_NAMES.items()
        if canonical_model_name in BASE_MODEL_OUTPUT_TOKEN_LIMITS
    },
}


def _get_cached_model_info(model: str) -> Mapping[str, Any]:
    """Get LiteLLM model metadata with caching.

    Args:
        model: Model identifier to look up.

    Returns:
        Dictionary of model metadata from LiteLLM.
    """
    if model not in _MODEL_METADATA_CACHE:
        # litellm does not re-export get_model_info in its type stubs; resolve
        # it dynamically like the connector does for other litellm attributes
        get_model_info = getattr(litellm, "get_model_info", None)
        if not callable(get_model_info):
            raise RuntimeError("litellm.get_model_info is unavailable")
        _MODEL_METADATA_CACHE[model] = get_model_info(model)
    return _MODEL_METADATA_CACHE[model]


def resolve_model_limit(model: str | None) -> tuple[bool, int | None]:
    """Identify whether model is recognised and report its output token limit.

    Args:
        model: Model identifier to resolve.

    Returns:
        Tuple of (model_recognized, token_limit). token_limit is None if not available.
    """
    if model is None:
        return False, None

    model_recognized = False
    model_candidates = [model]
    if model.startswith("argo:"):
        model_candidates.append(model.split(":", 1)[1])
        # Argo platform constraints trump vendor limits: the curated table
        # (with ARGO_MODEL_OUTPUT_TOKEN_OVERRIDES merged under the argo: key)
        # must win over LiteLLM's registry, otherwise e.g. Claude models get
        # vendor-sized max_tokens that Argo rejects without streaming
        argo_token_limit = MODEL_OUTPUT_TOKEN_LIMITS.get(model)
        if argo_token_limit is not None:
            logger.debug("Using Argo token limit for {}: {}", model, argo_token_limit)
            return True, int(argo_token_limit)

    for candidate_model_name in model_candidates:
        try:
            model_info = _get_cached_model_info(candidate_model_name)
        except Exception:
            continue
        else:
            model_recognized = True
            if model_info is None:
                continue
            token_limit = model_info.get("max_output_tokens") or model_info.get(
                "max_tokens"
            )
            if token_limit is not None:
                logger.debug(
                    "Using LiteLLM token limit for {}: {}",
                    candidate_model_name,
                    token_limit,
                )
                return True, int(token_limit)

    for candidate_model_name in model_candidates:
        fallback_token_limit = MODEL_OUTPUT_TOKEN_LIMITS.get(candidate_model_name)
        if fallback_token_limit is not None:
            logger.debug(
                "Using fallback token limit for {}: {}",
                candidate_model_name,
                fallback_token_limit,
            )
            return True, int(fallback_token_limit)

    return model_recognized, None


def compute_fractional_tokens(token_limit: int, fractional_ratio: float) -> int:
    """Convert fractional ratio of a token limit into a bounded positive count.

    Args:
        token_limit: Maximum token count for the model.
        fractional_ratio: Fraction of the limit to use (0 < ratio <= 1).

    Returns:
        Integer token count bounded between 1 and token_limit.
    """
    num_tokens = int(token_limit * fractional_ratio)
    if num_tokens <= 0:
        num_tokens = 1
    if num_tokens > token_limit:
        num_tokens = token_limit
    return num_tokens
