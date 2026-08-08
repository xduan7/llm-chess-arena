"""Connector that wraps LiteLLM for consistent API usage."""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Any, Iterable, TYPE_CHECKING

from loguru import logger

import litellm
from litellm import exceptions as litellm_exceptions

from llm_chess_arena.config.schema import ARGO_MODEL_CANONICAL_NAMES
from llm_chess_arena.exceptions import LLMPermanentError, LLMEmptyResponseError

if TYPE_CHECKING:
    from llm_chess_arena.rate_limiter import TokenBucketRateLimiter

litellm.suppress_debug_info = True


ARGO_MODEL_PREFIX = "argo:"
ARGO_DUMMY_API_KEY = "sk-argo-placeholder"


# models (OpenAI, Anthropic, Gemini) rather than erroring. Research code needs flexibility.
litellm.drop_params = True
# LiteLLM's verbose attribute/method availability varies across installations
set_verbose = getattr(litellm, "set_verbose", None)
if callable(set_verbose):
    set_verbose(False)
else:
    # Fallback to direct attribute if method unavailable
    setattr(litellm, "verbose", False)


class LLMConnector:
    """Wrapper around LiteLLM for testing isolation and API stability."""

    def __init__(
        self,
        model: str,
        temperature: float,
        max_num_tokens: int | None,
        request_timeout_in_seconds: float,
        max_api_request_retries: int,
        provider: str | None = None,
        api_base: str | None = None,
        rate_limiter: TokenBucketRateLimiter | None = None,
    ) -> None:
        """Configure a LiteLLM-backed connector for querying language models.

        Args:
            model: Provider-specific model identifier.
            temperature: Sampling temperature for completions.
            max_num_tokens: Maximum number of completion tokens to request.
            request_timeout_in_seconds: Request timeout, in seconds.
            max_api_request_retries: Maximum retries for transient failures.
            provider: Optional LiteLLM provider override (e.g., "anthropic").
            api_base: Optional custom API base URL for self-hosted endpoints.
            rate_limiter: Optional shared rate limiter (typically tournament-level) for
                coordinated throttling across multiple players/games.
        """
        self.model = model
        self.temperature = temperature
        self.max_num_tokens = max_num_tokens
        self.request_timeout_in_seconds = request_timeout_in_seconds
        self.max_api_request_retries = max_api_request_retries
        self.provider = provider
        self.api_base = api_base.rstrip("/") if api_base else None
        self._default_request_parameters: dict[str, Any] = {}
        self._rate_limiter = rate_limiter

        # Usage bookkeeping
        self._last_usage: UsageRecord | None = None
        self._total_usage = UsageRecord()

        if self.model.lower().startswith(ARGO_MODEL_PREFIX):
            self._setup_argo()

    @staticmethod
    def _sanitize_for_logging(message: str) -> str:
        """Sanitize exception messages to prevent API key exposure in logs.

        Args:
            message: Raw exception message that might contain sensitive data.

        Returns:
            str: Sanitized message with API keys redacted.
        """
        # Common API key patterns to redact
        patterns = [
            r"sk-[a-zA-Z0-9]{48}",  # OpenAI standard format
            r"sk-[a-zA-Z0-9-_]{10,}",  # Generic sk- prefix (min 10 chars)
            r"Bearer [a-zA-Z0-9-_.]{10,}",  # Bearer tokens
            r'api_key["\']?\s*[:=]\s*["\']?[a-zA-Z0-9-_]{6,}["\']?',  # api_key assignments
            r'authorization["\']?\s*[:=]\s*["\']?[a-zA-Z0-9-_.]{10,}["\']?',  # Authorization fields
        ]

        sanitized = message
        for pattern in patterns:
            sanitized = re.sub(
                pattern, "***REDACTED***", sanitized, flags=re.IGNORECASE
            )

        return sanitized

    def _get_endpoint_description(self) -> str:
        """Get a human-readable description of the configured endpoint."""
        if self.api_base:
            return self.api_base
        if self.provider:
            return f"{self.provider} ({self.model})"
        return self.model

    def _get_provider_for_rate_limiting(self) -> str:
        """Extract provider name for rate limiting.

        Returns:
            str: Provider name (e.g., "openai", "anthropic").
        """
        if self.provider:
            return self.provider.lower()

        # Infer from model name (e.g., "openai/gpt-4" -> "openai")
        if "/" in self.model:
            return self.model.split("/")[0].lower()

        # Default heuristics for common model names
        model_lower = self.model.lower()
        if "gpt" in model_lower:
            return "openai"
        elif "claude" in model_lower:
            return "anthropic"
        elif "gemini" in model_lower or "palm" in model_lower:
            return "google"

        return "unknown"

    def _setup_argo(self) -> None:
        """Configure Argo-specific connector settings based on model alias."""
        parts = self.model.split(":", maxsplit=1)
        alias = parts[1].strip() if len(parts) > 1 else ""
        if not alias:
            raise ValueError("Argo model alias missing after 'argo:'")

        # Validate that the argo model is in the allowed list
        full_model_name = f"{ARGO_MODEL_PREFIX}{alias}"
        if full_model_name not in ARGO_MODEL_CANONICAL_NAMES:
            available_models = sorted(
                [
                    model.replace(ARGO_MODEL_PREFIX, "")
                    for model in ARGO_MODEL_CANONICAL_NAMES.keys()
                ]
            )
            raise ValueError(
                f"Unrecognized argo model: '{alias}'. "
                f"Available argo models: {', '.join(available_models)}"
            )

        api_base_url = self.api_base or os.getenv("ARGO_API_BASE")
        if not api_base_url or not api_base_url.strip():
            raise ValueError("Argo models require connector.api_base or ARGO_API_BASE")
        self.api_base = api_base_url.strip().rstrip("/")
        self.model = full_model_name
        if self.provider and self.provider.lower() != "openai":
            logger.warning(
                "Ignoring provider {} for Argo model {}; using LiteLLM openai adapter",
                self.provider,
                self.model,
            )
        self.provider = None
        api_key = os.getenv("ARGO_API_KEY") or ARGO_DUMMY_API_KEY
        self._default_request_parameters.setdefault("api_key", api_key)
        self._default_request_parameters.setdefault("custom_llm_provider", "openai")
        logger.debug("Configured Argo model {} via {}", alias, self.api_base)

    def query(
        self,
        prompt: str,
        n: int = 1,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Execute a chat completion request against the configured provider.

        Args:
            prompt: Primary user message content.
            n: Number of completion samples to request.
            system_prompt: Optional system message for context framing.
            **kwargs: Additional provider-specific keyword arguments.

        Returns:
            list[str]: Text completions returned by the provider.

        Raises:
            ConnectionError: When the provider reports an error condition.
            TimeoutError: When the request exceeds the configured timeout.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        logger.debug(
            "Querying model {} with {} message(s) requesting {} completion(s)",
            self.model,
            len(messages),
            n,
        )

        completion_kwargs: dict[str, Any] = {
            "messages": messages,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_num_tokens,
            "timeout": self.request_timeout_in_seconds,
            "max_retries": 0,  # Handle retries ourselves for better logging
            "n": n,
            # LiteLLM's override parameter is custom_llm_provider; a bare
            # "provider" kwarg is silently dropped by drop_params
            "custom_llm_provider": self.provider,
            "api_base": self.api_base,
            **kwargs,
            **self._default_request_parameters,
        }
        completion_kwargs = {
            key: value for key, value in completion_kwargs.items() if value is not None
        }

        max_attempts = self.max_api_request_retries + 1
        endpoint = self._get_endpoint_description()

        # Acquire rate limit permit if limiter is configured
        if self._rate_limiter is not None:
            provider_name = self._get_provider_for_rate_limiting()
            permit_granted = self._rate_limiter.acquire_permit(
                provider_name, timeout_in_sec=self.request_timeout_in_seconds
            )
            if not permit_granted:
                raise TimeoutError(
                    f"Rate limiter timeout waiting for permit from {provider_name}"
                )

        for attempt in range(1, max_attempts + 1):
            try:
                response = litellm.completion(**completion_kwargs)
                contents = self._extract_response_contents(response)
                self._capture_usage(response)
                logger.debug(
                    "{} returned {} response(s): {}",
                    self.model,
                    len(contents),
                    contents,
                )
                return contents

            except LLMEmptyResponseError:
                # Let empty response errors bubble up to player for move retries
                raise
            except (
                litellm_exceptions.AuthenticationError,
                litellm_exceptions.InvalidRequestError,
                litellm_exceptions.BadRequestError,
                litellm_exceptions.ContentPolicyViolationError,
            ) as permanent_api_error:
                logger.error(
                    "Permanent API error (will not retry): {}",
                    self._sanitize_for_logging(str(permanent_api_error)),
                )
                raise LLMPermanentError(
                    f"LLM API request invalid: {permanent_api_error}"
                ) from permanent_api_error
            except (
                litellm_exceptions.Timeout,
                litellm_exceptions.RateLimitError,
                litellm_exceptions.ServiceUnavailableError,
                litellm_exceptions.InternalServerError,
                litellm_exceptions.APIError,
                litellm_exceptions.APIConnectionError,
            ) as transient_api_error:
                # Extract status code and error type for logging
                status_code = getattr(transient_api_error, "status_code", 0)
                if isinstance(status_code, str):
                    try:
                        status_code = int(status_code)
                    except (ValueError, TypeError):
                        status_code = 0

                error_type = (
                    type(transient_api_error).__name__.replace("Error", "").lower()
                )

                logger.warning(
                    "Network attempt {}/{} to {} failed - {} ({})",
                    attempt,
                    max_attempts,
                    endpoint,
                    error_type,
                    status_code or "unknown",
                )

                if attempt >= max_attempts:
                    raise ConnectionError(
                        f"{error_type.replace('_', ' ').title()} ({status_code or 'unknown'}) after {max_attempts} network attempts"
                    ) from transient_api_error

                # Exponential backoff with minute-based delays (1m, 2m, 4m, 8m, 16m cap)
                if attempt < max_attempts:
                    backoff_minutes = min(2 ** (attempt - 1), 16)
                    backoff_seconds = backoff_minutes * 60
                    logger.info(
                        "Retrying {} in {} minute(s)...",
                        endpoint,
                        backoff_minutes,
                    )
                    time.sleep(backoff_seconds)

            except Exception as unexpected_error:  # pragma: no cover
                error_type = (
                    type(unexpected_error).__name__.replace("Error", "").lower()
                )
                logger.error(
                    "Unexpected error connecting to {} - {}", endpoint, error_type
                )
                raise ConnectionError(f"Unexpected {error_type}") from unexpected_error

        # Should never reach here
        raise ConnectionError("Network retries exhausted")

    def _extract_response_contents(self, response: Any) -> list[str]:
        """Return cleaned completion strings from the LiteLLM response payload."""
        try:
            choices: Iterable[Any] = response.choices
        except AttributeError as response_attribute_error:  # pragma: no cover
            raise LLMEmptyResponseError(
                "LLM response missing choices payload"
            ) from response_attribute_error

        contents: list[str] = []
        for choice in choices:
            message = getattr(choice, "message", None)
            content = getattr(message, "content", None) if message is not None else None
            if content is None:
                raise LLMEmptyResponseError("LLM response missing content message")
            content_str = str(content).strip()
            if not content_str:
                raise LLMEmptyResponseError(
                    "LLM response contains empty content (check max_num_tokens setting)"
                )
            contents.append(content_str)

        if not contents:
            raise LLMEmptyResponseError("LLM response contained no choices")

        return contents

    # --- Usage tracking ---------------------------------------------------------------

    def get_last_usage(self) -> UsageRecord | None:
        """Return the most recent usage metrics captured from the provider.

        Returns:
            UsageRecord | None: Snapshot of the last request's usage, if available.
        """
        return self._last_usage.copy() if self._last_usage is not None else None

    def get_total_usage(self) -> UsageRecord:
        """Return cumulative usage metrics across all requests.

        Returns:
            UsageRecord: Aggregated usage across the lifetime of the connector.
        """
        return self._total_usage.copy()

    def reset_usage(self) -> None:
        """Reset usage bookkeeping to an empty state."""
        self._last_usage = None
        self._total_usage = UsageRecord()

    def _capture_usage(self, response: Any) -> None:
        """Persist per-call usage data into last and cumulative trackers."""
        try:
            usage_record = self._extract_usage(response)
        except Exception as usage_extraction_error:  # pragma: no cover
            logger.debug(
                "Could not read token usage from API response: {}",
                usage_extraction_error,
            )
            usage_record = None

        if usage_record is None:
            self._last_usage = None
            return

        self._last_usage = usage_record
        self._total_usage.add(usage_record)

    @staticmethod
    def _extract_usage(response: Any) -> UsageRecord | None:
        """Extract a UsageRecord from the completion response when available."""
        usage_payload = getattr(response, "usage", None)
        if usage_payload is None and isinstance(response, dict):
            usage_payload = response.get("usage")

        if usage_payload is None:
            return None

        prompt_tokens = LLMConnector._safe_int(usage_payload, "prompt_tokens")
        completion_tokens = LLMConnector._safe_int(usage_payload, "completion_tokens")
        total_tokens = LLMConnector._safe_int(usage_payload, "total_tokens")

        if total_tokens == 0 and (prompt_tokens or completion_tokens):
            total_tokens = prompt_tokens + completion_tokens

        cost = LLMConnector._safe_cost(response)

        return UsageRecord(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost=cost,
        )

    @staticmethod
    def _safe_int(payload: Any, key: str) -> int:
        """Safely coerce a usage field to ``int`` without raising exceptions."""
        if isinstance(payload, dict):
            value = payload.get(key, 0)
        else:
            value = getattr(payload, key, 0)

        if value is None:
            return 0
        if isinstance(value, (int, float)):
            return int(value)

        try:
            return int(str(value))
        except (TypeError, ValueError):  # pragma: no cover
            return 0

    @staticmethod
    def _safe_cost(response: Any) -> float:
        """Safely coerce cost metadata to ``float`` with broad compatibility."""
        # LiteLLM's completion_cost is not always available across installations
        completion_cost_fn = getattr(litellm, "completion_cost", None)
        cost_value: Any = 0.0

        try:
            if callable(completion_cost_fn):
                cost_value = completion_cost_fn(response)
            else:
                raise AttributeError("completion_cost not available")
        except Exception:
            cost_value = getattr(response, "cost", 0.0)
            if isinstance(cost_value, dict):
                cost_value = cost_value.get("total_cost", 0.0)

        if cost_value is None:
            return 0.0
        if isinstance(cost_value, (int, float)):
            return float(cost_value)

        try:
            return float(str(cost_value))
        except (TypeError, ValueError):  # pragma: no cover
            return 0.0


@dataclass
class UsageRecord:
    """Track token usage and cost details for a single request."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0

    def add(self, other: UsageRecord) -> None:
        """Accumulate token usage and cost figures.

        Args:
            other: Usage record whose values should be added to this instance.
        """
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens
        self.cost += other.cost

    def copy(self) -> UsageRecord:
        """Create a copy of the usage record for safe external consumption.

        Returns:
            UsageRecord: Duplicate usage record with identical values.
        """
        return UsageRecord(
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            total_tokens=self.total_tokens,
            cost=self.cost,
        )
