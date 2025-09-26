"""Connector that wraps LiteLLM for consistent API usage."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Iterable

from loguru import logger

import litellm
from litellm import exceptions as litellm_exceptions

from llm_chess_arena.core.policies import network_operation
from llm_chess_arena.exceptions import LLMPermanentError, LLMEmptyResponseError

litellm.suppress_debug_info = True


ARGO_MODEL_PREFIX = "argo:"
ARGO_DUMMY_API_KEY = "sk-argo-placeholder"


# Cross-provider robustness: silently ignore unsupported params when switching between
# models (OpenAI, Anthropic, Gemini) rather than erroring. Research code needs flexibility.
litellm.drop_params = True
# Disable verbose logging across litellm versions (set_verbose availability varies)
set_verbose = getattr(litellm, "set_verbose", None)
if callable(set_verbose):
    set_verbose(False)
    setattr(litellm, "verbose", False)
else:
    setattr(litellm, "verbose", False)


class LLMConnector:
    """Wrapper around LiteLLM for testing isolation and API stability."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        provider: str | None = None,
        api_base: str | None = None,
    ) -> None:
        """Configure a LiteLLM-backed connector for querying language models.

        Args:
            model: Provider-specific model identifier.
            temperature: Sampling temperature for completions.
            max_tokens: Maximum number of completion tokens to request.
            timeout: Request timeout, in seconds.
            max_retries: Maximum retries for transient failures.
            provider: Optional LiteLLM provider override (e.g., "anthropic").
            api_base: Optional custom API base URL for self-hosted endpoints.
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.provider = provider
        self.api_base = api_base.rstrip("/") if api_base else None
        self._default_request_parameters: dict[str, Any] = {}

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

    def _setup_argo(self) -> None:
        parts = self.model.split(":", maxsplit=1)
        alias = parts[1].strip() if len(parts) > 1 else ""
        if not alias:
            raise ValueError("Argo model alias missing after 'argo:'")

        base = self.api_base or os.getenv("ARGO_API_BASE")
        if not base or not base.strip():
            raise ValueError("Argo models require connector.api_base or ARGO_API_BASE")
        self.api_base = base.strip().rstrip("/")
        self.model = f"{ARGO_MODEL_PREFIX}{alias}"
        if self.provider and self.provider.lower() != "openai":
            logger.warning(
                "Ignoring provider {} for Argo model {}; using LiteLLM openai adapter",
                self.provider,
                self.model,
            )
        self.provider = None
        self._default_request_parameters.setdefault("api_key", ARGO_DUMMY_API_KEY)
        self._default_request_parameters.setdefault("custom_llm_provider", "openai")
        logger.debug("Configured Argo model {} via {}", alias, self.api_base)

    @network_operation
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

        logger.debug("Querying model {} with messages: {}", self.model, messages)

        completion_kwargs: dict[str, Any] = {
            "messages": messages,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "max_retries": 0,  # Handle retries ourselves for better logging
            "n": n,
            "provider": self.provider,
            "api_base": self.api_base,
            **kwargs,
            **self._default_request_parameters,
        }
        completion_kwargs = {
            key: value for key, value in completion_kwargs.items() if value is not None
        }

        max_attempts = self.max_retries + 1
        endpoint = self._get_endpoint_description()

        for attempt in range(1, max_attempts + 1):
            try:
                response = litellm.completion(**completion_kwargs)
                contents = self._extract_response_contents(response)
                self._capture_usage(response)
                logger.debug("{} response choices: {}", self.model, contents)
                return contents

            except LLMEmptyResponseError:
                # Let empty response errors bubble up to player for move retries
                raise
            except litellm_exceptions.Timeout as e:
                logger.warning("Request timed out after {}s", self.timeout)
                raise TimeoutError(f"Request timed out after {self.timeout}s") from e
            except (
                litellm_exceptions.AuthenticationError,
                litellm_exceptions.InvalidRequestError,
                litellm_exceptions.BadRequestError,
                litellm_exceptions.ContentPolicyViolationError,
            ) as e:
                logger.error(
                    "Permanent API error (will not retry): {}",
                    self._sanitize_for_logging(str(e)),
                )
                raise LLMPermanentError(f"LLM API request invalid: {e}") from e
            except (
                litellm_exceptions.RateLimitError,
                litellm_exceptions.ServiceUnavailableError,
                litellm_exceptions.InternalServerError,
                litellm_exceptions.APIError,
                litellm_exceptions.APIConnectionError,
            ) as e:
                status = getattr(e, "status_code", "unknown")
                error_type = type(e).__name__.replace("Error", "").lower()
                logger.warning(
                    "Network attempt {}/{} to {} failed - {} ({})",
                    attempt,
                    max_attempts,
                    endpoint,
                    error_type,
                    status,
                )

                if attempt >= max_attempts:
                    raise ConnectionError(
                        f"{error_type.replace('_', ' ').title()} ({status}) after {max_attempts} network attempts"
                    ) from e
                # Continue to next attempt
            except Exception as e:  # pragma: no cover - defensive guard
                error_type = type(e).__name__.replace("Error", "").lower()
                logger.error(
                    "Unexpected error connecting to {} - {}", endpoint, error_type
                )
                raise ConnectionError(f"Unexpected {error_type}") from e

        # Should never reach here
        raise ConnectionError("Network retries exhausted")

    def _extract_response_contents(self, response: Any) -> list[str]:
        try:
            choices: Iterable[Any] = response.choices
        except AttributeError as exc:  # pragma: no cover - defensive guard
            raise LLMEmptyResponseError("LLM response missing choices payload") from exc

        contents: list[str] = []
        for choice in choices:
            message = getattr(choice, "message", None)
            content = getattr(message, "content", None) if message is not None else None
            if content is None:
                raise LLMEmptyResponseError("LLM response missing content message")
            content_str = str(content).strip()
            if not content_str:
                raise LLMEmptyResponseError(
                    "LLM response contains empty content (check max_tokens setting)"
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
        try:
            usage_record = self._extract_usage(response)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.debug("Failed to extract usage data: {}", exc)
            usage_record = None

        if usage_record is None:
            self._last_usage = None
            return

        self._last_usage = usage_record
        self._total_usage.add(usage_record)

    @staticmethod
    def _extract_usage(response: Any) -> UsageRecord | None:
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
        except (TypeError, ValueError):  # pragma: no cover - safety net
            return 0

    @staticmethod
    def _safe_cost(response: Any) -> float:
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
        except (TypeError, ValueError):  # pragma: no cover - safety net
            return 0.0


@dataclass
class UsageRecord:
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
