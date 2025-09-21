"""Connector that wraps LiteLLM for consistent API usage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger

import litellm
from litellm import exceptions as litellm_exceptions

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
    """Wrapper around LiteLLM for testing isolation and API stability.

    Provides a thin abstraction over LiteLLM to enable easy mocking in tests
    and protect against future LiteLLM API changes.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        """Initialize the LiteLLM connector.

        Args:
            model: Model identifier (e.g., "gpt-4", "claude-3-opus", "gemini-pro").
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative).
            max_tokens: Maximum tokens in response.
            timeout: Request timeout in seconds.
            max_retries: Retry attempts for transient errors.
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self._last_usage: UsageRecord | None = None
        self._total_usage = UsageRecord()

    def query(
        self,
        prompt: str,
        n: int = 1,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Send prompt to LLM and return n completions.

        Args:
            prompt: User message to send.
            n: Number of completions to request.
            system_prompt: Optional system-level instructions.
            **kwargs: Additional params passed to litellm.completion.

        Returns:
            list[str]: Completion strings in provider order.

        Raises:
            TimeoutError: Request exceeded timeout.
            ConnectionError: API call failed.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        logger.debug("Querying model {} with messages: {}", self.model, messages)

        try:
            completion_kwargs: dict[str, Any] = {
                "messages": messages,
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "timeout": self.timeout,
                "max_retries": self.max_retries,
                "n": n,
                **kwargs,
            }
            completion_kwargs = {
                key: value
                for key, value in completion_kwargs.items()
                if value is not None
            }
            response = litellm.completion(**completion_kwargs)
            self._capture_usage(response)
            contents: list[str] = []
            for choice in response.choices:
                content = getattr(choice.message, "content", None)
                if content is None:
                    raise ConnectionError("LLM response missing content message")
                contents.append(str(content))
            logger.debug("{} response choices: {}", self.model, contents)
            return contents

        except litellm_exceptions.Timeout as e:
            logger.warning("Request timed out after {}s: {}", self.timeout, e)
            raise TimeoutError(f"Request timed out after {self.timeout}s") from e

        except (
            litellm_exceptions.RateLimitError,
            litellm_exceptions.ServiceUnavailableError,
            litellm_exceptions.InternalServerError,
        ) as e:
            logger.warning("Transient API error (may retry at higher level): {}", e)
            raise ConnectionError(f"LLM API temporarily unavailable: {e}") from e

        except (
            litellm_exceptions.AuthenticationError,
            litellm_exceptions.InvalidRequestError,
            litellm_exceptions.BadRequestError,
            litellm_exceptions.ContentPolicyViolationError,
        ) as e:
            logger.error("Permanent API error (will not retry): {}", e)
            raise ConnectionError(f"LLM API request invalid: {e}") from e

        except (
            litellm_exceptions.APIError,
            litellm_exceptions.APIConnectionError,
        ) as e:
            logger.error("API error occurred: {}", e)
            raise ConnectionError(f"LLM API call failed: {e}") from e

        except Exception as e:
            logger.error("Unexpected error during LLM API call: {}", e)
            raise ConnectionError(f"Unexpected error: {e}") from e

    def get_last_usage(self) -> UsageRecord | None:
        """Return usage for the most recent LiteLLM completion.

        Returns:
            UsageRecord | None: Usage metrics for the latest call when available.
        """

        return self._last_usage.copy() if self._last_usage is not None else None

    def get_total_usage(self) -> UsageRecord:
        """Return accumulated usage across all completions for this connector.

        Returns:
            UsageRecord: Aggregated usage counters.
        """

        return self._total_usage.copy()

    def reset_usage(self) -> None:
        """Reset usage statistics accumulated by this connector."""

        self._last_usage = None
        self._total_usage = UsageRecord()

    def _capture_usage(self, response: Any) -> None:
        """Extract usage metadata from a LiteLLM response and store it.

        Args:
            response: Response object returned from ``litellm.completion``.
        """

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
        """Translate LiteLLM usage metadata into a ``UsageRecord`` instance.

        Args:
            response: Response object returned from ``litellm.completion``.

        Returns:
            UsageRecord | None: Parsed usage record if metadata is available.
        """

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
        """Extract integer values from dict-like or attribute-like payloads."""

        value: Any
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
        """Best-effort extraction of monetary cost from a LiteLLM response."""

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
    """Tracks LiteLLM token usage and cost for a single or aggregated call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0

    def add(self, other: UsageRecord) -> None:
        """Accumulate usage statistics from another record.

        Args:
            other: Additional usage to add to this record.
        """

        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens
        self.cost += other.cost

    def copy(self) -> UsageRecord:
        """Return a shallow copy of the record for safe external usage."""

        return UsageRecord(
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            total_tokens=self.total_tokens,
            cost=self.cost,
        )
