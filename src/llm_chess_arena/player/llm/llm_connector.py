"""Connector that wraps LiteLLM for consistent API usage."""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Any

from loguru import logger

import httpx
import litellm
from litellm import exceptions as litellm_exceptions

ARGO_PROVIDER = "argo"
ARGO_DEFAULT_API_BASE = "https://apps.inside.anl.gov/argoapi/api/v1/resource"

# Argo models that do not accept temperature/top_p parameters
ARGO_NO_TEMPERATURE_MODELS = {
    "gpto1preview",
    "gpto1mini",
    "gpto1",
    "o1-preview",
    "gpto3mini",
    "gpto3",
    "gpto4mini",
}


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
            provider: Optional provider name (e.g., ``"argo"``) for custom routing.
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.provider = provider.lower() if isinstance(provider, str) else None
        self.api_base = api_base.rstrip("/") if api_base else None

        # Usage bookkeeping
        self._last_usage: UsageRecord | None = None
        self._total_usage = UsageRecord()

        # Argo-specific state (populated when provider == "argo")
        self._argo_username: str | None = None
        self._argo_endpoint: str | None = None
        self._argo_model: str | None = None
        self._argo_full_model: str | None = None
        self._argo_skip_temperature = False

        if self.provider == ARGO_PROVIDER:
            self._configure_argo_provider()

    @staticmethod
    def _normalise_argo_model_name(model: str) -> str:
        """Normalise optional version-suffixed Argo model names."""

        version_match = re.match(r"^(?P<base>.+?)-\d{8}$", model)
        if version_match:
            return version_match.group("base")
        return model

    def _configure_argo_provider(self) -> None:
        username = os.getenv("ARGO_USERNAME")
        if not username:
            raise ValueError(
                "ARGO_USERNAME environment variable must be set when using the Argo provider"
            )

        cleaned_model = self._normalise_argo_model_name(self.model)
        self._argo_username = username
        self._argo_model = cleaned_model
        self._argo_full_model = cleaned_model

        base = (self.api_base or ARGO_DEFAULT_API_BASE).rstrip("/")
        self._argo_endpoint = f"{base}/chat/"
        self._argo_skip_temperature = (
            cleaned_model.lower() in ARGO_NO_TEMPERATURE_MODELS
        )

        # Keep self.model aligned with the identifier that will be sent to Argo.
        self.model = cleaned_model

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

        if self.provider == ARGO_PROVIDER:
            return self._query_via_argo(messages=messages, n=n, extra_kwargs=kwargs)

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

        except Exception as e:  # pragma: no cover - defensive guard
            logger.error("Unexpected error during LLM API call: {}", e)
            raise ConnectionError(f"Unexpected error: {e}") from e

    # --- Argo helpers -----------------------------------------------------------------

    def _query_via_argo(
        self,
        *,
        messages: list[dict[str, str]],
        n: int,
        extra_kwargs: dict[str, Any],
    ) -> list[str]:
        if not self._argo_endpoint or not self._argo_username or not self._argo_model:
            raise ConnectionError("Argo provider is not configured correctly")

        logger.debug(
            "Querying Argo model {} with {} message(s)",
            self._argo_full_model or self._argo_model,
            len(messages),
        )

        payload_base = self._build_argo_payload(messages, extra_kwargs)
        results: list[str] = []

        with httpx.Client(timeout=self.timeout) as client:
            for attempt in range(max(1, n)):
                payload = payload_base.copy()
                data = self._send_argo_request(client, payload)
                results.append(self._extract_argo_content(data))

        return results

    def _build_argo_payload(
        self,
        messages: list[dict[str, str]],
        extra_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "user": self._argo_username,
            "model": self._argo_model,
            "messages": [msg.copy() for msg in messages],
        }

        max_tokens = extra_kwargs.get("max_tokens", self.max_tokens)
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        if not self._argo_skip_temperature:
            temperature = extra_kwargs.get("temperature", self.temperature)
            if temperature is not None:
                payload["temperature"] = temperature
            top_p = extra_kwargs.get("top_p")
            if top_p is not None:
                payload["top_p"] = top_p

        for forbidden in ("temperature", "top_p", "n", "return_full_response"):
            extra_kwargs.pop(forbidden, None)

        for key, value in extra_kwargs.items():
            if key in payload:
                continue
            payload[key] = value

        return payload

    def _send_argo_request(
        self, client: httpx.Client, payload: dict[str, Any]
    ) -> dict[str, Any]:
        assert self._argo_endpoint is not None
        last_error: str | None = None

        for attempt in range(max(1, self.max_retries)):
            try:
                response = client.post(
                    self._argo_endpoint,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
                return response.json()  # type: ignore[no-any-return]
            except httpx.HTTPStatusError as exc:
                last_error = f"HTTP {exc.response.status_code}: {exc.response.text}"
            except Exception as exc:  # pragma: no cover - defensive guard
                last_error = str(exc)

            if attempt < self.max_retries - 1:
                sleep_seconds = min(2**attempt, 5)
                logger.debug(
                    "Retrying Argo request in {}s after error: {}",
                    sleep_seconds,
                    last_error,
                )
                time.sleep(sleep_seconds)

        raise ConnectionError(
            f"Argo request failed after {self.max_retries} attempts: {last_error}"
        )

    @staticmethod
    def _extract_argo_content(data: dict[str, Any]) -> str:
        if "choices" in data and data["choices"]:
            choice = data["choices"][0]
            message = choice.get("message", {}) if isinstance(choice, dict) else {}
            content = message.get("content")
            if content:
                return str(content)

        if "content" in data and data["content"]:
            return str(data["content"])

        if "response" in data and data["response"]:
            return str(data["response"])

        raise ConnectionError(f"Unexpected response format from Argo: {data}")

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
