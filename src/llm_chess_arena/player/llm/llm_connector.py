"""Connector that wraps LiteLLM for consistent API usage."""

from __future__ import annotations

from typing import Any

from loguru import logger

import litellm
from litellm import exceptions as litellm_exceptions

# Cross-provider robustness: silently ignore unsupported params when switching between
# models (OpenAI, Anthropic, Gemini) rather than erroring. Research code needs flexibility.
litellm.drop_params = True
# Disable verbose logging across litellm versions (set_verbose availability varies)
if hasattr(litellm, "set_verbose"):
    setattr(litellm, "set_verbose", False)


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
            response = litellm.completion(**completion_kwargs)
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
