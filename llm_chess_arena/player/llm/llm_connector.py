from typing import Optional
from loguru import logger

import litellm

# Cross-provider robustness: silently ignore unsupported params when switching between
# models (OpenAI, Anthropic, Gemini) rather than erroring. Research code needs flexibility.
litellm.drop_params = True
litellm.set_verbose = False


class LLMConnector:
    """Wrapper around LiteLLM for testing isolation and API stability.

    Provides a thin abstraction over LiteLLM to enable easy mocking in tests
    and protect against future LiteLLM API changes.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        """Initialize LLM connector.

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
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> list[str]:
        """Send prompt to LLM and return n completions.

        Args:
            prompt: User message to send.
            n: Number of completions to request.
            system_prompt: Optional system-level instructions.
            **kwargs: Additional params passed to litellm.completion.

        Returns:
            List of completion strings in provider order.

        Raises:
            TimeoutError: Request exceeded timeout.
            ConnectionError: API call failed.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        logger.debug(f"Querying to {self.model} with the messages: {messages}")

        try:
            completion_kwarg = {
                "messages": messages,
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "timeout": self.timeout,
                "max_retries": self.max_retries,
                "n": n,
                **kwargs,
            }
            response = litellm.completion(**completion_kwarg)
            contents = [choice.message.content for choice in response.choices]
            logger.debug(f"{self.model} response choices: {contents}")
            return contents

        except litellm.Timeout as e:
            logger.warning(f"Request timed out after {self.timeout}s: {e}")
            raise TimeoutError(f"Request timed out after {self.timeout}s") from e

        except (
            litellm.RateLimitError,
            litellm.ServiceUnavailableError,
            litellm.InternalServerError,
        ) as e:
            logger.warning(f"Transient API error (may retry at higher level): {e}")
            raise ConnectionError(f"LLM API temporarily unavailable: {e}") from e

        except (
            litellm.AuthenticationError,
            litellm.InvalidRequestError,
            litellm.BadRequestError,
            litellm.ContentPolicyViolationError,
        ) as e:
            logger.error(f"Permanent API error (will not retry): {e}")
            raise ConnectionError(f"LLM API request invalid: {e}") from e

        except (litellm.APIError, litellm.APIConnectionError) as e:
            logger.error(f"API error occurred: {e}")
            raise ConnectionError(f"LLM API call failed: {e}") from e

        except Exception as e:
            logger.error(f"Unexpected error during LLM API call: {e}")
            raise ConnectionError(f"Unexpected error: {e}") from e
