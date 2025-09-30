"""Test double for the LLM connector to avoid network access."""

from __future__ import annotations

from typing import Any, Iterable

from llm_chess_arena.player.llm.connector import LLMConnector, UsageRecord


class MockLLMConnector(LLMConnector):
    """Mock LLM connector for testing without API calls."""

    def __init__(
        self,
        model: str = "mock-model",
        responses: list[str] | None = None,
        raise_on_query: Exception | None = None,
        **kwargs,
    ):
        """Set up canned responses, optional failure triggers, and usage data."""
        temperature = kwargs.pop("temperature", 0.7)
        max_num_tokens = kwargs.pop("max_num_tokens", None)
        request_timeout_in_seconds = kwargs.pop("request_timeout_in_seconds", 30.0)
        max_api_request_retries = kwargs.pop("max_api_request_retries", 3)
        usage_records = kwargs.pop("usage_records", None)
        super().__init__(
            model=model,
            temperature=temperature,
            max_num_tokens=max_num_tokens,
            request_timeout_in_seconds=request_timeout_in_seconds,
            max_api_request_retries=max_api_request_retries,
        )

        self.responses = responses or []
        self.raise_on_query = raise_on_query
        self.query_count = 0
        self.query_history = []
        self._usage_records = self._normalize_usage_records(usage_records)

    def query(
        self,
        prompt: str,
        n: int = 1,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Return predetermined responses or extract move from prompt."""
        self.query_count += 1  # Count each query call, not each response
        self.query_history.append(
            {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "n": n,
            }
        )

        if self.raise_on_query:
            raise self.raise_on_query

        responses = []
        for i in range(n):
            if self.responses:
                response = self.responses.pop(0)  # Take from front of list
                responses.append(response)
            else:
                # Extract first legal move from prompt for flexibility
                if "Legal moves:" in prompt:
                    lines = prompt.split("\n")
                    for line in lines:
                        if line.startswith("Legal moves:"):
                            moves = line.replace("Legal moves:", "").strip()
                            if moves:
                                first_move = moves.split(",")[0].strip()
                                responses.append(f"Final Answer: {first_move}")
                                break
                    else:
                        responses.append("Final Answer: e4")  # Default opening move
                else:
                    responses.append("Final Answer: e4")  # Default opening move

        self._assign_usage()
        return responses

    def get_model_info(self) -> dict[str, Any]:
        """Return mock model configuration."""
        return {
            "name": self.model,
            "provider": "Mock",
            "temperature": self.temperature,
            "timeout": self.timeout,
        }

    def reset_usage(self) -> None:
        """Reset accumulated usage statistics for the mock connector."""

        self._last_usage = None
        self._total_usage = UsageRecord()

    def _assign_usage(self) -> None:
        """Assign mock usage metrics to mimic LiteLLM accounting."""

        if not self._usage_records:
            self._last_usage = None
            return

        usage = self._usage_records.pop(0)
        self._last_usage = usage.copy()
        self._total_usage.add(usage)

    @staticmethod
    def _normalize_usage_records(
        usage_records: Iterable[UsageRecord | dict[str, Any]] | None,
    ) -> list[UsageRecord]:
        """Normalize configurable usage records into dataclass instances."""

        if usage_records is None:
            return []

        normalized: list[UsageRecord] = []
        for record in usage_records:
            if isinstance(record, UsageRecord):
                normalized.append(record)
                continue

            if isinstance(record, dict):
                normalized.append(
                    UsageRecord(
                        prompt_tokens=record.get("prompt_tokens", 0),
                        completion_tokens=record.get("completion_tokens", 0),
                        total_tokens=record.get("total_tokens", 0),
                        cost=record.get("cost", 0.0),
                    )
                )
                continue

            raise TypeError("usage_records must contain UsageRecord or mapping entries")

        return normalized
