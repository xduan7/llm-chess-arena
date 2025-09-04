from typing import Optional, Dict, Any, List
from llm_chess_arena.player.llm.llm_connector import LLMConnector


class MockLLMConnector(LLMConnector):
    """Mock LLM connector for testing without API calls."""

    def __init__(
        self,
        model: str = "mock-model",
        responses: Optional[List[str]] = None,
        raise_on_query: Optional[Exception] = None,
        **kwargs,
    ):
        """Initialize mock connector.

        Args:
            model: Model name for testing.
            responses: Predetermined responses to return in order.
            raise_on_query: Exception to raise on query (for error testing).
            **kwargs: Additional parameters captured but not used.
        """
        self.model = model
        self.responses = responses or []
        self.raise_on_query = raise_on_query
        self.query_count = 0
        self.query_history = []
        self.temperature = kwargs.get("temperature", 0.7)
        self.timeout = kwargs.get("timeout", 30.0)
        self.max_retries = kwargs.get("max_retries", 3)

    def query(
        self, prompt: str, system_prompt: Optional[str] = None, n: int = 1
    ) -> List[str]:
        """Return predetermined responses or extract move from prompt.

        Args:
            prompt: The prompt to send
            system_prompt: Optional system prompt
            n: Number of responses to generate (for voting)

        Returns:
            List of response strings
        """
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

        return responses

    def get_model_info(self) -> Dict[str, Any]:
        """Return mock model configuration."""
        return {
            "name": self.model,
            "provider": "Mock",
            "temperature": self.temperature,
            "timeout": self.timeout,
        }
