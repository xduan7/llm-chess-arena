import os
import pytest
from unittest.mock import patch, Mock
import litellm

from llm_chess_arena.player.llm.llm_connector import LLMConnector
from llm_chess_arena.config import load_env

load_env()


class TestLLMConnectorWithMockResponse:
    def test_query_returns_mocked_llm_response_content(self):
        with patch("litellm.completion") as mock_completion:
            mock_completion.return_value = Mock(
                choices=[Mock(message=Mock(content="Final Answer: e4"))]
            )

            connector = LLMConnector(
                model="gpt-3.5-turbo",
                temperature=0.7,
                max_tokens=150,
            )

            response = connector.query("What's your move?")

            assert response == ["Final Answer: e4"]
            mock_completion.assert_called_once()
            call_args = mock_completion.call_args

            assert call_args.kwargs["model"] == "gpt-3.5-turbo"
            assert call_args.kwargs["messages"] == [
                {"role": "user", "content": "What's your move?"}
            ]
            assert call_args.kwargs["temperature"] == 0.7
            assert call_args.kwargs["max_tokens"] == 150

    def test_query_includes_system_prompt_in_message_list(self):
        with patch("litellm.completion") as mock_completion:
            mock_completion.return_value = Mock(
                choices=[Mock(message=Mock(content="Response"))]
            )

            connector = LLMConnector(model="claude-3-haiku")

            connector.query("User prompt", system_prompt="You are a chess expert")

            call_args = mock_completion.call_args
            messages = call_args.kwargs["messages"]

            assert len(messages) == 2
            assert messages[0] == {
                "role": "system",
                "content": "You are a chess expert",
            }
            assert messages[1] == {"role": "user", "content": "User prompt"}

    def test_query_converts_litellm_timeout_to_standard_timeout_error(self):
        with patch("litellm.completion") as mock_completion:
            mock_completion.side_effect = litellm.Timeout(
                message="Request timed out", model="gpt-4", llm_provider="openai"
            )

            connector = LLMConnector(
                model="gpt-4",
                timeout=5.0,
                max_retries=1,
            )

            with pytest.raises(TimeoutError, match="Request timed out after 5.0s"):
                connector.query("Test prompt")

    def test_query_wraps_unexpected_exceptions_as_connection_error(self):
        with patch("litellm.completion") as mock_completion:
            mock_completion.side_effect = Exception("API error")

            connector = LLMConnector(
                model="gpt-3.5-turbo",
                max_retries=1,
            )

            with pytest.raises(ConnectionError, match="Unexpected error"):
                connector.query("Test prompt")


class TestLLMConnectorRetryLogic:
    def test_query_passes_retry_configuration_to_litellm(self):
        with patch("litellm.completion") as mock_completion:
            mock_completion.return_value = Mock(
                choices=[Mock(message=Mock(content="Success"))]
            )

            connector = LLMConnector(
                model="gpt-3.5-turbo",
                max_retries=5,
            )

            response = connector.query("Test")

            assert response == ["Success"]
            assert mock_completion.call_count >= 1

    def test_query_raises_error_after_all_retry_attempts_exhausted(self):
        with patch("litellm.completion") as mock_completion:
            mock_completion.side_effect = Exception("Persistent error")

            connector = LLMConnector(
                model="gpt-3.5-turbo",
                max_retries=2,
            )

            with pytest.raises(ConnectionError, match="Unexpected error"):
                connector.query("Test")

            assert mock_completion.call_count >= 1


class TestLLMConnectorConfiguration:
    def test_global_litellm_settings_are_configured_correctly(self):
        assert litellm.drop_params
        assert not litellm.set_verbose

    @patch("litellm.completion")
    def test_all_initialization_parameters_forwarded_to_litellm_completion(
        self, mock_completion
    ):
        mock_completion.return_value = Mock(
            choices=[Mock(message=Mock(content="Test"))]
        )

        connector = LLMConnector(
            model="gpt-4",
            temperature=0.5,
            max_tokens=100,
            timeout=20.0,
            max_retries=5,
        )

        connector.query("Test prompt")

        call_kwargs = mock_completion.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4"
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 100
        assert call_kwargs["timeout"] == 20.0


@pytest.mark.live
class TestLLMConnectorRealAPI:
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not set"
    )
    def test_openai_api_connection_returns_valid_response(self):
        openai_connector = LLMConnector(
            model="gpt-3.5-turbo",
            temperature=0.0,
            max_tokens=10,
            timeout=10.0,
        )

        llm_response = openai_connector.query(
            "Say 'connection successful' in 3 words or less"
        )

        assert llm_response is not None
        assert len(llm_response) > 0
        assert isinstance(llm_response, list)
        assert isinstance(llm_response[0], str)
        print(f"OpenAI response: {llm_response[0]}")

    @pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY"), reason="Anthropic API key not set"
    )
    def test_anthropic_api_connection_returns_valid_response(self):
        anthropic_connector = LLMConnector(
            model="claude-3-haiku-20240307",
            temperature=0.0,
            max_tokens=10,
            timeout=10.0,
        )

        llm_response = anthropic_connector.query(
            "Say 'connection successful' in 3 words or less"
        )

        assert llm_response is not None
        assert len(llm_response) > 0
        assert isinstance(llm_response, list)
        assert isinstance(llm_response[0], str)
        print(f"Anthropic response: {llm_response[0]}")

    @pytest.mark.skipif(
        not os.getenv("GOOGLE_API_KEY"), reason="Google API key not set"
    )
    def test_google_gemini_api_connection_returns_valid_response(self):
        gemini_connector = LLMConnector(
            model="gemini/gemini-2.0-flash-exp",
            temperature=0.0,
            max_tokens=10,
            timeout=10.0,
        )

        llm_response = gemini_connector.query(
            "Say 'connection successful' in 3 words or less"
        )

        assert llm_response is not None
        assert len(llm_response) > 0
        assert isinstance(llm_response, list)
        assert isinstance(llm_response[0], str)
        print(f"Google response: {llm_response[0]}")

    def test_llm_generates_valid_chess_opening_move(self):
        if os.getenv("OPENAI_API_KEY"):
            selected_model = "gpt-3.5-turbo"
        elif os.getenv("ANTHROPIC_API_KEY"):
            selected_model = "claude-3-haiku-20240307"
        elif os.getenv("GOOGLE_API_KEY"):
            selected_model = "gemini/gemini-2.0-flash-exp"
        else:
            pytest.skip("No API keys available")

        chess_llm_connector = LLMConnector(
            model=selected_model,
            temperature=0.0,
            max_tokens=20,
            timeout=10.0,
        )

        chess_move_prompt = "You are playing chess. The board is at the starting position. What is a good opening move? Reply with just the move in standard chess notation (e.g., 'e4')."
        llm_response = chess_llm_connector.query(chess_move_prompt)

        assert llm_response is not None
        assert len(llm_response) > 0
        assert isinstance(llm_response, list)

        common_chess_openings = ["e4", "d4", "Nf3", "c4", "e3", "d3"]
        response_contains_valid_opening = any(
            opening in llm_response[0] for opening in common_chess_openings
        )
        assert (
            response_contains_valid_opening
        ), f"Response doesn't contain a chess move: {llm_response[0]}"
        print(f"Chess move from {selected_model}: {llm_response[0]}")

    def test_system_prompt_influences_llm_response(self):
        if os.getenv("OPENAI_API_KEY"):
            selected_model = "gpt-3.5-turbo"
        elif os.getenv("ANTHROPIC_API_KEY"):
            selected_model = "claude-3-haiku-20240307"
        elif os.getenv("GOOGLE_API_KEY"):
            selected_model = "gemini/gemini-2.0-flash-exp"
        else:
            pytest.skip("No API keys available")

        chess_themed_connector = LLMConnector(
            model=selected_model,
            temperature=0.0,
            max_tokens=10,
            timeout=10.0,
        )

        llm_response_with_system_context = chess_themed_connector.query(
            "What are you?",
            system_prompt="You are a chess grandmaster. Always mention chess in your response.",
        )

        assert llm_response_with_system_context is not None
        assert len(llm_response_with_system_context) > 0
        assert isinstance(llm_response_with_system_context, list)

        response_mentions_chess = "chess" in llm_response_with_system_context[0].lower()
        assert (
            response_mentions_chess
        ), f"Response doesn't mention chess: {llm_response_with_system_context[0]}"
        print(
            f"System prompt response from {selected_model}: {llm_response_with_system_context[0]}"
        )
