"""Unit tests for the LiteLLM connector abstraction."""

import os
from unittest.mock import Mock, patch

import httpx
import litellm
import pytest

from llm_chess_arena.player.llm.llm_connector import (
    ARGO_DEFAULT_API_BASE,
    ARGO_PROVIDER,
    LLMConnector,
)
from llm_chess_arena.config import load_env

load_env()


class TestLLMConnectorWithMockResponse:
    """Unit tests that mock LiteLLM responses for deterministic behavior."""

    def test_query_returns_mocked_llm_response_content(self):
        """Connector should return content from mocked LiteLLM responses."""
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
        """System prompts must appear before user prompts."""
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
        """LiteLLM timeouts should map to TimeoutError for callers."""
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
        """Unexpected LiteLLM errors should raise ConnectionError."""
        with patch("litellm.completion") as mock_completion:
            mock_completion.side_effect = Exception("API error")

            connector = LLMConnector(
                model="gpt-3.5-turbo",
                max_retries=1,
            )

            with pytest.raises(ConnectionError, match="Unexpected error"):
                connector.query("Test prompt")


class TestLLMConnectorRetryLogic:
    """Retry configuration propagation and exhaustion handling."""

    def test_query_passes_retry_configuration_to_litellm(self):
        """Connector must forward retry settings to LiteLLM."""
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
        """Connector should raise once all retry attempts fail."""
        with patch("litellm.completion") as mock_completion:
            mock_completion.side_effect = Exception("Persistent error")

            connector = LLMConnector(
                model="gpt-3.5-turbo",
                max_retries=2,
            )

            with pytest.raises(ConnectionError, match="Unexpected error"):
                connector.query("Test")

            assert mock_completion.call_count >= 1

    def test_query_tracks_usage_statistics(self):
        """Connector should capture prompt/completion tokens and cost per call."""
        with (
            patch("litellm.completion") as mock_completion,
            patch("litellm.completion_cost") as mock_completion_cost,
        ):
            first_response = Mock()
            first_response.choices = [Mock(message=Mock(content="First"))]
            first_response.usage = {
                "prompt_tokens": 12,
                "completion_tokens": 34,
                "total_tokens": 46,
            }

            second_response = Mock()
            second_response.choices = [Mock(message=Mock(content="Second"))]
            second_response.usage = Mock(
                prompt_tokens=20,
                completion_tokens=10,
                total_tokens=30,
            )

            mock_completion.side_effect = [first_response, second_response]
            mock_completion_cost.side_effect = [0.0123, 0.0456]

            connector = LLMConnector(model="gpt-4o")

            connector.query("Test prompt")
            first_usage = connector.get_last_usage()

            assert first_usage is not None
            assert first_usage.prompt_tokens == 12
            assert first_usage.completion_tokens == 34
            assert first_usage.total_tokens == 46
            assert first_usage.cost == pytest.approx(0.0123)

            connector.query("Second prompt")
            second_usage = connector.get_last_usage()
            assert second_usage is not None
            assert second_usage.prompt_tokens == 20
            assert second_usage.completion_tokens == 10
            assert second_usage.total_tokens == 30
            assert second_usage.cost == pytest.approx(0.0456)

        totals = connector.get_total_usage()
        assert totals.prompt_tokens == 32
        assert totals.completion_tokens == 44
        assert totals.total_tokens == 76
        assert totals.cost == pytest.approx(0.0579)

    def test_reset_usage_clears_accumulated_totals(self):
        """Connector reset should clear last and total usage metrics."""
        with (
            patch("litellm.completion") as mock_completion,
            patch("litellm.completion_cost") as mock_completion_cost,
        ):
            response = Mock()
            response.choices = [Mock(message=Mock(content="Final"))]
            response.usage = {
                "prompt_tokens": 5,
                "completion_tokens": 7,
                "total_tokens": 12,
            }

            mock_completion.return_value = response
            mock_completion_cost.return_value = 0.01

            connector = LLMConnector(model="gpt-4")

            connector.query("Prompt")
            assert connector.get_last_usage() is not None

            connector.reset_usage()

            assert connector.get_last_usage() is None
            totals = connector.get_total_usage()
            assert totals.prompt_tokens == 0
            assert totals.completion_tokens == 0
            assert totals.total_tokens == 0
            assert totals.cost == pytest.approx(0.0)


class TestLLMConnectorArgoProvider:
    """Argo-specific provider integration tests."""

    @pytest.mark.parametrize(
        "provided,expected",
        [
            ("claudeopus4-20240229", "claudeopus4"),
            ("gpto4mini-20240101", "gpto4mini"),
            ("gemini25pro", "gemini25pro"),
        ],
    )
    def test_normalise_argo_model_name_strips_version_suffix(self, provided, expected):
        assert LLMConnector._normalise_argo_model_name(provided) == expected

    def test_argo_provider_requires_username_environment(self, monkeypatch):
        monkeypatch.delenv("ARGO_USERNAME", raising=False)

        with pytest.raises(
            ValueError, match="ARGO_USERNAME environment variable must be set"
        ):
            LLMConnector(model="gpt4", provider=ARGO_PROVIDER)

    @patch("httpx.Client")
    def test_argo_provider_issues_post_request_with_expected_payload(
        self, mock_client_cls, monkeypatch
    ):
        monkeypatch.setenv("ARGO_USERNAME", "chesstester")
        mock_client = mock_client_cls.return_value.__enter__.return_value
        mock_response = Mock(status_code=200)
        mock_response.json.return_value = {"response": "Move"}
        mock_response.raise_for_status.return_value = None
        mock_client.post.return_value = mock_response

        connector = LLMConnector(model="gpt4", provider=ARGO_PROVIDER)
        result = connector.query("Play a move")

        assert result == ["Move"]
        mock_client.post.assert_called_once()
        url_arg = mock_client.post.call_args.args[0]
        payload = mock_client.post.call_args.kwargs["json"]
        assert url_arg == f"{ARGO_DEFAULT_API_BASE}/chat/"
        assert payload["user"] == "chesstester"
        assert payload["model"] == "gpt4"
        assert payload["messages"][0]["content"].startswith("Play a move")

    @patch("httpx.Client")
    def test_argo_provider_trims_temperature_for_restricted_models(
        self, mock_client_cls, monkeypatch
    ):
        monkeypatch.setenv("ARGO_USERNAME", "chesstester")
        mock_client = mock_client_cls.return_value.__enter__.return_value
        mock_response = Mock(status_code=200)
        mock_response.json.return_value = {"response": "Move"}
        mock_response.raise_for_status.return_value = None
        mock_client.post.return_value = mock_response

        connector = LLMConnector(
            model="o1-preview",
            temperature=0.3,
            provider=ARGO_PROVIDER,
        )
        connector.query("Play a move", top_p=0.7, n=2)

        payload = mock_client.post.call_args.kwargs["json"]
        assert "temperature" not in payload
        assert "top_p" not in payload
        assert mock_client.post.call_count == 2

    @patch("httpx.Client")
    def test_argo_provider_respects_custom_api_base(self, mock_client_cls, monkeypatch):
        monkeypatch.setenv("ARGO_USERNAME", "chesstester")
        mock_client = mock_client_cls.return_value.__enter__.return_value
        mock_response = Mock(status_code=200)
        mock_response.json.return_value = {"response": "Move"}
        mock_response.raise_for_status.return_value = None
        mock_client.post.return_value = mock_response

        connector = LLMConnector(
            model="gemini25pro",
            provider=ARGO_PROVIDER,
            api_base="https://argo.example.com/api",
        )
        connector.query("Play a move")

        url_arg = mock_client.post.call_args.args[0]
        assert url_arg == "https://argo.example.com/api/chat/"
        payload = mock_client.post.call_args.kwargs["json"]
        assert payload["model"] == "gemini25pro"

    @patch("httpx.Client")
    def test_argo_provider_uses_default_api_base_when_none_provided(
        self, mock_client_cls, monkeypatch
    ):
        monkeypatch.setenv("ARGO_USERNAME", "chesstester")
        mock_client = mock_client_cls.return_value.__enter__.return_value
        mock_response = Mock(status_code=200)
        mock_response.json.return_value = {"response": "Move"}
        mock_response.raise_for_status.return_value = None
        mock_client.post.return_value = mock_response

        connector = LLMConnector(model="gpt4", provider=ARGO_PROVIDER)
        connector.query("Play a move")

        url_arg = mock_client.post.call_args.args[0]
        assert url_arg == f"{ARGO_DEFAULT_API_BASE}/chat/"

    @pytest.mark.parametrize(
        "model,api_base",
        [
            ("gpt4", None),
            ("gpto4mini", None),
            ("claudeopus4", "https://apps-dev.inside.anl.gov/argoapi/api/v1/resource"),
            ("gemini25pro", "https://apps-dev.inside.anl.gov/argoapi/api/v1/resource"),
        ],
    )
    @patch("httpx.Client")
    def test_argo_connector_sends_expected_model_id(
        self, mock_client_cls, monkeypatch, model, api_base
    ):
        monkeypatch.setenv("ARGO_USERNAME", "chesstester")
        mock_client = mock_client_cls.return_value.__enter__.return_value
        mock_response = Mock(status_code=200)
        mock_response.json.return_value = {"response": "Move"}
        mock_response.raise_for_status.return_value = None
        mock_client.post.return_value = mock_response

        connector_kwargs = {
            "model": model,
            "provider": ARGO_PROVIDER,
            "temperature": 0.5,
        }
        if api_base is not None:
            connector_kwargs["api_base"] = api_base

        connector = LLMConnector(**connector_kwargs)
        connector.query("Play a move")

        payload = mock_client.post.call_args.kwargs["json"]
        assert payload["model"] == model

    @patch("httpx.Client")
    @patch("time.sleep", return_value=None)
    def test_argo_provider_retries_on_http_error(
        self, mock_sleep, mock_client_cls, monkeypatch
    ):
        monkeypatch.setenv("ARGO_USERNAME", "chesstester")
        mock_client = mock_client_cls.return_value.__enter__.return_value
        dummy_request = httpx.Request("POST", "https://argo.example.com/chat/")
        dummy_response = httpx.Response(404, request=dummy_request)
        http_error = httpx.HTTPStatusError(
            "not found", request=dummy_request, response=dummy_response
        )

        success_response = Mock()
        success_response.json.return_value = {"response": "Move"}
        success_response.raise_for_status.return_value = None

        mock_client.post.side_effect = [http_error, http_error, success_response]

        connector = LLMConnector(model="gpt-4", provider=ARGO_PROVIDER, max_retries=3)
        result = connector.query("Play a move")

        assert result == ["Move"]
        assert mock_client.post.call_count == 3


class TestLLMConnectorConfiguration:
    """Configuration-level tests for the connector."""

    def test_global_litellm_settings_are_configured_correctly(self):
        """Global LiteLLM tweaks should match expected defaults."""
        assert litellm.drop_params
        assert getattr(litellm, "verbose", False) is False

    @patch("litellm.completion")
    def test_all_initialization_parameters_forwarded_to_litellm_completion(
        self, mock_completion
    ):
        """Initialization parameters must be passed through to LiteLLM."""
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
    """Live API smoke tests guarded by API keys."""

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not set"
    )
    def test_openai_api_connection_returns_valid_response(self):
        """Verify OpenAI connectivity when credentials are available."""
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

    @pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY"), reason="Anthropic API key not set"
    )
    def test_anthropic_api_connection_returns_valid_response(self):
        """Verify Anthropic connectivity when credentials are available."""
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

    @pytest.mark.skipif(
        not os.getenv("GOOGLE_API_KEY"), reason="Google API key not set"
    )
    def test_google_gemini_api_connection_returns_valid_response(self):
        """Verify Gemini connectivity when credentials are available."""
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
