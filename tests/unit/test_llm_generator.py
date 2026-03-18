"""Unit tests for lambdas.shared.llm_generator.LLMGenerator.

Uses ``unittest.mock`` to mock the Bedrock runtime client so that tests
run without any AWS credentials or network access.
"""

from __future__ import annotations

import io
import json
from unittest.mock import MagicMock, patch

import pytest

from lambdas.shared.constants import LLM_FALLBACK_MODEL_ID, LLM_PRIMARY_MODEL_ID
from lambdas.shared.llm_generator import (
    INITIAL_BACKOFF_S,
    MAX_RETRIES,
    RAG_PROMPT_TEMPLATE,
    SYSTEM_PROMPT,
    LLMGenerator,
    _is_retryable,
)
from lambdas.shared.models import GenerationConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bedrock_response(text: str) -> dict:
    """Build a mock Bedrock invoke_model response with the given text."""
    body = json.dumps({
        "content": [{"type": "text", "text": text}],
    }).encode()
    return {"body": io.BytesIO(body)}


def _stream_events(tokens: list[str]) -> list[dict]:
    """Build a list of mock streaming events from token strings."""
    events = []
    for token in tokens:
        payload = json.dumps({
            "type": "content_block_delta",
            "delta": {"text": token},
        }).encode()
        events.append({"chunk": {"bytes": payload}})
    return events


def _throttling_exception() -> Exception:
    """Create a mock Bedrock throttling exception."""
    exc = Exception("ThrottlingException: Rate exceeded")
    exc.response = {"Error": {"Code": "ThrottlingException"}}  # type: ignore[attr-defined]
    return exc


def _non_retryable_exception() -> Exception:
    """Create a non-retryable exception."""
    exc = Exception("ValidationException: Bad input")
    exc.response = {"Error": {"Code": "ValidationException"}}  # type: ignore[attr-defined]
    return exc


def _make_generator(mock_client: MagicMock | None = None) -> LLMGenerator:
    """Create an LLMGenerator with a mock Bedrock client."""
    client = mock_client or MagicMock()
    return LLMGenerator(bedrock_client=client)


# ===================================================================
# _is_retryable helper
# ===================================================================

class TestIsRetryable:
    def test_throttling_exception_is_retryable(self):
        assert _is_retryable(_throttling_exception()) is True

    def test_validation_exception_is_not_retryable(self):
        assert _is_retryable(_non_retryable_exception()) is False

    def test_generic_exception_not_retryable(self):
        assert _is_retryable(Exception("something else")) is False

    def test_throttling_in_message_is_retryable(self):
        exc = Exception("Request was Throttling limited")
        assert _is_retryable(exc) is True

    def test_service_unavailable_is_retryable(self):
        exc = Exception("ServiceUnavailableException")
        exc.response = {"Error": {"Code": "ServiceUnavailableException"}}  # type: ignore[attr-defined]
        assert _is_retryable(exc) is True


# ===================================================================
# Prompt templates
# ===================================================================

class TestPromptTemplates:
    def test_system_prompt_contains_guidelines(self):
        assert "ONLY" in SYSTEM_PROMPT
        assert "cite sources" in SYSTEM_PROMPT.lower()
        assert "fabricate" in SYSTEM_PROMPT.lower()
        assert "don't have enough information" in SYSTEM_PROMPT

    def test_rag_prompt_template_has_placeholders(self):
        assert "{context}" in RAG_PROMPT_TEMPLATE
        assert "{query}" in RAG_PROMPT_TEMPLATE

    def test_rag_prompt_template_renders(self):
        rendered = RAG_PROMPT_TEMPLATE.format(context="some context", query="what is X?")
        assert "some context" in rendered
        assert "what is X?" in rendered
        assert "Answer:" in rendered


# ===================================================================
# generate() — non-streaming
# ===================================================================

class TestGenerate:
    def test_generate_returns_text(self):
        client = MagicMock()
        client.invoke_model.return_value = _bedrock_response("Hello world")
        gen = _make_generator(client)

        result = gen.generate("ctx", "question?")

        assert result == "Hello world"
        client.invoke_model.assert_called_once()

    def test_generate_uses_config_params(self):
        client = MagicMock()
        client.invoke_model.return_value = _bedrock_response("ok")
        gen = _make_generator(client)

        config = GenerationConfig(
            model_id="custom-model",
            temperature=0.3,
            top_p=0.8,
            max_tokens=2048,
        )
        gen.generate("ctx", "q", config)

        call_kwargs = client.invoke_model.call_args
        body = json.loads(call_kwargs.kwargs["body"])
        assert body["temperature"] == 0.3
        assert body["top_p"] == 0.8
        assert body["max_tokens"] == 2048
        assert call_kwargs.kwargs["modelId"] == "custom-model"

    def test_generate_includes_system_prompt(self):
        client = MagicMock()
        client.invoke_model.return_value = _bedrock_response("answer")
        gen = _make_generator(client)

        gen.generate("ctx", "q")

        body = json.loads(client.invoke_model.call_args.kwargs["body"])
        assert body["system"] == SYSTEM_PROMPT

    def test_generate_formats_rag_prompt(self):
        client = MagicMock()
        client.invoke_model.return_value = _bedrock_response("answer")
        gen = _make_generator(client)

        gen.generate("my context", "my question")

        body = json.loads(client.invoke_model.call_args.kwargs["body"])
        user_msg = body["messages"][0]["content"]
        assert "my context" in user_msg
        assert "my question" in user_msg

    def test_generate_default_config(self):
        client = MagicMock()
        client.invoke_model.return_value = _bedrock_response("ok")
        gen = _make_generator(client)

        gen.generate("ctx", "q")

        call_kwargs = client.invoke_model.call_args
        assert call_kwargs.kwargs["modelId"] == LLM_PRIMARY_MODEL_ID
        body = json.loads(call_kwargs.kwargs["body"])
        assert body["temperature"] == 0.7
        assert body["top_p"] == 0.9
        assert body["max_tokens"] == 4096

    @patch("lambdas.shared.llm_generator.time.sleep")
    def test_generate_retries_on_throttling(self, mock_sleep):
        client = MagicMock()
        client.invoke_model.side_effect = [
            _throttling_exception(),
            _bedrock_response("recovered"),
        ]
        gen = _make_generator(client)

        result = gen.generate("ctx", "q")

        assert result == "recovered"
        assert client.invoke_model.call_count == 2
        # Backoff is INITIAL_BACKOFF_S * (2^0) + jitter in [0, 1)
        actual_wait = mock_sleep.call_args_list[0].args[0]
        assert INITIAL_BACKOFF_S <= actual_wait < INITIAL_BACKOFF_S + 1.0

    @patch("lambdas.shared.llm_generator.time.sleep")
    def test_generate_falls_back_to_haiku(self, mock_sleep):
        client = MagicMock()
        # Primary model fails all retries, fallback succeeds
        client.invoke_model.side_effect = [
            _throttling_exception(),
            _throttling_exception(),
            _throttling_exception(),
            _bedrock_response("haiku answer"),
        ]
        gen = _make_generator(client)

        result = gen.generate("ctx", "q")

        assert result == "haiku answer"
        # Last call should use fallback model
        last_call = client.invoke_model.call_args
        assert last_call.kwargs["modelId"] == LLM_FALLBACK_MODEL_ID

    def test_generate_raises_on_non_retryable_error(self):
        client = MagicMock()
        client.invoke_model.side_effect = _non_retryable_exception()
        gen = _make_generator(client)

        with pytest.raises(Exception, match="ValidationException"):
            gen.generate("ctx", "q")

    @patch("lambdas.shared.llm_generator.time.sleep")
    def test_generate_exponential_backoff_timing(self, mock_sleep):
        client = MagicMock()
        client.invoke_model.side_effect = [
            _throttling_exception(),
            _throttling_exception(),
            _bedrock_response("ok"),
        ]
        gen = _make_generator(client)

        gen.generate("ctx", "q")

        calls = [c.args[0] for c in mock_sleep.call_args_list]
        # Attempt 0: base * 2^0 + jitter => [1.0, 2.0)
        assert INITIAL_BACKOFF_S <= calls[0] < INITIAL_BACKOFF_S + 1.0
        # Attempt 1: base * 2^1 + jitter => [2.0, 3.0)
        assert INITIAL_BACKOFF_S * 2 <= calls[1] < INITIAL_BACKOFF_S * 2 + 1.0


# ===================================================================
# generate_stream() — streaming
# ===================================================================

class TestGenerateStream:
    def test_stream_yields_tokens(self):
        client = MagicMock()
        tokens = ["Hello", " ", "world"]
        client.invoke_model_with_response_stream.return_value = {
            "body": _stream_events(tokens),
        }
        gen = _make_generator(client)

        result = list(gen.generate_stream("ctx", "q"))

        assert result == tokens

    def test_stream_uses_config_params(self):
        client = MagicMock()
        client.invoke_model_with_response_stream.return_value = {
            "body": _stream_events(["ok"]),
        }
        gen = _make_generator(client)

        config = GenerationConfig(
            model_id="custom-model",
            temperature=0.5,
            top_p=0.85,
            max_tokens=1024,
        )
        list(gen.generate_stream("ctx", "q", config))

        call_kwargs = client.invoke_model_with_response_stream.call_args
        body = json.loads(call_kwargs.kwargs["body"])
        assert body["temperature"] == 0.5
        assert body["top_p"] == 0.85
        assert body["max_tokens"] == 1024
        assert call_kwargs.kwargs["modelId"] == "custom-model"

    @patch("lambdas.shared.llm_generator.time.sleep")
    def test_stream_retries_on_throttling(self, mock_sleep):
        client = MagicMock()
        client.invoke_model_with_response_stream.side_effect = [
            _throttling_exception(),
            {"body": _stream_events(["recovered"])},
        ]
        gen = _make_generator(client)

        result = list(gen.generate_stream("ctx", "q"))

        assert result == ["recovered"]
        assert client.invoke_model_with_response_stream.call_count == 2

    @patch("lambdas.shared.llm_generator.time.sleep")
    def test_stream_falls_back_to_haiku(self, mock_sleep):
        client = MagicMock()
        client.invoke_model_with_response_stream.side_effect = [
            _throttling_exception(),
            _throttling_exception(),
            _throttling_exception(),
            {"body": _stream_events(["haiku"])},
        ]
        gen = _make_generator(client)

        result = list(gen.generate_stream("ctx", "q"))

        assert result == ["haiku"]
        last_call = client.invoke_model_with_response_stream.call_args
        assert last_call.kwargs["modelId"] == LLM_FALLBACK_MODEL_ID

    def test_stream_raises_on_non_retryable_error(self):
        client = MagicMock()
        client.invoke_model_with_response_stream.side_effect = _non_retryable_exception()
        gen = _make_generator(client)

        with pytest.raises(Exception, match="ValidationException"):
            list(gen.generate_stream("ctx", "q"))

    def test_stream_skips_non_delta_events(self):
        client = MagicMock()
        events = [
            {"chunk": {"bytes": json.dumps({"type": "message_start"}).encode()}},
            {"chunk": {"bytes": json.dumps({"type": "content_block_delta", "delta": {"text": "hi"}}).encode()}},
            {"chunk": {"bytes": json.dumps({"type": "message_stop"}).encode()}},
        ]
        client.invoke_model_with_response_stream.return_value = {"body": events}
        gen = _make_generator(client)

        result = list(gen.generate_stream("ctx", "q"))

        assert result == ["hi"]


# ===================================================================
# Guardrails integration
# ===================================================================

class TestGuardrails:
    def test_guardrails_kwargs_when_enabled(self):
        client = MagicMock()
        client.invoke_model.return_value = _bedrock_response("safe answer")
        gen = _make_generator(client)

        config = GenerationConfig(
            guardrails_enabled=True,
            guardrail_id="gr-abc123",
        )
        gen.generate("ctx", "q", config)

        call_kwargs = client.invoke_model.call_args.kwargs
        assert call_kwargs["guardrailIdentifier"] == "gr-abc123"
        assert call_kwargs["guardrailVersion"] == "DRAFT"

    def test_no_guardrails_when_disabled(self):
        client = MagicMock()
        client.invoke_model.return_value = _bedrock_response("answer")
        gen = _make_generator(client)

        config = GenerationConfig(guardrails_enabled=False)
        gen.generate("ctx", "q", config)

        call_kwargs = client.invoke_model.call_args.kwargs
        assert "guardrailIdentifier" not in call_kwargs

    def test_no_guardrails_when_no_id(self):
        client = MagicMock()
        client.invoke_model.return_value = _bedrock_response("answer")
        gen = _make_generator(client)

        config = GenerationConfig(guardrails_enabled=True, guardrail_id=None)
        gen.generate("ctx", "q", config)

        call_kwargs = client.invoke_model.call_args.kwargs
        assert "guardrailIdentifier" not in call_kwargs

    def test_stream_guardrails_kwargs(self):
        client = MagicMock()
        client.invoke_model_with_response_stream.return_value = {
            "body": _stream_events(["ok"]),
        }
        gen = _make_generator(client)

        config = GenerationConfig(
            guardrails_enabled=True,
            guardrail_id="gr-xyz",
        )
        list(gen.generate_stream("ctx", "q", config))

        call_kwargs = client.invoke_model_with_response_stream.call_args.kwargs
        assert call_kwargs["guardrailIdentifier"] == "gr-xyz"


# ===================================================================
# Model fallback
# ===================================================================

class TestModelFallback:
    def test_primary_model_is_sonnet(self):
        assert "sonnet" in LLM_PRIMARY_MODEL_ID.lower()

    def test_fallback_model_is_haiku(self):
        assert "haiku" in LLM_FALLBACK_MODEL_ID.lower()

    @patch("lambdas.shared.llm_generator.time.sleep")
    def test_fallback_preserves_config(self, mock_sleep):
        client = MagicMock()
        # Primary fails with non-retryable on first call
        client.invoke_model.side_effect = [
            _non_retryable_exception(),
            _bedrock_response("fallback ok"),
        ]
        gen = _make_generator(client)

        config = GenerationConfig(
            temperature=0.2,
            top_p=0.5,
            max_tokens=512,
        )
        result = gen.generate("ctx", "q", config)

        assert result == "fallback ok"
        # Verify fallback call preserved temperature/top_p/max_tokens
        fallback_call = client.invoke_model.call_args
        body = json.loads(fallback_call.kwargs["body"])
        assert body["temperature"] == 0.2
        assert body["top_p"] == 0.5
        assert body["max_tokens"] == 512
        assert fallback_call.kwargs["modelId"] == LLM_FALLBACK_MODEL_ID

    @patch("lambdas.shared.llm_generator.time.sleep")
    def test_both_models_fail_raises(self, mock_sleep):
        client = MagicMock()
        client.invoke_model.side_effect = _non_retryable_exception()
        gen = _make_generator(client)

        with pytest.raises(Exception):
            gen.generate("ctx", "q")
