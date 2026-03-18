"""LLM Generator module for the AWS-native RAG service.

Provides the ``LLMGenerator`` class that invokes Amazon Bedrock Claude models
to generate natural language answers from retrieved context.  Supports both
non-streaming (``generate``) and streaming (``generate_stream``) modes, model
fallback from Sonnet to Haiku, retry with exponential backoff, and optional
Bedrock Guardrails integration.
"""

from __future__ import annotations

import json
import random
import time
from typing import Any, Generator

import boto3

from lambdas.shared.constants import LLM_FALLBACK_MODEL_ID, LLM_PRIMARY_MODEL_ID
from lambdas.shared.models import GenerationConfig
from lambdas.shared.utils import create_subsegment, get_structured_logger

logger = get_structured_logger(__name__)

# ---------------------------------------------------------------------------
# Retry / backoff configuration
# ---------------------------------------------------------------------------
MAX_RETRIES = 3
INITIAL_BACKOFF_S = 1.0
RETRYABLE_ERROR_CODES = frozenset({
    "ThrottlingException",
    "TooManyRequestsException",
    "ServiceUnavailableException",
    "ModelTimeoutException",
})

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Answer questions based ONLY on the "
    "provided context.\n\n"
    "Guidelines:\n"
    "- If the context doesn't contain the answer, say "
    '"I don\'t have enough information to answer this question."\n'
    "- Cite sources by referencing the document name and section.\n"
    "- Be concise and accurate.\n"
    "- Do not fabricate information."
)

RAG_PROMPT_TEMPLATE = (
    "Context:\n{context}\n\n"
    "Question: {query}\n\n"
    "Instructions:\n"
    "- Answer based only on the context above.\n"
    "- Cite the relevant document sections using [Source: document_name, page X].\n"
    "- If uncertain, acknowledge limitations.\n\n"
    "Answer:"
)


def _is_retryable(exc: Exception) -> bool:
    """Return ``True`` if *exc* looks like a transient Bedrock error."""
    error_code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
    if error_code in RETRYABLE_ERROR_CODES:
        return True
    # Some SDK versions surface throttling only in the message string.
    return "Throttling" in str(exc) or "throttl" in str(exc).lower()


class LLMGenerator:
    """Generate answers from retrieved context using Amazon Bedrock Claude.

    Parameters
    ----------
    bedrock_client : optional
        A pre-configured ``bedrock-runtime`` boto3 client.  When *None* a
        new client is created automatically.
    """

    def __init__(self, bedrock_client: Any | None = None) -> None:
        self._client = bedrock_client or boto3.client("bedrock-runtime")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        context: str,
        query: str,
        config: GenerationConfig | None = None,
    ) -> str:
        """Generate an answer from *context* (non-streaming).

        Tries the primary model first; on failure falls back to the
        fallback model.  Each model attempt retries up to ``MAX_RETRIES``
        times with exponential backoff on transient errors.
        """
        config = config or GenerationConfig()
        prompt = RAG_PROMPT_TEMPLATE.format(context=context, query=query)

        # Try primary model
        try:
            return self._invoke(prompt, config)
        except Exception as primary_exc:
            logger.warning(
                "Primary model (%s) failed: %s — falling back to %s",
                config.model_id,
                primary_exc,
                LLM_FALLBACK_MODEL_ID,
            )

        # Fallback model
        fallback_config = GenerationConfig(
            model_id=LLM_FALLBACK_MODEL_ID,
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
            guardrails_enabled=config.guardrails_enabled,
            guardrail_id=config.guardrail_id,
        )
        return self._invoke(prompt, fallback_config)

    def generate_stream(
        self,
        context: str,
        query: str,
        config: GenerationConfig | None = None,
    ) -> Generator[str, None, None]:
        """Generate an answer with streaming.  Yields token chunks.

        Falls back to the secondary model when the primary model fails.
        """
        config = config or GenerationConfig()
        prompt = RAG_PROMPT_TEMPLATE.format(context=context, query=query)

        # Try primary model
        try:
            yield from self._invoke_stream(prompt, config)
            return
        except Exception as primary_exc:
            logger.warning(
                "Primary model stream (%s) failed: %s — falling back to %s",
                config.model_id,
                primary_exc,
                LLM_FALLBACK_MODEL_ID,
            )

        # Fallback model
        fallback_config = GenerationConfig(
            model_id=LLM_FALLBACK_MODEL_ID,
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
            guardrails_enabled=config.guardrails_enabled,
            guardrail_id=config.guardrail_id,
        )
        yield from self._invoke_stream(prompt, fallback_config)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_body(self, prompt: str, config: GenerationConfig) -> str:
        """Build the JSON request body for the Bedrock Converse/Messages API."""
        body: dict[str, Any] = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": prompt}],
        }
        return json.dumps(body)

    def _guardrail_kwargs(self, config: GenerationConfig) -> dict[str, Any]:
        """Return extra kwargs for guardrails if enabled and configured."""
        if config.guardrails_enabled and config.guardrail_id:
            return {
                "guardrailIdentifier": config.guardrail_id,
                "guardrailVersion": "DRAFT",
            }
        return {}

    def _invoke(self, prompt: str, config: GenerationConfig) -> str:
        """Invoke Bedrock (non-streaming) with retry + exponential backoff."""
        body = self._build_body(prompt, config)
        extra_kwargs = self._guardrail_kwargs(config)

        last_exc: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                with create_subsegment("bedrock_invoke"):
                    response = self._client.invoke_model(
                        modelId=config.model_id,
                        contentType="application/json",
                        accept="application/json",
                        body=body,
                        **extra_kwargs,
                    )
                    raw = response["body"].read() if hasattr(response["body"], "read") else response["body"]
                    result = json.loads(raw)
                    # Claude Messages API returns content as a list of blocks
                    content_blocks = result.get("content", [])
                    return "".join(
                        block.get("text", "") for block in content_blocks if block.get("type") == "text"
                    )
            except Exception as exc:
                if _is_retryable(exc) and attempt < MAX_RETRIES - 1:
                    wait = INITIAL_BACKOFF_S * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(
                        "Bedrock invoke throttled (attempt %d/%d), retrying in %.1fs",
                        attempt + 1,
                        MAX_RETRIES,
                        wait,
                    )
                    time.sleep(wait)
                    last_exc = exc
                else:
                    raise

        raise RuntimeError(
            f"Bedrock invoke failed after {MAX_RETRIES} retries"
        ) from last_exc

    def _invoke_stream(self, prompt: str, config: GenerationConfig) -> Generator[str, None, None]:
        """Invoke Bedrock with streaming, retry + exponential backoff + jitter."""
        body = self._build_body(prompt, config)
        extra_kwargs = self._guardrail_kwargs(config)

        last_exc: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                with create_subsegment("bedrock_stream"):
                    response = self._client.invoke_model_with_response_stream(
                        modelId=config.model_id,
                        contentType="application/json",
                        accept="application/json",
                        body=body,
                        **extra_kwargs,
                    )
                    for event in response.get("body", []):
                        chunk = event.get("chunk")
                        if chunk:
                            payload = json.loads(chunk["bytes"].decode("utf-8"))
                            if payload.get("type") == "content_block_delta":
                                text = payload.get("delta", {}).get("text", "")
                                if text:
                                    yield text
                return  # success — exit retry loop
            except Exception as exc:
                if _is_retryable(exc) and attempt < MAX_RETRIES - 1:
                    wait = INITIAL_BACKOFF_S * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(
                        "Bedrock stream throttled (attempt %d/%d), retrying in %.1fs",
                        attempt + 1,
                        MAX_RETRIES,
                        wait,
                    )
                    time.sleep(wait)
                    last_exc = exc
                else:
                    raise

        raise RuntimeError(
            f"Bedrock stream failed after {MAX_RETRIES} retries"
        ) from last_exc
