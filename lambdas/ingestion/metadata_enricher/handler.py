"""Metadata Enricher Lambda handler for the ingestion pipeline.

Accepts a Step Functions event with document_id, s3_bucket, text (or s3_key to
extracted text), correlation_id, and pii_detection flag.  Invokes Amazon
Comprehend to extract entities, key phrases, dominant language, and optionally
PII entities.  Stores the resulting metadata JSON in S3 under
``processed/metadata/`` and returns a ``DocumentMetadata``-compatible dict.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any

import boto3

from lambdas.shared.constants import PROCESSED_METADATA_PREFIX
from lambdas.shared.models import DocumentMetadata, Entity
from lambdas.shared.utils import generate_correlation_id, get_structured_logger

logger = get_structured_logger(__name__)

# Comprehend has a 5000 UTF-8 byte limit for synchronous operations
_COMPREHEND_MAX_BYTES = 5000


# ---------------------------------------------------------------------------
# boto3 client helpers (extracted for testability)
# ---------------------------------------------------------------------------

def _get_comprehend_client():  # noqa: ANN202
    """Return a boto3 Comprehend client."""
    return boto3.client("comprehend")


def _get_s3_client():  # noqa: ANN202
    """Return a boto3 S3 client."""
    return boto3.client("s3")


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _truncate_for_comprehend(text: str) -> str:
    """Truncate *text* so its UTF-8 encoding fits within Comprehend's limit."""
    encoded = text.encode("utf-8")
    if len(encoded) <= _COMPREHEND_MAX_BYTES:
        return text
    truncated = encoded[:_COMPREHEND_MAX_BYTES].decode("utf-8", errors="ignore")
    return truncated


# ---------------------------------------------------------------------------
# Comprehend wrappers
# ---------------------------------------------------------------------------

def _detect_entities(
    text: str,
    language_code: str,
    comprehend: Any,
) -> list[Entity]:
    """Invoke Comprehend detect_entities and return a list of ``Entity``."""
    safe_text = _truncate_for_comprehend(text)
    response = comprehend.detect_entities(Text=safe_text, LanguageCode=language_code)
    return [
        Entity(
            text=e["Text"],
            type=e["Type"],
            score=round(e["Score"], 4),
        )
        for e in response.get("Entities", [])
    ]


def _detect_key_phrases(
    text: str,
    language_code: str,
    comprehend: Any,
) -> list[str]:
    """Invoke Comprehend detect_key_phrases and return phrase strings."""
    safe_text = _truncate_for_comprehend(text)
    response = comprehend.detect_key_phrases(Text=safe_text, LanguageCode=language_code)
    return [kp["Text"] for kp in response.get("KeyPhrases", [])]


def _detect_dominant_language(text: str, comprehend: Any) -> str:
    """Invoke Comprehend detect_dominant_language and return the top language code."""
    safe_text = _truncate_for_comprehend(text)
    response = comprehend.detect_dominant_language(Text=safe_text)
    languages = response.get("Languages", [])
    if not languages:
        return "en"  # fallback
    # Return the language with the highest score
    best = max(languages, key=lambda lang: lang.get("Score", 0))
    return best["LanguageCode"]


def _detect_pii_entities(
    text: str,
    language_code: str,
    comprehend: Any,
) -> list[dict]:
    """Invoke Comprehend detect_pii_entities and return PII entity dicts."""
    safe_text = _truncate_for_comprehend(text)
    response = comprehend.detect_pii_entities(Text=safe_text, LanguageCode=language_code)
    return [
        {
            "type": e["Type"],
            "score": round(e["Score"], 4),
            "begin_offset": e["BeginOffset"],
            "end_offset": e["EndOffset"],
        }
        for e in response.get("Entities", [])
    ]


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def _load_text_from_s3(s3_bucket: str, s3_key: str, s3_client: Any | None = None) -> str:
    """Load text content from an S3 object."""
    s3 = s3_client or _get_s3_client()
    obj = s3.get_object(Bucket=s3_bucket, Key=s3_key)
    return obj["Body"].read().decode("utf-8", errors="replace")


def _store_metadata(
    s3_bucket: str,
    document_id: str,
    metadata_dict: dict,
    s3_client: Any | None = None,
) -> str:
    """Store metadata JSON in S3 under ``processed/metadata/{document_id}.json``."""
    s3 = s3_client or _get_s3_client()
    output_key = f"{PROCESSED_METADATA_PREFIX}{document_id}.json"
    s3.put_object(
        Bucket=s3_bucket,
        Key=output_key,
        Body=json.dumps(metadata_dict, default=str).encode("utf-8"),
        ContentType="application/json",
    )
    return output_key


# ---------------------------------------------------------------------------
# Lambda handler
# ---------------------------------------------------------------------------

def handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    """Enrich a document with NLP metadata via Amazon Comprehend.

    Parameters
    ----------
    event : dict
        Expected keys from Step Functions:
        ``document_id``, ``s3_bucket``, and either ``text`` (inline) or
        ``s3_key`` (path to extracted text in S3).
        Optional: ``correlation_id``, ``pii_detection`` (bool, default False).
    context : object, optional
        Lambda context (unused).

    Returns
    -------
    dict
        DocumentMetadata fields plus ``output_key`` and ``correlation_id``.
    """
    correlation_id = event.get("correlation_id") or generate_correlation_id()
    extra = {"correlation_id": correlation_id}

    document_id: str = event["document_id"]
    s3_bucket: str = event["s3_bucket"]
    pii_detection: bool = event.get(
        "pii_detection",
        os.environ.get("PII_DETECTION_ENABLED", "false").lower() == "true",
    )

    logger.info(
        "Starting metadata enrichment for %s (pii_detection=%s)",
        document_id,
        pii_detection,
        extra=extra,
    )

    # Resolve text: inline or from S3
    text: str = event.get("text", "")
    if not text:
        s3_key: str = event.get("s3_key", "")
        if not s3_key:
            error_msg = "Either 'text' or 's3_key' must be provided in the event"
            logger.error(error_msg, extra=extra)
            return {
                "document_id": document_id,
                "s3_bucket": s3_bucket,
                "entities": [],
                "key_phrases": [],
                "language": "",
                "pii_detected": False,
                "pii_entities": [],
                "error": error_msg,
                "correlation_id": correlation_id,
            }
        try:
            text = _load_text_from_s3(s3_bucket, s3_key)
        except Exception as exc:
            error_msg = f"Failed to load text from s3://{s3_bucket}/{s3_key}: {exc}"
            logger.error(error_msg, extra=extra)
            return {
                "document_id": document_id,
                "s3_bucket": s3_bucket,
                "entities": [],
                "key_phrases": [],
                "language": "",
                "pii_detected": False,
                "pii_entities": [],
                "error": error_msg,
                "correlation_id": correlation_id,
            }

    comprehend = _get_comprehend_client()

    try:
        # Step 1: Detect dominant language
        language = _detect_dominant_language(text, comprehend)
        logger.info("Detected language '%s' for %s", language, document_id, extra=extra)

        # Step 2: Detect entities
        entities = _detect_entities(text, language, comprehend)
        logger.info("Detected %d entities for %s", len(entities), document_id, extra=extra)

        # Step 3: Detect key phrases
        key_phrases = _detect_key_phrases(text, language, comprehend)
        logger.info("Detected %d key phrases for %s", len(key_phrases), document_id, extra=extra)

        # Step 4: Optionally detect PII
        pii_detected = False
        pii_entities: list[dict] = []
        if pii_detection:
            pii_entities = _detect_pii_entities(text, language, comprehend)
            pii_detected = len(pii_entities) > 0
            logger.info(
                "PII detection for %s: detected=%s, count=%d",
                document_id,
                pii_detected,
                len(pii_entities),
                extra=extra,
            )

    except Exception as exc:
        error_msg = f"Comprehend enrichment failed for {document_id}: {exc}"
        logger.error(error_msg, extra=extra)
        return {
            "document_id": document_id,
            "s3_bucket": s3_bucket,
            "entities": [],
            "key_phrases": [],
            "language": "",
            "pii_detected": False,
            "pii_entities": [],
            "error": error_msg,
            "correlation_id": correlation_id,
        }

    # Build DocumentMetadata
    metadata = DocumentMetadata(
        document_id=document_id,
        entities=entities,
        key_phrases=key_phrases,
        language=language,
        pii_detected=pii_detected,
        pii_entities=pii_entities,
    )

    metadata_dict = asdict(metadata)

    # Store metadata JSON in S3
    output_key = ""
    try:
        output_key = _store_metadata(s3_bucket, document_id, metadata_dict)
        logger.info(
            "Stored metadata for %s at s3://%s/%s",
            document_id,
            s3_bucket,
            output_key,
            extra=extra,
        )
    except Exception as exc:
        logger.error(
            "Failed to store metadata for %s: %s",
            document_id,
            str(exc),
            extra=extra,
        )

    response = metadata_dict
    response["output_key"] = output_key
    response["correlation_id"] = correlation_id

    logger.info(
        "Metadata enrichment complete for %s: language=%s, entities=%d, key_phrases=%d, pii=%s",
        document_id,
        language,
        len(entities),
        len(key_phrases),
        pii_detected,
        extra=extra,
    )

    return response
