"""Validator Lambda handler for the ingestion pipeline.

Accepts a Step Functions event with bucket, key, and document_id.
Validates file type, S3 object existence, and classifies content type for routing.
"""

from __future__ import annotations

import os
from typing import Any

import boto3
from botocore.exceptions import ClientError

from lambdas.shared.constants import SUPPORTED_FILE_TYPES
from lambdas.shared.utils import generate_correlation_id, get_structured_logger

logger = get_structured_logger(__name__)

# File types routed to Textract for extraction
_TEXTRACT_TYPES: frozenset[str] = frozenset({"pdf", "png", "jpeg", "tiff"})

# File types handled by native Python libraries
_NATIVE_TYPES: frozenset[str] = frozenset({"txt", "csv", "html", "docx"})


def _get_s3_client():  # noqa: ANN202
    """Return a boto3 S3 client (extracted for testability)."""
    return boto3.client("s3")


def _extract_extension(key: str) -> str:
    """Extract the lowercase file extension from an S3 key (without the dot)."""
    _, ext = os.path.splitext(key)
    return ext.lstrip(".").lower()


def _classify_content_type(file_type: str) -> str:
    """Return 'textract' or 'native' based on the file type."""
    if file_type in _TEXTRACT_TYPES:
        return "textract"
    return "native"


def handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    """Validate an incoming document and classify its content type.

    Parameters
    ----------
    event : dict
        Expected keys: ``bucket``, ``key``, ``document_id``.
    context : object, optional
        Lambda context (unused).

    Returns
    -------
    dict
        Validation result with keys: document_id, s3_bucket, s3_key,
        file_type, content_type, file_size, valid, and optionally error.
    """
    correlation_id = event.get("correlation_id") or generate_correlation_id()
    extra = {"correlation_id": correlation_id}

    bucket: str = event["bucket"]
    key: str = event["key"]
    document_id: str = event["document_id"]

    logger.info("Validating document %s in s3://%s/%s", document_id, bucket, key, extra=extra)

    file_type = _extract_extension(key)

    # Check supported file type
    if file_type not in SUPPORTED_FILE_TYPES:
        error_msg = (
            f"Unsupported file type '.{file_type}'. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_FILE_TYPES))}"
        )
        logger.warning("Validation failed for %s: %s", document_id, error_msg, extra=extra)
        return {
            "document_id": document_id,
            "s3_bucket": bucket,
            "s3_key": key,
            "file_type": file_type,
            "content_type": None,
            "file_size": 0,
            "valid": False,
            "error": error_msg,
            "correlation_id": correlation_id,
        }

    # Verify S3 object exists and get size
    s3 = _get_s3_client()
    try:
        head = s3.head_object(Bucket=bucket, Key=key)
        file_size: int = head["ContentLength"]
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code", "Unknown")
        if error_code in ("404", "NoSuchKey"):
            error_msg = f"S3 object not found: s3://{bucket}/{key}"
        else:
            error_msg = f"Error accessing S3 object: {error_code}"
        logger.error("S3 error for %s: %s", document_id, error_msg, extra=extra)
        return {
            "document_id": document_id,
            "s3_bucket": bucket,
            "s3_key": key,
            "file_type": file_type,
            "content_type": None,
            "file_size": 0,
            "valid": False,
            "error": error_msg,
            "correlation_id": correlation_id,
        }

    content_type = _classify_content_type(file_type)

    logger.info(
        "Validation passed for %s: type=%s, content_type=%s, size=%d",
        document_id,
        file_type,
        content_type,
        file_size,
        extra=extra,
    )

    return {
        "document_id": document_id,
        "s3_bucket": bucket,
        "s3_key": key,
        "file_type": file_type,
        "content_type": content_type,
        "file_size": file_size,
        "valid": True,
        "correlation_id": correlation_id,
    }
