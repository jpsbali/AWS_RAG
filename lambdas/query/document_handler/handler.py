"""Document Handler Lambda — return document metadata and processing status.

Looks up S3 objects across raw/, processed/, and failed/ prefixes to determine
the current processing status and gather metadata for a given document_id.
"""

from __future__ import annotations

import json
import os
from typing import Any

import boto3

from lambdas.shared.constants import (
    FAILED_PREFIX,
    PROCESSED_CHUNKS_PREFIX,
    PROCESSED_METADATA_PREFIX,
    PROCESSED_TEXT_PREFIX,
    RAW_PREFIX,
)
from lambdas.shared.utils import generate_correlation_id, get_structured_logger

logger = get_structured_logger(__name__)

CORS_HEADERS = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type,Authorization,X-Amz-Date,X-Api-Key,X-Amz-Security-Token",
    "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
}


def _get_s3_client() -> Any:
    return boto3.client("s3")


def _error_response(status_code: int, code: str, message: str, request_id: str) -> dict[str, Any]:
    return {
        "statusCode": status_code,
        "headers": CORS_HEADERS,
        "body": json.dumps({"error": {"code": code, "message": message, "request_id": request_id}}),
    }


def _success_response(status_code: int, body: dict) -> dict[str, Any]:
    return {
        "statusCode": status_code,
        "headers": CORS_HEADERS,
        "body": json.dumps(body),
    }


def _list_objects(s3_client: Any, bucket: str, prefix: str) -> list[dict]:
    """List S3 objects under a prefix. Returns list of object summaries."""
    try:
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        return response.get("Contents", [])
    except Exception:
        return []


def _determine_status(
    raw_objects: list[dict],
    text_objects: list[dict],
    chunks_objects: list[dict],
    metadata_objects: list[dict],
    failed_objects: list[dict],
) -> str:
    """Determine document processing status from S3 prefix presence."""
    if failed_objects:
        return "failed"
    if chunks_objects and text_objects:
        return "completed"
    if text_objects or metadata_objects:
        return "processing"
    if raw_objects:
        return "uploaded"
    return "not_found"


def handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    """Handle a document status request from API Gateway.

    Expects pathParameters.id to contain the document_id.
    """
    correlation_id = generate_correlation_id()

    # Extract document_id from path parameters
    path_params = event.get("pathParameters") or {}
    document_id = path_params.get("id", "")

    if not document_id:
        return _error_response(400, "MISSING_DOCUMENT_ID", "Document ID is required.", correlation_id)

    bucket = os.environ.get("DOCUMENT_BUCKET", "")
    if not bucket:
        return _error_response(500, "CONFIG_ERROR", "DOCUMENT_BUCKET not configured.", correlation_id)

    s3_client = _get_s3_client()

    # Check each prefix for objects related to this document_id
    raw_objects = _list_objects(s3_client, bucket, f"{RAW_PREFIX}{document_id}/")
    text_objects = _list_objects(s3_client, bucket, f"{PROCESSED_TEXT_PREFIX}{document_id}/")
    chunks_objects = _list_objects(s3_client, bucket, f"{PROCESSED_CHUNKS_PREFIX}{document_id}/")
    metadata_objects = _list_objects(s3_client, bucket, f"{PROCESSED_METADATA_PREFIX}{document_id}/")
    failed_objects = _list_objects(s3_client, bucket, f"{FAILED_PREFIX}{document_id}/")

    status = _determine_status(raw_objects, text_objects, chunks_objects, metadata_objects, failed_objects)

    if status == "not_found":
        return _error_response(404, "DOCUMENT_NOT_FOUND", f"Document '{document_id}' not found.", correlation_id)

    # Build metadata response
    raw_file = raw_objects[0] if raw_objects else None
    result: dict[str, Any] = {
        "document_id": document_id,
        "status": status,
        "raw": {
            "key": raw_file["Key"] if raw_file else None,
            "size": raw_file["Size"] if raw_file else None,
            "last_modified": raw_file["LastModified"].isoformat() if raw_file and "LastModified" in raw_file else None,
        } if raw_file else None,
        "processing": {
            "text_extracted": len(text_objects) > 0,
            "chunks_generated": len(chunks_objects) > 0,
            "metadata_enriched": len(metadata_objects) > 0,
            "failed": len(failed_objects) > 0,
        },
    }

    logger.info("Document status: document_id=%s status=%s", document_id, status)
    return _success_response(200, result)
