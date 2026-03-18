"""Ingest Trigger Lambda — accept file upload and store in S3 raw/ prefix.

Supports two modes:
1. Base64-encoded file body in the request → decode and upload to S3.
2. Presigned URL request (file_name only, no body) → generate a presigned PUT URL.

Returns document_id and status on success.
"""

from __future__ import annotations

import base64
import json
import os
import uuid
from typing import Any

import boto3

from lambdas.shared.constants import RAW_PREFIX, SUPPORTED_CONTENT_TYPES, SUPPORTED_FILE_TYPES
from lambdas.shared.utils import generate_correlation_id, get_structured_logger

logger = get_structured_logger(__name__)

CORS_HEADERS = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type,Authorization,X-Amz-Date,X-Api-Key,X-Amz-Security-Token",
    "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
}

PRESIGNED_URL_EXPIRY = 3600  # 1 hour


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


def _extract_file_extension(file_name: str) -> str:
    """Extract lowercase file extension without the dot."""
    if "." not in file_name:
        return ""
    return file_name.rsplit(".", 1)[-1].lower()


def handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    """Handle an ingest request from API Gateway.

    Expects JSON body with:
      - file_name (required): original filename
      - content_type (optional): MIME type
      - file_content (optional): base64-encoded file bytes
      - metadata (optional): user-supplied metadata dict

    If file_content is provided, uploads directly to S3.
    Otherwise, returns a presigned URL for the client to upload.
    """
    correlation_id = generate_correlation_id()

    # Parse request body
    try:
        body = event.get("body", "{}")
        if isinstance(body, str):
            body = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        return _error_response(400, "INVALID_JSON", "Request body is not valid JSON.", correlation_id)

    file_name = body.get("file_name")
    if not file_name or not isinstance(file_name, str) or not file_name.strip():
        return _error_response(400, "MISSING_FILE_NAME", "file_name is required.", correlation_id)

    file_name = file_name.strip()

    # Validate file type
    extension = _extract_file_extension(file_name)
    if extension not in SUPPORTED_FILE_TYPES:
        return _error_response(
            400,
            "UNSUPPORTED_FILE_TYPE",
            f"File type '{extension}' is not supported. Supported types: {', '.join(sorted(SUPPORTED_FILE_TYPES))}",
            correlation_id,
        )

    # Resolve content type
    content_type = body.get("content_type", "")
    if not content_type:
        # Reverse lookup from extension
        for mime, ext in SUPPORTED_CONTENT_TYPES.items():
            if ext == extension:
                content_type = mime
                break
        if not content_type:
            content_type = "application/octet-stream"

    bucket = os.environ.get("DOCUMENT_BUCKET", "")
    if not bucket:
        return _error_response(500, "CONFIG_ERROR", "DOCUMENT_BUCKET not configured.", correlation_id)

    document_id = str(uuid.uuid4())
    s3_key = f"{RAW_PREFIX}{document_id}/{file_name}"
    s3_client = _get_s3_client()
    user_metadata = body.get("metadata", {})

    file_content = body.get("file_content")

    if file_content:
        # Mode 1: Direct upload from base64-encoded body
        try:
            file_bytes = base64.b64decode(file_content)
        except Exception:
            return _error_response(400, "INVALID_BASE64", "file_content is not valid base64.", correlation_id)

        try:
            s3_client.put_object(
                Bucket=bucket,
                Key=s3_key,
                Body=file_bytes,
                ContentType=content_type,
                Metadata={
                    "document_id": document_id,
                    "original_filename": file_name,
                    **{k: str(v) for k, v in user_metadata.items()},
                },
            )
        except Exception as exc:
            logger.error("S3 upload failed: %s", exc)
            return _error_response(500, "UPLOAD_FAILED", f"Failed to upload file to S3: {exc}", correlation_id)

        logger.info("Document uploaded: document_id=%s key=%s", document_id, s3_key)
        return _success_response(200, {
            "document_id": document_id,
            "status": "uploaded",
            "s3_key": s3_key,
            "message": "Document uploaded and queued for processing.",
        })
    else:
        # Mode 2: Generate presigned URL
        try:
            presigned_url = s3_client.generate_presigned_url(
                "put_object",
                Params={
                    "Bucket": bucket,
                    "Key": s3_key,
                    "ContentType": content_type,
                },
                ExpiresIn=PRESIGNED_URL_EXPIRY,
            )
        except Exception as exc:
            logger.error("Presigned URL generation failed: %s", exc)
            return _error_response(500, "PRESIGNED_URL_FAILED", f"Failed to generate presigned URL: {exc}", correlation_id)

        logger.info("Presigned URL generated: document_id=%s key=%s", document_id, s3_key)
        return _success_response(200, {
            "document_id": document_id,
            "status": "pending_upload",
            "upload_url": presigned_url,
            "s3_key": s3_key,
            "message": "Use the upload_url to PUT the file content.",
        })
