"""Unit tests for lambdas.query.ingest_trigger.handler."""

from __future__ import annotations

import base64
import json
import os
from unittest.mock import MagicMock, patch

import pytest

from lambdas.query.ingest_trigger.handler import (
    _extract_file_extension,
    handler,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(body: dict) -> dict:
    return {"body": json.dumps(body)}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("DOCUMENT_BUCKET", "test-bucket")
    monkeypatch.setenv("ENV_NAME", "test")


# ---------------------------------------------------------------------------
# _extract_file_extension
# ---------------------------------------------------------------------------

class TestExtractFileExtension:
    def test_pdf(self):
        assert _extract_file_extension("report.pdf") == "pdf"

    def test_uppercase(self):
        assert _extract_file_extension("IMAGE.PNG") == "png"

    def test_no_extension(self):
        assert _extract_file_extension("noext") == ""

    def test_multiple_dots(self):
        assert _extract_file_extension("archive.tar.gz") == "gz"

    def test_dotfile(self):
        assert _extract_file_extension(".hidden") == "hidden"


# ---------------------------------------------------------------------------
# Handler — validation errors
# ---------------------------------------------------------------------------

class TestHandlerValidation:
    def test_missing_body(self):
        result = handler({"body": "{}"})
        assert result["statusCode"] == 400
        body = json.loads(result["body"])
        assert body["error"]["code"] == "MISSING_FILE_NAME"

    def test_invalid_json_body(self):
        result = handler({"body": "not json{{"})
        assert result["statusCode"] == 400
        body = json.loads(result["body"])
        assert body["error"]["code"] == "INVALID_JSON"

    def test_unsupported_file_type(self):
        result = handler(_make_event({"file_name": "data.xyz"}))
        assert result["statusCode"] == 400
        body = json.loads(result["body"])
        assert body["error"]["code"] == "UNSUPPORTED_FILE_TYPE"
        assert "xyz" in body["error"]["message"]

    def test_empty_file_name(self):
        result = handler(_make_event({"file_name": "  "}))
        assert result["statusCode"] == 400
        body = json.loads(result["body"])
        assert body["error"]["code"] == "MISSING_FILE_NAME"

    def test_missing_bucket_env(self, monkeypatch):
        monkeypatch.delenv("DOCUMENT_BUCKET", raising=False)
        result = handler(_make_event({"file_name": "test.pdf"}))
        assert result["statusCode"] == 500
        body = json.loads(result["body"])
        assert body["error"]["code"] == "CONFIG_ERROR"


# ---------------------------------------------------------------------------
# Handler — base64 upload mode
# ---------------------------------------------------------------------------

class TestHandlerBase64Upload:
    @patch("lambdas.query.ingest_trigger.handler._get_s3_client")
    def test_successful_upload(self, mock_s3_factory):
        s3 = MagicMock()
        mock_s3_factory.return_value = s3

        content = base64.b64encode(b"hello world").decode()
        event = _make_event({
            "file_name": "test.pdf",
            "file_content": content,
        })
        result = handler(event)

        assert result["statusCode"] == 200
        body = json.loads(result["body"])
        assert body["status"] == "uploaded"
        assert "document_id" in body
        assert body["s3_key"].startswith("raw/")
        assert body["s3_key"].endswith("/test.pdf")
        s3.put_object.assert_called_once()

    @patch("lambdas.query.ingest_trigger.handler._get_s3_client")
    def test_upload_with_metadata(self, mock_s3_factory):
        s3 = MagicMock()
        mock_s3_factory.return_value = s3

        content = base64.b64encode(b"data").decode()
        event = _make_event({
            "file_name": "doc.txt",
            "file_content": content,
            "metadata": {"author": "test"},
        })
        result = handler(event)
        assert result["statusCode"] == 200

        call_kwargs = s3.put_object.call_args[1]
        assert call_kwargs["Metadata"]["author"] == "test"

    def test_invalid_base64(self):
        event = _make_event({
            "file_name": "test.pdf",
            "file_content": "not-valid-base64!!!",
        })
        result = handler(event)
        assert result["statusCode"] == 400
        body = json.loads(result["body"])
        assert body["error"]["code"] == "INVALID_BASE64"

    @patch("lambdas.query.ingest_trigger.handler._get_s3_client")
    def test_s3_upload_failure(self, mock_s3_factory):
        s3 = MagicMock()
        s3.put_object.side_effect = Exception("S3 error")
        mock_s3_factory.return_value = s3

        content = base64.b64encode(b"data").decode()
        event = _make_event({
            "file_name": "test.pdf",
            "file_content": content,
        })
        result = handler(event)
        assert result["statusCode"] == 500
        body = json.loads(result["body"])
        assert body["error"]["code"] == "UPLOAD_FAILED"


# ---------------------------------------------------------------------------
# Handler — presigned URL mode
# ---------------------------------------------------------------------------

class TestHandlerPresignedUrl:
    @patch("lambdas.query.ingest_trigger.handler._get_s3_client")
    def test_presigned_url_generation(self, mock_s3_factory):
        s3 = MagicMock()
        s3.generate_presigned_url.return_value = "https://s3.example.com/presigned"
        mock_s3_factory.return_value = s3

        event = _make_event({"file_name": "report.pdf"})
        result = handler(event)

        assert result["statusCode"] == 200
        body = json.loads(result["body"])
        assert body["status"] == "pending_upload"
        assert body["upload_url"] == "https://s3.example.com/presigned"
        assert "document_id" in body
        s3.generate_presigned_url.assert_called_once()

    @patch("lambdas.query.ingest_trigger.handler._get_s3_client")
    def test_presigned_url_failure(self, mock_s3_factory):
        s3 = MagicMock()
        s3.generate_presigned_url.side_effect = Exception("URL gen failed")
        mock_s3_factory.return_value = s3

        event = _make_event({"file_name": "report.pdf"})
        result = handler(event)
        assert result["statusCode"] == 500
        body = json.loads(result["body"])
        assert body["error"]["code"] == "PRESIGNED_URL_FAILED"


# ---------------------------------------------------------------------------
# CORS headers
# ---------------------------------------------------------------------------

class TestCorsHeaders:
    @patch("lambdas.query.ingest_trigger.handler._get_s3_client")
    def test_success_has_cors(self, mock_s3_factory):
        s3 = MagicMock()
        s3.generate_presigned_url.return_value = "https://example.com"
        mock_s3_factory.return_value = s3

        result = handler(_make_event({"file_name": "test.pdf"}))
        assert result["headers"]["Access-Control-Allow-Origin"] == "*"
        assert "GET,POST,OPTIONS" in result["headers"]["Access-Control-Allow-Methods"]

    def test_error_has_cors(self):
        result = handler(_make_event({"file_name": "bad.xyz"}))
        assert result["headers"]["Access-Control-Allow-Origin"] == "*"


# ---------------------------------------------------------------------------
# Error response format
# ---------------------------------------------------------------------------

class TestErrorFormat:
    def test_error_has_standard_fields(self):
        result = handler(_make_event({"file_name": "bad.xyz"}))
        body = json.loads(result["body"])
        error = body["error"]
        assert "code" in error
        assert "message" in error
        assert "request_id" in error
        assert len(error["request_id"]) > 0
