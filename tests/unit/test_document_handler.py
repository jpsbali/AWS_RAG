"""Unit tests for lambdas.query.document_handler.handler."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from lambdas.query.document_handler.handler import (
    _determine_status,
    _list_objects,
    handler,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(document_id: str) -> dict:
    return {"pathParameters": {"id": document_id}}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("DOCUMENT_BUCKET", "test-bucket")
    monkeypatch.setenv("ENV_NAME", "test")


# ---------------------------------------------------------------------------
# _determine_status
# ---------------------------------------------------------------------------

class TestDetermineStatus:
    def test_failed(self):
        assert _determine_status([], [], [], [], [{"Key": "f"}]) == "failed"

    def test_completed(self):
        assert _determine_status([{"Key": "r"}], [{"Key": "t"}], [{"Key": "c"}], [], []) == "completed"

    def test_processing_text_only(self):
        assert _determine_status([{"Key": "r"}], [{"Key": "t"}], [], [], []) == "processing"

    def test_processing_metadata_only(self):
        assert _determine_status([{"Key": "r"}], [], [], [{"Key": "m"}], []) == "processing"

    def test_uploaded(self):
        assert _determine_status([{"Key": "r"}], [], [], [], []) == "uploaded"

    def test_not_found(self):
        assert _determine_status([], [], [], [], []) == "not_found"

    def test_failed_takes_precedence(self):
        # Even if raw and processed exist, failed wins
        assert _determine_status([{"Key": "r"}], [{"Key": "t"}], [{"Key": "c"}], [], [{"Key": "f"}]) == "failed"


# ---------------------------------------------------------------------------
# Handler — validation
# ---------------------------------------------------------------------------

class TestHandlerValidation:
    def test_missing_document_id(self):
        result = handler({"pathParameters": {}})
        assert result["statusCode"] == 400
        body = json.loads(result["body"])
        assert body["error"]["code"] == "MISSING_DOCUMENT_ID"

    def test_no_path_parameters(self):
        result = handler({})
        assert result["statusCode"] == 400

    def test_missing_bucket_env(self, monkeypatch):
        monkeypatch.delenv("DOCUMENT_BUCKET", raising=False)
        result = handler(_make_event("doc-123"))
        assert result["statusCode"] == 500
        body = json.loads(result["body"])
        assert body["error"]["code"] == "CONFIG_ERROR"


# ---------------------------------------------------------------------------
# Handler — document lookup
# ---------------------------------------------------------------------------

class TestHandlerLookup:
    @patch("lambdas.query.document_handler.handler._get_s3_client")
    def test_document_not_found(self, mock_s3_factory):
        s3 = MagicMock()
        s3.list_objects_v2.return_value = {"Contents": []}
        mock_s3_factory.return_value = s3

        result = handler(_make_event("nonexistent-id"))
        assert result["statusCode"] == 404
        body = json.loads(result["body"])
        assert body["error"]["code"] == "DOCUMENT_NOT_FOUND"

    @patch("lambdas.query.document_handler.handler._get_s3_client")
    def test_uploaded_document(self, mock_s3_factory):
        s3 = MagicMock()
        now = datetime.now(timezone.utc)

        def list_side_effect(Bucket, Prefix):
            if Prefix.startswith("raw/"):
                return {"Contents": [{"Key": "raw/doc-1/file.pdf", "Size": 1024, "LastModified": now}]}
            return {"Contents": []}

        s3.list_objects_v2.side_effect = list_side_effect
        mock_s3_factory.return_value = s3

        result = handler(_make_event("doc-1"))
        assert result["statusCode"] == 200
        body = json.loads(result["body"])
        assert body["document_id"] == "doc-1"
        assert body["status"] == "uploaded"
        assert body["raw"]["size"] == 1024

    @patch("lambdas.query.document_handler.handler._get_s3_client")
    def test_completed_document(self, mock_s3_factory):
        s3 = MagicMock()
        now = datetime.now(timezone.utc)

        def list_side_effect(Bucket, Prefix):
            if Prefix.startswith("raw/"):
                return {"Contents": [{"Key": "raw/doc-1/file.pdf", "Size": 1024, "LastModified": now}]}
            if Prefix.startswith("processed/text/"):
                return {"Contents": [{"Key": "processed/text/doc-1/output.json"}]}
            if Prefix.startswith("processed/chunks/"):
                return {"Contents": [{"Key": "processed/chunks/doc-1/chunks.json"}]}
            return {"Contents": []}

        s3.list_objects_v2.side_effect = list_side_effect
        mock_s3_factory.return_value = s3

        result = handler(_make_event("doc-1"))
        assert result["statusCode"] == 200
        body = json.loads(result["body"])
        assert body["status"] == "completed"
        assert body["processing"]["text_extracted"] is True
        assert body["processing"]["chunks_generated"] is True

    @patch("lambdas.query.document_handler.handler._get_s3_client")
    def test_failed_document(self, mock_s3_factory):
        s3 = MagicMock()

        def list_side_effect(Bucket, Prefix):
            if Prefix.startswith("raw/"):
                return {"Contents": [{"Key": "raw/doc-1/file.pdf", "Size": 512}]}
            if Prefix.startswith("failed/"):
                return {"Contents": [{"Key": "failed/doc-1/file.pdf"}]}
            return {"Contents": []}

        s3.list_objects_v2.side_effect = list_side_effect
        mock_s3_factory.return_value = s3

        result = handler(_make_event("doc-1"))
        assert result["statusCode"] == 200
        body = json.loads(result["body"])
        assert body["status"] == "failed"
        assert body["processing"]["failed"] is True


# ---------------------------------------------------------------------------
# CORS and error format
# ---------------------------------------------------------------------------

class TestCorsAndErrorFormat:
    def test_success_has_cors(self):
        with patch("lambdas.query.document_handler.handler._get_s3_client") as mock_s3_factory:
            s3 = MagicMock()
            s3.list_objects_v2.return_value = {"Contents": []}
            mock_s3_factory.return_value = s3

            result = handler(_make_event("doc-1"))
            assert result["headers"]["Access-Control-Allow-Origin"] == "*"

    def test_error_has_standard_fields(self):
        result = handler({"pathParameters": {}})
        body = json.loads(result["body"])
        error = body["error"]
        assert "code" in error
        assert "message" in error
        assert "request_id" in error
