"""Unit tests for lambdas.ingestion.validator.handler."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import boto3
import pytest
from botocore.exceptions import ClientError
from moto import mock_aws

from lambdas.ingestion.validator.handler import (
    _classify_content_type,
    _extract_extension,
    handler,
)


# ---------------------------------------------------------------------------
# Helper: _extract_extension
# ---------------------------------------------------------------------------

class TestExtractExtension:
    def test_pdf(self):
        assert _extract_extension("raw/doc.pdf") == "pdf"

    def test_uppercase(self):
        assert _extract_extension("raw/doc.PDF") == "pdf"

    def test_nested_path(self):
        assert _extract_extension("raw/folder/sub/file.docx") == "docx"

    def test_no_extension(self):
        assert _extract_extension("raw/noext") == ""

    def test_dotfile(self):
        # os.path.splitext treats .hidden as a name with no extension
        assert _extract_extension("raw/.hidden") == ""

    def test_multiple_dots(self):
        assert _extract_extension("raw/archive.tar.gz") == "gz"


# ---------------------------------------------------------------------------
# Helper: _classify_content_type
# ---------------------------------------------------------------------------

class TestClassifyContentType:
    @pytest.mark.parametrize("ft", ["pdf", "png", "jpeg", "tiff"])
    def test_textract_types(self, ft: str):
        assert _classify_content_type(ft) == "textract"

    @pytest.mark.parametrize("ft", ["txt", "csv", "html", "docx"])
    def test_native_types(self, ft: str):
        assert _classify_content_type(ft) == "native"


# ---------------------------------------------------------------------------
# Handler integration tests (moto)
# ---------------------------------------------------------------------------

@mock_aws
class TestHandlerValid:
    """Tests for valid documents that should pass validation."""

    def _setup_s3(self, bucket: str, key: str, body: bytes = b"hello") -> None:
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=bucket)
        s3.put_object(Bucket=bucket, Key=key, Body=body)

    def test_valid_pdf(self):
        self._setup_s3("test-bucket", "raw/doc.pdf")
        result = handler(
            {"bucket": "test-bucket", "key": "raw/doc.pdf", "document_id": "doc-1"}
        )
        assert result["valid"] is True
        assert result["file_type"] == "pdf"
        assert result["content_type"] == "textract"
        assert result["file_size"] == 5
        assert result["document_id"] == "doc-1"
        assert "error" not in result

    def test_valid_txt(self):
        self._setup_s3("test-bucket", "raw/notes.txt", b"some text content")
        result = handler(
            {"bucket": "test-bucket", "key": "raw/notes.txt", "document_id": "doc-2"}
        )
        assert result["valid"] is True
        assert result["content_type"] == "native"
        assert result["file_type"] == "txt"

    def test_valid_csv(self):
        self._setup_s3("test-bucket", "raw/data.csv", b"a,b\n1,2")
        result = handler(
            {"bucket": "test-bucket", "key": "raw/data.csv", "document_id": "doc-3"}
        )
        assert result["valid"] is True
        assert result["content_type"] == "native"

    def test_valid_png(self):
        self._setup_s3("test-bucket", "raw/image.png", b"\x89PNG")
        result = handler(
            {"bucket": "test-bucket", "key": "raw/image.png", "document_id": "doc-4"}
        )
        assert result["valid"] is True
        assert result["content_type"] == "textract"

    def test_correlation_id_passthrough(self):
        self._setup_s3("test-bucket", "raw/doc.pdf")
        result = handler(
            {
                "bucket": "test-bucket",
                "key": "raw/doc.pdf",
                "document_id": "doc-5",
                "correlation_id": "custom-id-123",
            }
        )
        assert result["correlation_id"] == "custom-id-123"

    def test_correlation_id_generated(self):
        self._setup_s3("test-bucket", "raw/doc.pdf")
        result = handler(
            {"bucket": "test-bucket", "key": "raw/doc.pdf", "document_id": "doc-6"}
        )
        assert result["correlation_id"] is not None
        assert len(result["correlation_id"]) > 0


@mock_aws
class TestHandlerInvalid:
    """Tests for documents that should fail validation."""

    def test_unsupported_file_type(self):
        result = handler(
            {"bucket": "test-bucket", "key": "raw/file.exe", "document_id": "doc-bad"}
        )
        assert result["valid"] is False
        assert "Unsupported file type" in result["error"]
        assert "exe" in result["error"]
        assert "pdf" in result["error"]  # lists supported formats

    def test_s3_object_not_found(self):
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="test-bucket")
        result = handler(
            {"bucket": "test-bucket", "key": "raw/missing.pdf", "document_id": "doc-miss"}
        )
        assert result["valid"] is False
        assert result["file_size"] == 0
        assert result["error"] is not None

    def test_no_extension(self):
        result = handler(
            {"bucket": "test-bucket", "key": "raw/noext", "document_id": "doc-noext"}
        )
        assert result["valid"] is False
        assert "Unsupported file type" in result["error"]
