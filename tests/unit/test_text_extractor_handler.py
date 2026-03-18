"""Unit tests for lambdas.ingestion.text_extractor.handler."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import boto3
import pytest
from moto import mock_aws

from lambdas.ingestion.text_extractor.handler import (
    _extract_native_csv,
    _extract_native_txt,
    _parse_textract_blocks,
    handler,
    validate_quality,
)


# ---------------------------------------------------------------------------
# validate_quality
# ---------------------------------------------------------------------------

class TestValidateQuality:
    def test_meets_default_threshold(self):
        assert validate_quality("a" * 50) is True

    def test_below_default_threshold(self):
        assert validate_quality("a" * 49) is False

    def test_empty_string(self):
        assert validate_quality("") is False

    def test_custom_threshold(self):
        assert validate_quality("hello", min_threshold=5) is True
        assert validate_quality("hi", min_threshold=5) is False

    def test_exact_threshold(self):
        assert validate_quality("abcde", min_threshold=5) is True


# ---------------------------------------------------------------------------
# Native extraction helpers
# ---------------------------------------------------------------------------

class TestNativeTxt:
    def test_basic(self):
        assert _extract_native_txt(b"hello world") == "hello world"

    def test_utf8(self):
        text = "café résumé"
        assert _extract_native_txt(text.encode("utf-8")) == text

    def test_empty(self):
        assert _extract_native_txt(b"") == ""


class TestNativeCsv:
    def test_basic_csv(self):
        result = _extract_native_csv(b"a,b,c\n1,2,3")
        assert "a b c" in result
        assert "1 2 3" in result

    def test_single_column(self):
        result = _extract_native_csv(b"name\nAlice\nBob")
        assert "Alice" in result
        assert "Bob" in result


# ---------------------------------------------------------------------------
# Textract block parsing
# ---------------------------------------------------------------------------

class TestParseTextractBlocks:
    def test_lines_extracted(self):
        blocks = [
            {"BlockType": "LINE", "Text": "Hello", "Page": 1, "Confidence": 99.0},
            {"BlockType": "LINE", "Text": "World", "Page": 1, "Confidence": 98.0},
        ]
        result = _parse_textract_blocks(blocks, document_id="doc-1")
        assert result.text == "Hello\nWorld"
        assert result.pages == 1
        assert result.document_id == "doc-1"
        assert result.extraction_method == "textract"
        assert result.confidence > 0

    def test_multi_page(self):
        blocks = [
            {"BlockType": "LINE", "Text": "Page1", "Page": 1, "Confidence": 95.0},
            {"BlockType": "LINE", "Text": "Page2", "Page": 2, "Confidence": 95.0},
            {"BlockType": "LINE", "Text": "Page3", "Page": 3, "Confidence": 95.0},
        ]
        result = _parse_textract_blocks(blocks, document_id="doc-2")
        assert result.pages == 3

    def test_tables_and_forms(self):
        blocks = [
            {"BlockType": "TABLE", "Id": "t1", "Page": 1},
            {"BlockType": "KEY_VALUE_SET", "Id": "kv1", "Page": 1},
            {"BlockType": "LINE", "Text": "text", "Page": 1, "Confidence": 90.0},
        ]
        result = _parse_textract_blocks(blocks, document_id="doc-3")
        assert len(result.tables) == 1
        assert len(result.forms) == 1

    def test_empty_blocks(self):
        result = _parse_textract_blocks([], document_id="doc-4")
        assert result.text == ""
        assert result.pages == 0
        assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# Handler – native extraction (moto for S3)
# ---------------------------------------------------------------------------

@mock_aws
class TestHandlerNative:
    """Handler tests for native (non-Textract) extraction paths."""

    def _setup_s3(self, bucket: str, key: str, body: bytes) -> None:
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=bucket)
        s3.put_object(Bucket=bucket, Key=key, Body=body)

    def _make_event(self, **overrides) -> dict:
        base = {
            "document_id": "doc-1",
            "s3_bucket": "test-bucket",
            "s3_key": "raw/file.txt",
            "file_type": "txt",
            "content_type": "native",
            "correlation_id": "corr-123",
        }
        base.update(overrides)
        return base

    def test_txt_extraction(self):
        content = "This is a long enough text for quality validation to pass easily."
        self._setup_s3("test-bucket", "raw/file.txt", content.encode())
        result = handler(self._make_event())
        assert result["valid"] is True
        assert result["extraction_method"] == "native"
        assert result["text"] == content
        assert result["confidence"] == 1.0
        assert result["correlation_id"] == "corr-123"
        assert result["output_key"].startswith("processed/text/")

    def test_csv_extraction(self):
        csv_data = "name,age\nAlice,30\nBob,25\nCharlie,35\nDiana,28\nEve,22"
        self._setup_s3("test-bucket", "raw/data.csv", csv_data.encode())
        result = handler(self._make_event(
            s3_key="raw/data.csv", file_type="csv",
        ))
        assert result["valid"] is True
        assert "Alice" in result["text"]

    def test_quality_below_threshold(self):
        self._setup_s3("test-bucket", "raw/tiny.txt", b"hi")
        result = handler(self._make_event(
            s3_key="raw/tiny.txt", file_type="txt",
        ))
        assert result["valid"] is False
        assert result["text"] == "hi"

    def test_custom_min_threshold(self):
        self._setup_s3("test-bucket", "raw/short.txt", b"hello")
        result = handler(self._make_event(
            s3_key="raw/short.txt", file_type="txt", min_threshold=3,
        ))
        assert result["valid"] is True

    def test_stored_text_in_s3(self):
        content = "Enough text content to pass the quality threshold for extraction."
        self._setup_s3("test-bucket", "raw/file.txt", content.encode())
        result = handler(self._make_event())
        # Verify the text was stored in S3
        s3 = boto3.client("s3", region_name="us-east-1")
        stored = s3.get_object(Bucket="test-bucket", Key=result["output_key"])
        assert stored["Body"].read().decode() == content

    def test_unknown_content_type(self):
        self._setup_s3("test-bucket", "raw/file.txt", b"data")
        result = handler(self._make_event(content_type="unknown"))
        assert result["valid"] is False
        assert "Unknown content_type" in result["error"]

    def test_correlation_id_generated_when_missing(self):
        content = "Enough text content to pass the quality threshold for extraction."
        self._setup_s3("test-bucket", "raw/file.txt", content.encode())
        event = self._make_event()
        del event["correlation_id"]
        result = handler(event)
        assert result["correlation_id"] is not None
        assert len(result["correlation_id"]) > 0


# ---------------------------------------------------------------------------
# Handler – Textract extraction (mocked)
# ---------------------------------------------------------------------------

class TestHandlerTextract:
    """Handler tests for Textract extraction paths."""

    @mock_aws
    def test_textract_sync_pdf(self):
        # Set up S3 with a small PDF-like object
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="test-bucket")
        s3.put_object(Bucket="test-bucket", Key="raw/doc.pdf", Body=b"x" * 1000)

        mock_textract = MagicMock()
        mock_textract.analyze_document.return_value = {
            "Blocks": [
                {"BlockType": "LINE", "Text": "Hello from PDF line one", "Page": 1, "Confidence": 99.5},
                {"BlockType": "LINE", "Text": "Hello from PDF line two is here", "Page": 1, "Confidence": 98.0},
            ]
        }

        with patch(
            "lambdas.ingestion.text_extractor.handler._get_textract_client",
            return_value=mock_textract,
        ):
            result = handler({
                "document_id": "doc-pdf",
                "s3_bucket": "test-bucket",
                "s3_key": "raw/doc.pdf",
                "file_type": "pdf",
                "content_type": "textract",
                "correlation_id": "corr-pdf",
            })

        assert result["valid"] is True
        assert result["extraction_method"] == "textract"
        assert "Hello from PDF" in result["text"]
        mock_textract.analyze_document.assert_called_once()

    @mock_aws
    def test_textract_sync_image(self):
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="test-bucket")
        s3.put_object(Bucket="test-bucket", Key="raw/photo.png", Body=b"\x89PNG" + b"x" * 100)

        mock_textract = MagicMock()
        mock_textract.detect_document_text.return_value = {
            "Blocks": [
                {"BlockType": "LINE", "Text": "Text from image that is long enough to pass the quality threshold", "Page": 1, "Confidence": 95.0},
            ]
        }

        with patch(
            "lambdas.ingestion.text_extractor.handler._get_textract_client",
            return_value=mock_textract,
        ):
            result = handler({
                "document_id": "doc-img",
                "s3_bucket": "test-bucket",
                "s3_key": "raw/photo.png",
                "file_type": "png",
                "content_type": "textract",
                "correlation_id": "corr-img",
            })

        assert result["valid"] is True
        mock_textract.detect_document_text.assert_called_once()

    @mock_aws
    def test_textract_async_large_pdf(self):
        """PDFs estimated at >15 pages should use async Textract."""
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="test-bucket")
        # ~16 pages at 50KB/page = 800KB
        s3.put_object(Bucket="test-bucket", Key="raw/big.pdf", Body=b"x" * 800_000)

        mock_textract = MagicMock()
        mock_textract.start_document_analysis.return_value = {"JobId": "job-123"}
        mock_textract.get_document_analysis.return_value = {
            "JobStatus": "SUCCEEDED",
            "Blocks": [
                {"BlockType": "LINE", "Text": "Async extracted text that is long enough to pass threshold", "Page": 1, "Confidence": 97.0},
            ],
        }

        with patch(
            "lambdas.ingestion.text_extractor.handler._get_textract_client",
            return_value=mock_textract,
        ):
            result = handler({
                "document_id": "doc-big",
                "s3_bucket": "test-bucket",
                "s3_key": "raw/big.pdf",
                "file_type": "pdf",
                "content_type": "textract",
                "correlation_id": "corr-big",
            })

        assert result["valid"] is True
        mock_textract.start_document_analysis.assert_called_once()

    @mock_aws
    def test_textract_error_returns_invalid(self):
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="test-bucket")
        s3.put_object(Bucket="test-bucket", Key="raw/bad.pdf", Body=b"x" * 100)

        mock_textract = MagicMock()
        mock_textract.analyze_document.side_effect = RuntimeError("Textract boom")

        with patch(
            "lambdas.ingestion.text_extractor.handler._get_textract_client",
            return_value=mock_textract,
        ):
            result = handler({
                "document_id": "doc-err",
                "s3_bucket": "test-bucket",
                "s3_key": "raw/bad.pdf",
                "file_type": "pdf",
                "content_type": "textract",
                "correlation_id": "corr-err",
            })

        assert result["valid"] is False
        assert "Textract boom" in result["error"]
