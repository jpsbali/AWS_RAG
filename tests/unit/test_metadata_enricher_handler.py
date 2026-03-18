"""Unit tests for lambdas.ingestion.metadata_enricher.handler."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import boto3
import pytest
from moto import mock_aws

from lambdas.ingestion.metadata_enricher.handler import (
    _detect_dominant_language,
    _detect_entities,
    _detect_key_phrases,
    _detect_pii_entities,
    _truncate_for_comprehend,
    handler,
)
from lambdas.shared.models import Entity


# ---------------------------------------------------------------------------
# _truncate_for_comprehend
# ---------------------------------------------------------------------------

class TestTruncateForComprehend:
    def test_short_text_unchanged(self):
        text = "Hello world"
        assert _truncate_for_comprehend(text) == text

    def test_exact_limit(self):
        text = "a" * 5000
        assert _truncate_for_comprehend(text) == text

    def test_over_limit_truncated(self):
        text = "a" * 6000
        result = _truncate_for_comprehend(text)
        assert len(result.encode("utf-8")) <= 5000

    def test_multibyte_chars(self):
        # Each emoji is 4 bytes in UTF-8
        text = "\U0001f600" * 2000  # 8000 bytes
        result = _truncate_for_comprehend(text)
        assert len(result.encode("utf-8")) <= 5000


# ---------------------------------------------------------------------------
# _detect_dominant_language
# ---------------------------------------------------------------------------

class TestDetectDominantLanguage:
    def test_returns_top_language(self):
        mock_comprehend = MagicMock()
        mock_comprehend.detect_dominant_language.return_value = {
            "Languages": [
                {"LanguageCode": "en", "Score": 0.95},
                {"LanguageCode": "fr", "Score": 0.03},
            ]
        }
        assert _detect_dominant_language("Hello world", mock_comprehend) == "en"

    def test_fallback_when_empty(self):
        mock_comprehend = MagicMock()
        mock_comprehend.detect_dominant_language.return_value = {"Languages": []}
        assert _detect_dominant_language("", mock_comprehend) == "en"

    def test_picks_highest_score(self):
        mock_comprehend = MagicMock()
        mock_comprehend.detect_dominant_language.return_value = {
            "Languages": [
                {"LanguageCode": "es", "Score": 0.80},
                {"LanguageCode": "de", "Score": 0.90},
            ]
        }
        assert _detect_dominant_language("text", mock_comprehend) == "de"


# ---------------------------------------------------------------------------
# _detect_entities
# ---------------------------------------------------------------------------

class TestDetectEntities:
    def test_returns_entities(self):
        mock_comprehend = MagicMock()
        mock_comprehend.detect_entities.return_value = {
            "Entities": [
                {"Text": "Amazon", "Type": "ORGANIZATION", "Score": 0.9876},
                {"Text": "Seattle", "Type": "LOCATION", "Score": 0.9543},
            ]
        }
        result = _detect_entities("Amazon is in Seattle", "en", mock_comprehend)
        assert len(result) == 2
        assert isinstance(result[0], Entity)
        assert result[0].text == "Amazon"
        assert result[0].type == "ORGANIZATION"
        assert result[1].text == "Seattle"

    def test_empty_entities(self):
        mock_comprehend = MagicMock()
        mock_comprehend.detect_entities.return_value = {"Entities": []}
        result = _detect_entities("no entities here", "en", mock_comprehend)
        assert result == []


# ---------------------------------------------------------------------------
# _detect_key_phrases
# ---------------------------------------------------------------------------

class TestDetectKeyPhrases:
    def test_returns_phrases(self):
        mock_comprehend = MagicMock()
        mock_comprehend.detect_key_phrases.return_value = {
            "KeyPhrases": [
                {"Text": "machine learning", "Score": 0.99},
                {"Text": "cloud computing", "Score": 0.95},
            ]
        }
        result = _detect_key_phrases("machine learning and cloud computing", "en", mock_comprehend)
        assert result == ["machine learning", "cloud computing"]

    def test_empty_phrases(self):
        mock_comprehend = MagicMock()
        mock_comprehend.detect_key_phrases.return_value = {"KeyPhrases": []}
        result = _detect_key_phrases("", "en", mock_comprehend)
        assert result == []


# ---------------------------------------------------------------------------
# _detect_pii_entities
# ---------------------------------------------------------------------------

class TestDetectPiiEntities:
    def test_returns_pii(self):
        mock_comprehend = MagicMock()
        mock_comprehend.detect_pii_entities.return_value = {
            "Entities": [
                {"Type": "EMAIL", "Score": 0.9999, "BeginOffset": 0, "EndOffset": 15},
            ]
        }
        result = _detect_pii_entities("test@example.com", "en", mock_comprehend)
        assert len(result) == 1
        assert result[0]["type"] == "EMAIL"
        assert result[0]["begin_offset"] == 0
        assert result[0]["end_offset"] == 15

    def test_no_pii(self):
        mock_comprehend = MagicMock()
        mock_comprehend.detect_pii_entities.return_value = {"Entities": []}
        result = _detect_pii_entities("no pii here", "en", mock_comprehend)
        assert result == []


# ---------------------------------------------------------------------------
# Handler tests — inline text
# ---------------------------------------------------------------------------

def _make_comprehend_mock(
    language: str = "en",
    entities: list | None = None,
    key_phrases: list | None = None,
    pii_entities: list | None = None,
) -> MagicMock:
    """Build a mock Comprehend client with configurable responses."""
    mock = MagicMock()
    mock.detect_dominant_language.return_value = {
        "Languages": [{"LanguageCode": language, "Score": 0.99}]
    }
    mock.detect_entities.return_value = {
        "Entities": entities
        or [
            {"Text": "AWS", "Type": "ORGANIZATION", "Score": 0.95},
        ]
    }
    mock.detect_key_phrases.return_value = {
        "KeyPhrases": key_phrases
        or [
            {"Text": "cloud services", "Score": 0.92},
        ]
    }
    mock.detect_pii_entities.return_value = {
        "Entities": pii_entities or []
    }
    return mock


@mock_aws
class TestHandlerInlineText:
    """Handler tests when text is provided inline in the event."""

    def _setup_s3(self, bucket: str = "test-bucket") -> None:
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=bucket)

    def _make_event(self, **overrides) -> dict:
        base = {
            "document_id": "doc-1",
            "s3_bucket": "test-bucket",
            "text": "AWS provides cloud services in Seattle and worldwide.",
            "correlation_id": "corr-123",
        }
        base.update(overrides)
        return base

    def test_basic_enrichment(self):
        self._setup_s3()
        mock_comprehend = _make_comprehend_mock()

        with patch(
            "lambdas.ingestion.metadata_enricher.handler._get_comprehend_client",
            return_value=mock_comprehend,
        ):
            result = handler(self._make_event())

        assert result["document_id"] == "doc-1"
        assert result["language"] == "en"
        assert len(result["entities"]) == 1
        assert result["entities"][0]["text"] == "AWS"
        assert result["key_phrases"] == ["cloud services"]
        assert result["pii_detected"] is False
        assert result["pii_entities"] == []
        assert result["correlation_id"] == "corr-123"
        assert result["output_key"].startswith("processed/metadata/")

    def test_pii_detection_enabled_with_pii(self):
        self._setup_s3()
        mock_comprehend = _make_comprehend_mock(
            pii_entities=[
                {"Type": "EMAIL", "Score": 0.99, "BeginOffset": 0, "EndOffset": 10},
            ]
        )

        with patch(
            "lambdas.ingestion.metadata_enricher.handler._get_comprehend_client",
            return_value=mock_comprehend,
        ):
            result = handler(self._make_event(pii_detection=True))

        assert result["pii_detected"] is True
        assert len(result["pii_entities"]) == 1
        assert result["pii_entities"][0]["type"] == "EMAIL"
        mock_comprehend.detect_pii_entities.assert_called_once()

    def test_pii_detection_disabled(self):
        self._setup_s3()
        mock_comprehend = _make_comprehend_mock()

        with patch(
            "lambdas.ingestion.metadata_enricher.handler._get_comprehend_client",
            return_value=mock_comprehend,
        ):
            result = handler(self._make_event(pii_detection=False))

        assert result["pii_detected"] is False
        mock_comprehend.detect_pii_entities.assert_not_called()

    def test_metadata_stored_in_s3(self):
        self._setup_s3()
        mock_comprehend = _make_comprehend_mock()

        with patch(
            "lambdas.ingestion.metadata_enricher.handler._get_comprehend_client",
            return_value=mock_comprehend,
        ):
            result = handler(self._make_event())

        # Verify metadata was stored in S3
        s3 = boto3.client("s3", region_name="us-east-1")
        stored = s3.get_object(Bucket="test-bucket", Key=result["output_key"])
        stored_data = json.loads(stored["Body"].read().decode())
        assert stored_data["document_id"] == "doc-1"
        assert stored_data["language"] == "en"

    def test_correlation_id_generated_when_missing(self):
        self._setup_s3()
        mock_comprehend = _make_comprehend_mock()

        with patch(
            "lambdas.ingestion.metadata_enricher.handler._get_comprehend_client",
            return_value=mock_comprehend,
        ):
            event = self._make_event()
            del event["correlation_id"]
            result = handler(event)

        assert result["correlation_id"] is not None
        assert len(result["correlation_id"]) > 0

    def test_correlation_id_passthrough(self):
        self._setup_s3()
        mock_comprehend = _make_comprehend_mock()

        with patch(
            "lambdas.ingestion.metadata_enricher.handler._get_comprehend_client",
            return_value=mock_comprehend,
        ):
            result = handler(self._make_event(correlation_id="my-custom-id"))

        assert result["correlation_id"] == "my-custom-id"


# ---------------------------------------------------------------------------
# Handler tests — text loaded from S3
# ---------------------------------------------------------------------------

@mock_aws
class TestHandlerS3Text:
    """Handler tests when text is loaded from S3 via s3_key."""

    def _setup_s3_with_text(
        self, bucket: str = "test-bucket", key: str = "processed/text/doc-2.txt"
    ) -> None:
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=bucket)
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=b"Amazon Web Services provides cloud computing solutions.",
        )

    def test_loads_text_from_s3(self):
        self._setup_s3_with_text()
        mock_comprehend = _make_comprehend_mock()

        with patch(
            "lambdas.ingestion.metadata_enricher.handler._get_comprehend_client",
            return_value=mock_comprehend,
        ):
            result = handler({
                "document_id": "doc-2",
                "s3_bucket": "test-bucket",
                "s3_key": "processed/text/doc-2.txt",
                "correlation_id": "corr-s3",
            })

        assert result["document_id"] == "doc-2"
        assert result["language"] == "en"
        assert "error" not in result

    def test_missing_s3_key_and_text(self):
        result = handler({
            "document_id": "doc-3",
            "s3_bucket": "test-bucket",
            "correlation_id": "corr-err",
        })
        assert "error" in result
        assert "Either 'text' or 's3_key'" in result["error"]

    def test_s3_key_not_found(self):
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="test-bucket")

        result = handler({
            "document_id": "doc-4",
            "s3_bucket": "test-bucket",
            "s3_key": "processed/text/nonexistent.txt",
            "correlation_id": "corr-miss",
        })
        assert "error" in result
        assert result["entities"] == []


# ---------------------------------------------------------------------------
# Handler tests — error handling
# ---------------------------------------------------------------------------

@mock_aws
class TestHandlerErrors:
    """Handler tests for Comprehend error scenarios."""

    def _setup_s3(self, bucket: str = "test-bucket") -> None:
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=bucket)

    def test_comprehend_failure(self):
        self._setup_s3()
        mock_comprehend = MagicMock()
        mock_comprehend.detect_dominant_language.side_effect = RuntimeError("Comprehend down")

        with patch(
            "lambdas.ingestion.metadata_enricher.handler._get_comprehend_client",
            return_value=mock_comprehend,
        ):
            result = handler({
                "document_id": "doc-err",
                "s3_bucket": "test-bucket",
                "text": "Some text to process",
                "correlation_id": "corr-fail",
            })

        assert "error" in result
        assert "Comprehend" in result["error"]
        assert result["entities"] == []
        assert result["language"] == ""

    def test_pii_detection_default_false(self):
        self._setup_s3()
        mock_comprehend = _make_comprehend_mock()

        with patch(
            "lambdas.ingestion.metadata_enricher.handler._get_comprehend_client",
            return_value=mock_comprehend,
        ):
            result = handler({
                "document_id": "doc-default",
                "s3_bucket": "test-bucket",
                "text": "Some text",
            })

        assert result["pii_detected"] is False
        mock_comprehend.detect_pii_entities.assert_not_called()
