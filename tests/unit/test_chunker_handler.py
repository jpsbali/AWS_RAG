"""Unit tests for lambdas.ingestion.chunker.handler."""

from __future__ import annotations

import json
from unittest.mock import patch

import boto3
import pytest
from moto import mock_aws

from lambdas.ingestion.chunker.handler import (
    _chunk_fixed,
    _chunk_recursive,
    _chunk_semantic,
    _chunk_sentence_window,
    _token_count,
    chunk_text,
    handler,
)
from lambdas.shared.models import ChunkConfig


# ---------------------------------------------------------------------------
# _token_count
# ---------------------------------------------------------------------------

class TestTokenCount:
    def test_empty_string(self):
        assert _token_count("") == 0

    def test_single_word(self):
        assert _token_count("hello") == 1

    def test_multiple_words(self):
        assert _token_count("the quick brown fox") == 4

    def test_extra_whitespace(self):
        assert _token_count("  hello   world  ") == 2


# ---------------------------------------------------------------------------
# Fixed-size chunking
# ---------------------------------------------------------------------------

class TestFixedChunking:
    def _config(self, chunk_size: int = 10, overlap: int = 2) -> ChunkConfig:
        return ChunkConfig(strategy="fixed", chunk_size=chunk_size, overlap=overlap)

    def test_short_text_single_chunk(self):
        text = "one two three"
        chunks = _chunk_fixed(text, self._config(), "doc-1", [], [])
        assert len(chunks) == 1
        assert chunks[0].content == "one two three"
        assert chunks[0].chunk_index == 0
        assert chunks[0].document_id == "doc-1"

    def test_overlap_produces_overlapping_chunks(self):
        words = " ".join(f"w{i}" for i in range(20))
        chunks = _chunk_fixed(words, self._config(chunk_size=10, overlap=3), "doc-1", [], [])
        assert len(chunks) >= 2
        # Second chunk should start 7 words in (10 - 3 overlap)
        second_words = chunks[1].content.split()
        assert second_words[0] == "w7"

    def test_chunk_ids_sequential(self):
        words = " ".join(f"w{i}" for i in range(25))
        chunks = _chunk_fixed(words, self._config(chunk_size=10, overlap=0), "doc-1", [], [])
        for i, c in enumerate(chunks):
            assert c.chunk_index == i
            assert c.chunk_id == f"doc-1_chunk_{i}"

    def test_metadata_attached(self):
        text = "hello world foo bar"
        chunks = _chunk_fixed(text, self._config(), "doc-1", ["AWS"], ["cloud"])
        assert chunks[0].metadata.entities == ["AWS"]
        assert chunks[0].metadata.key_phrases == ["cloud"]

    def test_chunk_size_field(self):
        text = "one two three four five"
        chunks = _chunk_fixed(text, self._config(), "doc-1", [], [])
        assert chunks[0].chunk_size == 5


# ---------------------------------------------------------------------------
# Semantic chunking
# ---------------------------------------------------------------------------

class TestSemanticChunking:
    def _config(self, min_size: int = 2, max_size: int = 50) -> ChunkConfig:
        return ChunkConfig(
            strategy="semantic", min_chunk_size=min_size, max_chunk_size=max_size,
        )

    def test_splits_on_double_newline(self):
        text = "First paragraph with enough words.\n\nSecond paragraph with enough words."
        chunks = _chunk_semantic(text, self._config(min_size=2, max_size=10), "doc-1", [], [])
        assert len(chunks) >= 1

    def test_single_paragraph(self):
        text = "Just a single paragraph with some words."
        chunks = _chunk_semantic(text, self._config(), "doc-1", [], [])
        assert len(chunks) == 1
        assert "single paragraph" in chunks[0].content

    def test_respects_max_chunk_size(self):
        # Create text with two large paragraphs
        para1 = " ".join(f"word{i}" for i in range(20))
        para2 = " ".join(f"term{i}" for i in range(20))
        text = f"{para1}\n\n{para2}"
        chunks = _chunk_semantic(text, self._config(min_size=2, max_size=15), "doc-1", [], [])
        assert len(chunks) >= 2


# ---------------------------------------------------------------------------
# Recursive character chunking
# ---------------------------------------------------------------------------

class TestRecursiveChunking:
    def _config(self, chunk_size: int = 10) -> ChunkConfig:
        return ChunkConfig(strategy="recursive", chunk_size=chunk_size)

    def test_short_text_single_chunk(self):
        text = "hello world"
        chunks = _chunk_recursive(text, self._config(), "doc-1", [], [])
        assert len(chunks) == 1
        assert chunks[0].content == "hello world"

    def test_splits_on_paragraph_breaks(self):
        para1 = " ".join(f"a{i}" for i in range(8))
        para2 = " ".join(f"b{i}" for i in range(8))
        text = f"{para1}\n\n{para2}"
        chunks = _chunk_recursive(text, self._config(chunk_size=10), "doc-1", [], [])
        assert len(chunks) == 2

    def test_falls_through_separators(self):
        # Text with no paragraph breaks, only sentence breaks
        text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence here."
        chunks = _chunk_recursive(text, self._config(chunk_size=5), "doc-1", [], [])
        assert len(chunks) >= 2

    def test_chunk_ids_sequential(self):
        text = "a b c d e\n\nf g h i j"
        chunks = _chunk_recursive(text, self._config(chunk_size=6), "doc-1", [], [])
        for i, c in enumerate(chunks):
            assert c.chunk_index == i


# ---------------------------------------------------------------------------
# Sentence window chunking
# ---------------------------------------------------------------------------

class TestSentenceWindowChunking:
    def _config(self, window: int = 1) -> ChunkConfig:
        return ChunkConfig(strategy="sentence_window", window_size=window)

    def test_produces_one_chunk_per_sentence(self):
        text = "First sentence. Second sentence. Third sentence."
        chunks = _chunk_sentence_window(text, self._config(window=0), "doc-1", [], [])
        assert len(chunks) == 3

    def test_window_includes_neighbors(self):
        text = "Sentence A. Sentence B. Sentence C. Sentence D. Sentence E."
        chunks = _chunk_sentence_window(text, self._config(window=1), "doc-1", [], [])
        # Center sentence B (index 1) should include A, B, C
        assert "Sentence A." in chunks[1].content
        assert "Sentence B." in chunks[1].content
        assert "Sentence C." in chunks[1].content

    def test_window_clamps_at_boundaries(self):
        text = "First. Second. Third."
        chunks = _chunk_sentence_window(text, self._config(window=3), "doc-1", [], [])
        # First chunk window should not go negative
        assert len(chunks) == 3
        assert "First." in chunks[0].content

    def test_empty_text(self):
        chunks = _chunk_sentence_window("", self._config(), "doc-1", [], [])
        assert chunks == []


# ---------------------------------------------------------------------------
# chunk_text dispatcher
# ---------------------------------------------------------------------------

class TestChunkText:
    def test_dispatches_fixed(self):
        config = ChunkConfig(strategy="fixed", chunk_size=100, overlap=0)
        chunks = chunk_text("hello world", config, "doc-1")
        assert len(chunks) == 1

    def test_dispatches_semantic(self):
        config = ChunkConfig(strategy="semantic")
        chunks = chunk_text("hello world", config, "doc-1")
        assert len(chunks) >= 1

    def test_dispatches_recursive(self):
        config = ChunkConfig(strategy="recursive", chunk_size=100)
        chunks = chunk_text("hello world", config, "doc-1")
        assert len(chunks) == 1

    def test_dispatches_sentence_window(self):
        config = ChunkConfig(strategy="sentence_window", window_size=1)
        chunks = chunk_text("Hello there. How are you.", config, "doc-1")
        assert len(chunks) == 2

    def test_unknown_strategy_raises(self):
        config = ChunkConfig(strategy="unknown")
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            chunk_text("text", config, "doc-1")

    def test_entities_and_key_phrases_passed(self):
        config = ChunkConfig(strategy="fixed", chunk_size=100)
        chunks = chunk_text("hello world", config, "doc-1", ["AWS"], ["cloud"])
        assert chunks[0].metadata.entities == ["AWS"]
        assert chunks[0].metadata.key_phrases == ["cloud"]


# ---------------------------------------------------------------------------
# Handler tests — inline text
# ---------------------------------------------------------------------------

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
            "text": "The quick brown fox jumps over the lazy dog. " * 20,
            "correlation_id": "corr-123",
            "strategy": "fixed",
            "entities": ["fox"],
            "key_phrases": ["quick brown"],
        }
        base.update(overrides)
        return base

    def test_basic_chunking(self):
        self._setup_s3()
        result = handler(self._make_event())
        assert result["document_id"] == "doc-1"
        assert result["chunk_count"] > 0
        assert result["output_key"].startswith("processed/chunks/")
        assert result["correlation_id"] == "corr-123"
        assert "error" not in result

    def test_chunks_stored_in_s3(self):
        self._setup_s3()
        result = handler(self._make_event())
        s3 = boto3.client("s3", region_name="us-east-1")
        obj = s3.get_object(Bucket="test-bucket", Key=result["output_key"])
        stored = json.loads(obj["Body"].read().decode())
        assert isinstance(stored, list)
        assert len(stored) == result["chunk_count"]
        # Verify chunk structure
        first = stored[0]
        assert "chunk_id" in first
        assert "document_id" in first
        assert "content" in first
        assert "metadata" in first

    def test_default_strategy_is_fixed(self):
        self._setup_s3()
        event = self._make_event()
        del event["strategy"]
        result = handler(event)
        assert result["chunk_count"] > 0

    def test_semantic_strategy(self):
        self._setup_s3()
        result = handler(self._make_event(strategy="semantic"))
        assert result["chunk_count"] > 0

    def test_recursive_strategy(self):
        self._setup_s3()
        result = handler(self._make_event(strategy="recursive"))
        assert result["chunk_count"] > 0

    def test_sentence_window_strategy(self):
        self._setup_s3()
        result = handler(self._make_event(strategy="sentence_window"))
        assert result["chunk_count"] > 0

    def test_chunk_config_overrides(self):
        self._setup_s3()
        result = handler(self._make_event(
            chunk_config={"chunk_size": 5, "overlap": 1},
        ))
        assert result["chunk_count"] > 1

    def test_correlation_id_generated_when_missing(self):
        self._setup_s3()
        event = self._make_event()
        del event["correlation_id"]
        result = handler(event)
        assert result["correlation_id"] is not None
        assert len(result["correlation_id"]) > 0

    def test_correlation_id_passthrough(self):
        self._setup_s3()
        result = handler(self._make_event(correlation_id="my-id"))
        assert result["correlation_id"] == "my-id"


# ---------------------------------------------------------------------------
# Handler tests — text loaded from S3
# ---------------------------------------------------------------------------

@mock_aws
class TestHandlerS3Text:
    """Handler tests when text is loaded from S3 via s3_key."""

    def _setup_s3_with_text(
        self, bucket: str = "test-bucket", key: str = "processed/text/doc-2.txt",
    ) -> None:
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=bucket)
        s3.put_object(
            Bucket=bucket, Key=key,
            Body=b"Sentence one about AWS. Sentence two about cloud. Sentence three about data.",
        )

    def test_loads_text_from_s3(self):
        self._setup_s3_with_text()
        result = handler({
            "document_id": "doc-2",
            "s3_bucket": "test-bucket",
            "s3_key": "processed/text/doc-2.txt",
            "correlation_id": "corr-s3",
        })
        assert result["document_id"] == "doc-2"
        assert result["chunk_count"] > 0
        assert "error" not in result

    def test_missing_text_and_s3_key(self):
        result = handler({
            "document_id": "doc-3",
            "s3_bucket": "test-bucket",
            "correlation_id": "corr-err",
        })
        assert "error" in result
        assert result["chunk_count"] == 0

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
        assert result["chunk_count"] == 0


# ---------------------------------------------------------------------------
# Handler tests — error handling
# ---------------------------------------------------------------------------

@mock_aws
class TestHandlerErrors:
    def _setup_s3(self, bucket: str = "test-bucket") -> None:
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=bucket)

    def test_invalid_strategy(self):
        self._setup_s3()
        result = handler({
            "document_id": "doc-err",
            "s3_bucket": "test-bucket",
            "text": "Some text to chunk",
            "strategy": "nonexistent",
            "correlation_id": "corr-fail",
        })
        assert "error" in result
        assert result["chunk_count"] == 0

    def test_empty_text_produces_no_chunks(self):
        self._setup_s3()
        result = handler({
            "document_id": "doc-empty",
            "s3_bucket": "test-bucket",
            "text": "",
            "s3_key": "",
            "correlation_id": "corr-empty",
        })
        # No text and no s3_key → error
        assert "error" in result
        assert result["chunk_count"] == 0
