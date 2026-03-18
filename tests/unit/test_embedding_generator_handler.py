"""Unit tests for lambdas.ingestion.embedding_generator.handler."""

from __future__ import annotations

import json
import math
from unittest.mock import MagicMock, patch

import boto3
import fakeredis
import pytest
from moto import mock_aws

from lambdas.ingestion.embedding_generator.handler import (
    MAX_RETRIES,
    VALID_DIMENSIONS,
    _cache_key,
    _invoke_bedrock,
    generate_embeddings_for_chunks,
    get_cached_embedding,
    handler,
    normalize_embedding,
    set_cached_embedding,
)
from lambdas.shared.constants import (
    EMBEDDING_CACHE_PREFIX,
    EMBEDDING_CACHE_TTL,
    MAX_EMBEDDING_BATCH_SIZE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_embedding(dims: int = 1024) -> list[float]:
    """Return a simple non-normalised embedding vector."""
    return [1.0 / (i + 1) for i in range(dims)]


def _make_bedrock_response(embedding: list[float]) -> dict:
    """Build a mock Bedrock invoke_model return value."""
    return {"body": json.dumps({"embedding": embedding})}


def _make_chunks(n: int = 3) -> list[dict]:
    """Return *n* minimal chunk dicts."""
    return [
        {"chunk_id": f"doc-1_chunk_{i}", "content": f"chunk content number {i}"}
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# normalize_embedding
# ---------------------------------------------------------------------------

class TestNormalizeEmbedding:
    def test_unit_vector_unchanged(self):
        vec = [1.0, 0.0, 0.0]
        result = normalize_embedding(vec)
        assert result == pytest.approx([1.0, 0.0, 0.0])

    def test_l2_norm_is_one(self):
        vec = [3.0, 4.0]
        result = normalize_embedding(vec)
        norm = math.sqrt(sum(v * v for v in result))
        assert norm == pytest.approx(1.0)

    def test_high_dimensional(self):
        vec = _fake_embedding(1024)
        result = normalize_embedding(vec)
        norm = math.sqrt(sum(v * v for v in result))
        assert norm == pytest.approx(1.0)
        assert len(result) == 1024

    def test_zero_vector_unchanged(self):
        vec = [0.0, 0.0, 0.0]
        result = normalize_embedding(vec)
        assert result == [0.0, 0.0, 0.0]

    def test_negative_values(self):
        vec = [-3.0, 4.0]
        result = normalize_embedding(vec)
        norm = math.sqrt(sum(v * v for v in result))
        assert norm == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

class TestCacheHelpers:
    def setup_method(self):
        self.redis = fakeredis.FakeRedis(decode_responses=True)

    def test_cache_key_format(self):
        key = _cache_key("abc123")
        assert key == f"{EMBEDDING_CACHE_PREFIX}abc123"

    def test_set_and_get_cached_embedding(self):
        emb = [0.1, 0.2, 0.3]
        set_cached_embedding(self.redis, "hash1", emb)
        result = get_cached_embedding(self.redis, "hash1")
        assert result == emb

    def test_get_returns_none_on_miss(self):
        result = get_cached_embedding(self.redis, "nonexistent")
        assert result is None

    def test_get_returns_none_when_redis_is_none(self):
        result = get_cached_embedding(None, "hash1")
        assert result is None

    def test_set_noop_when_redis_is_none(self):
        # Should not raise
        set_cached_embedding(None, "hash1", [0.1])

    def test_ttl_is_set(self):
        set_cached_embedding(self.redis, "hash_ttl", [1.0])
        ttl = self.redis.ttl(_cache_key("hash_ttl"))
        assert ttl > 0
        assert ttl <= EMBEDDING_CACHE_TTL

    def test_get_handles_redis_error(self):
        broken = MagicMock()
        broken.get.side_effect = Exception("connection lost")
        result = get_cached_embedding(broken, "hash1")
        assert result is None

    def test_set_handles_redis_error(self):
        broken = MagicMock()
        broken.setex.side_effect = Exception("connection lost")
        # Should not raise
        set_cached_embedding(broken, "hash1", [0.1])


# ---------------------------------------------------------------------------
# _invoke_bedrock
# ---------------------------------------------------------------------------

class TestInvokeBedrock:
    def test_returns_normalised_embedding(self):
        raw = [3.0, 4.0]
        client = MagicMock()
        client.invoke_model.return_value = _make_bedrock_response(raw)
        result = _invoke_bedrock(client, "hello", 1024, "search_document")
        norm = math.sqrt(sum(v * v for v in result))
        assert norm == pytest.approx(1.0)

    @patch("lambdas.ingestion.embedding_generator.handler.time.sleep")
    def test_retries_on_throttling(self, mock_sleep):
        client = MagicMock()
        throttle_exc = Exception("ThrottlingException")
        client.invoke_model.side_effect = [
            throttle_exc,
            _make_bedrock_response([1.0, 0.0]),
        ]
        result = _invoke_bedrock(client, "hello", 1024, "search_document")
        assert len(result) == 2
        assert mock_sleep.call_count == 1

    @patch("lambdas.ingestion.embedding_generator.handler.time.sleep")
    def test_raises_after_max_retries(self, mock_sleep):
        client = MagicMock()
        client.invoke_model.side_effect = Exception("Throttling error")
        with pytest.raises(RuntimeError, match="throttled after"):
            _invoke_bedrock(client, "hello", 1024, "search_document")
        assert mock_sleep.call_count == MAX_RETRIES

    def test_non_throttle_error_raises_immediately(self):
        client = MagicMock()
        client.invoke_model.side_effect = ValueError("bad input")
        with pytest.raises(ValueError, match="bad input"):
            _invoke_bedrock(client, "hello", 1024, "search_document")


# ---------------------------------------------------------------------------
# generate_embeddings_for_chunks
# ---------------------------------------------------------------------------

class TestGenerateEmbeddingsForChunks:
    def test_all_cache_misses(self):
        chunks = _make_chunks(3)
        bedrock = MagicMock()
        bedrock.invoke_model.return_value = _make_bedrock_response(_fake_embedding(4))

        results, cached = generate_embeddings_for_chunks(
            chunks, 1024, "search_document",
            redis_client=None, bedrock_client=bedrock,
        )
        assert len(results) == 3
        assert cached == 0
        assert all(not r.cached for r in results)
        assert bedrock.invoke_model.call_count == 3

    def test_all_cache_hits(self):
        redis = fakeredis.FakeRedis(decode_responses=True)
        chunks = _make_chunks(2)
        emb = [0.5, 0.5]

        # Pre-populate cache
        from lambdas.shared.utils import compute_hash
        for chunk in chunks:
            h = compute_hash(chunk["content"])
            set_cached_embedding(redis, h, emb)

        bedrock = MagicMock()
        results, cached = generate_embeddings_for_chunks(
            chunks, 1024, "search_document",
            redis_client=redis, bedrock_client=bedrock,
        )
        assert len(results) == 2
        assert cached == 2
        assert all(r.cached for r in results)
        bedrock.invoke_model.assert_not_called()

    def test_mixed_cache_hits_and_misses(self):
        redis = fakeredis.FakeRedis(decode_responses=True)
        chunks = _make_chunks(3)
        emb = [0.5, 0.5]

        # Cache only the first chunk
        from lambdas.shared.utils import compute_hash
        h = compute_hash(chunks[0]["content"])
        set_cached_embedding(redis, h, emb)

        bedrock = MagicMock()
        bedrock.invoke_model.return_value = _make_bedrock_response(_fake_embedding(4))

        results, cached = generate_embeddings_for_chunks(
            chunks, 1024, "search_document",
            redis_client=redis, bedrock_client=bedrock,
        )
        assert len(results) == 3
        assert cached == 1
        assert results[0].cached is True
        assert results[1].cached is False
        assert results[2].cached is False
        assert bedrock.invoke_model.call_count == 2

    def test_caches_new_embeddings(self):
        redis = fakeredis.FakeRedis(decode_responses=True)
        chunks = _make_chunks(1)
        bedrock = MagicMock()
        bedrock.invoke_model.return_value = _make_bedrock_response([1.0, 0.0])

        generate_embeddings_for_chunks(
            chunks, 1024, "search_document",
            redis_client=redis, bedrock_client=bedrock,
        )

        # Verify it was cached
        from lambdas.shared.utils import compute_hash
        h = compute_hash(chunks[0]["content"])
        cached = get_cached_embedding(redis, h)
        assert cached is not None

    def test_batching_respects_max_size(self):
        """Chunks > MAX_EMBEDDING_BATCH_SIZE are processed in multiple batches."""
        n = MAX_EMBEDDING_BATCH_SIZE + 10
        chunks = _make_chunks(n)
        bedrock = MagicMock()
        bedrock.invoke_model.return_value = _make_bedrock_response([1.0])

        results, cached = generate_embeddings_for_chunks(
            chunks, 1024, "search_document",
            redis_client=None, bedrock_client=bedrock,
        )
        assert len(results) == n
        assert bedrock.invoke_model.call_count == n

    def test_chunk_ids_preserved(self):
        chunks = _make_chunks(2)
        bedrock = MagicMock()
        bedrock.invoke_model.return_value = _make_bedrock_response([1.0])

        results, _ = generate_embeddings_for_chunks(
            chunks, 1024, "search_document",
            redis_client=None, bedrock_client=bedrock,
        )
        assert results[0].chunk_id == "doc-1_chunk_0"
        assert results[1].chunk_id == "doc-1_chunk_1"


# ---------------------------------------------------------------------------
# Handler integration tests
# ---------------------------------------------------------------------------

@mock_aws
class TestHandler:
    """End-to-end handler tests with mocked Bedrock and fakeredis."""

    def _setup_s3_with_chunks(
        self,
        bucket: str = "test-bucket",
        key: str = "processed/chunks/doc-1.json",
        chunks: list[dict] | None = None,
    ) -> None:
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=bucket)
        data = _make_chunks(3) if chunks is None else chunks
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(data).encode("utf-8"),
        )

    def _make_event(self, **overrides) -> dict:
        base = {
            "document_id": "doc-1",
            "s3_bucket": "test-bucket",
            "chunks_key": "processed/chunks/doc-1.json",
            "correlation_id": "corr-123",
        }
        base.update(overrides)
        return base

    @patch("lambdas.ingestion.embedding_generator.handler._get_redis_client", return_value=None)
    @patch("lambdas.ingestion.embedding_generator.handler._get_bedrock_client")
    def test_basic_embedding_generation(self, mock_bedrock_factory, mock_redis_factory):
        self._setup_s3_with_chunks()
        bedrock = MagicMock()
        bedrock.invoke_model.return_value = _make_bedrock_response(_fake_embedding(4))
        mock_bedrock_factory.return_value = bedrock

        result = handler(self._make_event())

        assert result["document_id"] == "doc-1"
        assert result["embedding_count"] == 3
        assert result["cached_count"] == 0
        assert result["output_key"].startswith("processed/embeddings/")
        assert result["correlation_id"] == "corr-123"
        assert "error" not in result

    @patch("lambdas.ingestion.embedding_generator.handler._get_redis_client", return_value=None)
    @patch("lambdas.ingestion.embedding_generator.handler._get_bedrock_client")
    def test_embeddings_stored_in_s3(self, mock_bedrock_factory, mock_redis_factory):
        self._setup_s3_with_chunks()
        bedrock = MagicMock()
        bedrock.invoke_model.return_value = _make_bedrock_response([3.0, 4.0])
        mock_bedrock_factory.return_value = bedrock

        result = handler(self._make_event())

        s3 = boto3.client("s3", region_name="us-east-1")
        obj = s3.get_object(Bucket="test-bucket", Key=result["output_key"])
        stored = json.loads(obj["Body"].read().decode())
        assert isinstance(stored, list)
        assert len(stored) == 3
        # Each entry should have embedding and cached fields
        assert "embedding" in stored[0]
        assert "cached" in stored[0]

    @patch("lambdas.ingestion.embedding_generator.handler._get_redis_client", return_value=None)
    @patch("lambdas.ingestion.embedding_generator.handler._get_bedrock_client")
    def test_custom_dimensions(self, mock_bedrock_factory, mock_redis_factory):
        self._setup_s3_with_chunks()
        bedrock = MagicMock()
        bedrock.invoke_model.return_value = _make_bedrock_response(_fake_embedding(4))
        mock_bedrock_factory.return_value = bedrock

        result = handler(self._make_event(dimensions=256))
        assert result["embedding_count"] == 3
        assert "error" not in result

    @patch("lambdas.ingestion.embedding_generator.handler._get_redis_client", return_value=None)
    def test_invalid_dimensions(self, mock_redis_factory):
        self._setup_s3_with_chunks()
        result = handler(self._make_event(dimensions=999))
        assert "error" in result
        assert result["embedding_count"] == 0

    @patch("lambdas.ingestion.embedding_generator.handler._get_redis_client", return_value=None)
    def test_missing_chunks_key(self, mock_redis_factory):
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="test-bucket")
        result = handler(self._make_event(chunks_key="nonexistent.json"))
        assert "error" in result
        assert result["embedding_count"] == 0

    @patch("lambdas.ingestion.embedding_generator.handler._get_redis_client", return_value=None)
    @patch("lambdas.ingestion.embedding_generator.handler._get_bedrock_client")
    def test_empty_chunks(self, mock_bedrock_factory, mock_redis_factory):
        self._setup_s3_with_chunks(chunks=[])
        bedrock = MagicMock()
        mock_bedrock_factory.return_value = bedrock

        result = handler(self._make_event())
        assert result["embedding_count"] == 0
        assert result["cached_count"] == 0
        bedrock.invoke_model.assert_not_called()

    @patch("lambdas.ingestion.embedding_generator.handler._get_redis_client")
    @patch("lambdas.ingestion.embedding_generator.handler._get_bedrock_client")
    def test_with_redis_caching(self, mock_bedrock_factory, mock_redis_factory):
        self._setup_s3_with_chunks()
        redis = fakeredis.FakeRedis(decode_responses=True)
        mock_redis_factory.return_value = redis

        bedrock = MagicMock()
        bedrock.invoke_model.return_value = _make_bedrock_response([1.0, 0.0])
        mock_bedrock_factory.return_value = bedrock

        # First call — all cache misses
        result1 = handler(self._make_event())
        assert result1["cached_count"] == 0
        assert result1["embedding_count"] == 3

        # Second call — all cache hits
        result2 = handler(self._make_event())
        assert result2["cached_count"] == 3
        assert result2["embedding_count"] == 3

    @patch("lambdas.ingestion.embedding_generator.handler._get_redis_client", return_value=None)
    @patch("lambdas.ingestion.embedding_generator.handler._get_bedrock_client")
    def test_correlation_id_generated_when_missing(self, mock_bedrock_factory, mock_redis_factory):
        self._setup_s3_with_chunks()
        bedrock = MagicMock()
        bedrock.invoke_model.return_value = _make_bedrock_response([1.0])
        mock_bedrock_factory.return_value = bedrock

        event = self._make_event()
        del event["correlation_id"]
        result = handler(event)
        assert result["correlation_id"] is not None
        assert len(result["correlation_id"]) > 0

    @patch("lambdas.ingestion.embedding_generator.handler._get_redis_client", return_value=None)
    @patch("lambdas.ingestion.embedding_generator.handler._get_bedrock_client")
    def test_default_input_type(self, mock_bedrock_factory, mock_redis_factory):
        self._setup_s3_with_chunks(chunks=_make_chunks(1))
        bedrock = MagicMock()
        bedrock.invoke_model.return_value = _make_bedrock_response([1.0])
        mock_bedrock_factory.return_value = bedrock

        handler(self._make_event())

        # Verify the Bedrock call used the correct body
        call_kwargs = bedrock.invoke_model.call_args
        body = json.loads(call_kwargs.kwargs.get("body") or call_kwargs[1].get("body"))
        assert body["normalize"] is True
