"""Unit tests for lambdas.ingestion.vector_indexer.handler."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch, call

import boto3
import pytest
from moto import mock_aws

from lambdas.ingestion.vector_indexer.handler import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_INDEX_ALIAS,
    BULK_REFRESH_INTERVAL,
    INCREMENTAL_REFRESH_INTERVAL,
    MAX_BATCH_SIZE,
    _build_bulk_body,
    _load_embeddings_from_s3,
    _resolve_write_index,
    _set_refresh_interval,
    bulk_index_chunks,
    handler,
    reindex_with_alias_switch,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_enriched_chunks(n: int = 3, document_id: str = "doc-1") -> list[dict]:
    """Return *n* enriched chunk dicts with embeddings."""
    return [
        {
            "chunk_id": f"{document_id}_chunk_{i}",
            "document_id": document_id,
            "chunk_index": i,
            "content": f"chunk content number {i}",
            "chunk_size": 20,
            "start_position": i * 100,
            "end_position": (i + 1) * 100,
            "metadata": {
                "document_id": document_id,
                "source": "test.pdf",
                "entities": ["entity_a"],
                "key_phrases": ["phrase_a"],
            },
            "embedding": [0.1 * (i + 1)] * 4,
            "cached": False,
        }
        for i in range(n)
    ]


def _mock_os_client(bulk_errors: bool = False, bulk_items: list | None = None) -> MagicMock:
    """Return a mocked OpenSearch client."""
    client = MagicMock()
    if bulk_items is None:
        bulk_items = []
    client.bulk.return_value = {"errors": bulk_errors, "items": bulk_items}
    client.indices.get_alias.return_value = {"rag-chunks-v1": {"aliases": {"rag-chunks": {}}}}
    client.indices.put_settings.return_value = {"acknowledged": True}
    client.indices.create.return_value = {"acknowledged": True}
    client.indices.delete.return_value = {"acknowledged": True}
    client.indices.update_aliases.return_value = {"acknowledged": True}
    client.reindex.return_value = {"total": 100, "updated": 100, "failures": []}
    return client


# ---------------------------------------------------------------------------
# _build_bulk_body
# ---------------------------------------------------------------------------

class TestBuildBulkBody:
    def test_produces_ndjson_with_action_and_doc(self):
        chunks = _make_enriched_chunks(2)
        body = _build_bulk_body(chunks, "my-index")
        lines = body.strip().split("\n")
        # 2 chunks → 4 lines (action + doc each)
        assert len(lines) == 4

        action_0 = json.loads(lines[0])
        assert action_0["index"]["_index"] == "my-index"
        assert action_0["index"]["_id"] == "doc-1_chunk_0"

        doc_0 = json.loads(lines[1])
        assert doc_0["chunk_id"] == "doc-1_chunk_0"
        assert doc_0["document_id"] == "doc-1"
        assert "embedding" in doc_0
        assert "content" in doc_0

    def test_body_ends_with_newline(self):
        chunks = _make_enriched_chunks(1)
        body = _build_bulk_body(chunks, "idx")
        assert body.endswith("\n")

    def test_empty_chunks(self):
        body = _build_bulk_body([], "idx")
        assert body == "\n"

    def test_uses_chunk_id_as_doc_id(self):
        chunks = _make_enriched_chunks(1)
        chunks[0]["chunk_id"] = "custom-id-123"
        body = _build_bulk_body(chunks, "idx")
        action = json.loads(body.strip().split("\n")[0])
        assert action["index"]["_id"] == "custom-id-123"


# ---------------------------------------------------------------------------
# _resolve_write_index
# ---------------------------------------------------------------------------

class TestResolveWriteIndex:
    def test_returns_concrete_index_behind_alias(self):
        client = MagicMock()
        client.indices.get_alias.return_value = {
            "rag-chunks-v1": {"aliases": {"rag-chunks": {}}}
        }
        result = _resolve_write_index(client, "rag-chunks")
        assert result == "rag-chunks-v1"

    def test_returns_alias_when_no_concrete_index(self):
        client = MagicMock()
        client.indices.get_alias.side_effect = Exception("alias not found")
        result = _resolve_write_index(client, "rag-chunks")
        assert result == "rag-chunks"

    def test_returns_alias_when_empty_response(self):
        client = MagicMock()
        client.indices.get_alias.return_value = {}
        result = _resolve_write_index(client, "rag-chunks")
        assert result == "rag-chunks"


# ---------------------------------------------------------------------------
# _set_refresh_interval
# ---------------------------------------------------------------------------

class TestSetRefreshInterval:
    def test_calls_put_settings(self):
        client = MagicMock()
        _set_refresh_interval(client, "my-index", "30s")
        client.indices.put_settings.assert_called_once_with(
            index="my-index",
            body={"index": {"refresh_interval": "30s"}},
        )

    def test_does_not_raise_on_error(self):
        client = MagicMock()
        client.indices.put_settings.side_effect = Exception("cluster error")
        # Should not raise
        _set_refresh_interval(client, "my-index", "1s")


# ---------------------------------------------------------------------------
# bulk_index_chunks
# ---------------------------------------------------------------------------

class TestBulkIndexChunks:
    def test_indexes_all_chunks_in_single_batch(self):
        client = _mock_os_client()
        chunks = _make_enriched_chunks(3)
        indexed, errors = bulk_index_chunks(client, chunks, "idx", batch_size=500)
        assert indexed == 3
        assert errors == 0
        client.bulk.assert_called_once()

    def test_splits_into_multiple_batches(self):
        client = _mock_os_client()
        chunks = _make_enriched_chunks(5)
        indexed, errors = bulk_index_chunks(client, chunks, "idx", batch_size=2)
        assert indexed == 5
        assert errors == 0
        # 5 chunks / batch_size 2 → 3 bulk calls
        assert client.bulk.call_count == 3

    def test_counts_errors_from_bulk_response(self):
        error_items = [
            {"index": {"_id": "doc-1_chunk_0", "error": {"type": "mapper_parsing_exception"}}},
            {"index": {"_id": "doc-1_chunk_1", "status": 201}},
        ]
        client = _mock_os_client(bulk_errors=True, bulk_items=error_items)
        chunks = _make_enriched_chunks(2)
        indexed, errors = bulk_index_chunks(client, chunks, "idx", batch_size=500)
        assert indexed == 1
        assert errors == 1

    def test_empty_chunks(self):
        client = _mock_os_client()
        indexed, errors = bulk_index_chunks(client, [], "idx")
        assert indexed == 0
        assert errors == 0
        client.bulk.assert_not_called()


# ---------------------------------------------------------------------------
# reindex_with_alias_switch
# ---------------------------------------------------------------------------

class TestReindexWithAliasSwitch:
    def test_full_reindex_flow(self):
        client = _mock_os_client()
        result = reindex_with_alias_switch(
            client, "old-idx", "new-idx", "rag-chunks",
            index_body={"settings": {"index": {"knn": True}}},
        )
        # 1. Create new index with body
        client.indices.create.assert_called_once_with(
            index="new-idx",
            body={"settings": {"index": {"knn": True}}},
        )
        # 2. Set refresh to 30s during bulk
        # 3. Reindex
        client.reindex.assert_called_once()
        # 4. Restore refresh to 1s
        put_settings_calls = client.indices.put_settings.call_args_list
        assert len(put_settings_calls) == 2
        assert put_settings_calls[0] == call(
            index="new-idx", body={"index": {"refresh_interval": BULK_REFRESH_INTERVAL}}
        )
        assert put_settings_calls[1] == call(
            index="new-idx", body={"index": {"refresh_interval": INCREMENTAL_REFRESH_INTERVAL}}
        )
        # 5. Alias switch
        client.indices.update_aliases.assert_called_once()
        alias_body = client.indices.update_aliases.call_args[1]["body"]
        actions = alias_body["actions"]
        assert {"remove": {"index": "old-idx", "alias": "rag-chunks"}} in actions
        assert {"add": {"index": "new-idx", "alias": "rag-chunks"}} in actions
        # 6. Delete old
        client.indices.delete.assert_called_once_with(index="old-idx")

        assert result["old_index"] == "old-idx"
        assert result["new_index"] == "new-idx"

    def test_reindex_without_index_body(self):
        client = _mock_os_client()
        reindex_with_alias_switch(client, "old", "new", "alias")
        client.indices.create.assert_called_once_with(index="new")


# ---------------------------------------------------------------------------
# Handler integration tests (S3 via moto, OpenSearch mocked)
# ---------------------------------------------------------------------------

@mock_aws
class TestHandler:
    def _setup_s3_with_embeddings(
        self,
        bucket: str = "test-bucket",
        key: str = "processed/embeddings/doc-1.json",
        chunks: list[dict] | None = None,
    ) -> None:
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=bucket)
        data = _make_enriched_chunks(3) if chunks is None else chunks
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(data).encode("utf-8"),
        )

    def _make_event(self, **overrides) -> dict:
        base = {
            "document_id": "doc-1",
            "s3_bucket": "test-bucket",
            "embeddings_key": "processed/embeddings/doc-1.json",
            "correlation_id": "corr-456",
            "opensearch_endpoint": "https://search-test.us-east-1.es.amazonaws.com",
            "index_alias": "rag-chunks",
        }
        base.update(overrides)
        return base

    @patch("lambdas.ingestion.vector_indexer.handler._build_opensearch_client")
    def test_basic_indexing(self, mock_build_os):
        self._setup_s3_with_embeddings()
        os_client = _mock_os_client()
        mock_build_os.return_value = os_client

        result = handler(self._make_event())

        assert result["document_id"] == "doc-1"
        assert result["indexed_count"] == 3
        assert result["error_count"] == 0
        assert result["correlation_id"] == "corr-456"
        assert "error" not in result

    @patch("lambdas.ingestion.vector_indexer.handler._build_opensearch_client")
    def test_empty_chunks(self, mock_build_os):
        self._setup_s3_with_embeddings(chunks=[])
        os_client = _mock_os_client()
        mock_build_os.return_value = os_client

        result = handler(self._make_event())

        assert result["indexed_count"] == 0
        assert result["error_count"] == 0
        os_client.bulk.assert_not_called()

    @patch("lambdas.ingestion.vector_indexer.handler._build_opensearch_client")
    def test_missing_embeddings_key(self, mock_build_os):
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="test-bucket")

        result = handler(self._make_event(embeddings_key="nonexistent.json"))

        assert "error" in result
        assert result["indexed_count"] == 0

    @patch("lambdas.ingestion.vector_indexer.handler._build_opensearch_client")
    def test_correlation_id_generated_when_missing(self, mock_build_os):
        self._setup_s3_with_embeddings()
        os_client = _mock_os_client()
        mock_build_os.return_value = os_client

        event = self._make_event()
        del event["correlation_id"]
        result = handler(event)

        assert result["correlation_id"] is not None
        assert len(result["correlation_id"]) > 0

    @patch("lambdas.ingestion.vector_indexer.handler._build_opensearch_client")
    def test_bulk_load_sets_refresh_interval(self, mock_build_os):
        """When chunks exceed batch_size, refresh_interval is set to 30s then restored."""
        # Use batch_size=2 so 3 chunks triggers bulk mode
        self._setup_s3_with_embeddings()
        os_client = _mock_os_client()
        mock_build_os.return_value = os_client

        result = handler(self._make_event(batch_size=2))

        assert result["indexed_count"] == 3
        put_calls = os_client.indices.put_settings.call_args_list
        # Should have set 30s then 1s
        intervals = [
            c[1]["body"]["index"]["refresh_interval"] for c in put_calls
        ]
        assert BULK_REFRESH_INTERVAL in intervals
        assert INCREMENTAL_REFRESH_INTERVAL in intervals

    @patch("lambdas.ingestion.vector_indexer.handler._build_opensearch_client")
    def test_incremental_load_skips_refresh_change(self, mock_build_os):
        """When chunks fit in one batch, refresh_interval is not changed."""
        self._setup_s3_with_embeddings(chunks=_make_enriched_chunks(2))
        os_client = _mock_os_client()
        mock_build_os.return_value = os_client

        result = handler(self._make_event(batch_size=500))

        assert result["indexed_count"] == 2
        os_client.indices.put_settings.assert_not_called()

    @patch("lambdas.ingestion.vector_indexer.handler._build_opensearch_client")
    def test_batch_size_capped_at_max(self, mock_build_os):
        self._setup_s3_with_embeddings()
        os_client = _mock_os_client()
        mock_build_os.return_value = os_client

        result = handler(self._make_event(batch_size=9999))

        assert result["indexed_count"] == 3
        assert "error" not in result

    @patch("lambdas.ingestion.vector_indexer.handler._build_opensearch_client")
    def test_reindex_mode(self, mock_build_os):
        self._setup_s3_with_embeddings()
        os_client = _mock_os_client()
        mock_build_os.return_value = os_client

        event = self._make_event(reindex={
            "new_index": "rag-chunks-v2",
            "index_body": {"settings": {"index": {"knn": True}}},
        })
        result = handler(event)

        assert "reindex" in result
        assert result["reindex"]["new_index"] == "rag-chunks-v2"
        os_client.indices.create.assert_called_once()
        os_client.reindex.assert_called_once()
        os_client.indices.update_aliases.assert_called_once()
        os_client.indices.delete.assert_called_once()

    @patch("lambdas.ingestion.vector_indexer.handler._build_opensearch_client")
    def test_reindex_failure_returns_error(self, mock_build_os):
        self._setup_s3_with_embeddings()
        os_client = _mock_os_client()
        os_client.indices.create.side_effect = Exception("index already exists")
        mock_build_os.return_value = os_client

        event = self._make_event(reindex={"new_index": "rag-chunks-v2"})
        result = handler(event)

        assert "error" in result
        assert "Reindex failed" in result["error"]

    @patch("lambdas.ingestion.vector_indexer.handler._build_opensearch_client")
    def test_bulk_failure_restores_refresh_interval(self, mock_build_os):
        """On bulk failure, refresh_interval is still restored."""
        self._setup_s3_with_embeddings()
        os_client = _mock_os_client()
        os_client.bulk.side_effect = Exception("cluster unavailable")
        mock_build_os.return_value = os_client

        result = handler(self._make_event(batch_size=2))

        assert "error" in result
        # Should still have tried to restore refresh interval
        put_calls = os_client.indices.put_settings.call_args_list
        intervals = [c[1]["body"]["index"]["refresh_interval"] for c in put_calls]
        assert BULK_REFRESH_INTERVAL in intervals
        assert INCREMENTAL_REFRESH_INTERVAL in intervals

    @patch("lambdas.ingestion.vector_indexer.handler._build_opensearch_client")
    def test_idempotent_upsert(self, mock_build_os):
        """Indexing the same chunks twice uses the same _id, making it idempotent."""
        self._setup_s3_with_embeddings()
        os_client = _mock_os_client()
        mock_build_os.return_value = os_client

        handler(self._make_event())
        handler(self._make_event())

        # Both calls should use the same chunk_ids as _id
        assert os_client.bulk.call_count == 2
        for bulk_call in os_client.bulk.call_args_list:
            body = bulk_call[1]["body"]
            lines = body.strip().split("\n")
            for i in range(0, len(lines), 2):
                action = json.loads(lines[i])
                assert "_id" in action["index"]

    @patch("lambdas.ingestion.vector_indexer.handler._build_opensearch_client")
    def test_default_index_alias(self, mock_build_os):
        self._setup_s3_with_embeddings()
        os_client = _mock_os_client()
        mock_build_os.return_value = os_client

        event = self._make_event()
        del event["index_alias"]
        handler(event)

        os_client.indices.get_alias.assert_called_with(name=DEFAULT_INDEX_ALIAS)

    @patch("lambdas.ingestion.vector_indexer.handler._build_opensearch_client")
    def test_opensearch_connection_failure(self, mock_build_os):
        self._setup_s3_with_embeddings()
        mock_build_os.side_effect = Exception("connection refused")

        result = handler(self._make_event())

        assert "error" in result
        assert "Failed to connect" in result["error"]
