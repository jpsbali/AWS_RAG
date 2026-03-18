"""Vector Indexer Lambda handler for the ingestion pipeline.

Accepts a Step Functions event with document_id, s3_bucket, embeddings_key
(S3 key to enriched chunks+embeddings JSON), correlation_id, opensearch_endpoint,
and index_alias.  Loads enriched chunks from S3, connects to OpenSearch with
AWS SigV4 auth, bulk-indexes documents using chunk_id as _id for idempotent
upsert, and supports zero-downtime reindexing via alias switching.
"""

from __future__ import annotations

import json
import os
from typing import Any

import boto3

from lambdas.shared.utils import create_subsegment, generate_correlation_id, get_structured_logger

logger = get_structured_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_INDEX_ALIAS = "rag-chunks"
DEFAULT_BATCH_SIZE = 500
MAX_BATCH_SIZE = 1000
BULK_REFRESH_INTERVAL = "30s"
INCREMENTAL_REFRESH_INTERVAL = "1s"


# ---------------------------------------------------------------------------
# boto3 / OpenSearch client helpers
# ---------------------------------------------------------------------------

def _get_s3_client():  # noqa: ANN202
    return boto3.client("s3")


def _build_opensearch_client(endpoint: str) -> Any:
    """Build an OpenSearch client with AWS SigV4 authentication.

    Uses ``requests_aws4auth`` for IAM-based signing so the Lambda's
    execution role is used automatically.
    """
    from opensearchpy import OpenSearch, RequestsHttpConnection
    from requests_aws4auth import AWS4Auth

    region = os.environ.get("AWS_REGION", "us-east-1")
    credentials = boto3.Session().get_credentials()
    aws_auth = AWS4Auth(
        credentials.access_key,
        credentials.secret_key,
        region,
        "es",
        session_token=credentials.token,
    )

    scheme = "https"
    host = endpoint.replace("https://", "").replace("http://", "").rstrip("/")

    return OpenSearch(
        hosts=[{"host": host, "port": 443}],
        http_auth=aws_auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
    )


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def _load_embeddings_from_s3(
    s3_bucket: str,
    embeddings_key: str,
    s3_client: Any | None = None,
) -> list[dict]:
    """Load enriched chunks+embeddings JSON array from S3."""
    s3 = s3_client or _get_s3_client()
    obj = s3.get_object(Bucket=s3_bucket, Key=embeddings_key)
    return json.loads(obj["Body"].read().decode("utf-8"))


# ---------------------------------------------------------------------------
# Index / alias helpers
# ---------------------------------------------------------------------------

def _resolve_write_index(os_client: Any, alias: str) -> str:
    """Return the concrete index name behind *alias*, or *alias* itself."""
    try:
        alias_info = os_client.indices.get_alias(name=alias)
        # alias_info is {index_name: {aliases: {alias: {...}}}}
        indices = list(alias_info.keys())
        if indices:
            return indices[0]
    except Exception:
        pass
    return alias


def _set_refresh_interval(os_client: Any, index: str, interval: str) -> None:
    """Update the refresh_interval setting on *index*."""
    try:
        os_client.indices.put_settings(
            index=index,
            body={"index": {"refresh_interval": interval}},
        )
    except Exception as exc:
        logger.warning("Failed to set refresh_interval=%s on %s: %s", interval, index, exc)


# ---------------------------------------------------------------------------
# Reindexing support
# ---------------------------------------------------------------------------

def reindex_with_alias_switch(
    os_client: Any,
    old_index: str,
    new_index: str,
    alias: str,
    index_body: dict | None = None,
) -> dict[str, Any]:
    """Zero-downtime reindex: create new index → reindex → alias switch → delete old.

    Parameters
    ----------
    os_client : OpenSearch client
    old_index : current concrete index name
    new_index : new concrete index name to create
    alias     : alias to atomically switch
    index_body: optional mapping/settings for the new index

    Returns
    -------
    dict with reindex result summary
    """
    # 1. Create new index
    if index_body:
        os_client.indices.create(index=new_index, body=index_body)
    else:
        os_client.indices.create(index=new_index)

    # 2. Set refresh_interval to 30s during bulk reindex
    _set_refresh_interval(os_client, new_index, BULK_REFRESH_INTERVAL)

    # 3. Reindex from old to new
    reindex_body = {
        "source": {"index": old_index},
        "dest": {"index": new_index},
    }
    reindex_result = os_client.reindex(body=reindex_body, request_timeout=300)

    # 4. Restore refresh_interval to 1s
    _set_refresh_interval(os_client, new_index, INCREMENTAL_REFRESH_INTERVAL)

    # 5. Atomic alias switch
    os_client.indices.update_aliases(body={
        "actions": [
            {"remove": {"index": old_index, "alias": alias}},
            {"add": {"index": new_index, "alias": alias}},
        ]
    })

    # 6. Delete old index
    os_client.indices.delete(index=old_index)

    return {
        "old_index": old_index,
        "new_index": new_index,
        "alias": alias,
        "reindex_result": reindex_result,
    }


# ---------------------------------------------------------------------------
# Bulk indexing
# ---------------------------------------------------------------------------

def _build_bulk_body(chunks: list[dict], index: str) -> str:
    """Build an OpenSearch bulk request body from enriched chunks.

    Each chunk is upserted using its ``chunk_id`` as the document ``_id``,
    making the operation idempotent.
    """
    lines: list[str] = []
    for chunk in chunks:
        action = json.dumps({
            "index": {
                "_index": index,
                "_id": chunk["chunk_id"],
            }
        })
        doc = {
            "chunk_id": chunk["chunk_id"],
            "document_id": chunk["document_id"],
            "content": chunk["content"],
            "embedding": chunk["embedding"],
            "metadata": chunk.get("metadata", {}),
        }
        lines.append(action)
        lines.append(json.dumps(doc, default=str))
    # Bulk body must end with a newline
    return "\n".join(lines) + "\n"


def bulk_index_chunks(
    os_client: Any,
    chunks: list[dict],
    index: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[int, int]:
    """Bulk index *chunks* into *index* in batches.

    Returns (indexed_count, error_count).
    """
    indexed = 0
    errors = 0

    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start: batch_start + batch_size]
        body = _build_bulk_body(batch, index)
        response = os_client.bulk(body=body)

        if response.get("errors"):
            for item in response.get("items", []):
                action_result = item.get("index", {})
                if action_result.get("error"):
                    errors += 1
                    logger.warning(
                        "Bulk index error for %s: %s",
                        action_result.get("_id", "unknown"),
                        action_result["error"],
                    )
                else:
                    indexed += 1
        else:
            indexed += len(batch)

    return indexed, errors


# ---------------------------------------------------------------------------
# Lambda handler
# ---------------------------------------------------------------------------

def handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    """Index enriched chunks with embeddings into OpenSearch.

    Parameters
    ----------
    event : dict
        Expected keys from Step Functions:
        ``document_id``, ``s3_bucket``, ``embeddings_key`` (S3 key to
        enriched chunks+embeddings JSON).
        Optional: ``correlation_id``, ``opensearch_endpoint`` (or env var
        ``OPENSEARCH_ENDPOINT``), ``index_alias`` (default ``rag-chunks``),
        ``batch_size`` (default 500), ``reindex`` (dict with ``new_index``
        and optional ``index_body`` for zero-downtime reindexing).
    context : object, optional
        Lambda context (unused).

    Returns
    -------
    dict
        Result with ``document_id``, ``indexed_count``, ``error_count``,
        and ``correlation_id``.
    """
    correlation_id = event.get("correlation_id") or generate_correlation_id()
    extra = {"correlation_id": correlation_id}

    document_id: str = event["document_id"]
    s3_bucket: str = event["s3_bucket"]
    embeddings_key: str = event["embeddings_key"]
    index_alias: str = event.get("index_alias", DEFAULT_INDEX_ALIAS)
    batch_size: int = min(
        int(event.get("batch_size", DEFAULT_BATCH_SIZE)),
        MAX_BATCH_SIZE,
    )
    opensearch_endpoint: str = (
        event.get("opensearch_endpoint")
        or os.environ.get("OPENSEARCH_ENDPOINT", "")
    )

    logger.info(
        "Starting vector indexing for %s (alias=%s, batch_size=%d)",
        document_id, index_alias, batch_size, extra=extra,
    )

    # ------------------------------------------------------------------
    # 1. Load enriched chunks + embeddings from S3
    # ------------------------------------------------------------------
    try:
        chunks = _load_embeddings_from_s3(s3_bucket, embeddings_key)
    except Exception as exc:
        error_msg = f"Failed to load embeddings from s3://{s3_bucket}/{embeddings_key}: {exc}"
        logger.error(error_msg, extra=extra)
        return {
            "document_id": document_id,
            "indexed_count": 0,
            "error_count": 0,
            "error": error_msg,
            "correlation_id": correlation_id,
        }

    if not chunks:
        logger.warning("No chunks to index for %s", document_id, extra=extra)
        return {
            "document_id": document_id,
            "indexed_count": 0,
            "error_count": 0,
            "correlation_id": correlation_id,
        }

    # ------------------------------------------------------------------
    # 2. Connect to OpenSearch
    # ------------------------------------------------------------------
    try:
        os_client = _build_opensearch_client(opensearch_endpoint)
    except Exception as exc:
        error_msg = f"Failed to connect to OpenSearch at {opensearch_endpoint}: {exc}"
        logger.error(error_msg, extra=extra)
        return {
            "document_id": document_id,
            "indexed_count": 0,
            "error_count": 0,
            "error": error_msg,
            "correlation_id": correlation_id,
        }

    # ------------------------------------------------------------------
    # 3. Handle optional zero-downtime reindexing
    # ------------------------------------------------------------------
    reindex_config = event.get("reindex")
    if reindex_config:
        try:
            old_index = _resolve_write_index(os_client, index_alias)
            new_index = reindex_config["new_index"]
            index_body = reindex_config.get("index_body")
            reindex_result = reindex_with_alias_switch(
                os_client, old_index, new_index, index_alias, index_body,
            )
            logger.info(
                "Reindex complete: %s → %s (alias=%s)",
                old_index, new_index, index_alias, extra=extra,
            )
            return {
                "document_id": document_id,
                "indexed_count": 0,
                "error_count": 0,
                "reindex": reindex_result,
                "correlation_id": correlation_id,
            }
        except Exception as exc:
            error_msg = f"Reindex failed: {exc}"
            logger.error(error_msg, extra=extra)
            return {
                "document_id": document_id,
                "indexed_count": 0,
                "error_count": 0,
                "error": error_msg,
                "correlation_id": correlation_id,
            }

    # ------------------------------------------------------------------
    # 4. Resolve concrete index behind alias
    # ------------------------------------------------------------------
    write_index = _resolve_write_index(os_client, index_alias)

    # ------------------------------------------------------------------
    # 5. Set refresh_interval to 30s for bulk load
    # ------------------------------------------------------------------
    is_bulk = len(chunks) > batch_size
    if is_bulk:
        _set_refresh_interval(os_client, write_index, BULK_REFRESH_INTERVAL)

    # ------------------------------------------------------------------
    # 6. Bulk index
    # ------------------------------------------------------------------
    try:
        with create_subsegment("opensearch_bulk_index"):
            indexed_count, error_count = bulk_index_chunks(
                os_client, chunks, write_index, batch_size,
            )
    except Exception as exc:
        error_msg = f"Bulk indexing failed for {document_id}: {exc}"
        logger.error(error_msg, extra=extra)
        # Restore refresh interval even on failure
        if is_bulk:
            _set_refresh_interval(os_client, write_index, INCREMENTAL_REFRESH_INTERVAL)
        return {
            "document_id": document_id,
            "indexed_count": 0,
            "error_count": 0,
            "error": error_msg,
            "correlation_id": correlation_id,
        }

    # ------------------------------------------------------------------
    # 7. Restore refresh_interval to 1s
    # ------------------------------------------------------------------
    if is_bulk:
        _set_refresh_interval(os_client, write_index, INCREMENTAL_REFRESH_INTERVAL)

    logger.info(
        "Completed vector indexing for %s: %d indexed, %d errors",
        document_id, indexed_count, error_count, extra=extra,
    )

    return {
        "document_id": document_id,
        "indexed_count": indexed_count,
        "error_count": error_count,
        "correlation_id": correlation_id,
    }
