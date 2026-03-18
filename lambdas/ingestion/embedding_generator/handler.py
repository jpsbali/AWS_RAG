"""Embedding Generator Lambda handler for the ingestion pipeline.

Accepts a Step Functions event with document_id, s3_bucket, chunks_key (S3 key
to chunks JSON), correlation_id, dimensions (default 1024), and input_type
(default "search_document").  Loads chunks from S3, checks Redis cache for
existing embeddings, generates missing embeddings via Bedrock Titan v2 in
batches of up to 100, normalises each vector (L2), caches new embeddings in
Redis with a 30-day TTL, stores the enriched chunks+embeddings JSON back in S3,
and returns a result dict.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from typing import Any

import boto3

from lambdas.shared.constants import (
    EMBEDDING_CACHE_PREFIX,
    EMBEDDING_CACHE_TTL,
    EMBEDDING_MODEL_ID,
    DEFAULT_EMBEDDING_DIMENSIONS,
    MAX_EMBEDDING_BATCH_SIZE,
)
from lambdas.shared.models import EmbeddingResult
from lambdas.shared.utils import compute_hash, create_subsegment, generate_correlation_id, get_structured_logger

logger = get_structured_logger(__name__)

# ---------------------------------------------------------------------------
# Optional Redis import — allows the module to load even when redis is absent
# ---------------------------------------------------------------------------
try:
    import redis as _redis_mod  # noqa: F401
except ImportError:  # pragma: no cover
    _redis_mod = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Bedrock retry configuration
# ---------------------------------------------------------------------------
MAX_RETRIES = 5
INITIAL_BACKOFF_S = 2.0
THROTTLE_EXCEPTIONS = ("ThrottlingException", "TooManyRequestsException", "ServiceUnavailableException")

# ---------------------------------------------------------------------------
# Allowed dimension values
# ---------------------------------------------------------------------------
VALID_DIMENSIONS = {256, 512, 1024}


# ---------------------------------------------------------------------------
# boto3 client helpers
# ---------------------------------------------------------------------------

def _get_s3_client():  # noqa: ANN202
    return boto3.client("s3")


def _get_bedrock_client():  # noqa: ANN202
    return boto3.client("bedrock-runtime")


# ---------------------------------------------------------------------------
# L2 normalisation
# ---------------------------------------------------------------------------

def normalize_embedding(vector: list[float]) -> list[float]:
    """Return the L2-normalised version of *vector*."""
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        return vector
    return [v / norm for v in vector]


# ---------------------------------------------------------------------------
# Redis helpers
# ---------------------------------------------------------------------------

def _cache_key(text_hash: str) -> str:
    return f"{EMBEDDING_CACHE_PREFIX}{text_hash}"


def get_cached_embedding(
    redis_client: Any | None,
    text_hash: str,
) -> list[float] | None:
    """Return a cached embedding from Redis, or ``None`` on miss / error."""
    if redis_client is None:
        return None
    try:
        raw = redis_client.get(_cache_key(text_hash))
        if raw is not None:
            return json.loads(raw)
    except Exception:
        logger.warning("Redis cache read failed for %s", text_hash)
    return None


def set_cached_embedding(
    redis_client: Any | None,
    text_hash: str,
    embedding: list[float],
) -> None:
    """Store an embedding in Redis with the configured TTL."""
    if redis_client is None:
        return
    try:
        redis_client.setex(
            _cache_key(text_hash),
            EMBEDDING_CACHE_TTL,
            json.dumps(embedding),
        )
    except Exception:
        logger.warning("Redis cache write failed for %s", text_hash)


# ---------------------------------------------------------------------------
# Bedrock embedding call with retry
# ---------------------------------------------------------------------------

def _invoke_bedrock(
    bedrock_client: Any,
    text: str,
    dimensions: int,
    input_type: str,
) -> list[float]:
    """Call Bedrock Titan v2 for a single text and return the normalised embedding.

    Retries up to ``MAX_RETRIES`` times on throttling errors with exponential
    backoff.
    """
    body = json.dumps({
        "inputText": text,
        "dimensions": dimensions,
        "normalize": True,
    })

    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            with create_subsegment("bedrock_embed"):
                response = bedrock_client.invoke_model(
                    modelId=EMBEDDING_MODEL_ID,
                    contentType="application/json",
                    accept="application/json",
                    body=body,
                )
                result = json.loads(response["body"].read() if hasattr(response["body"], "read") else response["body"])
                embedding = result["embedding"]
                return normalize_embedding(embedding)
        except Exception as exc:
            error_code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
            if error_code in THROTTLE_EXCEPTIONS or "Throttling" in str(exc):
                wait = INITIAL_BACKOFF_S * (2 ** attempt)
                logger.warning(
                    "Bedrock throttled (attempt %d/%d), retrying in %.1fs",
                    attempt + 1, MAX_RETRIES, wait,
                )
                time.sleep(wait)
                last_exc = exc
            else:
                raise

    raise RuntimeError(f"Bedrock throttled after {MAX_RETRIES} retries") from last_exc


# ---------------------------------------------------------------------------
# Batch embedding generation
# ---------------------------------------------------------------------------

def generate_embeddings_for_chunks(
    chunks: list[dict],
    dimensions: int,
    input_type: str,
    redis_client: Any | None = None,
    bedrock_client: Any | None = None,
) -> tuple[list[EmbeddingResult], int]:
    """Generate embeddings for *chunks*, returning results and cached count.

    Parameters
    ----------
    chunks:
        List of chunk dicts (must contain ``chunk_id`` and ``content``).
    dimensions:
        Embedding vector size (256, 512, or 1024).
    input_type:
        Bedrock input type (``search_document`` or ``search_query``).
    redis_client:
        Optional Redis connection for caching.
    bedrock_client:
        Optional Bedrock runtime client (created if not supplied).

    Returns
    -------
    tuple[list[EmbeddingResult], int]
        (embedding_results, cached_count)
    """
    bedrock = bedrock_client or _get_bedrock_client()
    results: list[EmbeddingResult] = []
    cached_count = 0

    # 1. Check cache for every chunk
    uncached_indices: list[int] = []
    text_hashes: list[str] = []

    for idx, chunk in enumerate(chunks):
        content = chunk["content"]
        text_hash = compute_hash(content)
        text_hashes.append(text_hash)

        cached = get_cached_embedding(redis_client, text_hash)
        if cached is not None:
            results.append(EmbeddingResult(
                chunk_id=chunk["chunk_id"],
                embedding=cached,
                cached=True,
            ))
            cached_count += 1
        else:
            results.append(None)  # type: ignore[arg-type]  # placeholder
            uncached_indices.append(idx)

    # 2. Batch uncached chunks (up to MAX_EMBEDDING_BATCH_SIZE per batch)
    for batch_start in range(0, len(uncached_indices), MAX_EMBEDDING_BATCH_SIZE):
        batch_indices = uncached_indices[batch_start:batch_start + MAX_EMBEDDING_BATCH_SIZE]
        for idx in batch_indices:
            chunk = chunks[idx]
            embedding = _invoke_bedrock(bedrock, chunk["content"], dimensions, input_type)
            result = EmbeddingResult(
                chunk_id=chunk["chunk_id"],
                embedding=embedding,
                cached=False,
            )
            results[idx] = result

            # Cache the new embedding
            set_cached_embedding(redis_client, text_hashes[idx], embedding)

    return results, cached_count


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def _load_chunks_from_s3(
    s3_bucket: str,
    chunks_key: str,
    s3_client: Any | None = None,
) -> list[dict]:
    """Load chunks JSON array from S3."""
    s3 = s3_client or _get_s3_client()
    obj = s3.get_object(Bucket=s3_bucket, Key=chunks_key)
    return json.loads(obj["Body"].read().decode("utf-8"))


def _store_embeddings_to_s3(
    s3_bucket: str,
    document_id: str,
    chunks: list[dict],
    results: list[EmbeddingResult],
    s3_client: Any | None = None,
) -> str:
    """Store chunks enriched with their embeddings as JSON in S3.

    Returns the output S3 key.
    """
    s3 = s3_client or _get_s3_client()
    output_key = f"processed/embeddings/{document_id}.json"

    enriched = []
    for chunk, emb_result in zip(chunks, results):
        entry = dict(chunk)
        entry["embedding"] = emb_result.embedding
        entry["cached"] = emb_result.cached
        enriched.append(entry)

    s3.put_object(
        Bucket=s3_bucket,
        Key=output_key,
        Body=json.dumps(enriched, default=str).encode("utf-8"),
        ContentType="application/json",
    )
    return output_key


# ---------------------------------------------------------------------------
# Lambda handler
# ---------------------------------------------------------------------------

def handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    """Generate embeddings for document chunks and store results in S3.

    Parameters
    ----------
    event : dict
        Expected keys from Step Functions:
        ``document_id``, ``s3_bucket``, ``chunks_key`` (S3 key to chunks JSON).
        Optional: ``correlation_id``, ``dimensions`` (default 1024),
        ``input_type`` (default ``"search_document"``).
    context : object, optional
        Lambda context (unused).

    Returns
    -------
    dict
        Result with ``document_id``, ``embedding_count``, ``cached_count``,
        ``output_key``, and ``correlation_id``.
    """
    correlation_id = event.get("correlation_id") or generate_correlation_id()
    extra = {"correlation_id": correlation_id}

    document_id: str = event["document_id"]
    s3_bucket: str = event["s3_bucket"]
    chunks_key: str = event["chunks_key"]
    dimensions: int = int(event.get("dimensions", DEFAULT_EMBEDDING_DIMENSIONS))
    input_type: str = event.get("input_type", "search_document")

    # Validate dimensions
    if dimensions not in VALID_DIMENSIONS:
        error_msg = f"Invalid dimensions {dimensions}. Must be one of {sorted(VALID_DIMENSIONS)}"
        logger.error(error_msg, extra=extra)
        return {
            "document_id": document_id,
            "embedding_count": 0,
            "cached_count": 0,
            "output_key": "",
            "error": error_msg,
            "correlation_id": correlation_id,
        }

    logger.info(
        "Starting embedding generation for %s (dims=%d, input_type=%s)",
        document_id, dimensions, input_type, extra=extra,
    )

    # Load chunks from S3
    try:
        chunks = _load_chunks_from_s3(s3_bucket, chunks_key)
    except Exception as exc:
        error_msg = f"Failed to load chunks from s3://{s3_bucket}/{chunks_key}: {exc}"
        logger.error(error_msg, extra=extra)
        return {
            "document_id": document_id,
            "embedding_count": 0,
            "cached_count": 0,
            "output_key": "",
            "error": error_msg,
            "correlation_id": correlation_id,
        }

    if not chunks:
        logger.warning("No chunks found for %s", document_id, extra=extra)
        return {
            "document_id": document_id,
            "embedding_count": 0,
            "cached_count": 0,
            "output_key": "",
            "correlation_id": correlation_id,
        }

    # Connect to Redis (graceful degradation if unavailable)
    redis_client = _get_redis_client()

    # Generate embeddings
    try:
        results, cached_count = generate_embeddings_for_chunks(
            chunks=chunks,
            dimensions=dimensions,
            input_type=input_type,
            redis_client=redis_client,
            bedrock_client=None,  # will be created inside
        )
    except Exception as exc:
        error_msg = f"Embedding generation failed for {document_id}: {exc}"
        logger.error(error_msg, extra=extra)
        return {
            "document_id": document_id,
            "embedding_count": 0,
            "cached_count": 0,
            "output_key": "",
            "error": error_msg,
            "correlation_id": correlation_id,
        }

    # Store enriched chunks in S3
    try:
        output_key = _store_embeddings_to_s3(s3_bucket, document_id, chunks, results)
    except Exception as exc:
        error_msg = f"Failed to store embeddings for {document_id}: {exc}"
        logger.error(error_msg, extra=extra)
        return {
            "document_id": document_id,
            "embedding_count": len(results),
            "cached_count": cached_count,
            "output_key": "",
            "error": error_msg,
            "correlation_id": correlation_id,
        }

    logger.info(
        "Completed embeddings for %s: %d total, %d cached",
        document_id, len(results), cached_count, extra=extra,
    )

    return {
        "document_id": document_id,
        "embedding_count": len(results),
        "cached_count": cached_count,
        "output_key": output_key,
        "correlation_id": correlation_id,
    }


# ---------------------------------------------------------------------------
# Redis connection helper
# ---------------------------------------------------------------------------

def _get_redis_client() -> Any | None:
    """Return a Redis client or ``None`` if Redis is unavailable."""
    if _redis_mod is None:
        return None
    try:
        import os
        host = os.environ.get("REDIS_HOST", "localhost")
        port = int(os.environ.get("REDIS_PORT", "6379"))
        client = _redis_mod.Redis(host=host, port=port, decode_responses=True)
        client.ping()
        return client
    except Exception:
        logger.warning("Redis unavailable — proceeding without cache")
        return None
