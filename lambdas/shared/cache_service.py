"""Cache service wrapping ElastiCache Redis for the AWS-native RAG service.

Provides multi-tier caching for embeddings, query results, LLM responses,
and session context with graceful degradation when Redis is unavailable.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any

from lambdas.shared.constants import (
    EMBEDDING_CACHE_PREFIX,
    EMBEDDING_CACHE_TTL,
    LLM_RESPONSE_CACHE_PREFIX,
    LLM_RESPONSE_CACHE_TTL,
    QUERY_RESULT_CACHE_PREFIX,
    QUERY_RESULT_CACHE_TTL,
    SESSION_CACHE_PREFIX,
    SESSION_CACHE_TTL,
)
from lambdas.shared.models import QueryResponse, SourceCitation
from lambdas.shared.utils import create_subsegment, get_structured_logger

logger = get_structured_logger(__name__)

# ---------------------------------------------------------------------------
# Optional Redis import — allows the module to load even when redis is absent
# ---------------------------------------------------------------------------
try:
    import redis as _redis_mod
except ImportError:  # pragma: no cover
    _redis_mod = None  # type: ignore[assignment]


class CacheService:
    """Multi-tier cache backed by ElastiCache Redis.

    All public methods degrade gracefully: on any Redis error they return
    ``None`` (for reads) or silently do nothing (for writes) so that the
    calling code never has to handle cache failures.
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        ssl: bool | None = None,
    ) -> None:
        self._host = host or os.environ.get("REDIS_HOST", "localhost")
        self._port = port or int(os.environ.get("REDIS_PORT", "6379"))
        self._ssl = ssl if ssl is not None else (os.environ.get("REDIS_TLS", "false").lower() == "true")
        self._client: Any | None = None
        self._connect()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _connect(self) -> None:
        """Establish a Redis connection, or set client to ``None`` on failure."""
        if _redis_mod is None:
            logger.warning("redis package not installed — cache disabled")
            self._client = None
            return
        try:
            self._client = _redis_mod.Redis(
                host=self._host,
                port=self._port,
                ssl=self._ssl,
                decode_responses=True,
            )
            self._client.ping()
            logger.info("Connected to Redis at %s:%s (ssl=%s)", self._host, self._port, self._ssl)
        except Exception:
            logger.warning("Redis unavailable at %s:%s — cache disabled", self._host, self._port)
            self._client = None

    @property
    def available(self) -> bool:
        """Return ``True`` if the Redis connection is alive."""
        return self._client is not None

    # ------------------------------------------------------------------
    # Generic get / set
    # ------------------------------------------------------------------

    def get(self, key: str) -> str | None:
        """Get a cached value by key. Returns ``None`` on miss or error."""
        if self._client is None:
            return None
        try:
            with create_subsegment("redis_get"):
                return self._client.get(key)
        except Exception:
            logger.warning("Cache GET failed for key=%s", key)
            return None

    def set(self, key: str, value: str, ttl_seconds: int) -> None:
        """Set a cached value with the given TTL (seconds)."""
        if self._client is None:
            return
        try:
            with create_subsegment("redis_set"):
                self._client.setex(key, ttl_seconds, value)
        except Exception:
            logger.warning("Cache SET failed for key=%s", key)

    # ------------------------------------------------------------------
    # Embedding cache  (emb:{hash} → JSON list[float], 30-day TTL)
    # ------------------------------------------------------------------

    def get_embedding(self, text_hash: str) -> list[float] | None:
        """Return a cached embedding vector, or ``None`` on miss/error."""
        raw = self.get(f"{EMBEDDING_CACHE_PREFIX}{text_hash}")
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Failed to deserialise cached embedding for hash=%s", text_hash)
            return None

    def set_embedding(self, text_hash: str, embedding: list[float]) -> None:
        """Cache an embedding vector with a 30-day TTL."""
        self.set(
            f"{EMBEDDING_CACHE_PREFIX}{text_hash}",
            json.dumps(embedding),
            EMBEDDING_CACHE_TTL,
        )

    # ------------------------------------------------------------------
    # Query-response cache  (qr:{hash} → JSON QueryResponse, 24h TTL)
    # ------------------------------------------------------------------

    def get_query_response(self, query_hash: str) -> QueryResponse | None:
        """Return a cached ``QueryResponse``, or ``None`` on miss/error."""
        raw = self.get(f"{QUERY_RESULT_CACHE_PREFIX}{query_hash}")
        if raw is None:
            return None
        try:
            data = json.loads(raw)
            sources = [
                SourceCitation(**src) for src in data.get("sources", [])
            ]
            return QueryResponse(
                answer=data["answer"],
                sources=sources,
                query_embedding=data.get("query_embedding"),
                cached=True,
                latency_ms=data.get("latency_ms", 0),
            )
        except Exception:
            logger.warning("Failed to deserialise cached query response for hash=%s", query_hash)
            return None

    def set_query_response(self, query_hash: str, response: QueryResponse) -> None:
        """Cache a ``QueryResponse`` with a 24-hour TTL."""
        try:
            data = asdict(response)
            self.set(
                f"{QUERY_RESULT_CACHE_PREFIX}{query_hash}",
                json.dumps(data),
                QUERY_RESULT_CACHE_TTL,
            )
        except Exception:
            logger.warning("Failed to serialise query response for hash=%s", query_hash)

    # ------------------------------------------------------------------
    # LLM response cache  (llm:{hash} → string, 1h TTL)
    # ------------------------------------------------------------------

    def get_llm_response(self, query_hash: str) -> str | None:
        """Return a cached raw LLM response string, or ``None``."""
        return self.get(f"{LLM_RESPONSE_CACHE_PREFIX}{query_hash}")

    def set_llm_response(self, query_hash: str, response: str) -> None:
        """Cache a raw LLM response string with a 1-hour TTL."""
        self.set(
            f"{LLM_RESPONSE_CACHE_PREFIX}{query_hash}",
            response,
            LLM_RESPONSE_CACHE_TTL,
        )

    # ------------------------------------------------------------------
    # Session cache  (sess:{session_id} → JSON, 2h TTL)
    # ------------------------------------------------------------------

    def get_session(self, session_id: str) -> dict | None:
        """Return cached session data, or ``None``."""
        raw = self.get(f"{SESSION_CACHE_PREFIX}{session_id}")
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Failed to deserialise session for id=%s", session_id)
            return None

    def set_session(self, session_id: str, data: dict) -> None:
        """Cache session data with a 2-hour TTL."""
        try:
            self.set(
                f"{SESSION_CACHE_PREFIX}{session_id}",
                json.dumps(data),
                SESSION_CACHE_TTL,
            )
        except Exception:
            logger.warning("Failed to serialise session for id=%s", session_id)
