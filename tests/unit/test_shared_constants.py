"""Unit tests for lambdas.shared.constants."""

from lambdas.shared.constants import (
    ARCHIVES_PREFIX,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_DIMENSIONS,
    DEFAULT_MAX_CHUNK,
    DEFAULT_MIN_CHUNK,
    DEFAULT_OVERLAP,
    DEFAULT_WINDOW_SIZE,
    EMBEDDING_CACHE_PREFIX,
    EMBEDDING_CACHE_TTL,
    EMBEDDING_MODEL_ID,
    FAILED_PREFIX,
    LLM_FALLBACK_MODEL_ID,
    LLM_PRIMARY_MODEL_ID,
    LLM_RESPONSE_CACHE_PREFIX,
    LLM_RESPONSE_CACHE_TTL,
    MAX_EMBEDDING_BATCH_SIZE,
    MIN_EXTRACTION_QUALITY_THRESHOLD,
    PROCESSED_CHUNKS_PREFIX,
    PROCESSED_METADATA_PREFIX,
    PROCESSED_TEXT_PREFIX,
    QUALITY_THRESHOLD_CHARS,
    QUERY_RESULT_CACHE_PREFIX,
    QUERY_RESULT_CACHE_TTL,
    RAW_PREFIX,
    SESSION_CACHE_PREFIX,
    SESSION_CACHE_TTL,
    SUPPORTED_CONTENT_TYPES,
    SUPPORTED_FILE_TYPES,
)


def test_supported_file_types_is_frozenset():
    assert isinstance(SUPPORTED_FILE_TYPES, frozenset)
    assert "pdf" in SUPPORTED_FILE_TYPES
    assert "exe" not in SUPPORTED_FILE_TYPES
    assert len(SUPPORTED_FILE_TYPES) == 8


def test_supported_content_types_mapping():
    assert SUPPORTED_CONTENT_TYPES["application/pdf"] == "pdf"
    assert SUPPORTED_CONTENT_TYPES["text/plain"] == "txt"
    assert len(SUPPORTED_CONTENT_TYPES) == 8


def test_s3_prefixes():
    assert RAW_PREFIX == "raw/"
    assert PROCESSED_TEXT_PREFIX == "processed/text/"
    assert PROCESSED_CHUNKS_PREFIX == "processed/chunks/"
    assert PROCESSED_METADATA_PREFIX == "processed/metadata/"
    assert FAILED_PREFIX == "failed/"
    assert ARCHIVES_PREFIX == "archives/"


def test_cache_ttl_values():
    assert EMBEDDING_CACHE_TTL == 30 * 86400
    assert QUERY_RESULT_CACHE_TTL == 86400
    assert LLM_RESPONSE_CACHE_TTL == 3600
    assert SESSION_CACHE_TTL == 7200


def test_cache_key_prefixes():
    assert EMBEDDING_CACHE_PREFIX == "emb:"
    assert QUERY_RESULT_CACHE_PREFIX == "qr:"
    assert LLM_RESPONSE_CACHE_PREFIX == "llm:"
    assert SESSION_CACHE_PREFIX == "sess:"


def test_default_chunking_params():
    assert DEFAULT_CHUNK_SIZE == 1000
    assert DEFAULT_OVERLAP == 200
    assert DEFAULT_MIN_CHUNK == 200
    assert DEFAULT_MAX_CHUNK == 1500
    assert DEFAULT_WINDOW_SIZE == 3


def test_model_ids():
    assert EMBEDDING_MODEL_ID == "amazon.titan-embed-text-v2:0"
    assert "claude-3-5-sonnet" in LLM_PRIMARY_MODEL_ID
    assert "claude-3-5-haiku" in LLM_FALLBACK_MODEL_ID


def test_embedding_config():
    assert DEFAULT_EMBEDDING_DIMENSIONS == 1024
    assert MAX_EMBEDDING_BATCH_SIZE == 100


def test_quality_thresholds():
    assert MIN_EXTRACTION_QUALITY_THRESHOLD == 50
    assert QUALITY_THRESHOLD_CHARS == 50
