"""Unit tests for lambdas.shared.utils."""

import json
import logging

from lambdas.shared.utils import (
    build_cache_key,
    compute_hash,
    generate_correlation_id,
    get_structured_logger,
)


def test_compute_hash_deterministic():
    assert compute_hash("hello") == compute_hash("hello")


def test_compute_hash_different_inputs():
    assert compute_hash("a") != compute_hash("b")


def test_compute_hash_is_hex_sha256():
    h = compute_hash("test")
    assert len(h) == 64  # SHA-256 hex digest length
    assert all(c in "0123456789abcdef" for c in h)


def test_generate_correlation_id_format():
    cid = generate_correlation_id()
    parts = cid.split("-")
    assert len(parts) == 5  # UUID4 format: 8-4-4-4-12


def test_generate_correlation_id_unique():
    ids = {generate_correlation_id() for _ in range(100)}
    assert len(ids) == 100


def test_get_structured_logger_returns_logger():
    logger = get_structured_logger("test_logger")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_logger"


def test_structured_logger_json_output(capfd):
    logger = get_structured_logger("json_test")
    logger.info("hello")
    captured = capfd.readouterr()
    record = json.loads(captured.err.strip())
    assert record["level"] == "INFO"
    assert record["message"] == "hello"
    assert "timestamp" in record
    assert "correlation_id" in record


def test_build_cache_key_format():
    key = build_cache_key("emb:", "some text")
    assert key.startswith("emb:")
    assert len(key) == 4 + 64  # prefix + SHA-256 hex


def test_build_cache_key_deterministic():
    k1 = build_cache_key("qr:", "query", "filter")
    k2 = build_cache_key("qr:", "query", "filter")
    assert k1 == k2


def test_build_cache_key_different_parts():
    k1 = build_cache_key("emb:", "a")
    k2 = build_cache_key("emb:", "b")
    assert k1 != k2
