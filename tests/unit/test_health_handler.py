"""Unit tests for lambdas.query.health_handler.handler."""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, patch

import pytest

from lambdas.query.health_handler.handler import (
    _check_redis,
    _check_s3,
    handler,
)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("DOCUMENT_BUCKET", "test-bucket")
    monkeypatch.setenv("OPENSEARCH_ENDPOINT", "https://search.example.com")
    monkeypatch.setenv("REDIS_HOST", "localhost")
    monkeypatch.setenv("REDIS_PORT", "6379")
    monkeypatch.setenv("ENV_NAME", "test")


# ---------------------------------------------------------------------------
# _check_s3
# ---------------------------------------------------------------------------

class TestCheckS3:
    def test_healthy(self):
        s3 = MagicMock()
        result = _check_s3(s3, "test-bucket")
        assert result["status"] == "healthy"
        assert "latency_ms" in result
        s3.head_bucket.assert_called_once_with(Bucket="test-bucket")

    def test_unhealthy(self):
        s3 = MagicMock()
        s3.head_bucket.side_effect = Exception("Access denied")
        result = _check_s3(s3, "test-bucket")
        assert result["status"] == "unhealthy"
        assert "error" in result


# ---------------------------------------------------------------------------
# _check_redis
# ---------------------------------------------------------------------------

class TestCheckRedis:
    def test_no_host(self):
        result = _check_redis("", 6379)
        assert result["status"] == "unhealthy"
        assert "not configured" in result["error"]

    def test_healthy(self):
        mock_redis_mod = MagicMock()
        mock_client = MagicMock()
        mock_redis_mod.Redis.return_value = mock_client
        mock_client.ping.return_value = True

        with patch.dict("sys.modules", {"redis": mock_redis_mod}):
            result = _check_redis("localhost", 6379)
        assert result["status"] == "healthy"
        assert "latency_ms" in result

    def test_unhealthy(self):
        mock_redis_mod = MagicMock()
        mock_redis_mod.Redis.side_effect = Exception("Connection refused")

        with patch.dict("sys.modules", {"redis": mock_redis_mod}):
            result = _check_redis("localhost", 6379)
        assert result["status"] == "unhealthy"
        assert "error" in result


# ---------------------------------------------------------------------------
# Handler — all healthy
# ---------------------------------------------------------------------------

class TestHandlerAllHealthy:
    @patch("lambdas.query.health_handler.handler._check_redis")
    @patch("lambdas.query.health_handler.handler._check_opensearch")
    @patch("lambdas.query.health_handler.handler._get_s3_client")
    def test_all_healthy_returns_200(self, mock_s3_factory, mock_os_check, mock_redis_check):
        s3 = MagicMock()
        mock_s3_factory.return_value = s3

        mock_os_check.return_value = {"status": "healthy", "cluster_status": "green", "latency_ms": 10}
        mock_redis_check.return_value = {"status": "healthy", "latency_ms": 5}

        result = handler({})
        assert result["statusCode"] == 200
        body = json.loads(result["body"])
        assert body["status"] == "healthy"
        assert body["environment"] == "test"
        assert "components" in body
        assert body["components"]["opensearch"]["status"] == "healthy"
        assert body["components"]["redis"]["status"] == "healthy"
        assert "request_id" in body


# ---------------------------------------------------------------------------
# Handler — degraded
# ---------------------------------------------------------------------------

class TestHandlerDegraded:
    @patch("lambdas.query.health_handler.handler._check_redis")
    @patch("lambdas.query.health_handler.handler._check_opensearch")
    @patch("lambdas.query.health_handler.handler._get_s3_client")
    def test_redis_down_returns_503(self, mock_s3_factory, mock_os_check, mock_redis_check):
        s3 = MagicMock()
        mock_s3_factory.return_value = s3

        mock_os_check.return_value = {"status": "healthy", "latency_ms": 10}
        mock_redis_check.return_value = {"status": "unhealthy", "error": "Connection refused"}

        result = handler({})
        assert result["statusCode"] == 503
        body = json.loads(result["body"])
        assert body["status"] == "degraded"
        assert body["components"]["redis"]["status"] == "unhealthy"

    @patch("lambdas.query.health_handler.handler._check_redis")
    @patch("lambdas.query.health_handler.handler._check_opensearch")
    @patch("lambdas.query.health_handler.handler._get_s3_client")
    def test_opensearch_down_returns_503(self, mock_s3_factory, mock_os_check, mock_redis_check):
        s3 = MagicMock()
        mock_s3_factory.return_value = s3

        mock_os_check.return_value = {"status": "unhealthy", "error": "Timeout"}
        mock_redis_check.return_value = {"status": "healthy", "latency_ms": 5}

        result = handler({})
        assert result["statusCode"] == 503
        body = json.loads(result["body"])
        assert body["status"] == "degraded"

    @patch("lambdas.query.health_handler.handler._check_redis")
    @patch("lambdas.query.health_handler.handler._check_opensearch")
    @patch("lambdas.query.health_handler.handler._get_s3_client")
    def test_s3_down_returns_503(self, mock_s3_factory, mock_os_check, mock_redis_check):
        s3 = MagicMock()
        s3.head_bucket.side_effect = Exception("Access denied")
        mock_s3_factory.return_value = s3

        mock_os_check.return_value = {"status": "healthy", "latency_ms": 10}
        mock_redis_check.return_value = {"status": "healthy", "latency_ms": 5}

        result = handler({})
        assert result["statusCode"] == 503
        body = json.loads(result["body"])
        assert body["status"] == "degraded"
        assert body["components"]["s3"]["status"] == "unhealthy"


# ---------------------------------------------------------------------------
# Handler — missing config
# ---------------------------------------------------------------------------

class TestHandlerMissingConfig:
    @patch("lambdas.query.health_handler.handler._check_redis")
    @patch("lambdas.query.health_handler.handler._check_opensearch")
    @patch("lambdas.query.health_handler.handler._get_s3_client")
    def test_missing_bucket_still_returns(self, mock_s3_factory, mock_os_check, mock_redis_check, monkeypatch):
        monkeypatch.delenv("DOCUMENT_BUCKET", raising=False)
        mock_os_check.return_value = {"status": "healthy", "latency_ms": 10}
        mock_redis_check.return_value = {"status": "healthy", "latency_ms": 5}

        result = handler({})
        assert result["statusCode"] == 503
        body = json.loads(result["body"])
        assert body["components"]["s3"]["status"] == "unhealthy"


# ---------------------------------------------------------------------------
# CORS headers
# ---------------------------------------------------------------------------

class TestCorsHeaders:
    @patch("lambdas.query.health_handler.handler._check_redis")
    @patch("lambdas.query.health_handler.handler._check_opensearch")
    @patch("lambdas.query.health_handler.handler._get_s3_client")
    def test_response_has_cors(self, mock_s3_factory, mock_os_check, mock_redis_check):
        s3 = MagicMock()
        mock_s3_factory.return_value = s3
        mock_os_check.return_value = {"status": "healthy", "latency_ms": 10}
        mock_redis_check.return_value = {"status": "healthy", "latency_ms": 5}

        result = handler({})
        assert result["headers"]["Access-Control-Allow-Origin"] == "*"
        assert "GET,POST,OPTIONS" in result["headers"]["Access-Control-Allow-Methods"]
