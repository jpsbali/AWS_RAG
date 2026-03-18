"""Unit tests for lambdas.query.metrics_handler.handler."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from lambdas.query.metrics_handler.handler import (
    _build_metric_queries,
    handler,
)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("ENV_NAME", "test")


# ---------------------------------------------------------------------------
# _build_metric_queries
# ---------------------------------------------------------------------------

class TestBuildMetricQueries:
    def test_returns_all_metrics(self):
        queries = _build_metric_queries("dev", 300)
        assert len(queries) == 5
        ids = {q["Id"] for q in queries}
        assert "query_latency_avg" in ids
        assert "query_latency_p95" in ids
        assert "cache_hit_rate" in ids
        assert "error_count" in ids
        assert "invocation_count" in ids

    def test_period_is_set(self):
        queries = _build_metric_queries("dev", 600)
        for q in queries:
            assert q["MetricStat"]["Period"] == 600

    def test_lambda_metrics_use_function_name(self):
        queries = _build_metric_queries("prod", 300)
        error_query = next(q for q in queries if q["Id"] == "error_count")
        dims = error_query["MetricStat"]["Metric"]["Dimensions"]
        assert dims[0]["Name"] == "FunctionName"
        assert dims[0]["Value"] == "rag-query-handler-prod"


# ---------------------------------------------------------------------------
# Handler — success
# ---------------------------------------------------------------------------

class TestHandlerSuccess:
    @patch("lambdas.query.metrics_handler.handler._get_cloudwatch_client")
    def test_returns_metrics(self, mock_cw_factory):
        cw = MagicMock()
        now = datetime.now(timezone.utc)
        cw.get_metric_data.return_value = {
            "MetricDataResults": [
                {
                    "Id": "query_latency_avg",
                    "Label": "Query Latency (avg ms)",
                    "Values": [150.0, 120.0],
                    "Timestamps": [now, now],
                },
                {
                    "Id": "query_latency_p95",
                    "Label": "Query Latency (p95 ms)",
                    "Values": [500.0],
                    "Timestamps": [now],
                },
                {
                    "Id": "cache_hit_rate",
                    "Label": "Cache Hit Rate",
                    "Values": [0.75],
                    "Timestamps": [now],
                },
                {
                    "Id": "error_count",
                    "Label": "Lambda Errors",
                    "Values": [5.0],
                    "Timestamps": [now],
                },
                {
                    "Id": "invocation_count",
                    "Label": "Lambda Invocations",
                    "Values": [100.0],
                    "Timestamps": [now],
                },
            ]
        }
        mock_cw_factory.return_value = cw

        result = handler({})
        assert result["statusCode"] == 200
        body = json.loads(result["body"])

        assert "metrics" in body
        assert body["metrics"]["query_latency_avg"]["latest_value"] == 150.0
        assert body["metrics"]["cache_hit_rate"]["latest_value"] == 0.75
        assert body["computed"]["error_rate_percent"] == 5.0
        assert body["environment"] == "test"
        assert "request_id" in body
        assert "time_window" in body

    @patch("lambdas.query.metrics_handler.handler._get_cloudwatch_client")
    def test_empty_metrics(self, mock_cw_factory):
        cw = MagicMock()
        cw.get_metric_data.return_value = {"MetricDataResults": []}
        mock_cw_factory.return_value = cw

        result = handler({})
        assert result["statusCode"] == 200
        body = json.loads(result["body"])
        assert body["metrics"] == {}
        assert body["computed"]["error_rate_percent"] is None


# ---------------------------------------------------------------------------
# Handler — query parameters
# ---------------------------------------------------------------------------

class TestHandlerQueryParams:
    @patch("lambdas.query.metrics_handler.handler._get_cloudwatch_client")
    def test_custom_hours(self, mock_cw_factory):
        cw = MagicMock()
        cw.get_metric_data.return_value = {"MetricDataResults": []}
        mock_cw_factory.return_value = cw

        result = handler({"queryStringParameters": {"hours": "6"}})
        assert result["statusCode"] == 200
        body = json.loads(result["body"])
        assert body["time_window"]["hours"] == 6

    @patch("lambdas.query.metrics_handler.handler._get_cloudwatch_client")
    def test_hours_clamped_max(self, mock_cw_factory):
        cw = MagicMock()
        cw.get_metric_data.return_value = {"MetricDataResults": []}
        mock_cw_factory.return_value = cw

        result = handler({"queryStringParameters": {"hours": "100"}})
        body = json.loads(result["body"])
        assert body["time_window"]["hours"] == 24

    @patch("lambdas.query.metrics_handler.handler._get_cloudwatch_client")
    def test_hours_clamped_min(self, mock_cw_factory):
        cw = MagicMock()
        cw.get_metric_data.return_value = {"MetricDataResults": []}
        mock_cw_factory.return_value = cw

        result = handler({"queryStringParameters": {"hours": "0"}})
        body = json.loads(result["body"])
        assert body["time_window"]["hours"] == 1

    @patch("lambdas.query.metrics_handler.handler._get_cloudwatch_client")
    def test_invalid_hours_uses_default(self, mock_cw_factory):
        cw = MagicMock()
        cw.get_metric_data.return_value = {"MetricDataResults": []}
        mock_cw_factory.return_value = cw

        result = handler({"queryStringParameters": {"hours": "abc"}})
        body = json.loads(result["body"])
        assert body["time_window"]["hours"] == 1

    @patch("lambdas.query.metrics_handler.handler._get_cloudwatch_client")
    def test_no_query_params(self, mock_cw_factory):
        cw = MagicMock()
        cw.get_metric_data.return_value = {"MetricDataResults": []}
        mock_cw_factory.return_value = cw

        result = handler({})
        body = json.loads(result["body"])
        assert body["time_window"]["hours"] == 1


# ---------------------------------------------------------------------------
# Handler — CloudWatch failure
# ---------------------------------------------------------------------------

class TestHandlerFailure:
    @patch("lambdas.query.metrics_handler.handler._get_cloudwatch_client")
    def test_cloudwatch_error(self, mock_cw_factory):
        cw = MagicMock()
        cw.get_metric_data.side_effect = Exception("CW error")
        mock_cw_factory.return_value = cw

        result = handler({})
        assert result["statusCode"] == 500
        body = json.loads(result["body"])
        assert body["error"]["code"] == "METRICS_FETCH_FAILED"
        assert "request_id" in body["error"]


# ---------------------------------------------------------------------------
# CORS and error format
# ---------------------------------------------------------------------------

class TestCorsAndErrorFormat:
    @patch("lambdas.query.metrics_handler.handler._get_cloudwatch_client")
    def test_success_has_cors(self, mock_cw_factory):
        cw = MagicMock()
        cw.get_metric_data.return_value = {"MetricDataResults": []}
        mock_cw_factory.return_value = cw

        result = handler({})
        assert result["headers"]["Access-Control-Allow-Origin"] == "*"
        assert "GET,POST,OPTIONS" in result["headers"]["Access-Control-Allow-Methods"]

    @patch("lambdas.query.metrics_handler.handler._get_cloudwatch_client")
    def test_error_has_cors(self, mock_cw_factory):
        cw = MagicMock()
        cw.get_metric_data.side_effect = Exception("fail")
        mock_cw_factory.return_value = cw

        result = handler({})
        assert result["headers"]["Access-Control-Allow-Origin"] == "*"

    @patch("lambdas.query.metrics_handler.handler._get_cloudwatch_client")
    def test_error_has_standard_fields(self, mock_cw_factory):
        cw = MagicMock()
        cw.get_metric_data.side_effect = Exception("fail")
        mock_cw_factory.return_value = cw

        result = handler({})
        body = json.loads(result["body"])
        error = body["error"]
        assert "code" in error
        assert "message" in error
        assert "request_id" in error
