"""Metrics Handler Lambda — return system metrics summary from CloudWatch.

Queries CloudWatch for key RAG service metrics: query latency, cache hit rate,
error rates, and processing times over a configurable time window.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import boto3

from lambdas.shared.utils import generate_correlation_id, get_structured_logger

logger = get_structured_logger(__name__)

CORS_HEADERS = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type,Authorization,X-Amz-Date,X-Api-Key,X-Amz-Security-Token",
    "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
}

# Default time window for metrics queries (1 hour)
DEFAULT_PERIOD_HOURS = 1
METRIC_PERIOD_SECONDS = 300  # 5-minute granularity

# Metric definitions to query
METRIC_DEFINITIONS = [
    {
        "id": "query_latency_avg",
        "namespace": "RAG/QueryPipeline",
        "metric_name": "QueryLatency",
        "stat": "Average",
        "label": "Query Latency (avg ms)",
    },
    {
        "id": "query_latency_p95",
        "namespace": "RAG/QueryPipeline",
        "metric_name": "QueryLatency",
        "stat": "p95",
        "label": "Query Latency (p95 ms)",
    },
    {
        "id": "cache_hit_rate",
        "namespace": "RAG/QueryPipeline",
        "metric_name": "CacheHitRate",
        "stat": "Average",
        "label": "Cache Hit Rate",
    },
    {
        "id": "error_count",
        "namespace": "AWS/Lambda",
        "metric_name": "Errors",
        "stat": "Sum",
        "label": "Lambda Errors",
    },
    {
        "id": "invocation_count",
        "namespace": "AWS/Lambda",
        "metric_name": "Invocations",
        "stat": "Sum",
        "label": "Lambda Invocations",
    },
]


def _get_cloudwatch_client() -> Any:
    return boto3.client("cloudwatch")


def _error_response(status_code: int, code: str, message: str, request_id: str) -> dict[str, Any]:
    return {
        "statusCode": status_code,
        "headers": CORS_HEADERS,
        "body": json.dumps({"error": {"code": code, "message": message, "request_id": request_id}}),
    }


def _success_response(status_code: int, body: dict) -> dict[str, Any]:
    return {
        "statusCode": status_code,
        "headers": CORS_HEADERS,
        "body": json.dumps(body, default=str),
    }


def _build_metric_queries(env_name: str, period: int) -> list[dict]:
    """Build CloudWatch GetMetricData queries for all defined metrics."""
    queries = []
    for defn in METRIC_DEFINITIONS:
        query: dict[str, Any] = {
            "Id": defn["id"],
            "MetricStat": {
                "Metric": {
                    "Namespace": defn["namespace"],
                    "MetricName": defn["metric_name"],
                    "Dimensions": [{"Name": "Service", "Value": "QueryHandler"}]
                    if defn["namespace"] == "RAG/QueryPipeline"
                    else [{"Name": "FunctionName", "Value": f"rag-query-handler-{env_name}"}],
                },
                "Period": period,
                "Stat": defn["stat"],
            },
            "Label": defn["label"],
            "ReturnData": True,
        }
        queries.append(query)
    return queries


def handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    """Return system metrics summary from CloudWatch."""
    correlation_id = generate_correlation_id()
    env_name = os.environ.get("ENV_NAME", "dev")

    # Parse optional query parameters for time window
    query_params = event.get("queryStringParameters") or {}
    try:
        hours = int(query_params.get("hours", DEFAULT_PERIOD_HOURS))
        hours = max(1, min(hours, 24))  # Clamp to 1-24 hours
    except (ValueError, TypeError):
        hours = DEFAULT_PERIOD_HOURS

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=hours)

    cw_client = _get_cloudwatch_client()
    metric_queries = _build_metric_queries(env_name, METRIC_PERIOD_SECONDS)

    try:
        response = cw_client.get_metric_data(
            MetricDataQueries=metric_queries,
            StartTime=start_time,
            EndTime=end_time,
        )
    except Exception as exc:
        logger.error("CloudWatch query failed: %s", exc)
        return _error_response(500, "METRICS_FETCH_FAILED", f"Failed to fetch metrics: {exc}", correlation_id)

    # Parse results into a summary
    metrics_summary: dict[str, Any] = {}
    for result in response.get("MetricDataResults", []):
        metric_id = result.get("Id", "")
        values = result.get("Values", [])
        timestamps = result.get("Timestamps", [])
        label = result.get("Label", metric_id)

        latest_value = values[0] if values else None
        metrics_summary[metric_id] = {
            "label": label,
            "latest_value": latest_value,
            "data_points": len(values),
            "values": values[:12],  # Last 12 data points (1 hour at 5-min granularity)
            "timestamps": [t.isoformat() if hasattr(t, "isoformat") else str(t) for t in timestamps[:12]],
        }

    # Compute error rate if both error and invocation counts are available
    error_count = metrics_summary.get("error_count", {}).get("latest_value")
    invocation_count = metrics_summary.get("invocation_count", {}).get("latest_value")
    error_rate = None
    if error_count is not None and invocation_count and invocation_count > 0:
        error_rate = round(error_count / invocation_count * 100, 2)

    body = {
        "time_window": {
            "start": start_time.isoformat(),
            "end": end_time.isoformat(),
            "hours": hours,
        },
        "metrics": metrics_summary,
        "computed": {
            "error_rate_percent": error_rate,
        },
        "environment": env_name,
        "request_id": correlation_id,
    }

    logger.info("Metrics fetched for %d hours", hours)
    return _success_response(200, body)
