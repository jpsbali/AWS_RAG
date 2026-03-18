"""Health Handler Lambda — health check with component connectivity status.

Checks connectivity to S3, OpenSearch, and Redis, returning a 200 with
per-component status or a 503 if critical components are unhealthy.
"""

from __future__ import annotations

import json
import os
import time
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


def _get_s3_client() -> Any:
    return boto3.client("s3")


def _check_s3(s3_client: Any, bucket: str) -> dict[str, Any]:
    """Check S3 bucket accessibility."""
    start = time.time()
    try:
        s3_client.head_bucket(Bucket=bucket)
        latency_ms = int((time.time() - start) * 1000)
        return {"status": "healthy", "latency_ms": latency_ms}
    except Exception as exc:
        latency_ms = int((time.time() - start) * 1000)
        return {"status": "unhealthy", "latency_ms": latency_ms, "error": str(exc)}


def _check_opensearch(endpoint: str) -> dict[str, Any]:
    """Check OpenSearch cluster health."""
    if not endpoint:
        return {"status": "unhealthy", "error": "OPENSEARCH_ENDPOINT not configured"}
    start = time.time()
    try:
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
        host = endpoint.replace("https://", "").replace("http://", "").rstrip("/")
        client = OpenSearch(
            hosts=[{"host": host, "port": 443}],
            http_auth=aws_auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
        )
        info = client.cluster.health()
        latency_ms = int((time.time() - start) * 1000)
        cluster_status = info.get("status", "unknown")
        return {
            "status": "healthy" if cluster_status in ("green", "yellow") else "unhealthy",
            "cluster_status": cluster_status,
            "latency_ms": latency_ms,
        }
    except Exception as exc:
        latency_ms = int((time.time() - start) * 1000)
        return {"status": "unhealthy", "latency_ms": latency_ms, "error": str(exc)}


def _check_redis(host: str, port: int) -> dict[str, Any]:
    """Check Redis connectivity."""
    if not host:
        return {"status": "unhealthy", "error": "REDIS_HOST not configured"}
    start = time.time()
    try:
        import redis

        client = redis.Redis(host=host, port=port, socket_timeout=3, decode_responses=True)
        client.ping()
        latency_ms = int((time.time() - start) * 1000)
        return {"status": "healthy", "latency_ms": latency_ms}
    except Exception as exc:
        latency_ms = int((time.time() - start) * 1000)
        return {"status": "unhealthy", "latency_ms": latency_ms, "error": str(exc)}


def handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    """Health check endpoint returning component connectivity status."""
    correlation_id = generate_correlation_id()

    bucket = os.environ.get("DOCUMENT_BUCKET", "")
    opensearch_endpoint = os.environ.get("OPENSEARCH_ENDPOINT", "")
    redis_host = os.environ.get("REDIS_HOST", "")
    redis_port = int(os.environ.get("REDIS_PORT", "6379"))
    env_name = os.environ.get("ENV_NAME", "unknown")

    s3_client = _get_s3_client()

    # Run health checks
    s3_status = _check_s3(s3_client, bucket) if bucket else {"status": "unhealthy", "error": "DOCUMENT_BUCKET not configured"}
    opensearch_status = _check_opensearch(opensearch_endpoint)
    redis_status = _check_redis(redis_host, redis_port)

    components = {
        "s3": s3_status,
        "opensearch": opensearch_status,
        "redis": redis_status,
    }

    # Overall status: healthy only if all components are healthy
    all_healthy = all(c["status"] == "healthy" for c in components.values())
    overall_status = "healthy" if all_healthy else "degraded"
    status_code = 200 if all_healthy else 503

    body = {
        "status": overall_status,
        "environment": env_name,
        "components": components,
        "request_id": correlation_id,
    }

    logger.info("Health check: status=%s", overall_status)

    return {
        "statusCode": status_code,
        "headers": CORS_HEADERS,
        "body": json.dumps(body),
    }
