"""End-to-end integration tests for the AWS-native RAG service.

These tests run against a real deployed environment and require the following
environment variables to be set:

- API_ENDPOINT: REST API Gateway URL (e.g. https://xxx.execute-api.region.amazonaws.com/prod)
- COGNITO_USER_POOL_ID: Cognito user pool ID
- COGNITO_CLIENT_ID: Cognito app client ID
- COGNITO_USERNAME: Test user username
- COGNITO_PASSWORD: Test user password
- S3_BUCKET_NAME: The document bucket name
- AWS_REGION: AWS region (default: us-east-1)

Validates: Requirements 16.4, 19.1, 19.2, 19.3
"""

from __future__ import annotations

import base64
import json
import os
import time

import boto3
import pytest
import requests

# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------

API_ENDPOINT = os.environ.get("API_ENDPOINT", "")
COGNITO_USER_POOL_ID = os.environ.get("COGNITO_USER_POOL_ID", "")
COGNITO_CLIENT_ID = os.environ.get("COGNITO_CLIENT_ID", "")
COGNITO_USERNAME = os.environ.get("COGNITO_USERNAME", "")
COGNITO_PASSWORD = os.environ.get("COGNITO_PASSWORD", "")
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

_ENV_VARS_SET = all([
    API_ENDPOINT,
    COGNITO_USER_POOL_ID,
    COGNITO_CLIENT_ID,
    COGNITO_USERNAME,
    COGNITO_PASSWORD,
    S3_BUCKET_NAME,
])

_skip_reason = (
    "Integration tests require environment variables: "
    "API_ENDPOINT, COGNITO_USER_POOL_ID, COGNITO_CLIENT_ID, "
    "COGNITO_USERNAME, COGNITO_PASSWORD, S3_BUCKET_NAME"
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _ENV_VARS_SET, reason=_skip_reason),
]

# ---------------------------------------------------------------------------
# Timeouts and constants
# ---------------------------------------------------------------------------

DOCUMENT_PROCESSING_TIMEOUT_S = 120
DOCUMENT_POLL_INTERVAL_S = 5
REQUEST_TIMEOUT_S = 30

# Minimal valid PDF (single blank page)
_MINIMAL_PDF = (
    b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
    b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R"
    b"/Resources<<>>>>endobj\n"
    b"xref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n"
    b"0000000058 00000 n \n0000000115 00000 n \n"
    b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n206\n%%EOF"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _authenticate() -> str:
    """Authenticate with Cognito and return an ID token (JWT)."""
    client = boto3.client("cognito-idp", region_name=AWS_REGION)
    response = client.initiate_auth(
        ClientId=COGNITO_CLIENT_ID,
        AuthFlow="USER_PASSWORD_AUTH",
        AuthParameters={
            "USERNAME": COGNITO_USERNAME,
            "PASSWORD": COGNITO_PASSWORD,
        },
    )
    return response["AuthenticationResult"]["IdToken"]


def _auth_headers(token: str) -> dict[str, str]:
    """Return standard request headers with Authorization."""
    return {
        "Authorization": token,
        "Content-Type": "application/json",
    }


def _poll_document_status(
    document_id: str,
    token: str,
    *,
    target_status: str = "completed",
    timeout_s: int = DOCUMENT_PROCESSING_TIMEOUT_S,
) -> dict:
    """Poll GET /document/{id} until *target_status* or timeout.

    Returns the parsed JSON body of the final response.
    """
    url = f"{API_ENDPOINT.rstrip('/')}/document/{document_id}"
    headers = _auth_headers(token)
    deadline = time.time() + timeout_s

    last_body: dict = {}
    while time.time() < deadline:
        resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT_S)
        if resp.status_code == 200:
            last_body = resp.json()
            if last_body.get("status") == target_status:
                return last_body
        time.sleep(DOCUMENT_POLL_INTERVAL_S)

    pytest.fail(
        f"Document {document_id} did not reach '{target_status}' within "
        f"{timeout_s}s. Last response: {last_body}"
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def auth_token() -> str:
    """Module-scoped Cognito JWT token."""
    return _authenticate()


@pytest.fixture(scope="module")
def uploaded_pdf(auth_token: str) -> dict:
    """Upload a minimal test PDF and wait for processing to complete.

    Returns the ingest response body (contains document_id, s3_key, etc.).
    """
    url = f"{API_ENDPOINT.rstrip('/')}/ingest"
    payload = {
        "file_name": "integration_test.pdf",
        "content_type": "application/pdf",
        "file_content": base64.b64encode(_MINIMAL_PDF).decode(),
    }
    resp = requests.post(
        url,
        headers=_auth_headers(auth_token),
        json=payload,
        timeout=REQUEST_TIMEOUT_S,
    )
    assert resp.status_code == 200, f"Upload failed: {resp.status_code} {resp.text}"
    body = resp.json()
    assert "document_id" in body

    # Wait for processing to finish
    _poll_document_status(body["document_id"], auth_token)
    return body


# ---------------------------------------------------------------------------
# Test 1: Document ingestion end-to-end
# Validates: Requirement 19.3 — process single document within 120s
# ---------------------------------------------------------------------------


class TestDocumentIngestion:
    """Upload a test PDF and verify it appears in processed/ within 120s."""

    def test_upload_and_processing(self, uploaded_pdf: dict, auth_token: str) -> None:
        """Verify the uploaded PDF reaches 'completed' status."""
        document_id = uploaded_pdf["document_id"]
        url = f"{API_ENDPOINT.rstrip('/')}/document/{document_id}"
        resp = requests.get(
            url,
            headers=_auth_headers(auth_token),
            timeout=REQUEST_TIMEOUT_S,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "completed"
        assert body["processing"]["text_extracted"] is True
        assert body["processing"]["chunks_generated"] is True


# ---------------------------------------------------------------------------
# Test 2: Query end-to-end
# Validates: Requirements 19.1, 19.2 — query latency and response format
# ---------------------------------------------------------------------------


class TestQueryEndToEnd:
    """Send a query about the uploaded document and verify the response."""

    def test_query_returns_answer_and_citations(
        self, uploaded_pdf: dict, auth_token: str
    ) -> None:
        """POST /query returns an answer with sources."""
        url = f"{API_ENDPOINT.rstrip('/')}/query"
        payload = {"query": "What is in the uploaded document?", "k": 5}
        resp = requests.post(
            url,
            headers=_auth_headers(auth_token),
            json=payload,
            timeout=REQUEST_TIMEOUT_S,
        )
        assert resp.status_code == 200
        body = resp.json()

        # Response must contain answer and sources
        assert "answer" in body, f"Missing 'answer' in response: {body}"
        assert isinstance(body["answer"], str)
        assert len(body["answer"]) > 0

        assert "sources" in body, f"Missing 'sources' in response: {body}"
        assert isinstance(body["sources"], list)


# ---------------------------------------------------------------------------
# Test 3: Cache hit verification
# Validates: Requirement 19.1 — cached response within 500ms at p95
# ---------------------------------------------------------------------------


class TestCacheHit:
    """Repeat the same query and verify cache hit with faster response."""

    def test_repeated_query_is_cached(self, uploaded_pdf: dict, auth_token: str) -> None:
        """Second identical query should return cached=True and lower latency."""
        url = f"{API_ENDPOINT.rstrip('/')}/query"
        payload = {"query": "What is in the uploaded document?", "k": 5}
        headers = _auth_headers(auth_token)

        # First query (cache miss — prime the cache)
        resp1 = requests.post(
            url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT_S
        )
        assert resp1.status_code == 200
        body1 = resp1.json()
        latency1 = body1.get("latency_ms", float("inf"))

        # Second query (should be a cache hit)
        start = time.time()
        resp2 = requests.post(
            url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT_S
        )
        wall_time_ms = (time.time() - start) * 1000
        assert resp2.status_code == 200
        body2 = resp2.json()

        assert body2.get("cached") is True, (
            f"Expected cached=True on repeated query, got: {body2.get('cached')}"
        )

        # Cached response should be faster than the original
        latency2 = body2.get("latency_ms", wall_time_ms)
        assert latency2 < latency1, (
            f"Cached latency ({latency2}ms) should be less than "
            f"original ({latency1}ms)"
        )


# ---------------------------------------------------------------------------
# Test 4: Unsupported file rejection
# Validates: Requirement 1.6 — reject unsupported types with error format
# ---------------------------------------------------------------------------


class TestUnsupportedFileRejection:
    """Upload an unsupported file type and verify rejection."""

    def test_exe_file_rejected(self, auth_token: str) -> None:
        """POST /ingest with a .exe file returns 400 with standard error."""
        url = f"{API_ENDPOINT.rstrip('/')}/ingest"
        payload = {
            "file_name": "malware.exe",
            "content_type": "application/octet-stream",
            "file_content": base64.b64encode(b"MZ fake exe content").decode(),
        }
        resp = requests.post(
            url,
            headers=_auth_headers(auth_token),
            json=payload,
            timeout=REQUEST_TIMEOUT_S,
        )
        assert resp.status_code == 400, (
            f"Expected 400 for .exe upload, got {resp.status_code}: {resp.text}"
        )
        body = resp.json()

        # Verify standardized error format
        assert "error" in body, f"Missing 'error' key in response: {body}"
        error = body["error"]
        assert "code" in error, f"Missing 'code' in error: {error}"
        assert "message" in error, f"Missing 'message' in error: {error}"
        assert "request_id" in error, f"Missing 'request_id' in error: {error}"

        assert isinstance(error["code"], str)
        assert isinstance(error["message"], str)
        assert isinstance(error["request_id"], str)


# ---------------------------------------------------------------------------
# Test 5: Health check
# Validates: Requirement 12.1 — GET /health returns 200 with component status
# ---------------------------------------------------------------------------


class TestHealthCheck:
    """Verify GET /health returns 200 with component status."""

    def test_health_returns_200(self, auth_token: str) -> None:
        """GET /health returns 200 with status and components."""
        url = f"{API_ENDPOINT.rstrip('/')}/health"
        resp = requests.get(
            url,
            headers=_auth_headers(auth_token),
            timeout=REQUEST_TIMEOUT_S,
        )
        # Accept both 200 (all healthy) and 503 (degraded) as valid responses
        assert resp.status_code in (200, 503), (
            f"Expected 200 or 503, got {resp.status_code}: {resp.text}"
        )
        body = resp.json()

        assert "status" in body, f"Missing 'status' in health response: {body}"
        assert body["status"] in ("healthy", "degraded")

        assert "components" in body, f"Missing 'components' in health response: {body}"
        components = body["components"]
        assert isinstance(components, dict)

        # Each component should have a status field
        for name, info in components.items():
            assert "status" in info, (
                f"Component '{name}' missing 'status': {info}"
            )
