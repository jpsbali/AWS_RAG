"""Shared pytest fixtures for Moto, fakeredis, and mock AWS service clients."""
import json

import fakeredis
import pytest
from moto import mock_aws
from pathlib import Path
from unittest.mock import MagicMock

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def aws_credentials(monkeypatch):
    """Set dummy AWS credentials for Moto."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SECURITY_TOKEN", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")


@pytest.fixture
def mock_s3(aws_credentials):
    """Mocked S3 with the RAG documents bucket."""
    with mock_aws():
        import boto3

        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="rag-documents-test")
        yield s3


@pytest.fixture
def mock_redis():
    """In-memory Redis replacement."""
    return fakeredis.FakeRedis(decode_responses=True)


@pytest.fixture
def mock_bedrock_client():
    """Mocked Bedrock runtime client with fixture responses."""
    client = MagicMock()
    embedding_fixture = FIXTURES_DIR / "bedrock_embedding_response.json"
    if embedding_fixture.exists():
        embedding_response = json.loads(embedding_fixture.read_text())
        client.invoke_model.return_value = {
            "body": json.dumps(embedding_response).encode()
        }
    else:
        client.invoke_model.return_value = {
            "body": json.dumps({"embedding": [0.1] * 1024}).encode()
        }
    return client


@pytest.fixture
def mock_opensearch_client():
    """Mocked OpenSearch client."""
    client = MagicMock()
    search_fixture = FIXTURES_DIR / "opensearch_search_response.json"
    if search_fixture.exists():
        search_response = json.loads(search_fixture.read_text())
        client.search.return_value = search_response
    else:
        client.search.return_value = {"hits": {"total": {"value": 0}, "hits": []}}
    client.index.return_value = {"result": "created", "_id": "chunk_1"}
    client.bulk.return_value = {"errors": False, "items": []}
    return client
