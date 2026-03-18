# Development and Deployment Guide — AWS-Native RAG Service

## Prerequisites

Before you start, make sure you have the following installed and configured:

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) — fast Python package manager (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Node.js 18+ (required by AWS CDK CLI)
- AWS CDK CLI: `npm install -g aws-cdk`
- AWS CLI v2, configured with credentials (`aws configure`)
- Docker (for Streamlit frontend container and Lambda layer builds)
- Git

Your AWS account needs access to:
- Amazon Bedrock (with Claude 3.5 Sonnet and Titan Embeddings v2 model access enabled)
- Amazon OpenSearch Service
- Amazon Textract, Comprehend
- All standard services (S3, Lambda, Step Functions, API Gateway, ElastiCache, etc.)

---

## Project Structure

```
aws-native-rag-service/
├── app.py                          # CDK app entry point
├── cdk.json                        # CDK configuration (dev/prod context)
├── pyproject.toml                  # Project metadata, dependencies, and tool config
├── uv.lock                         # Locked dependency versions (committed to git)
├── buildspec.yml                   # CodeBuild build specification
│
├── stacks/                         # CDK infrastructure stacks
│   ├── network_stack.py            # VPC, subnets, security groups, VPC endpoints
│   ├── storage_stack.py            # S3 bucket, EventBridge rules
│   ├── security_stack.py           # KMS, Cognito, WAF, Secrets Manager
│   ├── search_stack.py             # OpenSearch domain
│   ├── cache_stack.py              # ElastiCache Redis cluster
│   ├── processing_stack.py         # Step Functions + ingestion Lambdas
│   ├── api_stack.py                # API Gateway (REST + WebSocket) + query Lambdas
│   ├── monitoring_stack.py         # CloudWatch dashboards, alarms, SNS, X-Ray
│   └── pipeline_stack.py           # CodePipeline + CodeBuild CI/CD
│
├── lambdas/                        # Lambda function source code
│   ├── shared/                     # Shared modules across all Lambdas
│   │   ├── models.py               # Dataclasses (Chunk, QueryRequest, etc.)
│   │   ├── constants.py            # Supported file types, model IDs, TTLs
│   │   ├── utils.py                # Hashing, correlation IDs, structured logger
│   │   ├── cache_service.py        # ElastiCache Redis client
│   │   └── llm_generator.py        # Bedrock LLM invocation (Claude)
│   │
│   ├── ingestion/                  # Ingestion pipeline Lambdas
│   │   ├── validator/handler.py
│   │   ├── text_extractor/handler.py
│   │   ├── metadata_enricher/handler.py
│   │   ├── chunker/handler.py
│   │   ├── embedding_generator/handler.py
│   │   └── vector_indexer/handler.py
│   │
│   ├── query/                      # Query pipeline + API Lambdas
│   │   ├── query_handler/handler.py
│   │   ├── ingest_trigger/handler.py
│   │   ├── document_handler/handler.py
│   │   ├── health_handler/handler.py
│   │   └── metrics_handler/handler.py
│   │
│   └── evaluation/                 # RAGAS evaluation Lambda
│       └── evaluator/handler.py
│
├── frontend/                       # Streamlit web application
│   ├── app.py
│   ├── pyproject.toml
│   └── Dockerfile
│
├── tests/
│   ├── conftest.py                 # Shared pytest fixtures (Moto, fakeredis, mocks)
│   ├── fixtures/                   # Response fixtures for Bedrock, Textract, etc.
│   │   ├── bedrock_embedding_response.json
│   │   ├── bedrock_llm_response.json
│   │   ├── textract_response.json
│   │   ├── comprehend_entities_response.json
│   │   ├── comprehend_phrases_response.json
│   │   └── opensearch_search_response.json
│   ├── unit/                       # Unit tests (Moto + mocks)
│   ├── property/                   # Property-based tests (Hypothesis + Moto)
│   └── integration/                # End-to-end tests (real AWS, CI/CD only)
│       └── test_e2e.py
│
└── .kiro/specs/aws-native-rag-service/   # Spec documents
    ├── requirements.md
    ├── design.md
    └── tasks.md
```

---

## Getting Started

### 1. Clone the repository and set up your environment

```bash
git clone <your-repo-url> aws-native-rag-service
cd aws-native-rag-service

# Install all dependencies (creates .venv automatically)
uv sync

# Activate the virtual environment
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows
```

`uv sync` reads `pyproject.toml`, resolves dependencies, creates a `.venv`, and installs everything — including dev dependencies. No separate `requirements.txt` files needed.

### 2. Bootstrap CDK (one-time per AWS account/region)

```bash
cdk bootstrap aws://<ACCOUNT_ID>/<REGION>
```

This creates the CDK toolkit stack in your AWS account, which is required for CDK deployments.

### 3. Configure environments

Environment-specific settings are managed via CDK context in `cdk.json`:

```json
{
  "context": {
    "dev": {
      "opensearch_instance_type": "t3.small.search",
      "opensearch_data_nodes": 2,
      "opensearch_master_nodes": 0,
      "cache_node_type": "cache.t3.micro",
      "cache_num_nodes": 1,
      "multi_az": false,
      "alarm_actions_enabled": false
    },
    "prod": {
      "opensearch_instance_type": "r6g.xlarge.search",
      "opensearch_data_nodes": 3,
      "opensearch_master_nodes": 3,
      "cache_node_type": "cache.r6g.large",
      "cache_num_shards": 2,
      "cache_replicas_per_shard": 1,
      "multi_az": true,
      "alarm_actions_enabled": true
    }
  }
}
```

You select the environment at deploy time:

```bash
cdk deploy --context env=dev     # Deploy dev environment
cdk deploy --context env=prod    # Deploy production environment
```

---

## Local Development Workflow

### Writing Lambda code

All Lambda functions live under `lambdas/`. Each function has its own directory with a `handler.py` entry point. Shared code (data models, cache client, LLM generator, utilities) lives in `lambdas/shared/`.

When you write or modify a Lambda:

1. Edit the handler in `lambdas/<pipeline>/<function>/handler.py`
2. Import shared modules from `lambdas/shared/`
3. Run unit and property tests locally
4. Commit and push — the CI/CD pipeline handles the rest

### Writing CDK infrastructure code

All CDK stacks live under `stacks/`. The CDK app entry point is `app.py`, which instantiates and wires all stacks together.

When you modify infrastructure:

1. Edit the relevant stack in `stacks/<stack_name>.py`
2. Run `cdk synth` to validate the CloudFormation template generates correctly
3. Run `cdk diff --context env=dev` to preview what will change
4. Commit and push — the pipeline deploys it

### Running tests locally

```bash
# Run all unit tests
pytest tests/unit/ -v

# Run all property-based tests (Hypothesis)
pytest tests/property/ -v

# Run a specific property test
pytest tests/property/test_chunking_properties.py -v

# Run with coverage
pytest tests/unit/ tests/property/ --cov=lambdas --cov-report=term-missing

# Validate CDK synthesizes cleanly
cdk synth --context env=dev
```

Property-based tests use Hypothesis with a minimum of 100 examples per property. Each test file maps to correctness properties defined in the design document.

### Linting and formatting

```bash
# Format code
black lambdas/ stacks/ tests/

# Lint
flake8 lambdas/ stacks/ tests/

# Type checking
mypy lambdas/ stacks/
```

---

## Local Testing with Moto (No AWS Account Required)

All unit and property-based tests run locally without hitting real AWS services. We use [Moto](https://docs.getmoto.org/) to mock AWS APIs and [fakeredis](https://github.com/cunla/fakeredis-py) for Redis. Services that Moto doesn't cover (Bedrock, OpenSearch data plane, Textract, Comprehend) are mocked with `unittest.mock`.

### Mocking strategy by service

| Service | Mock Library | Coverage |
|---------|-------------|----------|
| S3 (upload, get, lifecycle) | `moto` (`@mock_aws`) | Full — buckets, objects, events |
| EventBridge (rules, targets) | `moto` (`@mock_aws`) | Full — rules, put_events |
| Step Functions (state machines) | `moto` (`@mock_aws`) | Full — create, start, describe execution |
| Lambda (invoke, config) | `moto` (`@mock_aws`) | Full — create, invoke |
| SQS (DLQs) | `moto` (`@mock_aws`) | Full — queues, messages |
| SNS (notifications) | `moto` (`@mock_aws`) | Full — topics, publish |
| Cognito (user pools, auth) | `moto` (`@mock_aws`) | Full — user pools, tokens |
| IAM (roles, policies) | `moto` (`@mock_aws`) | Full — roles, policy validation |
| KMS (encryption keys) | `moto` (`@mock_aws`) | Full — create key, encrypt/decrypt |
| Secrets Manager | `moto` (`@mock_aws`) | Full — create, get secret |
| CloudWatch (metrics, logs) | `moto` (`@mock_aws`) | Full — put_metric_data, logs |
| Bedrock (embeddings, LLM) | `unittest.mock` | Mock `invoke_model` / `invoke_model_with_response_stream` with fixture responses |
| OpenSearch (search, index) | `unittest.mock` | Mock `opensearch-py` client methods (`index`, `search`, `bulk`) |
| Textract (OCR, tables) | `unittest.mock` | Mock `detect_document_text` / `analyze_document` with fixture responses |
| Comprehend (NER, phrases) | `unittest.mock` | Mock `detect_entities` / `detect_key_phrases` with fixture responses |
| ElastiCache Redis | `fakeredis` | Full — in-memory Redis replacement |

### Test fixtures

Create realistic response fixtures in `tests/fixtures/` for services that need manual mocking:

```
tests/
├── fixtures/
│   ├── bedrock_embedding_response.json    # Titan v2 embedding response
│   ├── bedrock_llm_response.json          # Claude generation response
│   ├── textract_response.json             # DetectDocumentText response
│   ├── comprehend_entities_response.json  # detect_entities response
│   ├── comprehend_phrases_response.json   # detect_key_phrases response
│   └── opensearch_search_response.json    # Hybrid search response
├── conftest.py                            # Shared pytest fixtures
├── unit/
└── property/
```

### Example: conftest.py with shared fixtures

```python
import pytest
import json
import fakeredis
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

    embedding_response = json.loads(
        (FIXTURES_DIR / "bedrock_embedding_response.json").read_text()
    )
    client.invoke_model.return_value = {
        "body": json.dumps(embedding_response).encode()
    }
    return client


@pytest.fixture
def mock_opensearch_client():
    """Mocked OpenSearch client."""
    client = MagicMock()

    search_response = json.loads(
        (FIXTURES_DIR / "opensearch_search_response.json").read_text()
    )
    client.search.return_value = search_response
    client.index.return_value = {"result": "created", "_id": "chunk_1"}
    client.bulk.return_value = {"errors": False, "items": []}
    return client
```

### Example: Unit test with Moto (S3 + validator)

```python
from moto import mock_aws
import boto3
import pytest


@mock_aws
def test_validator_accepts_pdf(aws_credentials):
    """Validator accepts PDF files uploaded to S3."""
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket="rag-documents-test")
    s3.put_object(
        Bucket="rag-documents-test",
        Key="raw/test-doc.pdf",
        Body=b"%PDF-1.4 fake content",
        ContentType="application/pdf",
    )

    from lambdas.ingestion.validator.handler import validate
    result = validate("rag-documents-test", "raw/test-doc.pdf")

    assert result["valid"] is True
    assert result["content_type"] == "application/pdf"


@mock_aws
def test_validator_rejects_unsupported_type(aws_credentials):
    """Validator rejects .exe files with descriptive error."""
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket="rag-documents-test")
    s3.put_object(
        Bucket="rag-documents-test",
        Key="raw/malware.exe",
        Body=b"MZ fake exe",
        ContentType="application/x-msdownload",
    )

    from lambdas.ingestion.validator.handler import validate
    result = validate("rag-documents-test", "raw/malware.exe")

    assert result["valid"] is False
    assert "supported formats" in result["error"].lower()
```

### Example: Property test with Moto + Hypothesis

```python
from hypothesis import given, strategies as st, settings
from moto import mock_aws
import boto3

SUPPORTED_TYPES = {"pdf", "png", "jpeg", "tiff", "docx", "txt", "csv", "html"}
ALL_EXTENSIONS = SUPPORTED_TYPES | {"exe", "zip", "tar", "mp4", "dll", "bat", "sh"}


# Property 1: File type acceptance is determined by the supported set
@settings(max_examples=100)
@given(ext=st.sampled_from(sorted(ALL_EXTENSIONS)))
@mock_aws
def test_file_type_acceptance(ext):
    """For any file extension, acceptance iff in supported set."""
    from lambdas.shared.constants import SUPPORTED_FILE_TYPES
    from lambdas.ingestion.validator.handler import is_supported_type

    result = is_supported_type(f"document.{ext}")
    assert result == (ext in SUPPORTED_FILE_TYPES)
```

### Example: Testing with fakeredis (cache service)

```python
import fakeredis
from lambdas.shared.cache_service import CacheService


def test_cache_round_trip():
    """Cached embedding can be retrieved by text hash."""
    redis = fakeredis.FakeRedis(decode_responses=True)
    cache = CacheService(redis_client=redis)

    embedding = [0.1, 0.2, 0.3] * 341 + [0.1]  # 1024 dims
    cache.set_embedding("abc123hash", embedding)

    result = cache.get_embedding("abc123hash")
    assert result == embedding


def test_cache_miss_returns_none():
    """Cache miss returns None, not an error."""
    redis = fakeredis.FakeRedis(decode_responses=True)
    cache = CacheService(redis_client=redis)

    result = cache.get_embedding("nonexistent")
    assert result is None
```

### Example: Testing Bedrock calls with unittest.mock

```python
from unittest.mock import MagicMock, patch
import json


def test_embedding_generator_calls_bedrock_with_correct_params():
    """Embedding generator uses search_document input type for indexing."""
    mock_client = MagicMock()
    mock_client.invoke_model.return_value = {
        "body": json.dumps({"embedding": [0.1] * 1024}).encode()
    }

    from lambdas.shared.embedding_service import generate_embedding
    result = generate_embedding(
        client=mock_client,
        text="test chunk",
        input_type="search_document",
        dimensions=1024,
    )

    call_body = json.loads(
        mock_client.invoke_model.call_args[1]["body"]
    )
    assert call_body["inputText"] == "test chunk"
    assert call_body["dimensions"] == 1024
    assert len(result) == 1024
```

### Running all local tests

```bash
# All tests run locally — no AWS account needed
uv run pytest tests/unit/ tests/property/ -v

# With coverage report
uv run pytest tests/unit/ tests/property/ --cov=lambdas --cov-report=term-missing

# Just property tests
uv run pytest tests/property/ -v --tb=short
```

### What requires a real AWS account

Only integration tests (`tests/integration/`) hit real AWS services. These run in the CI/CD pipeline after deploying to the dev environment, not locally.

| Test Type | Runs Locally | Needs AWS | Mock Library |
|-----------|-------------|-----------|-------------|
| Unit tests | Yes | No | Moto + fakeredis + unittest.mock |
| Property tests | Yes | No | Moto + fakeredis + unittest.mock |
| Integration tests | No | Yes (dev env) | None — real services |

---

## Git Workflow

### Branch strategy

```
main              ← Production-ready code. Merges trigger the CI/CD pipeline.
├── dev           ← Integration branch. Merges from feature branches.
│   ├── feature/ingestion-validator
│   ├── feature/query-handler
│   ├── feature/streamlit-frontend
│   └── fix/chunker-overlap-bug
```

- Create feature branches from `dev` for each task (e.g., `feature/task-5.1-validator-lambda`)
- Open a pull request to `dev` when your work is ready
- After code review and tests pass, merge to `dev`
- When `dev` is stable and tested, merge to `main` to trigger production deployment

### Commit conventions

Use conventional commits for clarity:

```
feat(ingestion): implement validator Lambda for file type checking
feat(query): add hybrid search with k-NN + BM25
fix(chunker): correct overlap calculation for fixed-size strategy
infra(search): add OpenSearch domain with HNSW index mapping
test(property): add Property 12 - embedding dimension validation
docs: update deployment guide with CDK context configuration
```

### Pull request checklist

Before merging, ensure:

- [ ] `cdk synth --context env=dev` succeeds
- [ ] `pytest tests/unit/ tests/property/ -v` passes
- [ ] `black --check .` and `flake8 .` pass
- [ ] New Lambda handlers have corresponding unit tests
- [ ] CDK stack changes have been reviewed for IAM policy scope (no wildcard resources)

---

## CI/CD Pipeline

The CI/CD pipeline is defined in `stacks/pipeline_stack.py` and uses AWS CodePipeline with CodeBuild. It triggers automatically on pushes to the `main` branch.

### Pipeline stages

```
┌──────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌────────────┐    ┌─────────────┐    ┌────────────┐
│  Source   │───▶│      Build       │───▶│  Integration    │───▶│ Deploy Dev │───▶│  Manual     │───▶│Deploy Prod │
│ (GitHub/  │    │   (CodeBuild)    │    │    Tests        │    │ (CDK)      │    │  Approval   │    │  (CDK)     │
│ CodeCommit│    │                  │    │  (CodeBuild)    │    │            │    │             │    │            │
└──────────┘    └──────────────────┘    └─────────────────┘    └────────────┘    └─────────────┘    └────────────┘
```

### Stage 1: Source

- Triggers on commits to `main` branch
- Source: GitHub (via CodeStar connection) or AWS CodeCommit

### Stage 2: Build (CodeBuild)

Defined in `buildspec.yml`:

```yaml
version: 0.2

phases:
  install:
    runtime-versions:
      python: 3.11
    commands:
      - curl -LsSf https://astral.sh/uv/install.sh | sh
      - export PATH="$HOME/.local/bin:$PATH"
      - uv sync

  pre_build:
    commands:
      - echo "Linting..."
      - uv run black --check lambdas/ stacks/ tests/
      - uv run flake8 lambdas/ stacks/ tests/

  build:
    commands:
      - echo "Running unit and property tests..."
      - uv run pytest tests/unit/ tests/property/ -v --cov=lambdas --cov-report=xml

      - echo "Synthesizing CDK stacks..."
      - uv run cdk synth --context env=dev --output cdk.out

  post_build:
    commands:
      - echo "Build complete"

artifacts:
  files:
    - '**/*'
  name: BuildArtifact

reports:
  coverage:
    files:
      - 'coverage.xml'
    file-format: 'COBERTURAXML'
```

What happens here:
- Installs `uv` and syncs all dependencies from `pyproject.toml`
- Runs `black` and `flake8` for code quality
- Runs all unit and property-based tests with coverage
- Synthesizes CDK CloudFormation templates
- If any step fails, the pipeline stops and notifies via SNS

### Stage 3: Integration Tests (CodeBuild)

Runs against the dev environment after deployment:

1. Uploads a test PDF to S3 → verifies it appears in `processed/` within 120 seconds
2. Sends a query via the API → verifies the response contains expected content and citations
3. Repeats the same query → verifies cache hit (faster response, `cached: true`)
4. Uploads an unsupported file type → verifies 400 rejection with correct error format
5. Calls GET /health → verifies 200 response

### Stage 4: Deploy Dev

- Runs `cdk deploy --context env=dev --require-approval never`
- Deploys all stacks to the dev environment
- Uses smaller instances (t3.small OpenSearch, cache.t3.micro Redis, single-AZ)

### Stage 5: Manual Approval

- A team member reviews the dev deployment
- Approves or rejects promotion to production
- Rejection stops the pipeline; approval continues to prod

### Stage 6: Deploy Prod

- Runs `cdk deploy --context env=prod --require-approval never`
- Deploys all stacks to production
- Uses production-sized instances (r6g.xlarge OpenSearch, r6g.large Redis, multi-AZ)
- Lambda aliases with traffic shifting for safe rollout

---

## Manual Deployment (without CI/CD)

If you need to deploy manually before the CI/CD pipeline is set up:

### Deploy to dev

```bash
# Activate virtual environment
source .venv/bin/activate

# Synthesize and review
cdk synth --context env=dev
cdk diff --context env=dev

# Deploy all stacks
cdk deploy --all --context env=dev

# Or deploy individual stacks in order
cdk deploy NetworkStack --context env=dev
cdk deploy SecurityStack --context env=dev
cdk deploy StorageStack --context env=dev
cdk deploy SearchStack --context env=dev
cdk deploy CacheStack --context env=dev
cdk deploy ProcessingStack --context env=dev
cdk deploy ApiStack --context env=dev
cdk deploy MonitoringStack --context env=dev
```

### Deploy to production

```bash
cdk deploy --all --context env=prod
```

### Stack deployment order

Stacks have dependencies, but CDK handles ordering automatically when you use `--all`. If deploying individually, follow this order:

1. `NetworkStack` — VPC, subnets, security groups (no dependencies)
2. `SecurityStack` — KMS, Cognito, WAF (no dependencies)
3. `StorageStack` — S3 bucket (depends on SecurityStack for KMS key)
4. `SearchStack` — OpenSearch (depends on NetworkStack, SecurityStack)
5. `CacheStack` — ElastiCache Redis (depends on NetworkStack, SecurityStack)
6. `ProcessingStack` — Step Functions + ingestion Lambdas (depends on all above)
7. `ApiStack` — API Gateway + query Lambdas (depends on all above)
8. `MonitoringStack` — Dashboards, alarms (depends on all above)
9. `PipelineStack` — CodePipeline (deploy last, only needed once)

### Tear down

```bash
# Destroy all stacks (dev)
cdk destroy --all --context env=dev

# Destroy a specific stack
cdk destroy ApiStack --context env=dev
```

Note: S3 buckets with data and OpenSearch domains with indices will require manual cleanup or `removal_policy=DESTROY` in dev.

---

## Deploying the Streamlit Frontend

The frontend is containerized and deployed via App Runner or ECS Fargate.

### Build and test locally

```bash
cd frontend

# Install dependencies
uv sync

# Run locally
uv run streamlit run app.py --server.port 8501
```

### Build Docker image

```bash
cd frontend
docker build -t rag-frontend:latest .
docker run -p 8501:8501 \
  -e API_GATEWAY_URL=https://<api-id>.execute-api.<region>.amazonaws.com/prod \
  -e COGNITO_DOMAIN=https://<domain>.auth.<region>.amazoncognito.com \
  -e COGNITO_CLIENT_ID=<client-id> \
  rag-frontend:latest
```

### Push to ECR and deploy

```bash
# Create ECR repository (one-time)
aws ecr create-repository --repository-name rag-frontend

# Login, tag, and push
aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
docker tag rag-frontend:latest <account>.dkr.ecr.<region>.amazonaws.com/rag-frontend:latest
docker push <account>.dkr.ecr.<region>.amazonaws.com/rag-frontend:latest
```

The CDK `FrontendStack` (or `ApiStack`) handles App Runner/Fargate deployment from the ECR image.

---

## Environment Variables

Lambda functions receive configuration via environment variables set in the CDK stacks:

| Variable | Description | Set By |
|----------|-------------|--------|
| `S3_BUCKET_NAME` | Document storage bucket name | StorageStack |
| `OPENSEARCH_ENDPOINT` | OpenSearch domain endpoint | SearchStack |
| `REDIS_ENDPOINT` | ElastiCache Redis endpoint | CacheStack |
| `BEDROCK_EMBEDDING_MODEL` | Embedding model ID (e.g., `amazon.titan-embed-text-v2:0`) | ProcessingStack |
| `BEDROCK_LLM_MODEL` | LLM model ID (e.g., `anthropic.claude-3-5-sonnet-20241022-v2:0`) | ApiStack |
| `EMBEDDING_DIMENSIONS` | Vector dimensions (256/512/1024) | ProcessingStack |
| `CHUNKING_STRATEGY` | Default chunking strategy | ProcessingStack |
| `SNS_TOPIC_ARN` | Alert notification topic | MonitoringStack |
| `COGNITO_USER_POOL_ID` | Cognito user pool ID | SecurityStack |
| `ENVIRONMENT` | `dev` or `prod` | CDK context |

These are wired automatically by CDK — you don't set them manually.

---

## Rollback Procedures

### Lambda rollback

Lambda functions use aliases with version tracking. To roll back:

```bash
# List versions
aws lambda list-versions-by-function --function-name rag-query-handler

# Point alias to previous version
aws lambda update-alias \
  --function-name rag-query-handler \
  --name production \
  --function-version <previous-version-number>
```

### Infrastructure rollback

```bash
# CDK automatically supports rollback on deployment failure
# To manually roll back, deploy a previous commit:
git checkout <previous-commit-hash>
cdk deploy --all --context env=prod
```

### OpenSearch index rollback

The system uses index aliases for zero-downtime switching:

```bash
# If a new index is bad, switch alias back to the old index
# This is handled by the vector_indexer Lambda's reindex() method
```

---

## Monitoring After Deployment

Once deployed, monitor the system via:

- CloudWatch Dashboard: `RAG-Service-Overview` — query latency, cache hit rate, error rates
- CloudWatch Alarms: Configured in MonitoringStack, notifications via SNS
- X-Ray Service Map: Visual trace of request flow through API Gateway → Lambda → OpenSearch → Bedrock
- CloudWatch Logs Insights: Query structured logs for debugging

### Useful log queries

```
-- Find slow queries (> 5 seconds)
fields @timestamp, query, latency_ms
| filter latency_ms > 5000
| sort latency_ms desc
| limit 20

-- Error breakdown by component
fields @timestamp, component, error_type
| filter level = "ERROR"
| stats count() by component, error_type

-- Cache hit rate over time
fields cache_hit
| stats avg(cache_hit) as hit_rate by bin(5m)
```

---

## Quick Reference

| Action | Command |
|--------|---------|
| Install dependencies | `uv sync` |
| Run unit tests | `pytest tests/unit/ -v` |
| Run property tests | `pytest tests/property/ -v` |
| Run all tests | `pytest tests/unit/ tests/property/ -v` |
| Lint | `black --check . && flake8 .` |
| Synthesize CDK | `cdk synth --context env=dev` |
| Preview changes | `cdk diff --context env=dev` |
| Deploy dev | `cdk deploy --all --context env=dev` |
| Deploy prod | `cdk deploy --all --context env=prod` |
| Destroy dev | `cdk destroy --all --context env=dev` |
| Run frontend locally | `cd frontend && uv run streamlit run app.py` |
