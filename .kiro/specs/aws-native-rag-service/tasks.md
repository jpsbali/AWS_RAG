# Implementation Plan: AWS-Native RAG Service

## Overview

Implement a production-grade, fully AWS-native RAG service using Python CDK for infrastructure, Python 3.11 Lambda functions for compute, and Hypothesis for property-based testing. The implementation proceeds from project scaffolding through infrastructure stacks, ingestion pipeline Lambdas, query pipeline Lambdas, API layer, frontend, evaluation pipeline, and CI/CD — each step building on the previous.

## Tasks

- [x] 1. Project scaffolding and shared utilities
  - [x] 1.1 Initialize CDK Python project and directory structure
    - Run `cdk init app --language python` in the project root
    - Create directory structure: `stacks/`, `lambdas/`, `lambdas/shared/`, `lambdas/ingestion/`, `lambdas/query/`, `lambdas/evaluation/`, `frontend/`, `tests/`, `tests/unit/`, `tests/property/`, `tests/integration/`, `tests/fixtures/`
    - Set up `pyproject.toml` with project metadata, CDK, boto3, hypothesis, pytest, moto, fakeredis, ragas dependencies (using uv for package management)
    - Configure `cdk.json` with context values for dev and prod environments
    - _Requirements: 16.1, 16.3_

  - [x] 1.2 Implement shared data models and constants
    - Create `lambdas/shared/models.py` with dataclasses: `ExtractionResult`, `DocumentMetadata`, `Chunk`, `ChunkConfig`, `ChunkMetadata`, `EmbeddingResult`, `QueryRequest`, `QueryResponse`, `SourceCitation`, `GenerationConfig`, `EvaluationReport`
    - Create `lambdas/shared/constants.py` with supported file types set, S3 prefix constants, cache TTL values, default chunking parameters, embedding model ID, LLM model IDs
    - Create `lambdas/shared/utils.py` with helper functions: SHA-256 hashing, correlation ID generation, structured JSON logger setup
    - _Requirements: 1.5, 6.6, 7.1, 10.5, 15.2_

  - [ ]* 1.3 Write property tests for file type validation (Property 1)
    - **Property 1: File type acceptance is determined by the supported set**
    - Generate random file extensions from supported + unsupported sets; assert accepted iff in supported set
    - **Validates: Requirements 1.5, 1.6**

- [x] 2. CDK infrastructure — Network, Storage, and Security stacks
  - [x] 2.1 Implement NetworkStack
    - Create `stacks/network_stack.py` with VPC (2 AZs, public/private subnets), security groups for Lambda, OpenSearch, ElastiCache, and VPC endpoints for S3, Bedrock, Comprehend, Textract, Step Functions, CloudWatch
    - _Requirements: 14.6, 16.1_

  - [x] 2.2 Implement StorageStack
    - Create `stacks/storage_stack.py` with S3 bucket (SSE-KMS encryption, versioning on `raw/` prefix, lifecycle policies for Intelligent-Tiering at 30d / Glacier at 90d / failed expiry at 180d), EventBridge notifications for `ObjectCreated` on `raw/` prefix, CORS configuration
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1_

  - [x] 2.3 Implement SecurityStack
    - Create `stacks/security_stack.py` with KMS customer-managed key (`alias/rag-encryption-key` with auto-rotation), Cognito user pool (password policy, optional MFA, OAuth 2.0 flows), WAF WebACL (SQL injection, XSS, rate-based rules at 2000 req/5min/IP), Secrets Manager secret for OpenSearch credentials
    - _Requirements: 14.1, 14.3, 14.4, 14.5_

  - [ ]* 2.4 Write property test for S3 encryption metadata (Property 2)
    - **Property 2: Uploaded documents are stored with KMS encryption**
    - Mock S3 put_object; verify SSE-KMS metadata is set with customer-managed key for random uploads
    - **Validates: Requirements 1.1**

- [x] 3. CDK infrastructure — Search, Cache, and Monitoring stacks
  - [x] 3.1 Implement SearchStack
    - Create `stacks/search_stack.py` with OpenSearch domain (3 data nodes r6g.xlarge, 3 dedicated masters r6g.large, multi-AZ, gp3 500GB, KMS encryption, TLS, VPC deployment), index template with knn_vector mapping (1024 dims, HNSW ef_construction=512, m=16), index alias configuration
    - _Requirements: 8.1, 8.2, 8.3, 8.7_

  - [x] 3.2 Implement CacheStack
    - Create `stacks/cache_stack.py` with ElastiCache Redis 7.x cluster (cache.r6g.large, 2 shards, 1 replica per shard, multi-AZ, KMS encryption at rest, TLS in transit, VPC private subnets)
    - _Requirements: 11.5_

  - [x] 3.3 Implement MonitoringStack
    - Create `stacks/monitoring_stack.py` with SNS topic for alerts, CloudWatch dashboard (query latency, cache hit rate, error rates, processing times), CloudWatch alarms (HighQueryLatency p95 > 10s, HighErrorRate > 5%, StepFunctionFailures > 0, OpenSearchClusterRed, CacheMemoryHigh > 90%), X-Ray tracing configuration
    - _Requirements: 15.1, 15.3, 15.4, 15.5_

- [x] 4. Checkpoint — Validate CDK stacks synthesize
  - Ensure `cdk synth` succeeds for all stacks, ask the user if questions arise.

- [x] 5. Ingestion pipeline — Validator and Text Extractor Lambdas
  - [x] 5.1 Implement Validator Lambda (`lambdas/ingestion/validator/handler.py`)
    - Validate file type against supported set (PDF, PNG, JPEG, TIFF, DOCX, TXT, CSV, HTML)
    - Validate file size and S3 object existence
    - Return validation result with content type classification for routing
    - Reject unsupported types with descriptive error listing supported formats
    - _Requirements: 1.5, 1.6, 4.5_

  - [x] 5.2 Implement Text Extractor Lambda (`lambdas/ingestion/text_extractor/handler.py`)
    - Route to Textract for PDF/image formats (DetectDocumentText, AnalyzeDocument for tables/forms, StartDocumentAnalysis for >15 pages)
    - Route to native Python libs for TXT (direct read), CSV (csv/pandas), HTML (BeautifulSoup), DOCX (python-docx)
    - Validate extraction quality against configurable minimum threshold (default: 50 chars)
    - Return `ExtractionResult` with text, pages, tables, forms, extraction_method, confidence
    - Store extracted text in S3 `processed/text/` prefix
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 3.5_

  - [ ]* 5.3 Write property tests for extraction routing and quality (Properties 7, 8)
    - **Property 7: Extraction method matches document content type**
    - Generate random content types; assert Textract used for PDF/images, native for text formats
    - **Property 8: Extraction quality validation enforces minimum threshold**
    - Generate random strings of varying lengths; assert pass iff length >= threshold
    - **Validates: Requirements 4.1, 4.4, 4.5**

- [x] 6. Ingestion pipeline — Metadata Enricher and Chunker Lambdas
  - [x] 6.1 Implement Metadata Enricher Lambda (`lambdas/ingestion/metadata_enricher/handler.py`)
    - Invoke Comprehend detect_entities, detect_key_phrases, detect_dominant_language
    - Optionally invoke detect_pii_entities when PII detection is enabled
    - Return `DocumentMetadata` with entities, key_phrases, language, pii_detected, pii_entities
    - Store metadata JSON in S3 `processed/metadata/` prefix
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 3.5_

  - [x] 6.2 Implement Chunker Lambda (`lambdas/ingestion/chunker/handler.py`)
    - Implement four chunking strategies: fixed-size, semantic, recursive character, sentence window
    - Fixed-size: split by token count (default 1000) with overlap (default 200)
    - Semantic: split at headers/paragraphs, enforce min 200 / max 1500 tokens
    - Recursive: apply separators in order ["\n\n", "\n", ". ", " "]
    - Sentence window: center on sentence with configurable window (default 3)
    - Attach metadata to each chunk: chunk_id, document_id, chunk_index, chunk_size, start/end positions, section_title, page_number, entities, key_phrases
    - Store chunks JSON in S3 `processed/chunks/` prefix
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 3.5_

  - [ ]* 6.3 Write property tests for metadata and chunking (Properties 9, 10, 11)
    - **Property 9: Metadata extraction produces complete structured output**
    - Generate random text; mock Comprehend; assert output contains entities, key_phrases, language; with PII enabled, assert pii_detected and pii_entities present
    - **Property 10: Chunks respect configured size bounds for any strategy**
    - Generate random texts with random ChunkConfig; assert all chunks within configured bounds
    - **Property 11: Every chunk contains required metadata fields**
    - Generate random chunks; assert chunk_id, document_id, chunk_index, chunk_size, start_position, end_position, and metadata fields present
    - **Validates: Requirements 5.1-5.5, 6.2, 6.3, 6.5, 6.6**

- [ ] 7. Ingestion pipeline — Embedding Generator and Vector Indexer Lambdas
  - [x] 7.1 Implement Embedding Generator Lambda (`lambdas/ingestion/embedding_generator/handler.py`)
    - Batch chunks into groups of up to 100
    - Check Redis cache for each chunk by SHA-256 hash of text
    - Generate embeddings via Bedrock Titan v2 (amazon.titan-embed-text-v2:0) with input_type=search_document
    - Support configurable dimensions (256, 512, 1024), default 1024
    - Normalize embeddings (L2 normalization)
    - Cache new embeddings in Redis with 30-day TTL
    - Retry on Bedrock throttling with exponential backoff up to 5 times
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7_

  - [x] 7.2 Implement Vector Indexer Lambda (`lambdas/ingestion/vector_indexer/handler.py`)
    - Bulk index chunks with embeddings into OpenSearch using chunk_id as document ID (idempotent upsert)
    - Batch size 500-1000 documents per bulk request
    - Set refresh_interval to 30s during bulk load, 1s for incremental
    - Support zero-downtime reindexing via alias switching
    - _Requirements: 8.4, 8.5, 8.6, 18.4_

  - [ ]* 7.3 Write property tests for embeddings (Properties 12, 13, 14, 15)
    - **Property 12: Embeddings have correct dimensions and are normalized**
    - Generate random text chunks with varying dimension configs; assert vector length matches config and L2 norm ≈ 1.0
    - **Property 13: Embedding input type matches use case**
    - Generate random calls in indexing vs query contexts; assert input_type is search_document or search_query respectively
    - **Property 14: Embedding batches do not exceed 100 chunks**
    - Generate random chunk lists (1-500); assert all batches have size <= 100
    - **Property 15: Embedding cache round-trip**
    - Generate random text → embed → cache → retrieve; assert retrieved embedding matches original
    - **Validates: Requirements 7.2, 7.3, 7.4, 7.5, 7.7**

- [ ] 8. Step Functions state machine definition
  - [x] 8.1 Implement ProcessingStack with Step Functions state machine
    - Create `stacks/processing_stack.py` with all 6 ingestion Lambda functions (validator, text_extractor, metadata_enricher, chunker, embedding_generator, vector_indexer)
    - Define Step Functions state machine: Validate → Choice (document type) → Extract (Textract or Native) → Quality Check → Extract Metadata → Chunk → Generate Embeddings → Index → Success
    - Configure per-step retry with exponential backoff (as per design: validator 3x/2-8s, extractor 3x/5-20s, enricher 3x/2-8s, chunker 2x/2-4s, embedder 5x/2-32s, indexer 3x/2-8s)
    - Configure Catch blocks routing to failure handler (move to failed/, publish SNS)
    - Wire EventBridge rule to trigger state machine on S3 raw/ ObjectCreated events
    - Configure Lambda memory, timeout, VPC, and IAM roles per design spec
    - _Requirements: 2.2, 2.3, 3.1, 3.2, 3.3, 3.4, 3.5, 14.2_

  - [ ]* 8.2 Write property tests for orchestration (Properties 3, 4, 5, 6)
    - **Property 3: EventBridge routes valid S3 events to Step Functions**
    - Generate random S3 events with varying prefixes and file types; assert routing iff raw/ prefix + supported type
    - **Property 4: Orchestrator executes pipeline steps in correct sequence**
    - Mock pipeline; process random documents; assert step order: validate → extract → metadata → chunk → embed → index
    - **Property 5: Failed documents after retry exhaustion are moved to failed/ with notification**
    - Inject random step failures; assert document moved to failed/ and SNS notification published
    - **Property 6: Intermediate results are persisted between pipeline steps**
    - Process random documents; assert intermediate outputs stored in S3 processed/ sub-prefixes between steps
    - **Validates: Requirements 2.2, 3.1, 3.4, 3.5**

- [x] 9. Checkpoint — Validate ingestion pipeline
  - Ensure all ingestion Lambda handlers import correctly, CDK synth succeeds with ProcessingStack, and all property tests pass. Ask the user if questions arise.

- [ ] 10. Query pipeline — Query Handler and LLM Generator Lambdas
  - [x] 10.1 Implement Cache Service (`lambdas/shared/cache_service.py`)
    - Implement `CacheService` class with get/set, get_embedding/set_embedding, get_query_response/set_query_response methods
    - Key patterns: `emb:{sha256}` (30d TTL), `qr:{sha256}` (24h TTL), `llm:{sha256}` (1h TTL), `sess:{session_id}` (2h TTL)
    - Handle Redis connection with TLS, graceful degradation on cache unavailability
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5_

  - [x] 10.2 Implement Query Handler Lambda (`lambdas/query/query_handler/handler.py`)
    - Step 1: Hash query + filters → check Redis cache for cached response
    - Step 2: On cache miss, generate query embedding via Bedrock Titan v2 (input_type=search_query)
    - Step 3: Execute hybrid search in OpenSearch (k-NN + BM25 + metadata filters)
    - Step 4: Optionally rerank results via Bedrock/SageMaker
    - Step 5: Apply MMR for diversity, select top 5-10 chunks
    - Step 6: Assemble context prompt with system instructions and retrieved chunks (respect context window limit)
    - Step 7: Invoke Bedrock Claude 3.5 Sonnet via InvokeModelWithResponseStream
    - Step 8: Cache full response in Redis (TTL: 1 hour)
    - Step 9: Return response with sources/citations
    - Emit structured JSON logs with correlation ID and publish CloudWatch custom metrics (QueryLatency, CacheHitRate)
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 10.1, 10.2, 10.3, 10.4, 11.1, 11.2, 11.3, 15.1, 15.2, 19.1, 19.2_

  - [x] 10.3 Implement LLM Generator module (`lambdas/shared/llm_generator.py`)
    - Implement `LLMGenerator` class with `generate()` and `generate_stream()` methods
    - System prompt template: context-only answers, cite sources, acknowledge limitations
    - RAG prompt template: context + question + instructions
    - Support configurable temperature, top_p, max_tokens via GenerationConfig
    - Primary model: Claude 3.5 Sonnet, fallback: Claude 3.5 Haiku
    - Retry up to 3 times with exponential backoff on Bedrock errors
    - Optional Bedrock Guardrails integration
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7_

  - [ ]* 10.4 Write property tests for query pipeline (Properties 16, 17, 18, 19, 20, 21, 22)
    - **Property 16: Hybrid search uses both vector and BM25 components**
    - Generate random queries; assert OpenSearch query includes both k-NN and BM25 components
    - **Property 17: Metadata filters are correctly applied to search results**
    - Generate random queries with random filters; assert all results satisfy filter conditions
    - **Property 18: Search results are ordered by score and limited to k**
    - Generate random result sets with random k; assert descending score order and count <= k
    - **Property 19: Assembled context respects LLM context window limit**
    - Generate random chunk sets of varying sizes; assert total tokens <= context window limit
    - **Property 20: LLM prompt includes system template and bounded chunk count**
    - Generate random queries with random chunks; assert prompt contains system template and 5-10 chunks
    - **Property 21: LLM generation parameters match configuration**
    - Generate random GenerationConfig values; assert Bedrock API call includes matching temperature, top_p, max_tokens
    - **Property 22: Cache round-trip for query responses**
    - Generate random queries; mock pipeline; assert second identical query returns cached response without re-invoking search/LLM
    - **Validates: Requirements 9.1, 9.2, 9.3, 9.6, 10.2, 10.3, 10.5, 11.1, 11.2, 11.3**

- [ ] 11. API Layer — API Gateway and Lambda handlers
  - [x] 11.1 Implement ApiStack with REST and WebSocket APIs
    - Create `stacks/api_stack.py` with REST API Gateway: POST /query (30s timeout), POST /ingest (10s), GET /document/{id} (10s), GET /health (5s), GET /metrics (10s)
    - WebSocket API: $connect, $disconnect, query routes (10min connection timeout, 5min idle)
    - Cognito authorizer for JWT validation on all endpoints
    - JSON schema request validators for POST /query and POST /ingest
    - Rate limiting: 1000 req/s per API key, 5000 burst
    - CORS configuration for Streamlit frontend
    - Access logging to CloudWatch, X-Ray tracing enabled
    - Attach WAF WebACL from SecurityStack
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.6, 12.7, 12.8, 14.1, 14.4_

  - [x] 11.2 Implement API Lambda handlers
    - `lambdas/query/ingest_trigger/handler.py`: Accept file upload, store in S3 raw/ prefix, return document_id and status
    - `lambdas/query/document_handler/handler.py`: Return document metadata and processing status by document_id
    - `lambdas/query/health_handler/handler.py`: Health check returning 200 with component status
    - `lambdas/query/metrics_handler/handler.py`: Return system metrics summary from CloudWatch
    - Standardized error response format: `{"error": {"code": str, "message": str, "request_id": str}}`
    - _Requirements: 12.1, 12.5, 12.7, 12.8_

  - [ ]* 11.3 Write property tests for API layer (Properties 23, 24, 25)
    - **Property 23: API request schema validation rejects invalid requests**
    - Generate random invalid request payloads (missing query field, invalid filter types); assert 400 response with validation error
    - **Property 24: API error responses follow standardized format**
    - Generate random error conditions; assert response contains error object with code, message, request_id fields
    - **Property 25: Unauthenticated requests are rejected**
    - Generate random requests with/without valid JWT; assert 401 for missing/invalid tokens, authenticated forwarding for valid tokens
    - **Validates: Requirements 12.3, 12.5, 14.1**

- [x] 12. Checkpoint — Validate query pipeline and API layer
  - Ensure all query Lambda handlers import correctly, CDK synth succeeds with ApiStack, and all property tests pass. Ask the user if questions arise.

- [ ] 13. Frontend application (Streamlit)
  - [x] 13.1 Implement Streamlit frontend
    - Create `frontend/app.py` with multi-page layout: Chat Interface, Document Upload, Document Browser
    - Chat Interface: query input, streamed response display via WebSocket, source citations with document links, conversation history
    - Document Upload: file picker (supported types), upload progress bar, processing status tracker via polling GET /document/{id}
    - Document Browser: list uploaded documents, view metadata, processing status
    - Authenticate via Cognito hosted UI (OAuth 2.0 / OIDC)
    - Create `frontend/Dockerfile` for containerized deployment
    - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.5_

  - [x] 13.2 Add frontend deployment to CDK
    - Add App Runner or ECS Fargate service to ApiStack or a new FrontendStack
    - Configure auto-scaling based on concurrent connections
    - Wire environment variables for API Gateway endpoint URL and Cognito config
    - _Requirements: 13.1_

- [ ] 14. RAGAS evaluation pipeline
  - [x] 14.1 Implement Evaluation Lambda (`lambdas/evaluation/evaluator/handler.py`)
    - Accept evaluation dataset S3 key as input
    - Load question-answer-context triples from S3
    - Run RAGAS evaluation with Bedrock as evaluator LLM: context_precision, context_recall, context_relevancy, faithfulness, answer_relevancy
    - Produce per-metric and aggregate scores
    - Store evaluation report JSON in S3
    - Publish metric results to CloudWatch custom metrics (`RAG/Evaluation` namespace)
    - _Requirements: 17.1, 17.2, 17.3, 17.4, 17.5_

  - [x] 14.2 Wire evaluation pipeline triggers
    - Add POST /evaluate endpoint to API Gateway for on-demand execution
    - Add EventBridge scheduled rule (default: weekly) to trigger evaluation Lambda
    - _Requirements: 17.6_

  - [ ]* 14.3 Write property tests for evaluation (Properties 27, 28)
    - **Property 27: Evaluation reports contain all required RAGAS metrics**
    - Generate random evaluation datasets; mock RAGAS; assert report contains all 5 metrics + aggregate score + S3 key
    - **Property 28: Evaluation pipeline accepts valid datasets**
    - Generate random valid datasets (JSON array of question-answer-context triples); assert pipeline processes without error
    - **Validates: Requirements 17.1, 17.2, 17.3, 17.4**

- [ ] 15. Observability and resilience wiring
  - [x] 15.1 Add structured logging and X-Ray tracing to all Lambdas
    - Add AWS X-Ray SDK instrumentation to all Lambda functions
    - Add custom X-Ray subsegments for cache checks, OpenSearch queries, Bedrock calls
    - Ensure all Lambda logs use structured JSON format with correlation IDs
    - Configure X-Ray trace sampling at 5% for production
    - _Requirements: 15.2, 15.3_

  - [x] 15.2 Implement DLQ and resilience patterns
    - Add SQS dead letter queues for: EventBridge rule failures, Lambda async invocations (max 2 retries), Step Functions task failures
    - Implement graceful degradation in Query Handler: return 503 + Retry-After when OpenSearch unavailable, proceed without cache when Redis unavailable
    - Implement Bedrock throttling retry with exponential backoff + jitter in LLM Generator
    - Ensure ingestion idempotency: use document_id/chunk_id as deduplication keys, overwrite on reprocessing
    - _Requirements: 18.1, 18.2, 18.3, 18.4, 18.5_

  - [ ]* 15.3 Write property tests for observability and resilience (Properties 26, 29, 30)
    - **Property 26: Observability output for every query execution**
    - Generate random query executions; assert structured JSON logs with correlation ID emitted and CloudWatch metrics published
    - **Property 29: Ingestion pipeline idempotency**
    - Generate random documents; process twice; assert same indexed result with no duplicate chunks
    - **Property 30: Lambda failure retry and DLQ routing**
    - Generate random Lambda failures; assert retry up to 2 times then route to SQS DLQ
    - **Validates: Requirements 15.1, 15.2, 18.1, 18.4**

- [x] 16. Checkpoint — Full system validation
  - Ensure all CDK stacks synthesize, all property tests pass, all Lambda handlers import correctly. Ask the user if questions arise.

- [ ] 17. CI/CD pipeline
  - [x] 17.1 Implement CI/CD with CodePipeline and CodeBuild
    - Create `stacks/pipeline_stack.py` with CodePipeline: Source (GitHub/CodeCommit main branch) → Build (CodeBuild: lint, unit tests, property tests, cdk synth) → Test (integration tests) → Deploy Dev (cdk deploy) → Manual Approval → Deploy Prod (cdk deploy)
    - CodeBuild buildspec: install uv, run uv sync, run pytest (unit + property tests), run cdk synth
    - Environment-specific config via CDK context or SSM Parameter Store
    - _Requirements: 16.1, 16.2, 16.3_

  - [x] 17.2 Implement integration tests
    - Create `tests/integration/test_e2e.py` with tests:
      1. Upload test PDF → verify appears in `processed/` within 120s
      2. Query with known question → verify answer contains expected content and citations
      3. Repeat query → verify cache hit and faster response
      4. Upload unsupported file → verify rejection with correct error format
      5. Verify GET /health returns 200
    - _Requirements: 16.4, 19.1, 19.2, 19.3_

- [x] 18. Final checkpoint — Complete system validation
  - Ensure all CDK stacks synthesize, all tests (unit, property, integration) pass, and the full pipeline is wired end-to-end. Ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation at key milestones
- Property tests validate the 30 correctness properties from the design document using Hypothesis
- All code is Python: CDK Python for infrastructure, Python 3.11 for Lambda functions
- Local testing uses Moto (AWS mocks), fakeredis (Redis mock), and unittest.mock (Bedrock, OpenSearch, Textract, Comprehend) — no AWS account needed for unit/property tests
- Unit tests and property tests are complementary — properties verify universal correctness, unit tests verify specific examples and edge cases
