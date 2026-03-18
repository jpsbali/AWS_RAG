# AWS-Native RAG Service — Specification Index

## Table of Contents

| # | Document | Location | Description |
|---|----------|----------|-------------|
| 1 | [Architecture Reference](#1-architecture-reference) | [aws_rag_infra.md](./aws_rag_infra.md) | Complete AWS-native RAG architecture covering 11 layers |
| 2 | [Project Plan](#2-project-plan) | [Cloud_Native_RAG_Project_Plan.md](./Cloud_Native_RAG_Project_Plan.md) | Community project plan — scope, teams, timeline |
| 3 | [Requirements](#3-requirements) | [requirements.md](../.kiro/specs/aws-native-rag-service/requirements.md) | 19 requirements with user stories and acceptance criteria |
| 4 | [Design](#4-design) | [design.md](../.kiro/specs/aws-native-rag-service/design.md) | Technical design — architecture diagrams, component interfaces, data models, 30 correctness properties |
| 5 | [Tasks](#5-tasks) | [tasks.md](../.kiro/specs/aws-native-rag-service/tasks.md) | 18 implementation tasks with sub-tasks, mapped to requirements |
| 6 | [Development & Deployment Guide](#6-development--deployment-guide) | [DEVELOPMENT_AND_DEPLOYMENT_GUIDE.md](./DEVELOPMENT_AND_DEPLOYMENT_GUIDE.md) | Developer workflow, local testing with Moto, CI/CD pipeline, deployment procedures |

---

## 1. Architecture Reference

**File:** [aws_rag_infra.md](./aws_rag_infra.md)

The foundational architecture document describing a production-ready RAG system built entirely on AWS-native services. Covers:

- Serverless-first, event-driven architecture with two primary pipelines (ingestion + query)
- 11 component layers: Data Ingestion (S3 + EventBridge), Document Processing (Lambda + Textract + Comprehend), Chunking, Embedding Generation (Bedrock Titan v2), Vector Database (OpenSearch), LLM Integration (Bedrock Claude), API Layer (API Gateway), Caching (ElastiCache Redis), Monitoring (CloudWatch + X-Ray), Security (IAM, Cognito, KMS, WAF), CI/CD (CDK + CodePipeline)
- Detailed configuration for each service including index mappings, IAM policies, and cluster sizing
- Cost estimates (~$2,118/month production) and optimization strategies
- Design patterns: event-driven, serverless-first, caching hierarchy, circuit breaker, retry with backoff, dead letter queues

---

## 2. Project Plan

**File:** [Cloud_Native_RAG_Project_Plan.md](./Cloud_Native_RAG_Project_Plan.md)

Community-driven project where 3 teams of 10 build the same RAG application on AWS, GCP, and Azure. Our focus is the AWS track. Key deliverables:

- Document Ingestion Pipeline (S3 upload, parse, chunk)
- Embedding Generation (Bedrock Titan v2)
- Vector Store (OpenSearch with hybrid search)
- Retrieval & Generation API
- Frontend Application (Streamlit)
- RAGAS Evaluation Pipeline

---

## 3. Requirements

**File:** [requirements.md](../.kiro/specs/aws-native-rag-service/requirements.md)

19 requirements covering all functional and non-functional aspects:

| Req # | Title | Summary |
|-------|-------|---------|
| 1 | Document Upload and Storage | S3 with KMS encryption, versioning, lifecycle policies, supported file types |
| 2 | Event-Driven Ingestion Trigger | EventBridge rules, S3 event routing to Step Functions, DLQ |
| 3 | Document Processing Orchestration | Step Functions workflow with retry/catch, intermediate S3 storage |
| 4 | Text Extraction | Textract for PDF/images, native libs for text formats, quality threshold |
| 5 | Metadata Extraction with NLP | Comprehend NER, key phrases, language detection, PII detection |
| 6 | Text Chunking | 4 strategies (fixed, semantic, recursive, sentence window), chunk metadata |
| 7 | Embedding Generation | Bedrock Titan v2, configurable dimensions, batching, caching, normalization |
| 8 | Vector Database Indexing | OpenSearch with HNSW, bulk/incremental indexing, alias-based reindexing |
| 9 | Hybrid Search and Retrieval | Vector + BM25, metadata filtering, reranking, MMR diversity |
| 10 | LLM Response Generation | Bedrock Claude 3.5 Sonnet, streaming, guardrails, prompt templates |
| 11 | Query Pipeline Caching | ElastiCache Redis — embeddings, results, responses, sessions |
| 12 | API Layer | API Gateway REST + WebSocket, schema validation, rate limiting, CORS |
| 13 | Frontend Application | Streamlit — chat interface, document upload, source citations |
| 14 | Authentication and Authorization | Cognito, IAM least-privilege, KMS, WAF, Secrets Manager |
| 15 | Monitoring and Observability | CloudWatch metrics/logs/alarms, X-Ray tracing, structured logging |
| 16 | CI/CD and Infrastructure as Code | AWS CDK (Python), CodePipeline, dev/prod environments |
| 17 | RAGAS Evaluation Pipeline | Context precision/recall, faithfulness, answer relevancy, scheduled runs |
| 18 | Error Handling and Resilience | Retry, DLQ, graceful degradation, idempotent ingestion |
| 19 | Performance | Cache hit < 500ms p95, full query < 15s p95, ingestion < 120s p95 |

Includes a detailed cost alternatives section comparing options for vector databases, LLMs, embedding models, compute, caching, and cluster sizing.

---

## 4. Design

**File:** [design.md](../.kiro/specs/aws-native-rag-service/design.md)

Technical design document with:

- 3 Mermaid architecture diagrams (high-level system, ingestion state machine, query sequence)
- Key design decisions table (OpenSearch, Bedrock Titan v2, Claude 3.5 Sonnet, Step Functions, Lambda, ElastiCache, CDK, Cognito)
- 17 component specifications with Python interfaces:
  - Storage (S3), Event Trigger (EventBridge), Orchestrator (Step Functions)
  - Text Extractor (Textract), Metadata Enricher (Comprehend), Chunking Engine
  - Embedding Generator (Bedrock), Vector Store (OpenSearch), Query Handler
  - LLM Generator (Bedrock Claude), Cache Layer (ElastiCache Redis)
  - API Layer (API Gateway), Frontend (Streamlit)
  - Auth & Security, Monitoring, CI/CD, RAGAS Evaluation
- 6 data model schemas: Document Record, Chunk Record, Query Log, Cache Entry, Step Functions Execution, Evaluation Report
- 30 correctness properties mapped to requirements (for property-based testing with Hypothesis)
- Error handling strategies for both ingestion and query pipelines
- Testing strategy: unit tests + property-based tests + integration tests

---

## 5. Tasks

**File:** [tasks.md](../.kiro/specs/aws-native-rag-service/tasks.md)

18 top-level implementation tasks with sub-tasks:

| Task | Title | Key Deliverables |
|------|-------|-----------------|
| 1 | Project scaffolding | CDK init, directory structure, pyproject.toml, shared models/constants |
| 2 | Network, Storage, Security stacks | VPC, S3 bucket, KMS, Cognito, WAF |
| 3 | Search, Cache, Monitoring stacks | OpenSearch domain, ElastiCache Redis, CloudWatch dashboards/alarms |
| 4 | Checkpoint | Validate CDK synth |
| 5 | Validator + Text Extractor | File type validation, Textract/native extraction |
| 6 | Metadata Enricher + Chunker | Comprehend NLP, 4 chunking strategies |
| 7 | Embedding Generator + Vector Indexer | Bedrock Titan v2 embeddings, OpenSearch bulk indexing |
| 8 | Step Functions state machine | Processing orchestration, EventBridge wiring, retry/catch |
| 9 | Checkpoint | Validate ingestion pipeline |
| 10 | Query Handler + LLM Generator | Cache service, hybrid search, Claude generation, streaming |
| 11 | API Layer | API Gateway REST + WebSocket, Lambda handlers |
| 12 | Checkpoint | Validate query pipeline + API |
| 13 | Frontend | Streamlit app, Docker, App Runner/Fargate deployment |
| 14 | RAGAS Evaluation | Evaluation Lambda, scheduled + on-demand triggers |
| 15 | Observability + Resilience | X-Ray tracing, DLQs, graceful degradation, idempotency |
| 16 | Checkpoint | Full system validation |
| 17 | CI/CD Pipeline | CodePipeline, CodeBuild, integration tests |
| 18 | Final Checkpoint | Complete system validation |

Property-based test tasks (marked optional with `*`) cover all 30 correctness properties using Hypothesis. Local testing uses Moto, fakeredis, and unittest.mock — no AWS account needed for unit/property tests.

---

## 6. Development & Deployment Guide

**File:** [DEVELOPMENT_AND_DEPLOYMENT_GUIDE.md](./DEVELOPMENT_AND_DEPLOYMENT_GUIDE.md)

Covers the full developer workflow:

- Prerequisites and environment setup (uv, CDK, AWS CLI)
- Project structure reference
- Local development workflow (writing Lambdas, CDK stacks, running tests)
- Local testing with Moto (mocking strategy per service, conftest.py fixtures, fakeredis for Redis, unittest.mock for Bedrock/OpenSearch/Textract/Comprehend)
- Git workflow (branch strategy, conventional commits, PR checklist)
- CI/CD pipeline (Source → Build → Integration Tests → Deploy Dev → Manual Approval → Deploy Prod)
- Manual deployment procedures and stack ordering
- Streamlit frontend containerization and ECR deployment
- Environment variables reference
- Rollback procedures (Lambda, infrastructure, OpenSearch index)
- Post-deployment monitoring and CloudWatch Logs Insights queries
