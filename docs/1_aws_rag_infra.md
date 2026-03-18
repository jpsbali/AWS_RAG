# AWS-Native RAG Service Architecture

## Table of Contents
- [Overview](#overview)
- [Architecture Diagram](#architecture-diagram)
- [Core Components](#core-components)
- [1. Data Ingestion Layer](#1-data-ingestion-layer)
- [2. Document Processing & Extraction](#2-document-processing--extraction)
- [3. Chunking & Text Processing](#3-chunking--text-processing)
- [4. Embedding Generation](#4-embedding-generation)
- [5. Vector Database](#5-vector-database)
- [6. LLM Integration](#6-llm-integration)
- [7. API & Application Layer](#7-api--application-layer)
- [8. Caching Layer](#8-caching-layer)
- [9. Monitoring & Observability](#9-monitoring--observability)
- [10. Security & Governance](#10-security--governance)
- [11. Deployment & CI/CD](#11-deployment--cicd)
- [Architecture Flow](#architecture-flow)
- [Design Patterns](#design-patterns)
- [Cost Optimization](#cost-optimization)

---

## Overview

This document outlines a complete, production-ready RAG (Retrieval Augmented Generation) system built entirely using AWS-native services. The architecture is designed to be:

- **Serverless-first**: Leveraging Lambda, API Gateway, and managed services
- **Scalable**: Auto-scaling capabilities across all layers
- **Cost-effective**: Pay-per-use model with intelligent caching
- **Secure**: End-to-end encryption and fine-grained access control
- **Observable**: Comprehensive monitoring and tracing

---

## Architecture Diagram

### High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           CLIENT LAYER                               │
│         Web App │ Mobile App │ API Clients │ Chatbot Interface      │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Amazon API Gateway + AWS WAF                      │
│              (Rate Limiting, Authentication, DDoS Protection)        │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                ┌────────────────┴────────────────┐
                │                                  │
                ▼                                  ▼
    ┌────────────────────┐            ┌────────────────────────┐
    │  INGESTION PIPELINE │            │    QUERY PIPELINE      │
    │     (Offline)       │            │      (Real-time)       │
    └────────────────────┘            └────────────────────────┘
```

### Detailed Ingestion Pipeline

```
┌──────────────────┐
│   Amazon S3      │ ← Document Upload
│  (Raw Storage)   │
└────────┬─────────┘
         │ (S3 Event Notification)
         ▼
┌────────────────────┐
│  EventBridge Rule  │
└────────┬───────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│              AWS Step Functions (Orchestration)              │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Step 1: Document Processing                         │   │
│  │  Lambda + Textract → Extract Text                   │   │
│  └─────────────────┬───────────────────────────────────┘   │
│                    │                                         │
│  ┌─────────────────▼───────────────────────────────────┐   │
│  │ Step 2: Metadata Extraction                         │   │
│  │  Lambda + Comprehend → NER, Entities, Key Phrases   │   │
│  └─────────────────┬───────────────────────────────────┘   │
│                    │                                         │
│  ┌─────────────────▼───────────────────────────────────┐   │
│  │ Step 3: Text Chunking                               │   │
│  │  Lambda → Semantic/Recursive/Fixed Chunking         │   │
│  └─────────────────┬───────────────────────────────────┘   │
│                    │                                         │
│  ┌─────────────────▼───────────────────────────────────┐   │
│  │ Step 4: Embedding Generation                        │   │
│  │  Lambda + Bedrock → Generate Vector Embeddings      │   │
│  └─────────────────┬───────────────────────────────────┘   │
│                    │                                         │
│  ┌─────────────────▼───────────────────────────────────┐   │
│  │ Step 5: Vector Storage                              │   │
│  │  Lambda → Index in OpenSearch Service               │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────┐
│  Amazon OpenSearch Service │
│  - Vector Index            │
│  - BM25 Lexical Index      │
│  - Metadata Filters        │
└────────────────────────────┘
```

### Detailed Query Pipeline

```
┌──────────────────┐
│   User Query     │
└────────┬─────────┘
         │
         ▼
┌────────────────────────────────┐
│    Amazon API Gateway          │
│  - Cognito Authorization       │
│  - Request Validation          │
│  - Rate Limiting               │
└────────┬───────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│              AWS Lambda (Query Handler)                      │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Step 1: Cache Check                                 │   │
│  │  ElastiCache Redis → Check for cached response      │   │
│  └─────────────────┬───────────────────────────────────┘   │
│                    │ (Cache Miss)                            │
│  ┌─────────────────▼───────────────────────────────────┐   │
│  │ Step 2: Query Embedding                             │   │
│  │  Bedrock → Generate query vector                    │   │
│  └─────────────────┬───────────────────────────────────┘   │
│                    │                                         │
│  ┌─────────────────▼───────────────────────────────────┐   │
│  │ Step 3: Hybrid Search                               │   │
│  │  OpenSearch → Vector + BM25 + Metadata Filters      │   │
│  └─────────────────┬───────────────────────────────────┘   │
│                    │                                         │
│  ┌─────────────────▼───────────────────────────────────┐   │
│  │ Step 4: Reranking (Optional)                        │   │
│  │  Bedrock/SageMaker → Cohere Rerank                  │   │
│  └─────────────────┬───────────────────────────────────┘   │
│                    │                                         │
│  ┌─────────────────▼───────────────────────────────────┐   │
│  │ Step 5: Context Assembly                            │   │
│  │  Retrieve chunks + Apply MMR/Diversity              │   │
│  └─────────────────┬───────────────────────────────────┘   │
│                    │                                         │
│  ┌─────────────────▼───────────────────────────────────┐   │
│  │ Step 6: LLM Generation                              │   │
│  │  Bedrock → Claude/Llama/Mistral with Context        │   │
│  └─────────────────┬───────────────────────────────────┘   │
│                    │                                         │
│  ┌─────────────────▼───────────────────────────────────┐   │
│  │ Step 7: Cache Response                              │   │
│  │  ElastiCache → Store for future queries             │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────┐
│  Response to Client  │
└──────────────────────┘
```

---

## Core Components

### System Components Overview

| Component | Purpose | AWS Services |
|-----------|---------|--------------|
| **Storage** | Document and data persistence | S3, DynamoDB |
| **Processing** | Document transformation and enrichment | Lambda, Step Functions, Textract, Comprehend |
| **Embedding** | Vector generation | Bedrock, SageMaker |
| **Retrieval** | Vector and hybrid search | OpenSearch Service, Aurora PostgreSQL |
| **Generation** | Answer synthesis | Bedrock (Claude, Llama, etc.) |
| **Serving** | API endpoints and routing | API Gateway, Lambda, App Runner |
| **Caching** | Performance optimization | ElastiCache (Redis) |
| **Monitoring** | System observability | CloudWatch, X-Ray, Grafana |
| **Security** | Access control and encryption | IAM, Cognito, KMS, WAF, Secrets Manager |

---

## 1. Data Ingestion Layer

### Purpose
Collect, receive, and route documents from various sources into the processing pipeline.

### AWS Services

#### Amazon S3 (Simple Storage Service)
- **Role**: Primary storage for all documents
- **Configuration**:
  - Bucket structure: `raw/`, `processed/`, `failed/`, `archives/`
  - Versioning enabled for audit trail
  - Lifecycle policies for cost optimization
  - Server-side encryption (SSE-KMS)
  - Event notifications to EventBridge
- **Storage Classes**:
  - S3 Standard for active documents
  - S3 Intelligent-Tiering for variable access patterns
  - S3 Glacier for archival

#### Amazon EventBridge
- **Role**: Event-driven orchestration trigger
- **Configuration**:
  - Rules to detect S3 object creation events
  - Pattern matching by file type/prefix
  - Route events to Step Functions
  - Dead letter queue for failed events
- **Event Patterns**:
  ```json
  {
    "source": ["aws.s3"],
    "detail-type": ["Object Created"],
    "detail": {
      "bucket": {"name": ["rag-documents-bucket"]}
    }
  }
  ```

#### AWS DataSync (Optional)
- **Role**: Automated data transfer from on-premises/other clouds
- **Use Cases**:
  - Scheduled sync from file servers
  - Migration of existing document repositories
  - Continuous sync from network shares

#### AWS Transfer Family (Optional)
- **Role**: SFTP/FTP/FTPS endpoint for document upload
- **Use Cases**:
  - Third-party integrations requiring FTP
  - Legacy systems integration
  - Secure file transfer workflows

### Data Flow
```
External Sources → S3 Bucket → EventBridge → Step Functions
                      ↓
                  Versioning
                      ↓
                 Lifecycle Rules
```

---

## 2. Document Processing & Extraction

### Purpose
Extract text, metadata, and structured information from various document formats.

### AWS Services

#### AWS Lambda
- **Role**: Serverless compute for document processing logic
- **Functions**:
  - `document-classifier`: Detect document type and route
  - `text-extractor`: Extract text using appropriate method
  - `metadata-enricher`: Add document metadata
  - `error-handler`: Process failed documents
- **Configuration**:
  - Runtime: Python 3.11+
  - Memory: 1024-3008 MB (based on document size)
  - Timeout: 5-15 minutes (for large documents)
  - Layers: Common libraries (boto3, PyPDF2, python-docx)
  - VPC: Configured if accessing OpenSearch/ElastiCache
  - Environment variables: S3 buckets, service endpoints

#### Amazon Textract
- **Role**: Extract text and structured data from documents
- **Capabilities**:
  - Text detection and OCR
  - Table extraction
  - Form data extraction
  - Handwriting recognition
  - Document layout analysis
- **Supported Formats**:
  - PDF, PNG, JPEG, TIFF
  - Scanned documents
  - Multi-page documents
- **API Operations**:
  - `DetectDocumentText`: Simple text extraction
  - `AnalyzeDocument`: Tables and forms
  - `StartDocumentAnalysis`: Async for large docs

#### Amazon Comprehend
- **Role**: Natural language processing and metadata extraction
- **Capabilities**:
  - Entity recognition (people, places, organizations)
  - Key phrase extraction
  - Sentiment analysis
  - Language detection
  - PII detection and redaction
  - Topic modeling
  - Custom entity recognition
- **Use Cases**:
  - Enrich chunks with entities for filtering
  - Detect document categories
  - Extract key topics for metadata
  - Identify sensitive information

#### AWS Step Functions
- **Role**: Orchestrate multi-step document processing workflow
- **State Machine Components**:
  - **Parallel States**: Process multiple documents concurrently
  - **Choice States**: Route based on document type
  - **Task States**: Invoke Lambda functions
  - **Retry/Catch**: Error handling and retries
  - **Map State**: Batch processing
- **Workflow Definition**:
  ```
  Start → Validate → Extract Text → Extract Metadata → Chunk → Embed → Index → Success
    ↓         ↓           ↓              ↓              ↓        ↓       ↓
  Error   Error       Error          Error          Error   Error   Error
    ↓         ↓           ↓              ↓              ↓        ↓       ↓
  Retry/DLQ  Retry/DLQ  Retry/DLQ     Retry/DLQ     Retry  Retry/DLQ  DLQ
  ```

### Processing Pipeline Components

| Step | Input | Processing | Output | AWS Service |
|------|-------|------------|--------|-------------|
| **Validation** | Raw file | Check format, size, corruption | Valid file or error | Lambda |
| **Text Extraction** | PDF/Image | OCR, text extraction | Plain text | Lambda + Textract |
| **Metadata Extraction** | Text | NER, key phrases, entities | JSON metadata | Lambda + Comprehend |
| **Quality Check** | Text | Validate extraction quality | Approved text | Lambda |
| **Storage** | Processed data | Store intermediate results | S3 location | S3 |

### Error Handling Strategy
- **Retry Logic**: 3 attempts with exponential backoff
- **Dead Letter Queue**: SNS topic for failed documents
- **Fallback**: Alternative extraction methods
- **Logging**: CloudWatch Logs for debugging

---

## 3. Chunking & Text Processing

### Purpose
Split documents into optimal-sized chunks for embedding and retrieval.

### AWS Services

#### AWS Lambda
- **Role**: Execute chunking algorithms
- **Functions**:
  - `semantic-chunker`: Content-aware splitting
  - `recursive-chunker`: Hierarchical splitting
  - `fixed-chunker`: Simple size-based splitting
  - `sentence-window-chunker`: Sentence-based with context windows
- **Configuration**:
  - Runtime: Python 3.11
  - Memory: 512-1024 MB
  - Libraries: langchain, llama-index, spaCy, nltk

### Chunking Strategies

#### 1. Fixed-Size Chunking
- **Method**: Split by character/token count
- **Parameters**:
  - Chunk size: 500-1000 tokens
  - Overlap: 50-200 tokens (10-20%)
- **Use Case**: Simple documents, consistent structure
- **Implementation**:
  ```python
  # Pseudocode
  chunk_size = 1000
  overlap = 200
  chunks = split_text(text, chunk_size, overlap)
  ```

#### 2. Semantic Chunking
- **Method**: Split by semantic boundaries (paragraphs, sections)
- **Parameters**:
  - Min chunk size: 200 tokens
  - Max chunk size: 1500 tokens
  - Boundary markers: headers, paragraph breaks
- **Use Case**: Structured documents with clear sections
- **Implementation**:
  - Detect section headers
  - Respect paragraph boundaries
  - Maintain semantic coherence

#### 3. Recursive Character Splitting
- **Method**: Hierarchical splitting with multiple separators
- **Separators Priority**:
  1. `\n\n` (paragraph breaks)
  2. `\n` (line breaks)
  3. `. ` (sentences)
  4. ` ` (words)
- **Use Case**: Mixed content types
- **Library**: LangChain RecursiveCharacterTextSplitter

#### 4. Sentence Window Chunking
- **Method**: Each chunk is a sentence with surrounding context
- **Parameters**:
  - Window size: 1-3 sentences before/after
  - Overlap: Natural sentence boundaries
- **Use Case**: Q&A systems, precise retrieval
- **Advantage**: Better context preservation

### Chunk Metadata

Each chunk includes:
```json
{
  "chunk_id": "doc123_chunk_5",
  "document_id": "doc123",
  "chunk_index": 5,
  "content": "chunk text...",
  "chunk_size": 523,
  "start_position": 2500,
  "end_position": 3023,
  "metadata": {
    "document_title": "...",
    "section": "Introduction",
    "page_number": 3,
    "entities": ["AWS", "Lambda"],
    "key_phrases": ["serverless computing"],
    "parent_chunk": "doc123_chunk_4",
    "child_chunks": ["doc123_chunk_5_1", "doc123_chunk_5_2"]
  }
}
```

### Storage
- **Intermediate**: S3 bucket (`processed/chunks/`)
- **Final**: Included in OpenSearch document
- **Format**: JSON with metadata

---

## 4. Embedding Generation

### Purpose
Convert text chunks into dense vector representations for semantic search.

### AWS Services

#### Amazon Bedrock
- **Role**: Managed service for foundation model APIs
- **Embedding Models Available**:
  
  | Model | Dimensions | Max Input | Use Case |
  |-------|------------|-----------|----------|
  | **Titan Embeddings v2** | 256, 512, 1024 | 8192 tokens | General purpose, multilingual |
  | **Titan Multimodal Embeddings** | 384, 1024 | Text + Images | Multi-modal RAG |
  | **Cohere Embed English v3** | 1024 | 512 tokens | English text, high quality |
  | **Cohere Embed Multilingual v3** | 1024 | 512 tokens | 100+ languages |

- **Configuration**:
  - Model ID: `amazon.titan-embed-text-v2:0`
  - Dimensions: 1024 (configurable)
  - Batch size: Up to 100 texts per API call
  - Input type: `search_document` for indexing, `search_query` for queries

#### Amazon SageMaker (Alternative)
- **Role**: Deploy custom embedding models
- **Use Cases**:
  - Fine-tuned embeddings for domain-specific data
  - Custom models (BERT, sentence-transformers)
  - Models not available in Bedrock
- **Deployment**:
  - Real-time endpoint for synchronous inference
  - Asynchronous endpoint for batch processing
  - Serverless inference for variable workloads
- **Popular Models**:
  - `sentence-transformers/all-MiniLM-L6-v2`
  - `sentence-transformers/all-mpnet-base-v2`
  - Custom fine-tuned BERT variants

#### AWS Batch (For Large-Scale Embedding)
- **Role**: Batch processing for millions of documents
- **Configuration**:
  - Compute environments: Spot instances for cost savings
  - Job queues: Priority-based processing
  - Job definitions: Container with embedding logic
- **Use Case**: Initial bulk indexing, reindexing

### Embedding Pipeline

#### Lambda Function: `generate-embeddings`
```python
# Pseudocode structure
def lambda_handler(event, context):
    # Input: chunks from previous step
    chunks = event['chunks']
    
    # Batch chunks (up to 100)
    batches = create_batches(chunks, batch_size=100)
    
    # Generate embeddings
    embeddings = []
    for batch in batches:
        response = bedrock_client.invoke_model(
            modelId='amazon.titan-embed-text-v2:0',
            body=json.dumps({
                'inputText': batch,
                'dimensions': 1024,
                'normalize': True
            })
        )
        embeddings.extend(response['embeddings'])
    
    # Return chunks with embeddings
    return {
        'chunks_with_embeddings': combine(chunks, embeddings)
    }
```

### Optimization Strategies

#### 1. Caching
- **Cache Embeddings**: Store in ElastiCache
- **Key**: Hash of chunk text
- **TTL**: Long-lived (weeks/months)
- **Benefit**: Avoid regeneration for duplicate content

#### 2. Batch Processing
- **Batch Size**: 50-100 chunks per API call
- **Parallel Batches**: Multiple concurrent Lambda invocations
- **Rate Limiting**: Implement exponential backoff

#### 3. Async Processing
- **SQS Queue**: Queue chunks for embedding
- **Worker Lambdas**: Poll queue and process
- **Scalability**: Auto-scale based on queue depth

### Cost Optimization
- **Bedrock Pricing**: Pay per input token
- **Titan v2**: ~$0.0001 per 1K tokens
- **Optimization**:
  - Cache embeddings
  - Use smaller dimensions if sufficient (512 vs 1024)
  - Batch requests to reduce API calls

---

## 5. Vector Database

### Purpose
Store and efficiently search vector embeddings with metadata filtering.

### AWS Services

#### Amazon OpenSearch Service (Recommended)
- **Role**: Primary vector database with hybrid search capabilities
- **Key Features**:
  - **Vector Engine**: k-NN plugin with HNSW/IVF algorithms
  - **Lexical Search**: BM25 for keyword matching
  - **Hybrid Search**: Combine vector + keyword results
  - **Metadata Filtering**: Pre-filter before vector search
  - **Scalability**: Auto-scaling, multi-AZ deployment

- **Index Configuration**:
  ```json
  {
    "mappings": {
      "properties": {
        "chunk_id": {"type": "keyword"},
        "content": {"type": "text", "analyzer": "standard"},
        "embedding": {
          "type": "knn_vector",
          "dimension": 1024,
          "method": {
            "name": "hnsw",
            "space_type": "cosinesimil",
            "engine": "nmslib",
            "parameters": {
              "ef_construction": 512,
              "m": 16
            }
          }
        },
        "metadata": {
          "properties": {
            "document_id": {"type": "keyword"},
            "source": {"type": "keyword"},
            "timestamp": {"type": "date"},
            "category": {"type": "keyword"},
            "entities": {"type": "keyword"}
          }
        }
      }
    },
    "settings": {
      "index": {
        "knn": true,
        "number_of_shards": 3,
        "number_of_replicas": 2
      }
    }
  }
  ```

- **Search Types**:
  
  **1. Vector Search (k-NN)**
  ```json
  {
    "size": 10,
    "query": {
      "knn": {
        "embedding": {
          "vector": [0.1, 0.2, ...],
          "k": 10
        }
      }
    }
  }
  ```

  **2. Hybrid Search (Vector + BM25)**
  ```json
  {
    "size": 10,
    "query": {
      "hybrid": {
        "queries": [
          {"knn": {"embedding": {"vector": [...], "k": 50}}},
          {"match": {"content": "query text"}}
        ]
      }
    }
  }
  ```

  **3. Filtered Search**
  ```json
  {
    "size": 10,
    "query": {
      "bool": {
        "must": [
          {"knn": {"embedding": {"vector": [...], "k": 10}}}
        ],
        "filter": [
          {"term": {"metadata.category": "technical"}},
          {"range": {"metadata.timestamp": {"gte": "2024-01-01"}}}
        ]
      }
    }
  }
  ```

- **Cluster Sizing**:
  - Development: 1 master, 2 data nodes (t3.small)
  - Production: 3 masters, 3+ data nodes (r6g.xlarge)
  - Storage: 100GB - 10TB+ depending on corpus size
  - Instance types: r6g (memory optimized) for vector search

#### Amazon Aurora PostgreSQL with pgvector (Alternative)
- **Role**: Relational database with vector extension
- **Key Features**:
  - Native SQL queries
  - ACID transactions
  - Familiar database paradigm
  - Vector similarity search via pgvector extension

- **Configuration**:
  ```sql
  -- Enable extension
  CREATE EXTENSION vector;
  
  -- Create table
  CREATE TABLE embeddings (
    id SERIAL PRIMARY KEY,
    chunk_id VARCHAR(255),
    content TEXT,
    embedding vector(1024),
    metadata JSONB
  );
  
  -- Create index
  CREATE INDEX ON embeddings 
  USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);
  
  -- Search query
  SELECT chunk_id, content, 
         1 - (embedding <=> query_vector) as similarity
  FROM embeddings
  WHERE metadata->>'category' = 'technical'
  ORDER BY embedding <=> query_vector
  LIMIT 10;
  ```

- **Use Case**: 
  - Smaller datasets (<1M vectors)
  - Need for ACID transactions
  - SQL-first organizations

#### Amazon MemoryDB for Redis (Alternative)
- **Role**: Ultra-low latency vector search
- **Key Features**:
  - Redis 7.0+ with vector similarity search
  - In-memory performance
  - Persistence and durability
  - Microsecond latency

- **Use Case**:
  - Real-time applications requiring <10ms latency
  - High-throughput scenarios
  - Smaller vector datasets

### Indexing Strategy

#### Initial Bulk Indexing
```
Step Functions → Lambda (Batch Index) → OpenSearch
- Batch size: 500-1000 documents
- Parallel indexing: 5-10 concurrent Lambdas
- Refresh interval: 30s during bulk load
```

#### Incremental Indexing
```
New Document → Process → Embed → Lambda → OpenSearch
- Real-time indexing
- Refresh interval: 1s
- Immediate availability
```

#### Reindexing Strategy
```
- Create new index with updated mapping
- Reindex alias pointing to old index
- Bulk reindex to new index
- Atomic alias switch
- Delete old index
```

### Performance Tuning

| Parameter | Development | Production |
|-----------|-------------|------------|
| **ef_construction** | 256 | 512 |
| **M** | 8 | 16 |
| **Shards** | 1 | 3-5 |
| **Replicas** | 1 | 2-3 |
| **Refresh Interval** | 5s | 1s |

---

## 6. LLM Integration

### Purpose
Generate natural language responses using retrieved context.

### AWS Services

#### Amazon Bedrock
- **Role**: Managed LLM inference service
- **Available Models**:

  | Model | Context Window | Use Case | Cost (per 1M tokens) |
  |-------|----------------|----------|----------------------|
  | **Claude 3.5 Sonnet** | 200K tokens | General purpose, best quality | Input: $3, Output: $15 |
  | **Claude 3.5 Haiku** | 200K tokens | Fast, cost-effective | Input: $0.25, Output: $1.25 |
  | **Claude 3 Opus** | 200K tokens | Complex reasoning | Input: $15, Output: $75 |
  | **Llama 3.1 70B** | 128K tokens | Open source alternative | Input: $0.99, Output: $0.99 |
  | **Mistral Large** | 128K tokens | Multilingual, technical | Input: $4, Output: $12 |
  | **Amazon Titan Text** | 32K tokens | AWS-native, cost-effective | Input: $0.50, Output: $1.50 |

- **API Configuration**:
  ```python
  # Pseudocode
  response = bedrock_runtime.invoke_model(
      modelId='anthropic.claude-3-5-sonnet-20241022-v2:0',
      body=json.dumps({
          'anthropic_version': 'bedrock-2023-05-31',
          'max_tokens': 4096,
          'temperature': 0.7,
          'top_p': 0.9,
          'messages': [
              {
                  'role': 'user',
                  'content': prompt
              }
          ],
          'system': system_prompt
      })
  )
  ```

- **Features**:
  - Streaming responses
  - System prompts for behavior control
  - Guardrails for content safety
  - Prompt caching (reduced cost for repeated context)
  - Multi-modal support (Claude 3.5 Sonnet)

#### Amazon SageMaker (Alternative)
- **Role**: Deploy custom or open-source LLMs
- **Use Cases**:
  - Fine-tuned models on proprietary data
  - Models not available in Bedrock
  - Custom inference logic
  - Cost optimization with reserved capacity

- **Deployment Options**:
  - **Real-time Endpoints**: ml.g5.xlarge, ml.g5.2xlarge
  - **Serverless Inference**: Auto-scaling, pay-per-use
  - **Async Endpoints**: Batch inference, queue-based

- **Popular Models**:
  - Llama 2/3 (7B, 13B, 70B)
  - Mistral (7B, 8x7B)
  - Falcon (7B, 40B)
  - Fine-tuned GPT variants

### Prompt Engineering

#### System Prompt Template
```
You are a helpful AI assistant. Answer questions based ONLY on the provided context.

Guidelines:
- If the context doesn't contain the answer, say "I don't have enough information"
- Cite sources by mentioning the document name
- Be concise and accurate
- Do not make up information
```

#### RAG Prompt Template
```
Context:
{retrieved_chunks}

Question: {user_question}

Instructions:
- Answer based only on the context above
- Cite the relevant document sections
- If uncertain, acknowledge limitations

Answer:
```

#### Advanced Techniques
1. **Few-Shot Prompting**: Include examples in system prompt
2. **Chain-of-Thought**: Request step-by-step reasoning
3. **Citations**: Force structured source attribution
4. **Hedging**: Encourage uncertainty expression

### Response Generation Strategies

#### 1. Standard Generation
```
Retrieve → Build Prompt → LLM → Response
- Single API call
- Fastest approach
- Best for simple queries
```

#### 2. Iterative Refinement
```
Retrieve → Generate Draft → Re-retrieve → Refine → Final Response
- Multiple LLM calls
- Better accuracy
- Higher latency
```

#### 3. Agentic Approach (Bedrock Agents)
```
Query → Agent → [Tool: Search] → [Tool: Calculator] → [Tool: Database] → Response
- Multi-step reasoning
- External tool integration
- Complex workflows
```

### Cost Optimization

#### 1. Prompt Caching
- **Bedrock Feature**: Cache system prompt and context
- **Benefit**: 90% cost reduction on cached tokens
- **TTL**: 5 minutes
- **Use Case**: Repeated queries with same context

#### 2. Model Selection
- **Simple queries**: Use Haiku (25x cheaper than Opus)
- **Complex reasoning**: Use Sonnet
- **Fallback strategy**: Try Haiku first, escalate if needed

#### 3. Context Window Management
- **Limit retrieved chunks**: 5-10 most relevant
- **Summarize long documents**: Pre-process with Haiku
- **Compress context**: Remove redundant information

---

## 7. API & Application Layer

### Purpose
Expose RAG functionality through secure, scalable APIs.

### AWS Services

#### Amazon API Gateway
- **Role**: Managed API endpoint
- **API Types**:
  
  **REST API**
  - Traditional request-response
  - Full API management features
  - Custom domains, API keys
  
  **HTTP API**
  - Lower latency, lower cost
  - Simpler configuration
  - JWT/OAuth 2.0 support
  
  **WebSocket API**
  - Bidirectional communication
  - Streaming responses
  - Chat applications

- **Configuration**:
  ```yaml
  Endpoints:
    - POST /query
    - POST /ingest
    - GET /document/{id}
    - GET /health
    - GET /metrics
  
  Features:
    - Request validation
    - Response transformation
    - CORS configuration
    - API keys / Usage plans
    - Rate limiting (1000 req/sec per key)
    - Request/response logging
  ```

- **Integration Types**:
  - **Lambda Proxy**: Full request/response control
  - **Lambda Non-Proxy**: Custom mapping templates
  - **HTTP Proxy**: Forward to other services
  - **VPC Link**: Private integrations

- **Security**:
  - AWS WAF integration
  - Cognito authorizers
  - IAM authorization
  - API keys and usage plans
  - Resource policies

#### AWS Lambda
- **Role**: Business logic execution
- **Key Functions**:

  | Function | Purpose | Trigger | Timeout |
  |----------|---------|---------|---------|
  | `query-handler` | Process user queries | API Gateway | 30s |
  | `ingest-trigger` | Initiate document ingestion | API Gateway | 5s |
  | `document-status` | Check processing status | API Gateway | 3s |
  | `feedback-collector` | Store user feedback | API Gateway | 5s |

- **Configuration Best Practices**:
  - **Memory**: 1024-3008 MB (proportional CPU)
  - **Timeout**: Query: 30s, Ingest: 5s
  - **Concurrency**: Reserved for production (avoid cold starts)
  - **VPC**: Only if accessing VPC resources (OpenSearch, ElastiCache)
  - **Environment Variables**: Service endpoints, config
  - **Layers**: Shared dependencies (opensearch-py, redis-py)

- **Query Handler Lambda Flow**:
  ```python
  def lambda_handler(event, context):
      # 1. Parse request
      query = event['body']['query']
      filters = event['body'].get('filters', {})
      
      # 2. Check cache
      cached = check_redis_cache(query)
      if cached:
          return cached
      
      # 3. Generate query embedding
      query_vector = bedrock_embed(query)
      
      # 4. Search OpenSearch
      results = opensearch_hybrid_search(
          vector=query_vector,
          text=query,
          filters=filters,
          top_k=10
      )
      
      # 5. Build context
      context = build_context(results)
      
      # 6. Generate response
      response = bedrock_generate(
          query=query,
          context=context
      )
      
      # 7. Cache result
      cache_response(query, response)
      
      # 8. Return
      return {
          'statusCode': 200,
          'body': json.dumps({
              'answer': response,
              'sources': results,
              'cached': False
          })
      }
  ```

#### AWS App Runner (Alternative for Containers)
- **Role**: Deploy containerized RAG application
- **Use Cases**:
  - FastAPI/Flask applications
  - Complex dependencies
  - Persistent connections
  - Custom runtime requirements

- **Configuration**:
  ```yaml
  Service:
    Name: rag-api-service
    Source:
      Type: ECR
      Image: account.dkr.ecr.region.amazonaws.com/rag-api:latest
    Instance:
      CPU: 1 vCPU
      Memory: 2 GB
    AutoScaling:
      MinSize: 1
      MaxSize: 10
      TargetConcurrency: 100
    HealthCheck:
      Path: /health
      Interval: 10s
  ```

- **Advantages**:
  - No cold starts
  - WebSocket support
  - Long-running connections
  - Easier debugging

#### Amazon ECS Fargate (Alternative)
- **Role**: Container orchestration without servers
- **Use Cases**:
  - Microservices architecture
  - Complex multi-container applications
  - Need for load balancing
  - Blue-green deployments

### API Endpoints Design

#### POST /query
```json
Request:
{
  "query": "What is AWS Lambda?",
  "filters": {
    "category": "documentation",
    "date_range": {"start": "2024-01-01", "end": "2024-12-31"}
  },
  "options": {
    "top_k": 5,
    "rerank": true,
    "stream": false
  }
}

Response:
{
  "answer": "AWS Lambda is a serverless compute service...",
  "sources": [
    {
      "chunk_id": "doc123_chunk_5",
      "content": "...",
      "document": "aws-lambda-guide.pdf",
      "score": 0.92,
      "page": 3
    }
  ],
  "metadata": {
    "latency_ms": 234,
    "cached": false,
    "model": "claude-3-5-sonnet",
    "tokens_used": 1523
  }
}
```

#### POST /ingest
```json
Request:
{
  "document_url": "s3://bucket/document.pdf",
  "metadata": {
    "category": "technical",
    "author": "John Doe",
    "tags": ["AWS", "serverless"]
  }
}

Response:
{
  "job_id": "ingest-12345",
  "status": "processing",
  "estimated_completion": "2024-01-30T10:30:00Z"
}
```

#### GET /document/{id}/status
```json
Response:
{
  "document_id": "doc123",
  "status": "completed",
  "progress": {
    "extraction": "completed",
    "chunking": "completed",
    "embedding": "completed",
    "indexing": "completed"
  },
  "chunks_created": 45,
  "processing_time_seconds": 12.3
}
```

---

## 8. Caching Layer

### Purpose
Reduce latency and costs by caching embeddings, queries, and responses.

### AWS Services

#### Amazon ElastiCache for Redis
- **Role**: In-memory caching for low-latency access
- **Cache Types**:

  **1. Query Embedding Cache**
  ```
  Key: hash(query_text)
  Value: embedding_vector
  TTL: 7 days
  Purpose: Avoid regenerating embeddings for repeat queries
  ```

  **2. Query Result Cache**
  ```
  Key: hash(query + filters)
  Value: {results, context, metadata}
  TTL: 1 hour
  Purpose: Return cached results for identical queries
  ```

  **3. LLM Response Cache**
  ```
  Key: hash(query + context + model)
  Value: generated_response
  TTL: 24 hours
  Purpose: Avoid LLM calls for identical questions
  ```

  **4. Session Cache**
  ```
  Key: session_id
  Value: conversation_history
  TTL: 30 minutes
  Purpose: Multi-turn conversations
  ```

- **Cluster Configuration**:
  
  **Development**:
  ```yaml
  Node Type: cache.t3.micro
  Nodes: 1
  Engine: Redis 7.0
  Multi-AZ: false
  Encryption: In-transit only
  ```

  **Production**:
  ```yaml
  Node Type: cache.r6g.large
  Nodes: 3 (1 primary, 2 replicas)
  Engine: Redis 7.0
  Multi-AZ: true
  Encryption: At-rest + In-transit
  Backup: Automated snapshots
  ```

- **Eviction Policy**: `allkeys-lru` (Least Recently Used)

#### Amazon CloudFront (Optional)
- **Role**: CDN for API responses and static assets
- **Use Cases**:
  - Cache GET endpoints globally
  - Reduce API Gateway costs
  - Improve response times for geographically distributed users
  - Serve static assets (UI, documentation)

- **Configuration**:
  ```yaml
  Origin: API Gateway
  Behaviors:
    - PathPattern: /query
      CachePolicy: Disabled (dynamic content)
    - PathPattern: /health
      CachePolicy: CachingOptimized (5 min TTL)
    - PathPattern: /static/*
      CachePolicy: CachingOptimized (24 hour TTL)
  ```

### Caching Strategy

#### Cache Hierarchy
```
Request → CloudFront (if enabled) → API Gateway → Lambda
                                                      ↓
                                               Check ElastiCache
                                                      ↓
                                              Cache Hit → Return
                                                      ↓
                                            Cache Miss → Process
                                                      ↓
                                              Update Cache → Return
```

#### Cache Key Design

**Query Cache Key**:
```python
cache_key = f"query:{hash(query_text)}:{hash(filters)}:{model}"
# Example: "query:a3f5e9:b2c4d6:claude-3-5-sonnet"
```

**Embedding Cache Key**:
```python
cache_key = f"embedding:{hash(text)}:{model}:{dimensions}"
# Example: "embedding:d4e6f8:titan-v2:1024"
```

#### Cache Warming Strategy

**1. Preload Common Queries**
```python
# During deployment or scheduled job
common_queries = [
    "What is AWS Lambda?",
    "How to deploy containers?",
    # ... top 100 queries
]

for query in common_queries:
    result = process_query(query)
    cache.set(cache_key(query), result, ttl=86400)
```

**2. Analytics-Driven Warming**
```
Daily Job:
  - Analyze query logs (CloudWatch Insights)
  - Identify top 100 queries
  - Pre-compute and cache results
  - Update cache before peak hours
```

### Cache Invalidation

**Strategies**:

1. **TTL-Based**: Automatic expiration
   - Query results: 1 hour
   - Embeddings: 7 days
   - LLM responses: 24 hours

2. **Event-Based**: Invalidate on data changes
   ```python
   # When document is reindexed
   invalidate_pattern("query:*")  # Clear all query caches
   ```

3. **Manual**: Admin endpoint for cache clearing
   ```
   POST /admin/cache/clear
   {
     "pattern": "query:*",
     "reason": "Index updated"
   }
   ```

### Performance Impact

| Metric | Without Cache | With Cache | Improvement |
|--------|---------------|------------|-------------|
| **Avg Latency** | 2.5s | 150ms | 94% faster |
| **P99 Latency** | 5.2s | 200ms | 96% faster |
| **LLM Cost** | $0.05/query | $0.005/query | 90% cheaper |
| **Throughput** | 100 QPS | 1000 QPS | 10x increase |

---

## 9. Monitoring & Observability

### Purpose
Track system health, performance, and usage for debugging and optimization.

### AWS Services

#### Amazon CloudWatch
- **Role**: Central monitoring and logging service
- **Components**:

  **1. CloudWatch Logs**
  - Log Groups per Lambda function
  - API Gateway access logs
  - Step Functions execution logs
  - OpenSearch slow query logs
  - Application logs (structured JSON)
  
  **Retention Policy**:
  ```
  Development: 7 days
  Production: 90 days
  Compliance: 1-7 years
  ```

  **2. CloudWatch Metrics**
  
  **Standard Metrics**:
  - Lambda: Invocations, Duration, Errors, Throttles
  - API Gateway: Count, Latency, 4xx/5xx errors
  - OpenSearch: CPU, Memory, Disk, Indexing rate
  - ElastiCache: CPU, Memory, Cache hits/misses
  - S3: Storage, Requests
  - Bedrock: Model invocations, Throttles

  **Custom Metrics**:
  ```python
  cloudwatch.put_metric_data(
      Namespace='RAG/Application',
      MetricData=[
          {
              'MetricName': 'QueryLatency',
              'Value': latency_ms,
              'Unit': 'Milliseconds',
              'Dimensions': [
                  {'Name': 'Environment', 'Value': 'production'},
                  {'Name': 'ModelName', 'Value': 'claude-3-5-sonnet'}
              ]
          },
          {
              'MetricName': 'RetrievalQuality',
              'Value': relevance_score,
              'Unit': 'None'
          },
          {
              'MetricName': 'CacheHitRate',
              'Value': cache_hits / total_requests,
              'Unit': 'Percent'
          }
      ]
  )
  ```

  **3. CloudWatch Dashboards**
  
  **System Overview Dashboard**:
  ```
  Widgets:
  - Total Queries (Count)
  - Average Latency (ms)
  - Error Rate (%)
  - Active Users
  - Cost per Query ($)
  - Cache Hit Rate (%)
  ```

  **Performance Dashboard**:
  ```
  Widgets:
  - Query Pipeline Latency (breakdown)
  - Embedding Generation Time
  - OpenSearch Query Time
  - LLM Inference Time
  - End-to-End Latency (P50, P95, P99)
  ```

  **Infrastructure Dashboard**:
  ```
  Widgets:
  - Lambda Concurrent Executions
  - OpenSearch CPU/Memory
  - ElastiCache Hit Rate
  - S3 Storage Used
  - API Gateway Throttles
  ```

  **4. CloudWatch Alarms**
  
  **Critical Alarms**:
  ```yaml
  - Name: HighErrorRate
    Metric: Errors
    Threshold: > 5%
    Period: 5 minutes
    Action: SNS → PagerDuty

  - Name: HighLatency
    Metric: Duration
    Threshold: > 5000ms
    Period: 5 minutes
    Action: SNS → Email

  - Name: OpenSearchClusterRed
    Metric: ClusterStatus.red
    Threshold: >= 1
    Period: 1 minute
    Action: SNS → PagerDuty

  - Name: LambdaThrottling
    Metric: Throttles
    Threshold: > 10
    Period: 5 minutes
    Action: SNS → Slack
  ```

  **Cost Alarms**:
  ```yaml
  - Name: DailyBudgetExceeded
    Metric: EstimatedCharges
    Threshold: > $100/day
    Action: SNS → Email + Slack
  ```

#### AWS X-Ray
- **Role**: Distributed tracing for request flows
- **Instrumentation**:
  ```python
  from aws_xray_sdk.core import xray_recorder
  from aws_xray_sdk.core import patch_all
  
  # Patch all supported libraries
  patch_all()
  
  @xray_recorder.capture('query_handler')
  def lambda_handler(event, context):
      
      with xray_recorder.capture('check_cache'):
          cached = redis_client.get(cache_key)
      
      with xray_recorder.capture('opensearch_search'):
          results = opensearch.search(...)
      
      with xray_recorder.capture('bedrock_generate'):
          response = bedrock.invoke_model(...)
      
      return response
  ```

- **Service Map**: Visual representation of request flow
  ```
  API Gateway → Lambda → [ElastiCache, OpenSearch, Bedrock]
  ```

- **Trace Analysis**:
  - Identify bottlenecks (which service is slow)
  - Detect errors and exceptions
  - Analyze cold start impact
  - Measure service dependencies

#### Amazon Managed Grafana (Optional)
- **Role**: Advanced visualization and analytics
- **Features**:
  - Multi-source dashboards (CloudWatch + custom)
  - Advanced querying and filtering
  - Alerting and notification
  - Team collaboration

- **Use Cases**:
  - Executive dashboards
  - Real-time operations monitoring
  - Historical trend analysis
  - Cost attribution

#### CloudWatch Logs Insights
- **Role**: Interactive log analytics
- **Query Examples**:

  **Find slow queries**:
  ```
  fields @timestamp, query, latency
  | filter latency > 2000
  | sort latency desc
  | limit 20
  ```

  **Error analysis**:
  ```
  fields @timestamp, @message
  | filter @message like /ERROR/
  | stats count() by error_type
  ```

  **Cache hit rate**:
  ```
  fields cache_hit
  | stats count() by cache_hit
  ```

  **Cost per query**:
  ```
  fields query, bedrock_cost, opensearch_cost
  | stats sum(bedrock_cost + opensearch_cost) as total_cost by bin(5m)
  ```

### Key Metrics to Track

#### Application Metrics
| Metric | Target | Alerting Threshold |
|--------|--------|-------------------|
| **Query Latency (P95)** | < 2s | > 5s |
| **Error Rate** | < 1% | > 5% |
| **Cache Hit Rate** | > 60% | < 40% |
| **Retrieval Precision** | > 0.8 | < 0.6 |
| **LLM Hallucination Rate** | < 5% | > 10% |

#### Infrastructure Metrics
| Metric | Target | Alerting Threshold |
|--------|--------|-------------------|
| **Lambda Cold Starts** | < 5% | > 15% |
| **OpenSearch CPU** | < 70% | > 85% |
| **ElastiCache Memory** | < 80% | > 90% |
| **API Gateway 5xx** | < 0.1% | > 1% |

#### Business Metrics
| Metric | Target | Review Frequency |
|--------|--------|-----------------|
| **Daily Active Users** | Track growth | Daily |
| **Queries per User** | > 5/session | Weekly |
| **User Satisfaction** | > 4/5 | Weekly |
| **Cost per Query** | < $0.05 | Daily |

### Logging Best Practices

#### Structured Logging
```python
import json
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def log_query(query, latency, cached, error=None):
    log_entry = {
        'timestamp': datetime.utcnow().isoformat(),
        'event': 'query_processed',
        'query': query,
        'latency_ms': latency,
        'cached': cached,
        'model': 'claude-3-5-sonnet',
        'environment': 'production'
    }
    
    if error:
        log_entry['error'] = str(error)
        log_entry['level'] = 'ERROR'
    else:
        log_entry['level'] = 'INFO'
    
    logger.info(json.dumps(log_entry))
```

#### Correlation IDs
```python
# API Gateway passes request ID
request_id = event['requestContext']['requestId']

# Add to all logs
logger.info(json.dumps({
    'request_id': request_id,
    'message': 'Processing query'
}))

# Pass to downstream services
headers = {'X-Request-ID': request_id}
```

---

## 10. Security & Governance

### Purpose
Ensure data protection, access control, and compliance.

### AWS Services

#### AWS Identity and Access Management (IAM)
- **Role**: Access control and permissions
- **IAM Policies**:

  **Lambda Execution Role**:
  ```json
  {
    "Version": "2012-10-17",
    "Statement": [
      {
        "Effect": "Allow",
        "Action": [
          "s3:GetObject",
          "s3:PutObject"
        ],
        "Resource": "arn:aws:s3:::rag-documents/*"
      },
      {
        "Effect": "Allow",
        "Action": [
          "bedrock:InvokeModel"
        ],
        "Resource": "arn:aws:bedrock:*:*:model/*"
      },
      {
        "Effect": "Allow",
        "Action": [
          "es:ESHttpPost",
          "es:ESHttpPut",
          "es:ESHttpGet"
        ],
        "Resource": "arn:aws:es:region:account:domain/rag-vectors/*"
      },
      {
        "Effect": "Allow",
        "Action": [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ],
        "Resource": "*"
      }
    ]
  }
  ```

  **Principle of Least Privilege**: Each service gets only necessary permissions

#### Amazon Cognito
- **Role**: User authentication and authorization
- **Features**:
  - User pools for authentication
  - Identity pools for AWS resource access
  - OAuth 2.0 / SAML / Social identity providers
  - Multi-factor authentication (MFA)
  - Custom authentication flows

- **Configuration**:
  ```yaml
  UserPool:
    Name: rag-users
    MFA: Optional
    PasswordPolicy:
      MinimumLength: 12
      RequireUppercase: true
      RequireNumbers: true
      RequireSymbols: true
    
    CustomAttributes:
      - organization
      - access_level
    
    AppClients:
      - Name: web-app
        AuthFlows: [USER_PASSWORD_AUTH]
      - Name: mobile-app
        AuthFlows: [USER_SRP_AUTH]
  ```

- **API Gateway Integration**:
  ```yaml
  Authorizer:
    Type: COGNITO_USER_POOLS
    UserPoolArn: arn:aws:cognito-idp:region:account:userpool/pool-id
    IdentitySource: method.request.header.Authorization
  ```

#### AWS Secrets Manager
- **Role**: Store and rotate sensitive credentials
- **Stored Secrets**:
  - Third-party API keys
  - Database passwords (if using Aurora)
  - OpenSearch credentials
  - External service tokens

- **Secret Rotation**:
  ```yaml
  Secret:
    Name: /rag/opensearch/credentials
    RotationSchedule: 90 days
    RotationLambda: rotate-opensearch-credentials
  ```

- **Lambda Access**:
  ```python
  import boto3
  
  secrets_client = boto3.client('secretsmanager')
  
  def get_secret(secret_name):
      response = secrets_client.get_secret_value(
          SecretId=secret_name
      )
      return json.loads(response['SecretString'])
  ```

#### AWS Key Management Service (KMS)
- **Role**: Encryption key management
- **Encryption Scope**:
  - S3 buckets (server-side encryption)
  - OpenSearch domain (at-rest encryption)
  - ElastiCache (at-rest + in-transit)
  - EBS volumes (Lambda in VPC)
  - Secrets Manager

- **Key Configuration**:
  ```yaml
  CustomerManagedKey:
    Alias: rag/encryption-key
    Description: Master key for RAG system encryption
    KeyPolicy:
      - Sid: Enable IAM policies
        Principal: {AWS: "arn:aws:iam::account:root"}
        Action: kms:*
      
      - Sid: Allow services to use key
        Principal:
          Service: 
            - s3.amazonaws.com
            - es.amazonaws.com
            - lambda.amazonaws.com
        Action:
          - kms:Decrypt
          - kms:GenerateDataKey
    
    KeyRotation: Enabled (automatic annual rotation)
  ```

#### AWS WAF (Web Application Firewall)
- **Role**: Protect API Gateway from attacks
- **Rules**:

  **1. Rate Limiting**:
  ```yaml
  - Name: RateLimitRule
    Priority: 1
    Statement:
      RateBasedStatement:
        Limit: 1000  # requests per 5 minutes
        AggregateKeyType: IP
    Action: Block
  ```

  **2. SQL Injection Protection**:
  ```yaml
  - Name: SQLiProtection
    Priority: 2
    Statement:
      SqliMatchStatement:
        FieldToMatch: {Body: {}}
        TextTransformations:
          - {Priority: 0, Type: URL_DECODE}
    Action: Block
  ```

  **3. Geo-Blocking** (if needed):
  ```yaml
  - Name: GeoBlock
    Priority: 3
    Statement:
      GeoMatchStatement:
        CountryCodes: [CN, RU]  # Example
    Action: Block
  ```

  **4. IP Whitelist** (for admin endpoints):
  ```yaml
  - Name: AdminIPWhitelist
    Priority: 4
    Statement:
      NotStatement:
        Statement:
          IPSetReferenceStatement:
            Arn: arn:aws:wafv2:region:account:ipset/admin-ips
    Action: Block
    Scope: /admin/*
  ```

#### Amazon Macie (Optional)
- **Role**: Discover and protect sensitive data
- **Use Cases**:
  - Scan S3 buckets for PII/PHI
  - Classify documents by sensitivity
  - Alert on sensitive data exposure
  - Compliance reporting

- **Sensitive Data Types**:
  - Credit card numbers
  - Social security numbers
  - Email addresses
  - Phone numbers
  - Medical records
  - Credentials

#### AWS CloudTrail
- **Role**: Audit logging for all AWS API calls
- **Configuration**:
  ```yaml
  Trail:
    Name: rag-audit-trail
    S3BucketName: rag-audit-logs
    IncludeGlobalEvents: true
    IsMultiRegion: true
    EnableLogFileValidation: true
    
    EventSelectors:
      - ReadWriteType: All
        IncludeManagementEvents: true
        DataResources:
          - Type: AWS::S3::Object
            Values: ["arn:aws:s3:::rag-documents/*"]
          - Type: AWS::Lambda::Function
            Values: ["arn:aws:lambda:*:*:function/rag-*"]
  ```

- **Monitoring**:
  - Track all document access
  - Monitor configuration changes
  - Detect unauthorized API calls
  - Compliance auditing

#### AWS Config
- **Role**: Resource configuration tracking and compliance
- **Rules**:
  ```yaml
  ConfigRules:
    - s3-bucket-encryption-enabled
    - lambda-function-settings-check
    - opensearch-encrypted-at-rest
    - iam-password-policy
    - approved-amis-by-tag
  ```

### Data Protection

#### Encryption Strategy

**At Rest**:
- S3: SSE-KMS with customer-managed keys
- OpenSearch: Node-to-node encryption + at-rest
- ElastiCache: At-rest encryption enabled
- EBS (Lambda): Encrypted volumes
- Secrets Manager: KMS encryption

**In Transit**:
- API Gateway: TLS 1.2+
- Lambda → OpenSearch: HTTPS
- Lambda → Bedrock: HTTPS
- Lambda → ElastiCache: TLS
- All inter-service: AWS PrivateLink (optional)

#### Network Security

**VPC Configuration** (if needed):
```yaml
VPC:
  CIDR: 10.0.0.0/16
  Subnets:
    Public:
      - 10.0.1.0/24  # AZ1
      - 10.0.2.0/24  # AZ2
    Private:
      - 10.0.10.0/24  # AZ1 - Lambda
      - 10.0.11.0/24  # AZ2 - Lambda
      - 10.0.20.0/24  # AZ1 - OpenSearch
      - 10.0.21.0/24  # AZ2 - OpenSearch
  
  SecurityGroups:
    Lambda:
      Ingress: []  # No inbound
      Egress:
        - Protocol: HTTPS
          Destination: 0.0.0.0/0
    
    OpenSearch:
      Ingress:
        - Protocol: HTTPS
          Source: Lambda-SG
      Egress: []
    
    ElastiCache:
      Ingress:
        - Protocol: Redis (6379)
          Source: Lambda-SG
      Egress: []
```

**VPC Endpoints** (reduce NAT costs):
```yaml
VPCEndpoints:
  - Service: com.amazonaws.region.s3 (Gateway)
  - Service: com.amazonaws.region.bedrock (Interface)
  - Service: com.amazonaws.region.secretsmanager (Interface)
```

#### Access Control Layers

```
Layer 1: AWS WAF → Block malicious traffic
Layer 2: API Gateway → API key / Cognito auth
Layer 3: IAM Roles → Service-level permissions
Layer 4: Resource Policies → Fine-grained access
Layer 5: Encryption → Data protection
```

### Compliance Considerations

#### GDPR
- Data minimization in logs
- User consent for data processing
- Right to be forgotten (delete user data)
- Data portability (export user data)
- Data retention policies

#### HIPAA (if applicable)
- BAA with AWS
- Enable CloudTrail logging
- Encrypt all data at rest and in transit
- Access controls and audit logs
- Incident response procedures

#### SOC 2
- Security monitoring and alerting
- Access control and authentication
- Change management procedures
- Incident response and disaster recovery
- Regular security assessments

---

## 11. Deployment & CI/CD

### Purpose
Automate infrastructure provisioning and application deployment.

### AWS Services

#### AWS CloudFormation / AWS CDK
- **Role**: Infrastructure as Code
- **AWS CDK (Recommended)**:
  ```python
  from aws_cdk import (
      Stack,
      aws_s3 as s3,
      aws_lambda as lambda_,
      aws_apigateway as apigw,
      aws_opensearch as opensearch,
      aws_iam as iam
  )
  
  class RagStack(Stack):
      def __init__(self, scope, id, **kwargs):
          super().__init__(scope, id, **kwargs)
          
          # S3 Bucket
          documents_bucket = s3.Bucket(
              self, "DocumentsBucket",
              encryption=s3.BucketEncryption.KMS,
              versioned=True,
              lifecycle_rules=[...]
          )
          
          # Lambda Functions
          query_handler = lambda_.Function(
              self, "QueryHandler",
              runtime=lambda_.Runtime.PYTHON_3_11,
              handler="index.handler",
              code=lambda_.Code.from_asset("lambda/query"),
              memory_size=1024,
              timeout=Duration.seconds(30),
              environment={
                  "OPENSEARCH_ENDPOINT": opensearch_domain.domain_endpoint,
                  "BEDROCK_MODEL": "claude-3-5-sonnet"
              }
          )
          
          # OpenSearch Domain
          opensearch_domain = opensearch.Domain(
              self, "VectorDB",
              version=opensearch.EngineVersion.OPENSEARCH_2_11,
              capacity={
                  "data_nodes": 3,
                  "data_node_instance_type": "r6g.large.search"
              },
              ebs={
                  "volume_size": 100,
                  "volume_type": ec2.EbsDeviceVolumeType.GP3
              },
              encryption_at_rest={"enabled": True},
              node_to_node_encryption=True
          )
          
          # API Gateway
          api = apigw.RestApi(
              self, "RagAPI",
              rest_api_name="RAG Service API"
          )
          
          query_integration = apigw.LambdaIntegration(query_handler)
          api.root.add_resource("query").add_method("POST", query_integration)
  ```

#### AWS CodePipeline
- **Role**: Continuous deployment orchestration
- **Pipeline Stages**:

  ```yaml
  Pipeline:
    Name: rag-deployment-pipeline
    
    Stages:
      - Name: Source
        Actions:
          - Name: SourceCode
            ActionType: Source
            Provider: GitHub / CodeCommit
            Configuration:
              Repository: rag-system
              Branch: main
      
      - Name: Build
        Actions:
          - Name: BuildAndTest
            ActionType: Build
            Provider: CodeBuild
            Configuration:
              ProjectName: rag-build-project
      
      - Name: DeployDev
        Actions:
          - Name: DeployToDev
            ActionType: Deploy
            Provider: CloudFormation
            Configuration:
              StackName: rag-stack-dev
              TemplatePath: BuildArtifact::template.yaml
              RoleArn: arn:aws:iam::account:role/CloudFormationRole
      
      - Name: IntegrationTests
        Actions:
          - Name: RunTests
            ActionType: Build
            Provider: CodeBuild
            Configuration:
              ProjectName: rag-integration-tests
      
      - Name: ApprovalForProd
        Actions:
          - Name: ManualApproval
            ActionType: Approval
      
      - Name: DeployProd
        Actions:
          - Name: DeployToProduction
            ActionType: Deploy
            Provider: CloudFormation
            Configuration:
              StackName: rag-stack-prod
              TemplatePath: BuildArtifact::template.yaml
  ```

#### AWS CodeBuild
- **Role**: Build and test execution
- **buildspec.yml**:
  ```yaml
  version: 0.2
  
  phases:
    install:
      runtime-versions:
        python: 3.11
      commands:
        - pip install -r requirements.txt
        - pip install pytest pytest-cov
    
    pre_build:
      commands:
        - echo "Running linters..."
        - black --check .
        - flake8 .
        - mypy .
    
    build:
      commands:
        - echo "Running unit tests..."
        - pytest tests/unit --cov=src --cov-report=xml
        
        - echo "Building Lambda deployment packages..."
        - cd lambda/query && zip -r ../../query-handler.zip .
        - cd lambda/ingest && zip -r ../../ingest-handler.zip .
        
        - echo "Synthesizing CDK stack..."
        - cdk synth --output cdk.out
    
    post_build:
      commands:
        - echo "Build completed"
  
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

#### Deployment Strategies

**1. Blue-Green Deployment** (Lambda):
```yaml
Lambda Function:
  Alias: Production
  Versions:
    - Version: $LATEST (Blue - current)
    - Version: 2 (Green - new)
  
  Deployment:
    Type: AllAtOnce / Linear / Canary
    
  Linear10PercentEvery10Minutes:
    - 0 min: 10% traffic to Green
    - 10 min: 20% traffic to Green
    - ...
    - 90 min: 100% traffic to Green
  
  Rollback:
    Trigger: CloudWatch Alarm (error rate > 5%)
    Action: Route 100% traffic back to Blue
```

**2. Canary Deployment** (API Gateway):
```yaml
API Gateway Stage:
  Name: production
  DeploymentId: latest-deployment
  
  CanarySettings:
    PercentTraffic: 10
    UseStageCache: false
  
  Progression:
    - Deploy new version with 10% traffic
    - Monitor metrics for 30 minutes
    - If healthy: Promote canary to 100%
    - If unhealthy: Rollback to previous version
```

### Environment Strategy

```yaml
Environments:
  Development:
    Purpose: Developer testing
    Infrastructure: Minimal (1 OpenSearch node, t3.small)
    Data: Synthetic test data
    Access: Development team
  
  Staging:
    Purpose: Pre-production validation
    Infrastructure: Production-like (3 OpenSearch nodes, r6g.large)
    Data: Sanitized production data
    Access: QA team + stakeholders
  
  Production:
    Purpose: Live customer traffic
    Infrastructure: Fully redundant, multi-AZ
    Data: Real customer data
    Access: Operations team only
```

### Rollback Procedures

**Lambda Rollback**:
```bash
# Automated rollback via CodeDeploy
aws deploy stop-deployment --deployment-id d-XXXXX

# Manual version rollback
aws lambda update-alias \
  --function-name query-handler \
  --name production \
  --function-version 5  # Previous stable version
```

**Infrastructure Rollback**:
```bash
# CloudFormation rollback
aws cloudformation rollback-stack --stack-name rag-stack-prod

# Or deploy previous template version
cdk deploy --version-tag v1.2.3
```

---

## Architecture Flow

### End-to-End Request Flow

#### Ingestion Flow
```
1. User uploads document to S3
   ↓
2. S3 triggers EventBridge rule
   ↓
3. EventBridge starts Step Functions workflow
   ↓
4. Step Functions orchestrates:
   a. Lambda + Textract: Extract text
   b. Lambda + Comprehend: Extract metadata
   c. Lambda: Chunk text
   d. Lambda + Bedrock: Generate embeddings
   e. Lambda: Index in OpenSearch
   ↓
5. Document indexed and searchable
   ↓
6. SNS notification: "Document XYZ processed successfully"
```

#### Query Flow
```
1. User sends query via web/mobile app
   ↓
2. API Gateway receives request
   ↓
3. Cognito validates JWT token
   ↓
4. WAF checks for malicious patterns
   ↓
5. API Gateway invokes Lambda
   ↓
6. Lambda checks ElastiCache for cached response
   ↓ (Cache miss)
7. Lambda calls Bedrock to embed query
   ↓
8. Lambda searches OpenSearch (hybrid: vector + BM25)
   ↓
9. Lambda optionally reranks results
   ↓
10. Lambda assembles context from top results
   ↓
11. Lambda calls Bedrock to generate response
   ↓
12. Lambda caches response in ElastiCache
   ↓
13. Lambda returns response to API Gateway
   ↓
14. API Gateway returns to client
   ↓
15. CloudWatch logs all metrics
```

---

## Design Patterns

### 1. Event-Driven Architecture
- **Pattern**: S3 events trigger processing pipelines
- **Benefits**: Decoupled components, automatic scaling
- **Implementation**: EventBridge + Step Functions + Lambda

### 2. Serverless First
- **Pattern**: Use managed services and Lambda over EC2
- **Benefits**: No server management, pay-per-use, auto-scaling
- **Trade-offs**: Cold starts, execution time limits

### 3. Caching Hierarchy
- **Pattern**: Multi-layer caching (CloudFront → ElastiCache → Source)
- **Benefits**: Reduced latency, lower costs
- **Implementation**: Cache frequently accessed data at each layer

### 4. Circuit Breaker
- **Pattern**: Fail fast when dependencies are down
- **Benefits**: Prevent cascade failures
- **Implementation**: Lambda error handling + exponential backoff

### 5. Retry with Exponential Backoff
- **Pattern**: Automatically retry failed operations with increasing delays
- **Benefits**: Handle transient failures
- **Implementation**: Step Functions retry config, Lambda custom logic

### 6. Dead Letter Queue
- **Pattern**: Route failed messages to separate queue for analysis
- **Benefits**: Don't lose failed requests, debug issues
- **Implementation**: SQS DLQ for Lambda, SNS for notifications

---

## Cost Optimization

### Cost Breakdown (Estimated Monthly - Production)

| Service | Usage | Monthly Cost |
|---------|-------|--------------|
| **Bedrock (Embeddings)** | 10M tokens | $100 |
| **Bedrock (LLM - Sonnet)** | 50M input tokens, 10M output | $300 |
| **OpenSearch** | 3x r6g.large nodes | $900 |
| **ElastiCache** | 3x r6g.large nodes | $600 |
| **Lambda** | 1M invocations, 512MB, 2s avg | $40 |
| **API Gateway** | 10M requests | $35 |
| **S3 Storage** | 1TB documents | $23 |
| **Data Transfer** | 500GB outbound | $45 |
| **CloudWatch** | Logs + Metrics | $50 |
| **Step Functions** | 100K executions | $25 |
| **Total** | | **~$2,118/month** |

### Optimization Strategies

#### 1. Use Smaller Embedding Dimensions
- Titan v2: 1024 → 512 dimensions
- Benefit: 50% smaller storage, faster search
- Trade-off: Slightly lower accuracy

#### 2. Implement Aggressive Caching
- Cache hit rate 60% → 90%
- Reduce Bedrock calls by 30%
- Save ~$100/month on LLM costs

#### 3. Use Spot Instances for Batch Processing
- AWS Batch with Spot for bulk embedding
- Save up to 90% on compute costs
- Suitable for non-time-sensitive workloads

#### 4. S3 Intelligent-Tiering
- Automatically move infrequent access data
- Save 40-70% on storage costs
- Zero retrieval fees

#### 5. Right-Size OpenSearch Cluster
- Start small (r6g.large), monitor usage
- Use UltraWarm for historical data
- Consider Aurora pgvector for smaller workloads

#### 6. Use Bedrock Prompt Caching
- Cache system prompts and context
- Save 90% on cached token costs
- Especially valuable for repetitive queries

#### 7. Lambda Memory Optimization
- Use AWS Compute Optimizer recommendations
- Right-size memory allocation
- Over-provisioning wastes money, under-provisioning increases duration

#### 8. Reserved Capacity (for stable workloads)
- OpenSearch Reserved Instances: 35-75% savings
- ElastiCache Reserved Nodes: 35-55% savings
- Commit to 1-year term for predictable workloads

---

## Next Steps

### Phase 1: MVP (Weeks 1-4)
- Set up AWS account and basic infrastructure
- Implement simple text ingestion pipeline
- Deploy basic vector search with OpenSearch
- Create minimal query API

### Phase 2: Production Features (Weeks 5-8)
- Add hybrid search (vector + BM25)
- Implement caching layer
- Set up monitoring and alerting
- Deploy to production

### Phase 3: Optimization (Weeks 9-12)
- Fine-tune retrieval quality
- Optimize costs
- Implement advanced features (reranking, multi-modal)
- Comprehensive testing and evaluation

### Phase 4: Scale & Enhance (Weeks 13+)
- Handle increased traffic
- Add new document types
- Improve user experience
- Continuous optimization

---

## Conclusion

This AWS-native RAG architecture provides:

✅ **Fully Serverless**: No servers to manage  
✅ **Scalable**: Handles from 10 to 10M queries/day  
✅ **Cost-Effective**: Pay only for what you use  
✅ **Secure**: End-to-end encryption, compliance-ready  
✅ **Observable**: Comprehensive monitoring and tracing  
✅ **Maintainable**: Infrastructure as Code, automated deployments  

**Ready to build with confidence on AWS!**

---

## Document Information

- **Title**: AWS-Native RAG Service Architecture
- **Version**: 1.0
- **Last Updated**: January 30, 2026
- **Author**: Paul - Senior AI Consultant at KrishAI Technologies
- **Purpose**: Complete architectural documentation for building production-ready RAG systems on AWS

---