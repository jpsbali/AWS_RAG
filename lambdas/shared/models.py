"""Shared data models for the AWS-native RAG service."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Entity:
    """A named entity extracted by Comprehend."""

    text: str
    type: str
    score: float


@dataclass
class ExtractionResult:
    """Result of text extraction from a document."""

    document_id: str
    text: str
    pages: int
    tables: list[dict]
    forms: list[dict]
    extraction_method: str  # "textract" | "native"
    confidence: float


@dataclass
class DocumentMetadata:
    """NLP-enriched metadata for a processed document."""

    document_id: str
    entities: list[Entity]
    key_phrases: list[str]
    language: str
    pii_detected: bool
    pii_entities: list


@dataclass
class ChunkConfig:
    """Configuration for text chunking."""

    strategy: str  # "fixed" | "semantic" | "recursive" | "sentence_window"
    chunk_size: int = 1000
    overlap: int = 200
    min_chunk_size: int = 200
    max_chunk_size: int = 1500
    window_size: int = 3


@dataclass
class ChunkMetadata:
    """Metadata attached to an individual chunk."""

    section_title: str | None
    page_number: int | None
    entities: list[str]
    key_phrases: list[str]


@dataclass
class Chunk:
    """A text chunk with positional and metadata information."""

    chunk_id: str
    document_id: str
    chunk_index: int
    content: str
    chunk_size: int
    start_position: int
    end_position: int
    metadata: ChunkMetadata


@dataclass
class EmbeddingResult:
    """Result of embedding generation for a chunk."""

    chunk_id: str
    embedding: list[float]
    cached: bool


@dataclass
class SourceCitation:
    """A source citation linking an answer back to a document chunk."""

    document_id: str
    chunk_id: str
    content_snippet: str
    score: float
    page_number: int | None


@dataclass
class QueryRequest:
    """Incoming query request from a client."""

    query: str
    filters: dict | None = None
    k: int = 10
    rerank: bool = False
    temperature: float = 0.7
    max_tokens: int = 4096
    stream: bool = True


@dataclass
class QueryResponse:
    """Response returned to the client after query processing."""

    answer: str
    sources: list[SourceCitation]
    query_embedding: list[float] | None
    cached: bool
    latency_ms: int


@dataclass
class GenerationConfig:
    """Configuration for Bedrock LLM generation."""

    model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 4096
    guardrails_enabled: bool = True
    guardrail_id: str | None = None


@dataclass
class EvaluationReport:
    """Report produced by a RAGAS evaluation run."""

    run_id: str
    timestamp: str
    dataset_size: int
    metrics: dict
    per_question_scores: list[dict]
    s3_report_key: str
