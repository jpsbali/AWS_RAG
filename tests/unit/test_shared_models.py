"""Unit tests for lambdas.shared.models."""

from lambdas.shared.models import (
    Chunk,
    ChunkConfig,
    ChunkMetadata,
    DocumentMetadata,
    EmbeddingResult,
    Entity,
    EvaluationReport,
    ExtractionResult,
    GenerationConfig,
    QueryRequest,
    QueryResponse,
    SourceCitation,
)


def test_extraction_result_fields():
    result = ExtractionResult(
        document_id="doc1",
        text="hello world",
        pages=2,
        tables=[{"col": "val"}],
        forms=[{"field": "value"}],
        extraction_method="textract",
        confidence=0.95,
    )
    assert result.document_id == "doc1"
    assert result.confidence == 0.95
    assert result.extraction_method == "textract"


def test_document_metadata_with_entities():
    entity = Entity(text="AWS", type="ORGANIZATION", score=0.99)
    meta = DocumentMetadata(
        document_id="doc1",
        entities=[entity],
        key_phrases=["cloud computing"],
        language="en",
        pii_detected=False,
        pii_entities=[],
    )
    assert meta.entities[0].text == "AWS"
    assert meta.language == "en"
    assert meta.pii_detected is False


def test_chunk_config_defaults():
    config = ChunkConfig(strategy="fixed")
    assert config.chunk_size == 1000
    assert config.overlap == 200
    assert config.min_chunk_size == 200
    assert config.max_chunk_size == 1500
    assert config.window_size == 3


def test_chunk_with_metadata():
    meta = ChunkMetadata(
        section_title="Intro",
        page_number=1,
        entities=["AWS"],
        key_phrases=["serverless"],
    )
    chunk = Chunk(
        chunk_id="doc1_chunk_0",
        document_id="doc1",
        chunk_index=0,
        content="Some text",
        chunk_size=100,
        start_position=0,
        end_position=100,
        metadata=meta,
    )
    assert chunk.chunk_id == "doc1_chunk_0"
    assert chunk.metadata.section_title == "Intro"


def test_embedding_result():
    result = EmbeddingResult(chunk_id="c1", embedding=[0.1, 0.2], cached=True)
    assert result.cached is True
    assert len(result.embedding) == 2


def test_query_request_defaults():
    req = QueryRequest(query="What is Lambda?")
    assert req.k == 10
    assert req.rerank is False
    assert req.temperature == 0.7
    assert req.max_tokens == 4096
    assert req.stream is True
    assert req.filters is None


def test_query_response():
    citation = SourceCitation(
        document_id="doc1",
        chunk_id="c1",
        content_snippet="snippet",
        score=0.9,
        page_number=3,
    )
    resp = QueryResponse(
        answer="Lambda is serverless.",
        sources=[citation],
        query_embedding=None,
        cached=False,
        latency_ms=450,
    )
    assert resp.sources[0].score == 0.9
    assert resp.cached is False


def test_generation_config_defaults():
    config = GenerationConfig()
    assert config.model_id == "anthropic.claude-3-5-sonnet-20241022-v2:0"
    assert config.temperature == 0.7
    assert config.top_p == 0.9
    assert config.max_tokens == 4096
    assert config.guardrails_enabled is True
    assert config.guardrail_id is None


def test_evaluation_report():
    report = EvaluationReport(
        run_id="eval-001",
        timestamp="2024-01-15T12:00:00Z",
        dataset_size=50,
        metrics={"faithfulness": 0.92},
        per_question_scores=[{"q": 1, "score": 0.9}],
        s3_report_key="evaluation/reports/eval-001.json",
    )
    assert report.dataset_size == 50
    assert report.metrics["faithfulness"] == 0.92
