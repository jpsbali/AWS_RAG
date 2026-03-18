"""Unit tests for lambdas.query.query_handler.handler."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from unittest.mock import MagicMock, patch

import pytest

from lambdas.query.query_handler.handler import (
    CONTEXT_WINDOW_LIMIT,
    MAX_CHUNKS_FOR_CONTEXT,
    MIN_CHUNKS_FOR_CONTEXT,
    MMR_LAMBDA,
    RAG_PROMPT_TEMPLATE,
    SYSTEM_PROMPT,
    _build_query_hash,
    _cosine_similarity,
    apply_mmr,
    assemble_context_prompt,
    build_citations,
    build_hybrid_search_query,
    execute_hybrid_search,
    generate_query_embedding,
    handler,
    invoke_llm,
    rerank_results,
)
from lambdas.shared.models import GenerationConfig, QueryResponse, SourceCitation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_embedding(dims: int = 4) -> list[float]:
    """Return a simple embedding vector."""
    return [1.0 / (i + 1) for i in range(dims)]


def _make_hit(
    chunk_id: str = "chunk_0",
    document_id: str = "doc-1",
    content: str = "Some content",
    score: float = 0.9,
    embedding: list[float] | None = None,
    page_number: int | None = 1,
) -> dict:
    """Build a mock OpenSearch hit dict."""
    return {
        "_score": score,
        "_source": {
            "chunk_id": chunk_id,
            "document_id": document_id,
            "content": content,
            "embedding": embedding or _fake_embedding(),
            "metadata": {"page_number": page_number, "category": "technical"},
        },
    }


def _make_api_event(body: dict) -> dict:
    """Build a mock API Gateway event."""
    return {"body": json.dumps(body)}


def _make_bedrock_embedding_response(embedding: list[float]) -> dict:
    return {"body": json.dumps({"embedding": embedding})}


def _make_bedrock_stream_response(text: str) -> dict:
    """Build a mock InvokeModelWithResponseStream response."""
    chunks = []
    for char in text:
        payload = json.dumps({
            "type": "content_block_delta",
            "delta": {"text": char},
        }).encode("utf-8")
        chunks.append({"chunk": {"bytes": payload}})
    return {"body": chunks}


# ---------------------------------------------------------------------------
# _build_query_hash
# ---------------------------------------------------------------------------

class TestBuildQueryHash:
    def test_same_query_same_hash(self):
        h1 = _build_query_hash("hello", None)
        h2 = _build_query_hash("hello", None)
        assert h1 == h2

    def test_different_query_different_hash(self):
        h1 = _build_query_hash("hello", None)
        h2 = _build_query_hash("world", None)
        assert h1 != h2

    def test_filters_affect_hash(self):
        h1 = _build_query_hash("hello", {"category": "a"})
        h2 = _build_query_hash("hello", {"category": "b"})
        assert h1 != h2

    def test_none_filters_vs_empty_different(self):
        h1 = _build_query_hash("hello", None)
        h2 = _build_query_hash("hello", {})
        assert h1 != h2


# ---------------------------------------------------------------------------
# generate_query_embedding
# ---------------------------------------------------------------------------

class TestGenerateQueryEmbedding:
    def test_returns_embedding(self):
        emb = [0.1, 0.2, 0.3]
        client = MagicMock()
        client.invoke_model.return_value = _make_bedrock_embedding_response(emb)
        result = generate_query_embedding(client, "test query")
        assert result == emb

    def test_calls_bedrock_with_correct_params(self):
        client = MagicMock()
        client.invoke_model.return_value = _make_bedrock_embedding_response([1.0])
        generate_query_embedding(client, "test query", dimensions=512)
        call_kwargs = client.invoke_model.call_args
        body = json.loads(call_kwargs.kwargs.get("body", call_kwargs[1].get("body", "")))
        assert body["inputText"] == "test query"
        assert body["dimensions"] == 512
        assert body["normalize"] is True


# ---------------------------------------------------------------------------
# build_hybrid_search_query
# ---------------------------------------------------------------------------

class TestBuildHybridSearchQuery:
    def test_basic_query_structure(self):
        vec = [0.1, 0.2]
        q = build_hybrid_search_query(vec, "test", k=10)
        assert q["size"] == 10
        assert "hybrid" in q["query"]
        queries = q["query"]["hybrid"]["queries"]
        assert len(queries) == 2
        # k-NN component
        assert "knn" in queries[0]
        assert queries[0]["knn"]["embedding"]["vector"] == vec
        # BM25 component
        assert "match" in queries[1]
        assert queries[1]["match"]["content"] == "test"

    def test_no_filters(self):
        q = build_hybrid_search_query([0.1], "test", k=5)
        assert "post_filter" not in q

    def test_term_filter(self):
        q = build_hybrid_search_query([0.1], "test", k=5, filters={"category": "tech"})
        assert "post_filter" in q
        clauses = q["post_filter"]["bool"]["filter"]
        assert len(clauses) == 1
        assert clauses[0] == {"term": {"metadata.category": "tech"}}

    def test_range_filter(self):
        q = build_hybrid_search_query(
            [0.1], "test", k=5,
            filters={"timestamp": {"gte": "2024-01-01"}},
        )
        clauses = q["post_filter"]["bool"]["filter"]
        assert clauses[0] == {"range": {"metadata.timestamp": {"gte": "2024-01-01"}}}

    def test_terms_filter_for_list(self):
        q = build_hybrid_search_query(
            [0.1], "test", k=5,
            filters={"document_ids": ["doc-1", "doc-2"]},
        )
        clauses = q["post_filter"]["bool"]["filter"]
        assert clauses[0] == {"terms": {"metadata.document_ids": ["doc-1", "doc-2"]}}

    def test_multiple_filters(self):
        q = build_hybrid_search_query(
            [0.1], "test", k=5,
            filters={"category": "tech", "timestamp": {"gte": "2024-01-01"}},
        )
        clauses = q["post_filter"]["bool"]["filter"]
        assert len(clauses) == 2


# ---------------------------------------------------------------------------
# _cosine_similarity
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert _cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_zero_vector(self):
        assert _cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


# ---------------------------------------------------------------------------
# apply_mmr
# ---------------------------------------------------------------------------

class TestApplyMMR:
    def test_returns_all_when_fewer_than_top_n(self):
        hits = [_make_hit(chunk_id=f"c{i}") for i in range(3)]
        result = apply_mmr([1.0, 0.0, 0.0, 0.0], hits, top_n=10)
        assert len(result) == 3

    def test_selects_top_n(self):
        hits = [_make_hit(chunk_id=f"c{i}", embedding=[float(i), 0.0, 0.0, 0.0]) for i in range(8)]
        result = apply_mmr([1.0, 0.0, 0.0, 0.0], hits, top_n=5)
        assert len(result) == 5

    def test_promotes_diversity(self):
        # Two very similar hits and one different
        hits = [
            _make_hit(chunk_id="similar_1", embedding=[1.0, 0.0, 0.0, 0.0], score=0.9),
            _make_hit(chunk_id="similar_2", embedding=[0.99, 0.01, 0.0, 0.0], score=0.85),
            _make_hit(chunk_id="diverse", embedding=[0.0, 1.0, 0.0, 0.0], score=0.8),
        ]
        query_emb = [0.7, 0.7, 0.0, 0.0]
        result = apply_mmr(query_emb, hits, top_n=2, lambda_param=0.5)
        chunk_ids = [h["_source"]["chunk_id"] for h in result]
        # With lambda=0.5, diversity matters — the diverse hit should be selected
        assert "diverse" in chunk_ids

    def test_empty_hits(self):
        result = apply_mmr([1.0], [], top_n=5)
        assert result == []

    def test_handles_missing_embeddings(self):
        hits = [_make_hit(chunk_id="c0", embedding=None)]
        hits[0]["_source"]["embedding"] = None
        result = apply_mmr([1.0, 0.0], hits, top_n=5)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# assemble_context_prompt
# ---------------------------------------------------------------------------

class TestAssembleContextPrompt:
    def test_includes_query_and_chunks(self):
        hits = [_make_hit(content="Chunk A"), _make_hit(chunk_id="c1", content="Chunk B")]
        prompt = assemble_context_prompt("What is X?", hits)
        assert "What is X?" in prompt
        assert "Chunk A" in prompt
        assert "Chunk B" in prompt

    def test_includes_source_references(self):
        hits = [_make_hit(document_id="doc-1", chunk_id="chunk_0")]
        prompt = assemble_context_prompt("query", hits)
        assert "doc-1" in prompt
        assert "chunk_0" in prompt

    def test_respects_context_window_limit(self):
        # Create a hit with very large content
        huge_content = "x" * (CONTEXT_WINDOW_LIMIT + 1000)
        hits = [
            _make_hit(content=huge_content),
            _make_hit(chunk_id="c1", content="Should not appear"),
        ]
        prompt = assemble_context_prompt("query", hits)
        assert "Should not appear" not in prompt

    def test_empty_hits(self):
        prompt = assemble_context_prompt("query", [])
        assert "query" in prompt


# ---------------------------------------------------------------------------
# invoke_llm
# ---------------------------------------------------------------------------

class TestInvokeLLM:
    def test_collects_streamed_response(self):
        client = MagicMock()
        client.invoke_model_with_response_stream.return_value = _make_bedrock_stream_response("Hello world")
        config = GenerationConfig()
        result = invoke_llm(client, "test prompt", config)
        assert result == "Hello world"

    def test_passes_correct_params(self):
        client = MagicMock()
        client.invoke_model_with_response_stream.return_value = _make_bedrock_stream_response("ok")
        config = GenerationConfig(temperature=0.5, max_tokens=2048, top_p=0.8)
        invoke_llm(client, "prompt", config)
        call_kwargs = client.invoke_model_with_response_stream.call_args
        body = json.loads(call_kwargs.kwargs.get("body", call_kwargs[1].get("body", "")))
        assert body["temperature"] == 0.5
        assert body["max_tokens"] == 2048
        assert body["top_p"] == 0.8
        assert body["system"] == SYSTEM_PROMPT

    @patch("lambdas.query.query_handler.handler.time.sleep")
    def test_retries_on_throttling(self, mock_sleep):
        client = MagicMock()
        client.invoke_model_with_response_stream.side_effect = [
            Exception("ThrottlingException"),
            _make_bedrock_stream_response("ok"),
        ]
        result = invoke_llm(client, "prompt", GenerationConfig())
        assert result == "ok"
        assert mock_sleep.call_count == 1

    @patch("lambdas.query.query_handler.handler.time.sleep")
    def test_raises_after_max_retries(self, mock_sleep):
        client = MagicMock()
        client.invoke_model_with_response_stream.side_effect = Exception("Throttling error")
        with pytest.raises(RuntimeError, match="throttled after"):
            invoke_llm(client, "prompt", GenerationConfig())

    def test_non_throttle_error_raises_immediately(self):
        client = MagicMock()
        client.invoke_model_with_response_stream.side_effect = ValueError("bad")
        with pytest.raises(ValueError, match="bad"):
            invoke_llm(client, "prompt", GenerationConfig())


# ---------------------------------------------------------------------------
# build_citations
# ---------------------------------------------------------------------------

class TestBuildCitations:
    def test_builds_citations_from_hits(self):
        hits = [
            _make_hit(chunk_id="c0", document_id="doc-1", content="Hello world", score=0.95, page_number=3),
            _make_hit(chunk_id="c1", document_id="doc-2", content="Foo bar", score=0.8, page_number=None),
        ]
        citations = build_citations(hits)
        assert len(citations) == 2
        assert citations[0].document_id == "doc-1"
        assert citations[0].chunk_id == "c0"
        assert citations[0].score == 0.95
        assert citations[0].page_number == 3
        assert citations[0].content_snippet == "Hello world"
        assert citations[1].page_number is None

    def test_truncates_long_content(self):
        long_content = "x" * 500
        hits = [_make_hit(content=long_content)]
        citations = build_citations(hits)
        assert len(citations[0].content_snippet) == 200

    def test_empty_hits(self):
        assert build_citations([]) == []


# ---------------------------------------------------------------------------
# rerank_results
# ---------------------------------------------------------------------------

class TestRerankResults:
    def test_reranks_by_score(self):
        hits = [_make_hit(chunk_id="c0"), _make_hit(chunk_id="c1")]
        client = MagicMock()
        client.invoke_model.return_value = {
            "body": json.dumps({
                "results": [
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.5},
                ]
            })
        }
        result = rerank_results(client, "query", hits)
        assert result[0]["_source"]["chunk_id"] == "c1"
        assert result[1]["_source"]["chunk_id"] == "c0"

    def test_falls_back_on_error(self):
        hits = [_make_hit(chunk_id="c0"), _make_hit(chunk_id="c1")]
        client = MagicMock()
        client.invoke_model.side_effect = Exception("rerank failed")
        result = rerank_results(client, "query", hits)
        # Should return original order
        assert result[0]["_source"]["chunk_id"] == "c0"


# ---------------------------------------------------------------------------
# Handler integration tests
# ---------------------------------------------------------------------------

class TestHandler:
    """End-to-end handler tests with mocked services."""

    @patch("lambdas.query.query_handler.handler._get_cloudwatch_client")
    @patch("lambdas.query.query_handler.handler._get_cache_service")
    @patch("lambdas.query.query_handler.handler._build_opensearch_client")
    @patch("lambdas.query.query_handler.handler._get_bedrock_client")
    def test_cache_miss_full_pipeline(
        self, mock_bedrock_factory, mock_os_factory, mock_cache_factory, mock_cw_factory,
    ):
        # Setup mocks
        bedrock = MagicMock()
        bedrock.invoke_model.return_value = _make_bedrock_embedding_response([0.1, 0.2])
        bedrock.invoke_model_with_response_stream.return_value = _make_bedrock_stream_response("The answer is 42.")
        mock_bedrock_factory.return_value = bedrock

        os_client = MagicMock()
        os_client.search.return_value = {
            "hits": {"hits": [_make_hit(chunk_id="c0"), _make_hit(chunk_id="c1")]},
        }
        mock_os_factory.return_value = os_client

        cache = MagicMock()
        cache.available = True
        cache.get_query_response.return_value = None
        mock_cache_factory.return_value = cache

        cw = MagicMock()
        mock_cw_factory.return_value = cw

        event = _make_api_event({"query": "What is the meaning of life?"})
        result = handler(event)

        assert result["statusCode"] == 200
        body = json.loads(result["body"])
        assert body["answer"] == "The answer is 42."
        assert body["cached"] is False
        assert len(body["sources"]) == 2
        assert body["latency_ms"] >= 0

        # Verify cache was written
        cache.set_query_response.assert_called_once()

        # Verify CloudWatch metrics published
        cw.put_metric_data.assert_called_once()

    @patch("lambdas.query.query_handler.handler._get_cloudwatch_client")
    @patch("lambdas.query.query_handler.handler._get_cache_service")
    def test_cache_hit_returns_cached(self, mock_cache_factory, mock_cw_factory):
        cached_resp = QueryResponse(
            answer="Cached answer",
            sources=[SourceCitation("doc-1", "c0", "snippet", 0.9, 1)],
            query_embedding=None,
            cached=True,
            latency_ms=10,
        )
        cache = MagicMock()
        cache.available = True
        cache.get_query_response.return_value = cached_resp
        mock_cache_factory.return_value = cache

        cw = MagicMock()
        mock_cw_factory.return_value = cw

        event = _make_api_event({"query": "cached query"})
        result = handler(event)

        assert result["statusCode"] == 200
        body = json.loads(result["body"])
        assert body["answer"] == "Cached answer"
        assert body["cached"] is True

    def test_invalid_request_missing_query(self):
        event = _make_api_event({"filters": {}})
        result = handler(event)
        assert result["statusCode"] == 400
        body = json.loads(result["body"])
        assert "error" in body
        assert body["error"]["code"] == "INVALID_REQUEST"

    def test_invalid_request_bad_json(self):
        event = {"body": "not json{{{"}
        result = handler(event)
        assert result["statusCode"] == 400

    @patch("lambdas.query.query_handler.handler._get_cloudwatch_client")
    @patch("lambdas.query.query_handler.handler._get_cache_service")
    @patch("lambdas.query.query_handler.handler._get_bedrock_client")
    def test_embedding_failure_returns_500(
        self, mock_bedrock_factory, mock_cache_factory, mock_cw_factory,
    ):
        bedrock = MagicMock()
        bedrock.invoke_model.side_effect = Exception("Bedrock down")
        mock_bedrock_factory.return_value = bedrock

        cache = MagicMock()
        cache.available = True
        cache.get_query_response.return_value = None
        mock_cache_factory.return_value = cache

        event = _make_api_event({"query": "test"})
        result = handler(event)
        assert result["statusCode"] == 500
        body = json.loads(result["body"])
        assert body["error"]["code"] == "EMBEDDING_FAILED"

    @patch("lambdas.query.query_handler.handler._get_cloudwatch_client")
    @patch("lambdas.query.query_handler.handler._get_cache_service")
    @patch("lambdas.query.query_handler.handler._build_opensearch_client")
    @patch("lambdas.query.query_handler.handler._get_bedrock_client")
    def test_opensearch_failure_returns_503(
        self, mock_bedrock_factory, mock_os_factory, mock_cache_factory, mock_cw_factory,
    ):
        bedrock = MagicMock()
        bedrock.invoke_model.return_value = _make_bedrock_embedding_response([0.1])
        mock_bedrock_factory.return_value = bedrock

        os_client = MagicMock()
        os_client.search.side_effect = Exception("OpenSearch unavailable")
        mock_os_factory.return_value = os_client

        cache = MagicMock()
        cache.available = True
        cache.get_query_response.return_value = None
        mock_cache_factory.return_value = cache

        event = _make_api_event({"query": "test"})
        result = handler(event)
        assert result["statusCode"] == 503
        assert result["headers"]["Retry-After"] == "30"
        body = json.loads(result["body"])
        assert body["error"]["code"] == "SEARCH_UNAVAILABLE"

    @patch("lambdas.query.query_handler.handler._get_cloudwatch_client")
    @patch("lambdas.query.query_handler.handler._get_cache_service")
    @patch("lambdas.query.query_handler.handler._build_opensearch_client")
    @patch("lambdas.query.query_handler.handler._get_bedrock_client")
    def test_no_results_returns_404(
        self, mock_bedrock_factory, mock_os_factory, mock_cache_factory, mock_cw_factory,
    ):
        bedrock = MagicMock()
        bedrock.invoke_model.return_value = _make_bedrock_embedding_response([0.1])
        mock_bedrock_factory.return_value = bedrock

        os_client = MagicMock()
        os_client.search.return_value = {"hits": {"hits": []}}
        mock_os_factory.return_value = os_client

        cache = MagicMock()
        cache.available = True
        cache.get_query_response.return_value = None
        mock_cache_factory.return_value = cache

        event = _make_api_event({"query": "test"})
        result = handler(event)
        assert result["statusCode"] == 404

    @patch("lambdas.query.query_handler.handler._get_cloudwatch_client")
    @patch("lambdas.query.query_handler.handler._get_cache_service")
    @patch("lambdas.query.query_handler.handler._build_opensearch_client")
    @patch("lambdas.query.query_handler.handler._get_bedrock_client")
    def test_llm_failure_returns_500(
        self, mock_bedrock_factory, mock_os_factory, mock_cache_factory, mock_cw_factory,
    ):
        bedrock = MagicMock()
        bedrock.invoke_model.return_value = _make_bedrock_embedding_response([0.1])
        bedrock.invoke_model_with_response_stream.side_effect = ValueError("LLM error")
        mock_bedrock_factory.return_value = bedrock

        os_client = MagicMock()
        os_client.search.return_value = {"hits": {"hits": [_make_hit()]}}
        mock_os_factory.return_value = os_client

        cache = MagicMock()
        cache.available = True
        cache.get_query_response.return_value = None
        mock_cache_factory.return_value = cache

        event = _make_api_event({"query": "test"})
        result = handler(event)
        assert result["statusCode"] == 500
        body = json.loads(result["body"])
        assert body["error"]["code"] == "GENERATION_FAILED"

    @patch("lambdas.query.query_handler.handler._get_cloudwatch_client")
    @patch("lambdas.query.query_handler.handler._get_cache_service")
    @patch("lambdas.query.query_handler.handler._build_opensearch_client")
    @patch("lambdas.query.query_handler.handler._get_bedrock_client")
    def test_rerank_enabled(
        self, mock_bedrock_factory, mock_os_factory, mock_cache_factory, mock_cw_factory,
    ):
        bedrock = MagicMock()
        bedrock.invoke_model.side_effect = [
            # First call: embedding
            _make_bedrock_embedding_response([0.1, 0.2]),
            # Second call: rerank
            {"body": json.dumps({"results": [
                {"index": 1, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.5},
            ]})},
        ]
        bedrock.invoke_model_with_response_stream.return_value = _make_bedrock_stream_response("answer")
        mock_bedrock_factory.return_value = bedrock

        os_client = MagicMock()
        os_client.search.return_value = {
            "hits": {"hits": [_make_hit(chunk_id="c0"), _make_hit(chunk_id="c1")]},
        }
        mock_os_factory.return_value = os_client

        cache = MagicMock()
        cache.available = True
        cache.get_query_response.return_value = None
        mock_cache_factory.return_value = cache

        cw = MagicMock()
        mock_cw_factory.return_value = cw

        event = _make_api_event({"query": "test", "rerank": True})
        result = handler(event)
        assert result["statusCode"] == 200

    @patch("lambdas.query.query_handler.handler._get_cloudwatch_client")
    @patch("lambdas.query.query_handler.handler._get_cache_service")
    @patch("lambdas.query.query_handler.handler._build_opensearch_client")
    @patch("lambdas.query.query_handler.handler._get_bedrock_client")
    def test_cache_unavailable_still_works(
        self, mock_bedrock_factory, mock_os_factory, mock_cache_factory, mock_cw_factory,
    ):
        bedrock = MagicMock()
        bedrock.invoke_model.return_value = _make_bedrock_embedding_response([0.1])
        bedrock.invoke_model_with_response_stream.return_value = _make_bedrock_stream_response("ok")
        mock_bedrock_factory.return_value = bedrock

        os_client = MagicMock()
        os_client.search.return_value = {"hits": {"hits": [_make_hit()]}}
        mock_os_factory.return_value = os_client

        cache = MagicMock()
        cache.available = False
        mock_cache_factory.return_value = cache

        cw = MagicMock()
        mock_cw_factory.return_value = cw

        event = _make_api_event({"query": "test"})
        result = handler(event)
        assert result["statusCode"] == 200
        # Cache write should not be called when unavailable
        cache.set_query_response.assert_not_called()
