"""Query Handler Lambda for the RAG query pipeline.

Receives an API Gateway event with a query request, orchestrates the full
query pipeline: cache check → embed → hybrid search → rerank → MMR →
assemble prompt → LLM generation → cache → return response with citations.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict
from typing import Any

import boto3

from lambdas.shared.cache_service import CacheService
from lambdas.shared.constants import (
    DEFAULT_EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL_ID,
    LLM_PRIMARY_MODEL_ID,
    LLM_RESPONSE_CACHE_TTL,
)
from lambdas.shared.models import (
    GenerationConfig,
    QueryRequest,
    QueryResponse,
    SourceCitation,
)
from lambdas.shared.utils import (
    compute_hash,
    configure_xray_sampling,
    create_subsegment,
    generate_correlation_id,
    get_structured_logger,
    patch_aws_sdk,
)

logger = get_structured_logger(__name__)

# X-Ray instrumentation
patch_aws_sdk()
configure_xray_sampling()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_INDEX_ALIAS = "rag-chunks"
DEFAULT_K = 10
MMR_LAMBDA = 0.7  # balance relevance vs diversity (1.0 = pure relevance)
MIN_CHUNKS_FOR_CONTEXT = 5
MAX_CHUNKS_FOR_CONTEXT = 10
CONTEXT_WINDOW_LIMIT = 180_000  # chars (~45k tokens, safe for Claude 200k)
MAX_RETRIES = 3
INITIAL_BACKOFF_S = 1.0
THROTTLE_EXCEPTIONS = ("ThrottlingException", "TooManyRequestsException", "ServiceUnavailableException")

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a helpful AI assistant. Answer questions based ONLY on the provided context.

Guidelines:
- If the context doesn't contain the answer, say "I don't have enough information to answer this question."
- Cite sources by referencing the document name and section.
- Be concise and accurate.
- Do not fabricate information."""

RAG_PROMPT_TEMPLATE = """Context:
{context}

Question: {query}

Instructions:
- Answer based only on the context above.
- Cite the relevant document sections using [Source: document_id, chunk_id].
- If uncertain, acknowledge limitations.

Answer:"""


# ---------------------------------------------------------------------------
# Client helpers
# ---------------------------------------------------------------------------

def _get_bedrock_client():  # noqa: ANN202
    return boto3.client("bedrock-runtime")


def _get_cloudwatch_client():  # noqa: ANN202
    return boto3.client("cloudwatch")


def _build_opensearch_client(endpoint: str) -> Any:
    """Build an OpenSearch client with AWS SigV4 authentication."""
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
    return OpenSearch(
        hosts=[{"host": host, "port": 443}],
        http_auth=aws_auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
    )


# ---------------------------------------------------------------------------
# Step 1: Cache check
# ---------------------------------------------------------------------------

def _build_query_hash(query: str, filters: dict | None) -> str:
    """Hash query + filters to produce a cache key."""
    parts = query
    if filters is None:
        parts += "|no_filters"
    else:
        parts += "|" + json.dumps(filters, sort_keys=True)
    return compute_hash(parts)


# ---------------------------------------------------------------------------
# Step 2: Generate query embedding
# ---------------------------------------------------------------------------

def generate_query_embedding(
    bedrock_client: Any,
    query: str,
    dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
) -> list[float]:
    """Generate a query embedding via Bedrock Titan v2 with input_type=search_query."""
    body = json.dumps({
        "inputText": query,
        "dimensions": dimensions,
        "normalize": True,
    })
    response = bedrock_client.invoke_model(
        modelId=EMBEDDING_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=body,
    )
    result = json.loads(
        response["body"].read() if hasattr(response["body"], "read") else response["body"]
    )
    return result["embedding"]


# ---------------------------------------------------------------------------
# Step 3: Hybrid search in OpenSearch
# ---------------------------------------------------------------------------

def build_hybrid_search_query(
    query_vector: list[float],
    query_text: str,
    k: int,
    filters: dict | None = None,
) -> dict:
    """Build an OpenSearch hybrid search query (k-NN + BM25 + metadata filters)."""
    query_body: dict[str, Any] = {
        "size": k,
        "query": {
            "hybrid": {
                "queries": [
                    {"knn": {"embedding": {"vector": query_vector, "k": min(k * 5, 50)}}},
                    {"match": {"content": query_text}},
                ],
            },
        },
    }
    if filters:
        filter_clauses = []
        for field, value in filters.items():
            if isinstance(value, dict) and any(op in value for op in ("gte", "lte", "gt", "lt")):
                filter_clauses.append({"range": {f"metadata.{field}": value}})
            elif isinstance(value, list):
                filter_clauses.append({"terms": {f"metadata.{field}": value}})
            else:
                filter_clauses.append({"term": {f"metadata.{field}": value}})
        query_body["post_filter"] = {"bool": {"filter": filter_clauses}}
    return query_body


def execute_hybrid_search(
    os_client: Any,
    query_vector: list[float],
    query_text: str,
    k: int,
    filters: dict | None,
    index: str,
) -> list[dict]:
    """Execute hybrid search and return raw hit dicts."""
    query_body = build_hybrid_search_query(query_vector, query_text, k, filters)
    response = os_client.search(index=index, body=query_body)
    return response.get("hits", {}).get("hits", [])


# ---------------------------------------------------------------------------
# Step 4: Reranking (optional)
# ---------------------------------------------------------------------------

def rerank_results(
    bedrock_client: Any,
    query: str,
    hits: list[dict],
) -> list[dict]:
    """Rerank search results using Bedrock. Falls back to original order on error."""
    try:
        documents = [
            {"text": hit["_source"].get("content", "")} for hit in hits
        ]
        body = json.dumps({
            "query": query,
            "documents": documents,
        })
        response = bedrock_client.invoke_model(
            modelId="cohere.rerank-v3-5:0",
            contentType="application/json",
            accept="application/json",
            body=body,
        )
        result = json.loads(
            response["body"].read() if hasattr(response["body"], "read") else response["body"]
        )
        ranked_indices = [r["index"] for r in sorted(result.get("results", []), key=lambda x: x.get("relevance_score", 0), reverse=True)]
        return [hits[i] for i in ranked_indices if i < len(hits)]
    except Exception:
        logger.warning("Reranking failed, using original order")
        return hits


# ---------------------------------------------------------------------------
# Step 5: MMR (Maximal Marginal Relevance)
# ---------------------------------------------------------------------------

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def apply_mmr(
    query_embedding: list[float],
    hits: list[dict],
    top_n: int = MAX_CHUNKS_FOR_CONTEXT,
    lambda_param: float = MMR_LAMBDA,
) -> list[dict]:
    """Apply Maximal Marginal Relevance to select diverse, relevant chunks.

    MMR score = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))
    """
    if len(hits) <= top_n:
        return hits

    # Extract embeddings from hits
    embeddings = []
    for hit in hits:
        emb = hit.get("_source", {}).get("embedding")
        if emb:
            embeddings.append(emb)
        else:
            embeddings.append([])

    selected: list[int] = []
    remaining = list(range(len(hits)))

    # Precompute query similarities
    query_sims = [
        _cosine_similarity(query_embedding, emb) if emb else 0.0
        for emb in embeddings
    ]

    for _ in range(top_n):
        if not remaining:
            break

        best_idx = -1
        best_score = -float("inf")

        for idx in remaining:
            relevance = query_sims[idx]

            # Max similarity to already-selected documents
            max_sim_to_selected = 0.0
            for sel_idx in selected:
                if embeddings[idx] and embeddings[sel_idx]:
                    sim = _cosine_similarity(embeddings[idx], embeddings[sel_idx])
                    max_sim_to_selected = max(max_sim_to_selected, sim)

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx >= 0:
            selected.append(best_idx)
            remaining.remove(best_idx)

    return [hits[i] for i in selected]


# ---------------------------------------------------------------------------
# Step 6: Assemble context prompt
# ---------------------------------------------------------------------------

def assemble_context_prompt(
    query: str,
    hits: list[dict],
) -> str:
    """Assemble the RAG prompt with system instructions and retrieved chunks."""
    context_parts: list[str] = []
    total_chars = 0

    for i, hit in enumerate(hits):
        source = hit.get("_source", {})
        content = source.get("content", "")
        doc_id = source.get("document_id", "unknown")
        chunk_id = source.get("chunk_id", f"chunk_{i}")
        metadata = source.get("metadata", {})
        page = metadata.get("page_number", "N/A")

        chunk_text = f"[Source: {doc_id}, {chunk_id}, page {page}]\n{content}"

        if total_chars + len(chunk_text) > CONTEXT_WINDOW_LIMIT:
            break

        context_parts.append(chunk_text)
        total_chars += len(chunk_text)

    context = "\n\n---\n\n".join(context_parts)
    return RAG_PROMPT_TEMPLATE.format(context=context, query=query)


# ---------------------------------------------------------------------------
# Step 7: Invoke Bedrock Claude for generation
# ---------------------------------------------------------------------------

def invoke_llm(
    bedrock_client: Any,
    prompt: str,
    config: GenerationConfig,
) -> str:
    """Invoke Bedrock Claude 3.5 Sonnet and return the full response text.

    Uses InvokeModelWithResponseStream for Lambda (collects all chunks).
    Retries on throttling with exponential backoff.
    """
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": prompt}],
    })

    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = bedrock_client.invoke_model_with_response_stream(
                modelId=config.model_id,
                contentType="application/json",
                accept="application/json",
                body=body,
            )
            # Collect streamed chunks into full response
            answer_parts: list[str] = []
            for event in response.get("body", []):
                chunk = event.get("chunk")
                if chunk:
                    payload = json.loads(chunk["bytes"].decode("utf-8"))
                    if payload.get("type") == "content_block_delta":
                        delta = payload.get("delta", {})
                        answer_parts.append(delta.get("text", ""))
            return "".join(answer_parts)
        except Exception as exc:
            error_code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
            if error_code in THROTTLE_EXCEPTIONS or "Throttling" in str(exc):
                wait = INITIAL_BACKOFF_S * (2 ** attempt)
                logger.warning(
                    "Bedrock LLM throttled (attempt %d/%d), retrying in %.1fs",
                    attempt + 1, MAX_RETRIES, wait,
                )
                time.sleep(wait)
                last_exc = exc
            else:
                raise

    raise RuntimeError(f"Bedrock LLM throttled after {MAX_RETRIES} retries") from last_exc


# ---------------------------------------------------------------------------
# Build source citations
# ---------------------------------------------------------------------------

def build_citations(hits: list[dict]) -> list[SourceCitation]:
    """Build SourceCitation objects from OpenSearch hits."""
    citations: list[SourceCitation] = []
    for hit in hits:
        source = hit.get("_source", {})
        metadata = source.get("metadata", {})
        content = source.get("content", "")
        snippet = content[:200] if len(content) > 200 else content
        citations.append(SourceCitation(
            document_id=source.get("document_id", ""),
            chunk_id=source.get("chunk_id", ""),
            content_snippet=snippet,
            score=hit.get("_score", 0.0),
            page_number=metadata.get("page_number"),
        ))
    return citations


# ---------------------------------------------------------------------------
# CloudWatch metrics
# ---------------------------------------------------------------------------

def _publish_metrics(
    cw_client: Any,
    latency_ms: int,
    cache_hit: bool,
    correlation_id: str,
) -> None:
    """Publish QueryLatency and CacheHitRate custom metrics to CloudWatch."""
    try:
        cw_client.put_metric_data(
            Namespace="RAG/QueryPipeline",
            MetricData=[
                {
                    "MetricName": "QueryLatency",
                    "Value": latency_ms,
                    "Unit": "Milliseconds",
                    "Dimensions": [
                        {"Name": "Service", "Value": "QueryHandler"},
                    ],
                },
                {
                    "MetricName": "CacheHitRate",
                    "Value": 1.0 if cache_hit else 0.0,
                    "Unit": "Count",
                    "Dimensions": [
                        {"Name": "Service", "Value": "QueryHandler"},
                    ],
                },
            ],
        )
    except Exception:
        logger.warning("Failed to publish CloudWatch metrics for %s", correlation_id)


# ---------------------------------------------------------------------------
# Lambda handler
# ---------------------------------------------------------------------------

def handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    """Handle a query request from API Gateway.

    Parameters
    ----------
    event : dict
        API Gateway event with ``body`` containing QueryRequest fields:
        ``query`` (required), ``filters``, ``k``, ``rerank``, ``temperature``,
        ``max_tokens``, ``stream``.
    context : object, optional
        Lambda context (unused).

    Returns
    -------
    dict
        API Gateway response with statusCode and body containing QueryResponse.
    """
    start_time = time.time()
    correlation_id = generate_correlation_id()
    extra = {"correlation_id": correlation_id}

    # Parse request body
    try:
        body = event.get("body", "{}")
        if isinstance(body, str):
            body = json.loads(body)
        request = QueryRequest(
            query=body["query"],
            filters=body.get("filters"),
            k=int(body.get("k", DEFAULT_K)),
            rerank=bool(body.get("rerank", False)),
            temperature=float(body.get("temperature", 0.7)),
            max_tokens=int(body.get("max_tokens", 4096)),
            stream=bool(body.get("stream", True)),
        )
    except (KeyError, json.JSONDecodeError, TypeError) as exc:
        logger.error("Invalid request: %s", exc, extra=extra)
        return {
            "statusCode": 400,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "error": {
                    "code": "INVALID_REQUEST",
                    "message": f"Invalid request body: {exc}",
                    "request_id": correlation_id,
                }
            }),
        }

    logger.info("Query received: %s", request.query[:100], extra=extra)

    # Initialize services
    cache = _get_cache_service()
    query_hash = _build_query_hash(request.query, request.filters)
    cache_hit = False

    # ------------------------------------------------------------------
    # Step 1: Check cache
    # ------------------------------------------------------------------
    with create_subsegment("cache_check"):
        cached_response = cache.get_query_response(query_hash) if cache.available else None
    if cached_response is not None:
        cache_hit = True
        latency_ms = int((time.time() - start_time) * 1000)
        cached_response.latency_ms = latency_ms
        cached_response.cached = True
        logger.info("Cache hit for query hash=%s", query_hash, extra=extra)

        _publish_metrics(_get_cloudwatch_client(), latency_ms, cache_hit, correlation_id)

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(asdict(cached_response)),
        }

    # ------------------------------------------------------------------
    # Step 2: Generate query embedding
    # ------------------------------------------------------------------
    try:
        bedrock_client = _get_bedrock_client()
        with create_subsegment("embedding_generation"):
            query_embedding = generate_query_embedding(bedrock_client, request.query)
    except Exception as exc:
        logger.error("Embedding generation failed: %s", exc, extra=extra)
        return _error_response(500, "EMBEDDING_FAILED", str(exc), correlation_id)

    # ------------------------------------------------------------------
    # Step 3: Hybrid search in OpenSearch
    # ------------------------------------------------------------------
    try:
        opensearch_endpoint = os.environ.get("OPENSEARCH_ENDPOINT", "")
        index_alias = os.environ.get("OPENSEARCH_INDEX", DEFAULT_INDEX_ALIAS)
        os_client = _build_opensearch_client(opensearch_endpoint)
        with create_subsegment("opensearch_search"):
            hits = execute_hybrid_search(
                os_client, query_embedding, request.query, request.k, request.filters, index_alias,
            )
    except Exception as exc:
        logger.error("OpenSearch search failed: %s", exc, extra=extra)
        return _error_response(503, "SEARCH_UNAVAILABLE", str(exc), correlation_id, retry_after=30)

    if not hits:
        logger.warning("No search results for query", extra=extra)
        return _error_response(404, "NO_RESULTS", "No relevant documents found.", correlation_id)

    # ------------------------------------------------------------------
    # Step 4: Optional reranking
    # ------------------------------------------------------------------
    if request.rerank:
        hits = rerank_results(bedrock_client, request.query, hits)

    # ------------------------------------------------------------------
    # Step 5: Apply MMR for diversity
    # ------------------------------------------------------------------
    selected_hits = apply_mmr(
        query_embedding, hits, top_n=min(MAX_CHUNKS_FOR_CONTEXT, max(MIN_CHUNKS_FOR_CONTEXT, request.k)),
    )

    # ------------------------------------------------------------------
    # Step 6: Assemble context prompt
    # ------------------------------------------------------------------
    prompt = assemble_context_prompt(request.query, selected_hits)

    # ------------------------------------------------------------------
    # Step 7: Invoke LLM
    # ------------------------------------------------------------------
    gen_config = GenerationConfig(
        model_id=LLM_PRIMARY_MODEL_ID,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )
    try:
        with create_subsegment("llm_generation"):
            answer = invoke_llm(bedrock_client, prompt, gen_config)
    except Exception as exc:
        logger.error("LLM generation failed: %s", exc, extra=extra)
        return _error_response(500, "GENERATION_FAILED", str(exc), correlation_id)

    # ------------------------------------------------------------------
    # Step 8: Build response and cache
    # ------------------------------------------------------------------
    citations = build_citations(selected_hits)
    latency_ms = int((time.time() - start_time) * 1000)

    response = QueryResponse(
        answer=answer,
        sources=citations,
        query_embedding=query_embedding,
        cached=False,
        latency_ms=latency_ms,
    )

    # Cache the response (TTL: 1 hour)
    if cache.available:
        cache.set_query_response(query_hash, response)

    # ------------------------------------------------------------------
    # Step 9: Publish metrics and return
    # ------------------------------------------------------------------
    _publish_metrics(_get_cloudwatch_client(), latency_ms, cache_hit, correlation_id)

    logger.info(
        "Query completed in %dms, %d sources", latency_ms, len(citations), extra=extra,
    )

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(asdict(response)),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_cache_service() -> CacheService:
    """Return a CacheService instance (graceful degradation)."""
    try:
        return CacheService()
    except Exception:
        logger.warning("CacheService unavailable — proceeding without cache")
        # Return a stub that reports unavailable
        svc = object.__new__(CacheService)
        svc._client = None
        return svc


def _error_response(
    status_code: int,
    code: str,
    message: str,
    request_id: str,
    retry_after: int | None = None,
) -> dict[str, Any]:
    """Build a standardized API Gateway error response."""
    headers: dict[str, Any] = {"Content-Type": "application/json"}
    if retry_after is not None:
        headers["Retry-After"] = str(retry_after)
    return {
        "statusCode": status_code,
        "headers": headers,
        "body": json.dumps({
            "error": {
                "code": code,
                "message": message,
                "request_id": request_id,
            }
        }),
    }
