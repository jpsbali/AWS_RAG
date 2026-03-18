"""Chunker Lambda handler for the ingestion pipeline.

Accepts a Step Functions event with document_id, s3_bucket, text (or s3_key to
extracted text), correlation_id, strategy, chunk_config overrides, entities, and
key_phrases.  Splits text into chunks using one of four strategies (fixed-size,
semantic, recursive, sentence_window), attaches metadata, stores the resulting
JSON in S3 under ``processed/chunks/``, and returns a result dict.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict
from typing import Any

import boto3

from lambdas.shared.constants import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_CHUNK,
    DEFAULT_MIN_CHUNK,
    DEFAULT_OVERLAP,
    DEFAULT_WINDOW_SIZE,
    PROCESSED_CHUNKS_PREFIX,
)
from lambdas.shared.models import Chunk, ChunkConfig, ChunkMetadata
from lambdas.shared.utils import generate_correlation_id, get_structured_logger

logger = get_structured_logger(__name__)


# ---------------------------------------------------------------------------
# boto3 client helpers
# ---------------------------------------------------------------------------

def _get_s3_client():  # noqa: ANN202
    """Return a boto3 S3 client."""
    return boto3.client("s3")


# ---------------------------------------------------------------------------
# Token helpers (simple word-based approximation)
# ---------------------------------------------------------------------------

def _token_count(text: str) -> int:
    """Approximate token count by splitting on whitespace."""
    return len(text.split())


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def _load_text_from_s3(s3_bucket: str, s3_key: str, s3_client: Any | None = None) -> str:
    """Load text content from an S3 object."""
    s3 = s3_client or _get_s3_client()
    obj = s3.get_object(Bucket=s3_bucket, Key=s3_key)
    return obj["Body"].read().decode("utf-8", errors="replace")


def _store_chunks(
    s3_bucket: str,
    document_id: str,
    chunks: list[Chunk],
    s3_client: Any | None = None,
) -> str:
    """Store chunks as a JSON array in S3 under ``processed/chunks/{document_id}.json``."""
    s3 = s3_client or _get_s3_client()
    output_key = f"{PROCESSED_CHUNKS_PREFIX}{document_id}.json"
    payload = [asdict(c) for c in chunks]
    s3.put_object(
        Bucket=s3_bucket,
        Key=output_key,
        Body=json.dumps(payload, default=str).encode("utf-8"),
        ContentType="application/json",
    )
    return output_key


# ---------------------------------------------------------------------------
# Chunk builder helper
# ---------------------------------------------------------------------------

def _build_chunk(
    document_id: str,
    chunk_index: int,
    content: str,
    start_position: int,
    end_position: int,
    entities: list[str],
    key_phrases: list[str],
    section_title: str | None = None,
    page_number: int | None = None,
) -> Chunk:
    """Create a ``Chunk`` instance with metadata."""
    chunk_id = f"{document_id}_chunk_{chunk_index}"
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=chunk_index,
        content=content,
        chunk_size=_token_count(content),
        start_position=start_position,
        end_position=end_position,
        metadata=ChunkMetadata(
            section_title=section_title,
            page_number=page_number,
            entities=entities,
            key_phrases=key_phrases,
        ),
    )


# ---------------------------------------------------------------------------
# Chunking strategies
# ---------------------------------------------------------------------------

def _chunk_fixed(text: str, config: ChunkConfig, document_id: str, entities: list[str], key_phrases: list[str]) -> list[Chunk]:
    """Fixed-size chunking: split by token count with overlap."""
    words = text.split()
    chunks: list[Chunk] = []
    chunk_size = config.chunk_size
    overlap = config.overlap
    idx = 0
    chunk_index = 0

    while idx < len(words):
        end_idx = min(idx + chunk_size, len(words))
        chunk_words = words[idx:end_idx]
        content = " ".join(chunk_words)

        # Calculate character positions in original text
        start_pos = _char_position_of_word(text, words, idx)
        end_pos = _char_position_of_word(text, words, end_idx - 1) + len(words[end_idx - 1]) if end_idx > 0 else 0

        chunks.append(_build_chunk(
            document_id=document_id,
            chunk_index=chunk_index,
            content=content,
            start_position=start_pos,
            end_position=end_pos,
            entities=entities,
            key_phrases=key_phrases,
        ))
        chunk_index += 1

        # Advance by (chunk_size - overlap) words
        step = max(1, chunk_size - overlap)
        idx += step

    return chunks


def _char_position_of_word(text: str, words: list[str], word_index: int) -> int:
    """Return the character offset in *text* where *words[word_index]* starts.

    Uses a simple forward scan to find each word's position.
    """
    pos = 0
    for i in range(word_index):
        found = text.find(words[i], pos)
        if found == -1:
            break
        pos = found + len(words[i])
    # Find the target word
    found = text.find(words[word_index], pos)
    return found if found != -1 else pos


def _chunk_semantic(text: str, config: ChunkConfig, document_id: str, entities: list[str], key_phrases: list[str]) -> list[Chunk]:
    """Semantic chunking: split at headers/paragraphs, enforce min/max token bounds."""
    # Split on markdown-style headers or double newlines (paragraph breaks)
    sections = re.split(r"(?=^#{1,6}\s)|(?:\n\n+)", text, flags=re.MULTILINE)
    sections = [s.strip() for s in sections if s and s.strip()]

    chunks: list[Chunk] = []
    chunk_index = 0
    current_section_title: str | None = None

    buffer = ""
    buffer_start = 0

    for section in sections:
        # Detect section title (markdown header)
        header_match = re.match(r"^(#{1,6})\s+(.*)", section)
        if header_match:
            current_section_title = header_match.group(2).strip()

        candidate = (buffer + " " + section).strip() if buffer else section
        candidate_tokens = _token_count(candidate)

        if candidate_tokens > config.max_chunk_size and buffer:
            # Flush buffer as a chunk if it meets minimum
            if _token_count(buffer) >= config.min_chunk_size:
                start_pos = text.find(buffer[:50], buffer_start)
                if start_pos == -1:
                    start_pos = buffer_start
                end_pos = start_pos + len(buffer)
                chunks.append(_build_chunk(
                    document_id=document_id,
                    chunk_index=chunk_index,
                    content=buffer,
                    start_position=start_pos,
                    end_position=end_pos,
                    entities=entities,
                    key_phrases=key_phrases,
                    section_title=current_section_title,
                ))
                chunk_index += 1
            buffer = section
            buffer_start = text.find(section[:50], buffer_start) if section else buffer_start
        else:
            buffer = candidate
            if not chunks and chunk_index == 0:
                buffer_start = 0

    # Flush remaining buffer
    if buffer.strip():
        tokens = _token_count(buffer)
        if tokens < config.min_chunk_size and chunks:
            # Merge with last chunk if too small
            last = chunks[-1]
            merged = last.content + " " + buffer
            chunks[-1] = _build_chunk(
                document_id=document_id,
                chunk_index=last.chunk_index,
                content=merged,
                start_position=last.start_position,
                end_position=last.end_position + len(buffer) + 1,
                entities=entities,
                key_phrases=key_phrases,
                section_title=current_section_title,
            )
        else:
            start_pos = text.find(buffer[:50], buffer_start) if buffer else buffer_start
            if start_pos == -1:
                start_pos = buffer_start
            end_pos = start_pos + len(buffer)
            chunks.append(_build_chunk(
                document_id=document_id,
                chunk_index=chunk_index,
                content=buffer,
                start_position=start_pos,
                end_position=end_pos,
                entities=entities,
                key_phrases=key_phrases,
                section_title=current_section_title,
            ))

    return chunks


def _chunk_recursive(text: str, config: ChunkConfig, document_id: str, entities: list[str], key_phrases: list[str]) -> list[Chunk]:
    """Recursive character splitting: apply separators in priority order."""
    separators = ["\n\n", "\n", ". ", " "]
    raw_chunks = _recursive_split(text, separators, config.chunk_size)

    chunks: list[Chunk] = []
    pos = 0
    for chunk_index, content in enumerate(raw_chunks):
        start_pos = text.find(content[:50], pos) if content else pos
        if start_pos == -1:
            start_pos = pos
        end_pos = start_pos + len(content)
        chunks.append(_build_chunk(
            document_id=document_id,
            chunk_index=chunk_index,
            content=content,
            start_position=start_pos,
            end_position=end_pos,
            entities=entities,
            key_phrases=key_phrases,
        ))
        pos = end_pos

    return chunks


def _recursive_split(text: str, separators: list[str], max_tokens: int) -> list[str]:
    """Recursively split *text* using *separators* until each piece fits within *max_tokens*."""
    if _token_count(text) <= max_tokens or not separators:
        return [text.strip()] if text.strip() else []

    sep = separators[0]
    remaining_seps = separators[1:]
    parts = text.split(sep)

    result: list[str] = []
    buffer = ""

    for part in parts:
        candidate = (buffer + sep + part) if buffer else part
        if _token_count(candidate) <= max_tokens:
            buffer = candidate
        else:
            if buffer.strip():
                if _token_count(buffer) <= max_tokens:
                    result.append(buffer.strip())
                else:
                    result.extend(_recursive_split(buffer, remaining_seps, max_tokens))
            buffer = part

    if buffer.strip():
        if _token_count(buffer) <= max_tokens:
            result.append(buffer.strip())
        else:
            result.extend(_recursive_split(buffer, remaining_seps, max_tokens))

    return result


def _chunk_sentence_window(text: str, config: ChunkConfig, document_id: str, entities: list[str], key_phrases: list[str]) -> list[Chunk]:
    """Sentence window chunking: center on each sentence with a configurable window."""
    # Split into sentences using a simple regex
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return []

    window = config.window_size
    chunks: list[Chunk] = []

    for i, _sentence in enumerate(sentences):
        start_idx = max(0, i - window)
        end_idx = min(len(sentences), i + window + 1)
        window_sentences = sentences[start_idx:end_idx]
        content = " ".join(window_sentences)

        # Calculate character positions
        start_pos = text.find(window_sentences[0][:50]) if window_sentences else 0
        if start_pos == -1:
            start_pos = 0
        end_pos = start_pos + len(content)

        chunks.append(_build_chunk(
            document_id=document_id,
            chunk_index=i,
            content=content,
            start_position=start_pos,
            end_position=end_pos,
            entities=entities,
            key_phrases=key_phrases,
        ))

    return chunks


# ---------------------------------------------------------------------------
# Strategy dispatcher
# ---------------------------------------------------------------------------

_STRATEGIES = {
    "fixed": _chunk_fixed,
    "semantic": _chunk_semantic,
    "recursive": _chunk_recursive,
    "sentence_window": _chunk_sentence_window,
}


def chunk_text(
    text: str,
    config: ChunkConfig,
    document_id: str,
    entities: list[str] | None = None,
    key_phrases: list[str] | None = None,
) -> list[Chunk]:
    """Split *text* into chunks using the strategy specified in *config*."""
    strategy_fn = _STRATEGIES.get(config.strategy)
    if strategy_fn is None:
        raise ValueError(
            f"Unknown chunking strategy '{config.strategy}'. "
            f"Supported: {', '.join(sorted(_STRATEGIES))}"
        )
    return strategy_fn(
        text, config, document_id,
        entities or [], key_phrases or [],
    )


# ---------------------------------------------------------------------------
# Lambda handler
# ---------------------------------------------------------------------------

def handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    """Chunk document text and store results in S3.

    Parameters
    ----------
    event : dict
        Expected keys from Step Functions:
        ``document_id``, ``s3_bucket``, and either ``text`` (inline) or
        ``s3_key`` (path to extracted text in S3).
        Optional: ``correlation_id``, ``strategy`` (default ``"fixed"``),
        ``chunk_config`` (dict of overrides), ``entities`` (list),
        ``key_phrases`` (list).
    context : object, optional
        Lambda context (unused).

    Returns
    -------
    dict
        Result with ``chunk_count``, ``output_key``, and ``correlation_id``.
    """
    correlation_id = event.get("correlation_id") or generate_correlation_id()
    extra = {"correlation_id": correlation_id}

    document_id: str = event["document_id"]
    s3_bucket: str = event["s3_bucket"]
    strategy: str = event.get("strategy", "fixed")
    entities: list[str] = event.get("entities", [])
    key_phrases: list[str] = event.get("key_phrases", [])

    logger.info(
        "Starting chunking for %s (strategy=%s)",
        document_id, strategy, extra=extra,
    )

    # Build ChunkConfig from defaults + optional overrides
    overrides = event.get("chunk_config", {}) or {}
    config = ChunkConfig(
        strategy=strategy,
        chunk_size=int(overrides.get("chunk_size", DEFAULT_CHUNK_SIZE)),
        overlap=int(overrides.get("overlap", DEFAULT_OVERLAP)),
        min_chunk_size=int(overrides.get("min_chunk_size", DEFAULT_MIN_CHUNK)),
        max_chunk_size=int(overrides.get("max_chunk_size", DEFAULT_MAX_CHUNK)),
        window_size=int(overrides.get("window_size", DEFAULT_WINDOW_SIZE)),
    )

    # Resolve text: inline or from S3
    text: str = event.get("text", "")
    if not text:
        s3_key: str = event.get("s3_key", "")
        if not s3_key:
            error_msg = "Either 'text' or 's3_key' must be provided"
            logger.error(error_msg, extra=extra)
            return {
                "document_id": document_id,
                "chunk_count": 0,
                "output_key": "",
                "error": error_msg,
                "correlation_id": correlation_id,
            }
        try:
            text = _load_text_from_s3(s3_bucket, s3_key)
        except Exception as exc:
            error_msg = f"Failed to load text from s3://{s3_bucket}/{s3_key}: {exc}"
            logger.error(error_msg, extra=extra)
            return {
                "document_id": document_id,
                "chunk_count": 0,
                "output_key": "",
                "error": error_msg,
                "correlation_id": correlation_id,
            }

    # Chunk the text
    try:
        chunks = chunk_text(text, config, document_id, entities, key_phrases)
    except Exception as exc:
        error_msg = f"Chunking failed for {document_id}: {exc}"
        logger.error(error_msg, extra=extra)
        return {
            "document_id": document_id,
            "chunk_count": 0,
            "output_key": "",
            "error": error_msg,
            "correlation_id": correlation_id,
        }

    logger.info(
        "Produced %d chunks for %s using strategy '%s'",
        len(chunks), document_id, strategy, extra=extra,
    )

    # Store chunks in S3
    output_key = ""
    try:
        output_key = _store_chunks(s3_bucket, document_id, chunks)
        logger.info(
            "Stored chunks for %s at s3://%s/%s",
            document_id, s3_bucket, output_key, extra=extra,
        )
    except Exception as exc:
        logger.error(
            "Failed to store chunks for %s: %s",
            document_id, str(exc), extra=extra,
        )

    return {
        "document_id": document_id,
        "chunk_count": len(chunks),
        "output_key": output_key,
        "correlation_id": correlation_id,
    }
