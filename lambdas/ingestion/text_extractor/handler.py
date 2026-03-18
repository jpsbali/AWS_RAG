"""Text Extractor Lambda handler for the ingestion pipeline.

Accepts a Step Functions event with document_id, s3_bucket, s3_key, file_type,
content_type, and correlation_id.  Routes to Amazon Textract for PDF/image
formats and to native Python libraries for TXT, CSV, HTML, and DOCX.

Validates extraction quality against a configurable minimum character threshold
(default 50), stores extracted text in S3 ``processed/text/``, and returns an
``ExtractionResult``-compatible dict.
"""

from __future__ import annotations

import csv
import io
import json
import os
import time
from dataclasses import asdict
from typing import Any

import boto3
from botocore.exceptions import ClientError

from lambdas.shared.constants import (
    MIN_EXTRACTION_QUALITY_THRESHOLD,
    PROCESSED_TEXT_PREFIX,
)
from lambdas.shared.models import ExtractionResult
from lambdas.shared.utils import generate_correlation_id, get_structured_logger

# ---------------------------------------------------------------------------
# Optional third-party imports (may not be installed in all environments)
# ---------------------------------------------------------------------------
try:
    from bs4 import BeautifulSoup  # type: ignore[import-untyped]

    _BS4_AVAILABLE = True
except ImportError:
    _BS4_AVAILABLE = False

try:
    import docx as _docx_module  # type: ignore[import-untyped]

    _DOCX_AVAILABLE = True
except ImportError:
    _DOCX_AVAILABLE = False

logger = get_structured_logger(__name__)

# ---------------------------------------------------------------------------
# File-type sets
# ---------------------------------------------------------------------------
_TEXTRACT_TYPES: frozenset[str] = frozenset({"pdf", "png", "jpeg", "tiff"})
_NATIVE_TYPES: frozenset[str] = frozenset({"txt", "csv", "html", "docx"})

# Textract async threshold (pages)
_ASYNC_PAGE_THRESHOLD = 15


# ---------------------------------------------------------------------------
# boto3 client helpers (extracted for testability)
# ---------------------------------------------------------------------------

def _get_s3_client():  # noqa: ANN202
    return boto3.client("s3")


def _get_textract_client():  # noqa: ANN202
    return boto3.client("textract")


# ---------------------------------------------------------------------------
# Quality validation
# ---------------------------------------------------------------------------

def validate_quality(text: str, min_threshold: int = MIN_EXTRACTION_QUALITY_THRESHOLD) -> bool:
    """Return ``True`` if *text* meets the minimum character-length threshold."""
    return len(text) >= min_threshold


# ---------------------------------------------------------------------------
# Textract extraction helpers
# ---------------------------------------------------------------------------

def _extract_textract_sync(
    s3_bucket: str,
    s3_key: str,
    file_type: str,
    textract: Any,
) -> ExtractionResult:
    """Use synchronous Textract APIs (DetectDocumentText / AnalyzeDocument)."""
    s3_obj = {"S3Object": {"Bucket": s3_bucket, "Name": s3_key}}

    # Use AnalyzeDocument for PDFs (may contain tables/forms)
    if file_type == "pdf":
        response = textract.analyze_document(
            Document=s3_obj,
            FeatureTypes=["TABLES", "FORMS"],
        )
    else:
        # Images – basic text detection
        response = textract.detect_document_text(Document=s3_obj)

    return _parse_textract_response(response, document_id="")


def _extract_textract_async(
    s3_bucket: str,
    s3_key: str,
    textract: Any,
) -> ExtractionResult:
    """Use asynchronous Textract StartDocumentAnalysis for large documents."""
    start_resp = textract.start_document_analysis(
        DocumentLocation={"S3Object": {"Bucket": s3_bucket, "Name": s3_key}},
        FeatureTypes=["TABLES", "FORMS"],
    )
    job_id = start_resp["JobId"]

    # Poll until complete
    while True:
        status_resp = textract.get_document_analysis(JobId=job_id)
        status = status_resp["JobStatus"]
        if status == "SUCCEEDED":
            break
        if status == "FAILED":
            raise RuntimeError(f"Textract async job {job_id} failed")
        time.sleep(2)

    # Collect all pages of results
    blocks: list[dict] = status_resp.get("Blocks", [])
    next_token = status_resp.get("NextToken")
    while next_token:
        page_resp = textract.get_document_analysis(JobId=job_id, NextToken=next_token)
        blocks.extend(page_resp.get("Blocks", []))
        next_token = page_resp.get("NextToken")

    return _parse_textract_blocks(blocks, document_id="")


def _parse_textract_response(response: dict, document_id: str) -> ExtractionResult:
    """Parse a synchronous Textract response into an ``ExtractionResult``."""
    return _parse_textract_blocks(response.get("Blocks", []), document_id)


def _parse_textract_blocks(blocks: list[dict], document_id: str) -> ExtractionResult:
    """Parse Textract blocks into an ``ExtractionResult``."""
    lines: list[str] = []
    tables: list[dict] = []
    forms: list[dict] = []
    pages: set[int] = set()
    confidences: list[float] = []

    for block in blocks:
        block_type = block.get("BlockType", "")
        page = block.get("Page", 1)
        pages.add(page)

        if block_type == "LINE":
            lines.append(block.get("Text", ""))
            confidences.append(block.get("Confidence", 0.0))
        elif block_type == "TABLE":
            tables.append(block)
        elif block_type == "KEY_VALUE_SET":
            forms.append(block)

    avg_confidence = (sum(confidences) / len(confidences) / 100.0) if confidences else 0.0

    return ExtractionResult(
        document_id=document_id,
        text="\n".join(lines),
        pages=len(pages),
        tables=tables,
        forms=forms,
        extraction_method="textract",
        confidence=round(avg_confidence, 4),
    )


# ---------------------------------------------------------------------------
# Textract routing
# ---------------------------------------------------------------------------

def _extract_via_textract(
    s3_bucket: str,
    s3_key: str,
    file_type: str,
    document_id: str,
    textract: Any | None = None,
    s3_client: Any | None = None,
) -> ExtractionResult:
    """Route to the appropriate Textract API based on document size."""
    textract = textract or _get_textract_client()
    s3 = s3_client or _get_s3_client()

    # Estimate page count for PDFs to decide sync vs async
    use_async = False
    if file_type == "pdf":
        try:
            head = s3.head_object(Bucket=s3_bucket, Key=s3_key)
            # Rough heuristic: ~50 KB per page for a typical PDF
            size_bytes = head.get("ContentLength", 0)
            estimated_pages = max(1, size_bytes // 50_000)
            if estimated_pages > _ASYNC_PAGE_THRESHOLD:
                use_async = True
        except ClientError:
            pass  # fall back to sync

    if use_async:
        result = _extract_textract_async(s3_bucket, s3_key, textract)
    else:
        result = _extract_textract_sync(s3_bucket, s3_key, file_type, textract)

    result.document_id = document_id
    return result


# ---------------------------------------------------------------------------
# Native extraction helpers
# ---------------------------------------------------------------------------

def _extract_native_txt(body: bytes) -> str:
    """Extract text from a plain-text file."""
    return body.decode("utf-8", errors="replace")


def _extract_native_csv(body: bytes) -> str:
    """Extract text from a CSV file."""
    text_content = body.decode("utf-8", errors="replace")
    reader = csv.reader(io.StringIO(text_content))
    rows = [" ".join(row) for row in reader]
    return "\n".join(rows)


def _extract_native_html(body: bytes) -> str:
    """Extract text from an HTML file using BeautifulSoup."""
    if not _BS4_AVAILABLE:
        raise ImportError(
            "beautifulsoup4 is required for HTML extraction. "
            "Install it with: pip install beautifulsoup4"
        )
    html_content = body.decode("utf-8", errors="replace")
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text(separator="\n", strip=True)


def _extract_native_docx(body: bytes) -> str:
    """Extract text from a DOCX file using python-docx."""
    if not _DOCX_AVAILABLE:
        raise ImportError(
            "python-docx is required for DOCX extraction. "
            "Install it with: pip install python-docx"
        )
    doc = _docx_module.Document(io.BytesIO(body))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


_NATIVE_EXTRACTORS: dict[str, Any] = {
    "txt": _extract_native_txt,
    "csv": _extract_native_csv,
    "html": _extract_native_html,
    "docx": _extract_native_docx,
}


def _extract_via_native(
    s3_bucket: str,
    s3_key: str,
    file_type: str,
    document_id: str,
    s3_client: Any | None = None,
) -> ExtractionResult:
    """Extract text using native Python libraries (no Textract)."""
    s3 = s3_client or _get_s3_client()
    obj = s3.get_object(Bucket=s3_bucket, Key=s3_key)
    body: bytes = obj["Body"].read()

    extractor_fn = _NATIVE_EXTRACTORS.get(file_type)
    if extractor_fn is None:
        raise ValueError(f"No native extractor for file type: {file_type}")

    text = extractor_fn(body)

    return ExtractionResult(
        document_id=document_id,
        text=text,
        pages=1,
        tables=[],
        forms=[],
        extraction_method="native",
        confidence=1.0,
    )


# ---------------------------------------------------------------------------
# S3 storage helper
# ---------------------------------------------------------------------------

def _store_extracted_text(
    s3_bucket: str,
    document_id: str,
    text: str,
    s3_client: Any | None = None,
) -> str:
    """Store extracted text in S3 under ``processed/text/{document_id}.txt``."""
    s3 = s3_client or _get_s3_client()
    output_key = f"{PROCESSED_TEXT_PREFIX}{document_id}.txt"
    s3.put_object(
        Bucket=s3_bucket,
        Key=output_key,
        Body=text.encode("utf-8"),
        ContentType="text/plain",
    )
    return output_key


# ---------------------------------------------------------------------------
# Lambda handler
# ---------------------------------------------------------------------------

def handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    """Extract text from a document and store the result in S3.

    Parameters
    ----------
    event : dict
        Expected keys from Step Functions:
        ``document_id``, ``s3_bucket``, ``s3_key``, ``file_type``,
        ``content_type`` (``"textract"`` | ``"native"``), ``correlation_id``.
    context : object, optional
        Lambda context (unused).

    Returns
    -------
    dict
        ExtractionResult fields plus ``valid``, ``output_key``, and
        ``correlation_id``.
    """
    correlation_id = event.get("correlation_id") or generate_correlation_id()
    extra = {"correlation_id": correlation_id}

    document_id: str = event["document_id"]
    s3_bucket: str = event["s3_bucket"]
    s3_key: str = event["s3_key"]
    file_type: str = event["file_type"]
    content_type: str = event["content_type"]  # "textract" or "native"
    min_threshold: int = int(
        event.get("min_threshold", os.environ.get("MIN_EXTRACTION_THRESHOLD", MIN_EXTRACTION_QUALITY_THRESHOLD))
    )

    logger.info(
        "Starting text extraction for %s (type=%s, content_type=%s)",
        document_id,
        file_type,
        content_type,
        extra=extra,
    )

    try:
        if content_type == "textract":
            result = _extract_via_textract(s3_bucket, s3_key, file_type, document_id)
        elif content_type == "native":
            result = _extract_via_native(s3_bucket, s3_key, file_type, document_id)
        else:
            raise ValueError(f"Unknown content_type: {content_type}")
    except Exception as exc:
        logger.error(
            "Extraction failed for %s: %s",
            document_id,
            str(exc),
            extra=extra,
        )
        return {
            "document_id": document_id,
            "s3_bucket": s3_bucket,
            "s3_key": s3_key,
            "text": "",
            "pages": 0,
            "tables": [],
            "forms": [],
            "extraction_method": content_type,
            "confidence": 0.0,
            "valid": False,
            "error": str(exc),
            "correlation_id": correlation_id,
        }

    # Quality validation
    is_valid = validate_quality(result.text, min_threshold)

    if not is_valid:
        logger.warning(
            "Extraction quality below threshold for %s: %d chars (min %d)",
            document_id,
            len(result.text),
            min_threshold,
            extra=extra,
        )

    # Store extracted text in S3
    output_key = ""
    try:
        output_key = _store_extracted_text(s3_bucket, document_id, result.text)
        logger.info(
            "Stored extracted text for %s at s3://%s/%s",
            document_id,
            s3_bucket,
            output_key,
            extra=extra,
        )
    except Exception as exc:
        logger.error(
            "Failed to store extracted text for %s: %s",
            document_id,
            str(exc),
            extra=extra,
        )

    response = asdict(result)
    response["valid"] = is_valid
    response["output_key"] = output_key
    response["correlation_id"] = correlation_id

    logger.info(
        "Extraction complete for %s: valid=%s, pages=%d, method=%s",
        document_id,
        is_valid,
        result.pages,
        result.extraction_method,
        extra=extra,
    )

    return response
