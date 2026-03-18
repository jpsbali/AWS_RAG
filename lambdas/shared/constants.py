"""Shared constants for the AWS-native RAG service."""

# ---------------------------------------------------------------------------
# Supported file types
# ---------------------------------------------------------------------------
SUPPORTED_FILE_TYPES: frozenset[str] = frozenset(
    {"pdf", "png", "jpeg", "tiff", "docx", "txt", "csv", "html"}
)

SUPPORTED_CONTENT_TYPES: dict[str, str] = {
    "application/pdf": "pdf",
    "image/png": "png",
    "image/jpeg": "jpeg",
    "image/tiff": "tiff",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "text/plain": "txt",
    "text/csv": "csv",
    "text/html": "html",
}

# ---------------------------------------------------------------------------
# S3 prefix constants
# ---------------------------------------------------------------------------
RAW_PREFIX = "raw/"
PROCESSED_TEXT_PREFIX = "processed/text/"
PROCESSED_CHUNKS_PREFIX = "processed/chunks/"
PROCESSED_METADATA_PREFIX = "processed/metadata/"
FAILED_PREFIX = "failed/"
ARCHIVES_PREFIX = "archives/"

# ---------------------------------------------------------------------------
# Cache TTL values (seconds)
# ---------------------------------------------------------------------------
EMBEDDING_CACHE_TTL = 30 * 86400       # 30 days
QUERY_RESULT_CACHE_TTL = 86400         # 24 hours
LLM_RESPONSE_CACHE_TTL = 3600          # 1 hour
SESSION_CACHE_TTL = 7200               # 2 hours

# ---------------------------------------------------------------------------
# Cache key prefixes
# ---------------------------------------------------------------------------
EMBEDDING_CACHE_PREFIX = "emb:"
QUERY_RESULT_CACHE_PREFIX = "qr:"
LLM_RESPONSE_CACHE_PREFIX = "llm:"
SESSION_CACHE_PREFIX = "sess:"

# ---------------------------------------------------------------------------
# Default chunking parameters
# ---------------------------------------------------------------------------
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_OVERLAP = 200
DEFAULT_MIN_CHUNK = 200
DEFAULT_MAX_CHUNK = 1500
DEFAULT_WINDOW_SIZE = 3

# ---------------------------------------------------------------------------
# Model IDs
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
LLM_PRIMARY_MODEL_ID = "anthropic.claude-3-5-sonnet-20241022-v2:0"
LLM_FALLBACK_MODEL_ID = "anthropic.claude-3-5-haiku-20241022-v1:0"

# ---------------------------------------------------------------------------
# Embedding configuration
# ---------------------------------------------------------------------------
DEFAULT_EMBEDDING_DIMENSIONS = 1024
MAX_EMBEDDING_BATCH_SIZE = 100

# ---------------------------------------------------------------------------
# Quality thresholds
# ---------------------------------------------------------------------------
MIN_EXTRACTION_QUALITY_THRESHOLD = 50
QUALITY_THRESHOLD_CHARS = 50
