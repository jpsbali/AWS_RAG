"""Shared utility functions for the AWS-native RAG service."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Generator


def compute_hash(text: str) -> str:
    """Return the SHA-256 hex digest of *text*."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def generate_correlation_id() -> str:
    """Generate a UUID4 string for request tracing."""
    return str(uuid.uuid4())


class _JsonFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", None),
        }
        return json.dumps(log_entry, default=str)


def get_structured_logger(name: str) -> logging.Logger:
    """Return a logger that outputs structured JSON with timestamp, level, message, and correlation_id."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(_JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def build_cache_key(prefix: str, *parts: str) -> str:
    """Build a cache key from *prefix* and the SHA-256 hash of the concatenated *parts*."""
    combined = ":".join(parts)
    return f"{prefix}{compute_hash(combined)}"


# ---------------------------------------------------------------------------
# AWS X-Ray instrumentation helpers (graceful degradation)
# ---------------------------------------------------------------------------

_xray_available = False
_xray_recorder = None

try:
    from aws_xray_sdk.core import xray_recorder as _recorder
    from aws_xray_sdk.core import patch

    _xray_available = True
    _xray_recorder = _recorder
except ImportError:
    pass


def patch_aws_sdk() -> None:
    """Patch boto3/botocore with X-Ray SDK for automatic trace capture.

    No-op when aws-xray-sdk is not installed.
    """
    if not _xray_available:
        return
    try:
        patch(["boto3", "botocore"])
    except Exception:
        pass


def configure_xray_sampling() -> None:
    """Configure X-Ray trace sampling at 5% for production.

    No-op when aws-xray-sdk is not installed or not in production.
    """
    if not _xray_available or _xray_recorder is None:
        return
    try:
        env = os.environ.get("ENV_NAME", "dev")
        if env == "prod":
            from aws_xray_sdk.core.sampling.local.sampler import LocalSampler

            sampling_rules = {
                "version": 2,
                "default": {"fixed_target": 1, "rate": 0.05},
                "rules": [],
            }
            _xray_recorder.sampler = LocalSampler(sampling_rules)
    except Exception:
        pass


@contextmanager
def create_subsegment(name: str) -> Generator[Any, None, None]:
    """Context manager that creates an X-Ray subsegment.

    Yields the subsegment (or ``None`` when X-Ray is unavailable).
    Gracefully degrades to a no-op when aws-xray-sdk is not installed
    or when there is no active segment.
    """
    if not _xray_available or _xray_recorder is None:
        yield None
        return
    subsegment = None
    try:
        subsegment = _xray_recorder.begin_subsegment(name)
    except Exception:
        pass
    try:
        yield subsegment
    except Exception as exc:
        if subsegment is not None:
            try:
                subsegment.add_exception(exc, exc.__traceback__)
            except Exception:
                pass
        raise
    finally:
        if subsegment is not None:
            try:
                _xray_recorder.end_subsegment()
            except Exception:
                pass
