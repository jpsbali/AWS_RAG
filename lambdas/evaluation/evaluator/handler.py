"""Evaluation Lambda — run RAGAS evaluation on a dataset and publish results.

Loads question-answer-context triples from S3, runs evaluation metrics
(context_precision, context_recall, context_relevancy, faithfulness,
answer_relevancy), stores the report in S3, and publishes scores to
CloudWatch custom metrics under the ``RAG/Evaluation`` namespace.

When the ``ragas`` library is available it is used as the evaluation engine
with Bedrock as the evaluator LLM.  When ``ragas`` is *not* installed (e.g.
in a stripped-down Lambda deployment) a lightweight fallback computes
simplified proxy metrics so the pipeline still produces a valid report.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import boto3

from lambdas.shared.models import EvaluationReport
from lambdas.shared.utils import generate_correlation_id, get_structured_logger

logger = get_structured_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EVALUATION_METRICS = [
    "context_precision",
    "context_recall",
    "context_relevancy",
    "faithfulness",
    "answer_relevancy",
]
CW_NAMESPACE = "RAG/Evaluation"
REPORT_PREFIX = "evaluations/reports"

# ---------------------------------------------------------------------------
# AWS client helpers (easily mockable)
# ---------------------------------------------------------------------------

def _get_s3_client() -> Any:
    return boto3.client("s3")


def _get_cloudwatch_client() -> Any:
    return boto3.client("cloudwatch")


# ---------------------------------------------------------------------------
# Try to import RAGAS — fall back to simplified evaluation if unavailable
# ---------------------------------------------------------------------------
_RAGAS_AVAILABLE = False
try:
    from ragas import evaluate as ragas_evaluate
    from datasets import Dataset

    # Prefer the newer import path (ragas >= 0.2); fall back to legacy path.
    try:
        from ragas.metrics.collections import (
            answer_relevancy as ragas_answer_relevancy,
            context_precision as ragas_context_precision,
            context_recall as ragas_context_recall,
            context_relevance as context_relevance,
            faithfulness as ragas_faithfulness,
        )
    except ImportError:
        from ragas.metrics import (
            answer_relevancy as ragas_answer_relevancy,
            context_precision as ragas_context_precision,
            context_recall as ragas_context_recall,
            context_relevancy as ragas_context_relevancy,
            faithfulness as ragas_faithfulness,
        )

    _RAGAS_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _load_dataset(s3_client: Any, bucket: str, key: str) -> list[dict]:
    """Download and parse the evaluation dataset JSON from S3.

    Expected format — a JSON array of objects::

        [
            {
                "question": "...",
                "answer": "...",
                "contexts": ["..."],
                "ground_truth": "..."
            },
            ...
        ]
    """
    response = s3_client.get_object(Bucket=bucket, Key=key)
    body = response["Body"].read().decode("utf-8")
    dataset = json.loads(body)
    if not isinstance(dataset, list) or len(dataset) == 0:
        raise ValueError("Evaluation dataset must be a non-empty JSON array")
    return dataset


# ---------------------------------------------------------------------------
# RAGAS evaluation path
# ---------------------------------------------------------------------------

def _run_ragas_evaluation(dataset: list[dict]) -> dict[str, Any]:
    """Run full RAGAS evaluation using the ragas library."""
    hf_dataset = Dataset.from_dict(
        {
            "question": [item["question"] for item in dataset],
            "answer": [item["answer"] for item in dataset],
            "contexts": [item["contexts"] for item in dataset],
            "ground_truth": [item.get("ground_truth", "") for item in dataset],
        }
    )

    metrics = [
        ragas_context_precision,
        ragas_context_recall,
        ragas_context_relevancy,
        ragas_faithfulness,
        ragas_answer_relevancy,
    ]

    result = ragas_evaluate(hf_dataset, metrics=metrics)

    scores: dict[str, float] = {}
    for metric_name in EVALUATION_METRICS:
        scores[metric_name] = float(result.get(metric_name, 0.0))

    per_question: list[dict] = []
    if hasattr(result, "to_pandas"):
        df = result.to_pandas()
        for _, row in df.iterrows():
            entry: dict[str, Any] = {}
            for metric_name in EVALUATION_METRICS:
                if metric_name in row:
                    entry[metric_name] = float(row[metric_name])
            per_question.append(entry)
    else:
        per_question = [{m: scores.get(m, 0.0) for m in EVALUATION_METRICS} for _ in dataset]

    return {"scores": scores, "per_question": per_question}


# ---------------------------------------------------------------------------
# Fallback (simplified) evaluation path
# ---------------------------------------------------------------------------

def _keyword_overlap(text_a: str, text_b: str) -> float:
    """Return Jaccard similarity of word sets between two strings."""
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def _context_coverage(answer: str, contexts: list[str]) -> float:
    """Fraction of answer words found in the concatenated contexts."""
    answer_words = set(answer.lower().split())
    if not answer_words:
        return 0.0
    context_words: set[str] = set()
    for ctx in contexts:
        context_words.update(ctx.lower().split())
    if not context_words:
        return 0.0
    return len(answer_words & context_words) / len(answer_words)


def _run_fallback_evaluation(dataset: list[dict]) -> dict[str, Any]:
    """Compute simplified proxy metrics when RAGAS is not available."""
    per_question: list[dict] = []

    for item in dataset:
        question = item.get("question", "")
        answer = item.get("answer", "")
        contexts = item.get("contexts", [])
        ground_truth = item.get("ground_truth", "")

        combined_context = " ".join(contexts)

        # Proxy: context_precision — overlap between question and contexts
        ctx_precision = _keyword_overlap(question, combined_context)
        # Proxy: context_recall — overlap between ground_truth and contexts
        ctx_recall = _keyword_overlap(ground_truth, combined_context) if ground_truth else ctx_precision
        # Proxy: context_relevancy — average of precision and recall
        ctx_relevancy = (ctx_precision + ctx_recall) / 2.0
        # Proxy: faithfulness — how much of the answer is covered by contexts
        faith = _context_coverage(answer, contexts)
        # Proxy: answer_relevancy — overlap between question and answer
        ans_rel = _keyword_overlap(question, answer)

        per_question.append(
            {
                "context_precision": round(ctx_precision, 4),
                "context_recall": round(ctx_recall, 4),
                "context_relevancy": round(ctx_relevancy, 4),
                "faithfulness": round(faith, 4),
                "answer_relevancy": round(ans_rel, 4),
            }
        )

    # Aggregate: mean of per-question scores
    scores: dict[str, float] = {}
    for metric_name in EVALUATION_METRICS:
        values = [q[metric_name] for q in per_question]
        scores[metric_name] = round(sum(values) / len(values), 4) if values else 0.0

    return {"scores": scores, "per_question": per_question}


# ---------------------------------------------------------------------------
# Report storage & CloudWatch publishing
# ---------------------------------------------------------------------------

def _store_report(s3_client: Any, bucket: str, report: EvaluationReport) -> None:
    """Serialise the evaluation report as JSON and upload to S3."""
    report_dict = {
        "run_id": report.run_id,
        "timestamp": report.timestamp,
        "dataset_size": report.dataset_size,
        "metrics": report.metrics,
        "per_question_scores": report.per_question_scores,
        "s3_report_key": report.s3_report_key,
    }
    s3_client.put_object(
        Bucket=bucket,
        Key=report.s3_report_key,
        Body=json.dumps(report_dict, indent=2),
        ContentType="application/json",
    )


def _publish_metrics(cw_client: Any, report: EvaluationReport, env_name: str) -> None:
    """Publish per-metric scores to CloudWatch as custom metrics."""
    metric_data: list[dict[str, Any]] = []
    for metric_name, score in report.metrics.items():
        metric_data.append(
            {
                "MetricName": metric_name,
                "Value": float(score),
                "Unit": "None",
                "Dimensions": [
                    {"Name": "Environment", "Value": env_name},
                    {"Name": "RunId", "Value": report.run_id},
                ],
            }
        )

    # Also publish an aggregate score
    if report.metrics:
        aggregate = sum(report.metrics.values()) / len(report.metrics)
        metric_data.append(
            {
                "MetricName": "aggregate_score",
                "Value": round(aggregate, 4),
                "Unit": "None",
                "Dimensions": [
                    {"Name": "Environment", "Value": env_name},
                    {"Name": "RunId", "Value": report.run_id},
                ],
            }
        )

    cw_client.put_metric_data(Namespace=CW_NAMESPACE, MetricData=metric_data)


# ---------------------------------------------------------------------------
# Lambda handler
# ---------------------------------------------------------------------------

def handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    """Evaluate a RAG dataset and produce a scored report.

    Parameters
    ----------
    event : dict
        Must contain ``dataset_s3_key``.  Optionally ``bucket`` to override
        the default ``DOCUMENT_BUCKET`` environment variable.
    context : Any
        Lambda context (unused).

    Returns
    -------
    dict
        ``{"run_id", "s3_report_key", "metrics", "dataset_size", "evaluation_mode"}``
    """
    correlation_id = generate_correlation_id()
    env_name = os.environ.get("ENV_NAME", "dev")
    default_bucket = os.environ.get("DOCUMENT_BUCKET", "rag-documents-test")

    dataset_s3_key = event.get("dataset_s3_key")
    if not dataset_s3_key:
        logger.error("Missing dataset_s3_key in event", extra={"correlation_id": correlation_id})
        raise ValueError("Event must contain 'dataset_s3_key'")

    bucket = event.get("bucket") or default_bucket
    run_id = f"eval-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{correlation_id[:8]}"
    report_key = f"{REPORT_PREFIX}/{run_id}.json"

    logger.info(
        "Starting evaluation run %s (dataset=%s, bucket=%s)",
        run_id,
        dataset_s3_key,
        bucket,
        extra={"correlation_id": correlation_id},
    )

    s3_client = _get_s3_client()
    cw_client = _get_cloudwatch_client()

    # 1. Load dataset
    dataset = _load_dataset(s3_client, bucket, key=dataset_s3_key)

    # 2. Run evaluation (RAGAS or fallback)
    if _RAGAS_AVAILABLE:
        evaluation_mode = "ragas"
        result = _run_ragas_evaluation(dataset)
    else:
        evaluation_mode = "fallback"
        result = _run_fallback_evaluation(dataset)

    scores = result["scores"]
    per_question = result["per_question"]

    # 3. Build report
    report = EvaluationReport(
        run_id=run_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        dataset_size=len(dataset),
        metrics=scores,
        per_question_scores=per_question,
        s3_report_key=report_key,
    )

    # 4. Store report in S3
    _store_report(s3_client, bucket, report)

    # 5. Publish CloudWatch metrics
    _publish_metrics(cw_client, report, env_name)

    logger.info(
        "Evaluation run %s complete — mode=%s, dataset_size=%d",
        run_id,
        evaluation_mode,
        len(dataset),
        extra={"correlation_id": correlation_id},
    )

    return {
        "run_id": run_id,
        "s3_report_key": report_key,
        "metrics": scores,
        "dataset_size": len(dataset),
        "evaluation_mode": evaluation_mode,
    }
