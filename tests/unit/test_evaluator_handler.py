"""Unit tests for lambdas.evaluation.evaluator.handler."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from moto import mock_aws

import boto3

from lambdas.evaluation.evaluator.handler import (
    _context_coverage,
    _keyword_overlap,
    _load_dataset,
    _publish_metrics,
    _run_fallback_evaluation,
    _store_report,
    handler,
    EVALUATION_METRICS,
    CW_NAMESPACE,
)
from lambdas.shared.models import EvaluationReport


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_DATASET = [
    {
        "question": "What is AWS Lambda?",
        "answer": "AWS Lambda is a serverless compute service that runs code.",
        "contexts": [
            "AWS Lambda lets you run code without provisioning servers.",
            "Lambda automatically scales your application.",
        ],
        "ground_truth": "AWS Lambda is a serverless compute service.",
    },
    {
        "question": "What is Amazon S3?",
        "answer": "Amazon S3 is an object storage service.",
        "contexts": ["Amazon S3 provides scalable object storage in the cloud."],
        "ground_truth": "Amazon S3 is an object storage service for the cloud.",
    },
]


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("ENV_NAME", "test")
    monkeypatch.setenv("DOCUMENT_BUCKET", "rag-documents-test")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")


@pytest.fixture
def s3_with_dataset():
    """Mocked S3 bucket with a sample evaluation dataset uploaded."""
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="rag-documents-test")
        s3.put_object(
            Bucket="rag-documents-test",
            Key="evaluations/dataset.json",
            Body=json.dumps(SAMPLE_DATASET),
        )
        yield s3


# ---------------------------------------------------------------------------
# _keyword_overlap
# ---------------------------------------------------------------------------

class TestKeywordOverlap:
    def test_identical_strings(self):
        assert _keyword_overlap("hello world", "hello world") == 1.0

    def test_no_overlap(self):
        assert _keyword_overlap("hello world", "foo bar") == 0.0

    def test_partial_overlap(self):
        score = _keyword_overlap("hello world foo", "hello bar baz")
        # intersection = {"hello"}, union = {"hello","world","foo","bar","baz"} → 1/5
        assert score == pytest.approx(0.2)

    def test_empty_string(self):
        assert _keyword_overlap("", "hello") == 0.0
        assert _keyword_overlap("hello", "") == 0.0

    def test_case_insensitive(self):
        assert _keyword_overlap("Hello World", "hello world") == 1.0


# ---------------------------------------------------------------------------
# _context_coverage
# ---------------------------------------------------------------------------

class TestContextCoverage:
    def test_full_coverage(self):
        assert _context_coverage("hello world", ["hello world foo bar"]) == 1.0

    def test_no_coverage(self):
        assert _context_coverage("hello world", ["foo bar baz"]) == 0.0

    def test_partial_coverage(self):
        score = _context_coverage("hello world foo", ["hello bar"])
        # answer words = {"hello","world","foo"}, context words = {"hello","bar"}
        # overlap = {"hello"} → 1/3
        assert score == pytest.approx(1 / 3)

    def test_empty_answer(self):
        assert _context_coverage("", ["some context"]) == 0.0

    def test_empty_contexts(self):
        assert _context_coverage("hello", []) == 0.0


# ---------------------------------------------------------------------------
# _load_dataset
# ---------------------------------------------------------------------------

class TestLoadDataset:
    def test_loads_valid_dataset(self, s3_with_dataset):
        dataset = _load_dataset(s3_with_dataset, "rag-documents-test", "evaluations/dataset.json")
        assert len(dataset) == 2
        assert dataset[0]["question"] == "What is AWS Lambda?"

    def test_rejects_empty_array(self, s3_with_dataset):
        s3_with_dataset.put_object(
            Bucket="rag-documents-test",
            Key="evaluations/empty.json",
            Body=json.dumps([]),
        )
        with pytest.raises(ValueError, match="non-empty"):
            _load_dataset(s3_with_dataset, "rag-documents-test", "evaluations/empty.json")

    def test_rejects_non_array(self, s3_with_dataset):
        s3_with_dataset.put_object(
            Bucket="rag-documents-test",
            Key="evaluations/bad.json",
            Body=json.dumps({"not": "an array"}),
        )
        with pytest.raises(ValueError, match="non-empty"):
            _load_dataset(s3_with_dataset, "rag-documents-test", "evaluations/bad.json")


# ---------------------------------------------------------------------------
# _run_fallback_evaluation
# ---------------------------------------------------------------------------

class TestFallbackEvaluation:
    def test_returns_all_metrics(self):
        result = _run_fallback_evaluation(SAMPLE_DATASET)
        for metric in EVALUATION_METRICS:
            assert metric in result["scores"]
            assert isinstance(result["scores"][metric], float)

    def test_per_question_count_matches_dataset(self):
        result = _run_fallback_evaluation(SAMPLE_DATASET)
        assert len(result["per_question"]) == len(SAMPLE_DATASET)

    def test_per_question_has_all_metrics(self):
        result = _run_fallback_evaluation(SAMPLE_DATASET)
        for entry in result["per_question"]:
            for metric in EVALUATION_METRICS:
                assert metric in entry

    def test_scores_between_zero_and_one(self):
        result = _run_fallback_evaluation(SAMPLE_DATASET)
        for metric, score in result["scores"].items():
            assert 0.0 <= score <= 1.0, f"{metric} out of range: {score}"

    def test_handles_missing_ground_truth(self):
        dataset = [
            {
                "question": "What is Lambda?",
                "answer": "A compute service.",
                "contexts": ["Lambda runs code."],
            }
        ]
        result = _run_fallback_evaluation(dataset)
        assert len(result["per_question"]) == 1
        for metric in EVALUATION_METRICS:
            assert metric in result["scores"]


# ---------------------------------------------------------------------------
# _store_report
# ---------------------------------------------------------------------------

class TestStoreReport:
    def test_stores_report_json(self, s3_with_dataset):
        report = EvaluationReport(
            run_id="eval-test-001",
            timestamp="2024-01-15T12:00:00Z",
            dataset_size=2,
            metrics={"context_precision": 0.85, "faithfulness": 0.9},
            per_question_scores=[{"context_precision": 0.85}],
            s3_report_key="evaluations/reports/eval-test-001.json",
        )
        _store_report(s3_with_dataset, "rag-documents-test", report)

        obj = s3_with_dataset.get_object(
            Bucket="rag-documents-test",
            Key="evaluations/reports/eval-test-001.json",
        )
        body = json.loads(obj["Body"].read().decode("utf-8"))
        assert body["run_id"] == "eval-test-001"
        assert body["metrics"]["context_precision"] == 0.85
        assert body["dataset_size"] == 2


# ---------------------------------------------------------------------------
# _publish_metrics
# ---------------------------------------------------------------------------

class TestPublishMetrics:
    def test_publishes_all_metrics_plus_aggregate(self):
        cw = MagicMock()
        report = EvaluationReport(
            run_id="eval-test-002",
            timestamp="2024-01-15T12:00:00Z",
            dataset_size=2,
            metrics={
                "context_precision": 0.8,
                "context_recall": 0.7,
                "context_relevancy": 0.75,
                "faithfulness": 0.9,
                "answer_relevancy": 0.85,
            },
            per_question_scores=[],
            s3_report_key="evaluations/reports/eval-test-002.json",
        )
        _publish_metrics(cw, report, "test")

        cw.put_metric_data.assert_called_once()
        call_kwargs = cw.put_metric_data.call_args
        assert call_kwargs.kwargs["Namespace"] == CW_NAMESPACE
        metric_data = call_kwargs.kwargs["MetricData"]
        # 5 metrics + 1 aggregate
        assert len(metric_data) == 6
        names = {m["MetricName"] for m in metric_data}
        assert "aggregate_score" in names
        for metric in EVALUATION_METRICS:
            assert metric in names

    def test_dimensions_include_env_and_run_id(self):
        cw = MagicMock()
        report = EvaluationReport(
            run_id="eval-test-003",
            timestamp="2024-01-15T12:00:00Z",
            dataset_size=1,
            metrics={"context_precision": 0.8},
            per_question_scores=[],
            s3_report_key="evaluations/reports/eval-test-003.json",
        )
        _publish_metrics(cw, report, "prod")

        metric_data = cw.put_metric_data.call_args.kwargs["MetricData"]
        for m in metric_data:
            dim_names = {d["Name"] for d in m["Dimensions"]}
            assert "Environment" in dim_names
            assert "RunId" in dim_names


# ---------------------------------------------------------------------------
# Handler — RAGAS path (mocked)
# ---------------------------------------------------------------------------

class TestHandlerRagasPath:
    @patch("lambdas.evaluation.evaluator.handler._RAGAS_AVAILABLE", True)
    @patch("lambdas.evaluation.evaluator.handler._run_ragas_evaluation")
    @patch("lambdas.evaluation.evaluator.handler._get_cloudwatch_client")
    @patch("lambdas.evaluation.evaluator.handler._get_s3_client")
    def test_uses_ragas_when_available(self, mock_s3_factory, mock_cw_factory, mock_ragas_eval):
        s3 = MagicMock()
        s3.get_object.return_value = {
            "Body": MagicMock(read=lambda: json.dumps(SAMPLE_DATASET).encode())
        }
        mock_s3_factory.return_value = s3
        mock_cw_factory.return_value = MagicMock()

        mock_ragas_eval.return_value = {
            "scores": {m: 0.85 for m in EVALUATION_METRICS},
            "per_question": [{m: 0.85 for m in EVALUATION_METRICS} for _ in SAMPLE_DATASET],
        }

        result = handler({"dataset_s3_key": "evaluations/dataset.json"})

        assert result["evaluation_mode"] == "ragas"
        assert result["dataset_size"] == 2
        mock_ragas_eval.assert_called_once()
        for metric in EVALUATION_METRICS:
            assert metric in result["metrics"]


# ---------------------------------------------------------------------------
# Handler — fallback path (end-to-end with moto S3)
# ---------------------------------------------------------------------------

class TestHandlerFallbackPath:
    @patch("lambdas.evaluation.evaluator.handler._RAGAS_AVAILABLE", False)
    @patch("lambdas.evaluation.evaluator.handler._get_cloudwatch_client")
    def test_fallback_end_to_end(self, mock_cw_factory, s3_with_dataset):
        mock_cw_factory.return_value = MagicMock()

        with patch("lambdas.evaluation.evaluator.handler._get_s3_client", return_value=s3_with_dataset):
            result = handler({"dataset_s3_key": "evaluations/dataset.json"})

        assert result["evaluation_mode"] == "fallback"
        assert result["dataset_size"] == 2
        assert "run_id" in result
        assert "s3_report_key" in result
        for metric in EVALUATION_METRICS:
            assert metric in result["metrics"]

        # Verify report was stored in S3
        report_obj = s3_with_dataset.get_object(
            Bucket="rag-documents-test",
            Key=result["s3_report_key"],
        )
        report_body = json.loads(report_obj["Body"].read().decode("utf-8"))
        assert report_body["run_id"] == result["run_id"]
        assert report_body["dataset_size"] == 2

    @patch("lambdas.evaluation.evaluator.handler._RAGAS_AVAILABLE", False)
    @patch("lambdas.evaluation.evaluator.handler._get_cloudwatch_client")
    def test_cloudwatch_metrics_published(self, mock_cw_factory, s3_with_dataset):
        cw = MagicMock()
        mock_cw_factory.return_value = cw

        with patch("lambdas.evaluation.evaluator.handler._get_s3_client", return_value=s3_with_dataset):
            handler({"dataset_s3_key": "evaluations/dataset.json"})

        cw.put_metric_data.assert_called_once()
        call_kwargs = cw.put_metric_data.call_args.kwargs
        assert call_kwargs["Namespace"] == CW_NAMESPACE

    @patch("lambdas.evaluation.evaluator.handler._RAGAS_AVAILABLE", False)
    @patch("lambdas.evaluation.evaluator.handler._get_cloudwatch_client")
    def test_bucket_override(self, mock_cw_factory):
        mock_cw_factory.return_value = MagicMock()

        with mock_aws():
            s3 = boto3.client("s3", region_name="us-east-1")
            s3.create_bucket(Bucket="custom-bucket")
            s3.put_object(
                Bucket="custom-bucket",
                Key="eval/data.json",
                Body=json.dumps(SAMPLE_DATASET),
            )
            with patch("lambdas.evaluation.evaluator.handler._get_s3_client", return_value=s3):
                result = handler({"dataset_s3_key": "eval/data.json", "bucket": "custom-bucket"})

        assert result["dataset_size"] == 2


# ---------------------------------------------------------------------------
# Handler — error cases
# ---------------------------------------------------------------------------

class TestHandlerErrors:
    def test_missing_dataset_key_raises(self):
        with pytest.raises(ValueError, match="dataset_s3_key"):
            handler({})

    def test_empty_dataset_key_raises(self):
        with pytest.raises(ValueError, match="dataset_s3_key"):
            handler({"dataset_s3_key": ""})
