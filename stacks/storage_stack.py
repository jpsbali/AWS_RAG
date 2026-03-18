"""StorageStack: S3 bucket with encryption, versioning, lifecycle, and EventBridge notifications."""
import aws_cdk as cdk
from aws_cdk import (
    aws_events as events,
    aws_events_targets as targets,
    aws_kms as kms,
    aws_s3 as s3,
    aws_stepfunctions as sfn,
)
from constructs import Construct


class StorageStack(cdk.Stack):
    """S3 document bucket with SSE-KMS, versioning, lifecycle policies, and EventBridge integration."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        encryption_key: kms.IKey,
        env_name: str = "dev",
        **kwargs,
    ):
        super().__init__(scope, construct_id, **kwargs)

        self.env_name = env_name

        # --- S3 Bucket ---
        self.document_bucket = s3.Bucket(
            self,
            "DocumentBucket",
            bucket_name=f"rag-documents-{cdk.Aws.ACCOUNT_ID}-{cdk.Aws.REGION}-{env_name}",
            encryption=s3.BucketEncryption.KMS,
            encryption_key=encryption_key,
            versioned=True,
            event_bridge_enabled=True,
            removal_policy=cdk.RemovalPolicy.RETAIN if env_name == "prod" else cdk.RemovalPolicy.DESTROY,
            auto_delete_objects=env_name != "prod",
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            cors=[
                s3.CorsRule(
                    allowed_methods=[s3.HttpMethods.GET, s3.HttpMethods.PUT, s3.HttpMethods.POST],
                    allowed_origins=["*"],  # Tighten to Streamlit domain in prod
                    allowed_headers=["*"],
                    max_age=3600,
                )
            ],
        )

        # --- Lifecycle Rules ---
        # raw/ → Intelligent-Tiering at 30d → Glacier at 90d
        self.document_bucket.add_lifecycle_rule(
            id="RawToIntelligentTiering",
            prefix="raw/",
            transitions=[
                s3.Transition(
                    storage_class=s3.StorageClass.INTELLIGENT_TIERING,
                    transition_after=cdk.Duration.days(30),
                ),
                s3.Transition(
                    storage_class=s3.StorageClass.GLACIER,
                    transition_after=cdk.Duration.days(90),
                ),
            ],
        )

        # processed/ → Intelligent-Tiering at 60d
        self.document_bucket.add_lifecycle_rule(
            id="ProcessedToIntelligentTiering",
            prefix="processed/",
            transitions=[
                s3.Transition(
                    storage_class=s3.StorageClass.INTELLIGENT_TIERING,
                    transition_after=cdk.Duration.days(60),
                ),
            ],
        )

        # failed/ → Expire at 180d
        self.document_bucket.add_lifecycle_rule(
            id="FailedExpiry",
            prefix="failed/",
            expiration=cdk.Duration.days(180),
        )

        # --- EventBridge Rule for raw/ uploads ---
        self.upload_rule = events.Rule(
            self,
            "RawUploadRule",
            rule_name=f"rag-raw-upload-{env_name}",
            description="Trigger ingestion pipeline on document upload to raw/ prefix",
            event_pattern=events.EventPattern(
                source=["aws.s3"],
                detail_type=["Object Created"],
                detail={
                    "bucket": {"name": [self.document_bucket.bucket_name]},
                    "object": {"key": [{"prefix": "raw/"}]},
                },
            ),
        )

        # --- Outputs ---
        cdk.CfnOutput(self, "DocumentBucketName", value=self.document_bucket.bucket_name)
        cdk.CfnOutput(self, "DocumentBucketArn", value=self.document_bucket.bucket_arn)

    def add_event_target(self, state_machine: sfn.IStateMachine) -> None:
        """Wire the EventBridge rule to a Step Functions state machine target."""
        self.upload_rule.add_target(targets.SfnStateMachine(state_machine))
