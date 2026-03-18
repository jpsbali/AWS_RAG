"""ProcessingStack: Step Functions state machine with 6 ingestion Lambdas and EventBridge trigger."""
import aws_cdk as cdk
from aws_cdk import (
    aws_ec2 as ec2,
    aws_events as events,
    aws_events_targets as targets,
    aws_iam as iam,
    aws_kms as kms,
    aws_lambda as _lambda,
    aws_lambda_destinations as destinations,
    aws_opensearchservice as opensearch,
    aws_s3 as s3,
    aws_sns as sns,
    aws_sqs as sqs,
    aws_stepfunctions as sfn,
    aws_stepfunctions_tasks as sfn_tasks,
)
from constructs import Construct


class ProcessingStack(cdk.Stack):
    """Step Functions orchestrator with 6 ingestion Lambda functions and per-step retry/catch."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        vpc: ec2.IVpc,
        lambda_sg: ec2.ISecurityGroup,
        document_bucket_name: str,
        document_bucket_arn: str,
        encryption_key: kms.IKey,
        opensearch_domain: opensearch.IDomain,
        opensearch_endpoint: str,
        cache_cluster: object,
        alert_topic: sns.ITopic,
        env_name: str = "dev",
        **kwargs,
    ):
        super().__init__(scope, construct_id, **kwargs)

        self.env_name = env_name

        # --- SQS Dead Letter Queues ---

        # DLQ for EventBridge rule delivery failures
        self.eventbridge_dlq = sqs.Queue(
            self,
            "EventBridgeDLQ",
            queue_name=f"rag-eventbridge-dlq-{env_name}",
            retention_period=cdk.Duration.days(14),
            encryption=sqs.QueueEncryption.KMS,
            encryption_master_key=encryption_key,
        )

        # DLQ for Lambda async invocation failures (max 2 retries)
        self.lambda_async_dlq = sqs.Queue(
            self,
            "LambdaAsyncDLQ",
            queue_name=f"rag-lambda-async-dlq-{env_name}",
            retention_period=cdk.Duration.days(14),
            encryption=sqs.QueueEncryption.KMS,
            encryption_master_key=encryption_key,
        )

        # DLQ for Step Functions task failures
        self.stepfunctions_dlq = sqs.Queue(
            self,
            "StepFunctionsDLQ",
            queue_name=f"rag-sfn-dlq-{env_name}",
            retention_period=cdk.Duration.days(14),
            encryption=sqs.QueueEncryption.KMS,
            encryption_master_key=encryption_key,
        )

        # Import the document bucket by name to avoid cross-stack cyclic refs
        document_bucket = s3.Bucket.from_bucket_attributes(
            self,
            "ImportedDocumentBucket",
            bucket_name=document_bucket_name,
            bucket_arn=document_bucket_arn,
        )

        # --- Common Lambda configuration ---
        private_subnets = ec2.SubnetSelection(
            subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
        )
        lambda_code = _lambda.Code.from_asset("lambdas")
        runtime = _lambda.Runtime.PYTHON_3_11

        common_env = {
            "DOCUMENT_BUCKET": document_bucket_name,
            "OPENSEARCH_ENDPOINT": opensearch_endpoint,
            "REDIS_HOST": getattr(cache_cluster, "redis_endpoint", ""),
            "REDIS_PORT": getattr(cache_cluster, "redis_port", "6379"),
            "ENV_NAME": env_name,
        }

        # --- Lambda Functions ---

        # 1. Validator Lambda (512 MB, 30s)
        self.validator_fn = _lambda.Function(
            self, "ValidatorFn",
            function_name=f"rag-validator-{env_name}",
            runtime=runtime,
            handler="ingestion.validator.handler.handler",
            code=lambda_code,
            memory_size=512,
            timeout=cdk.Duration.seconds(30),
            vpc=vpc,
            vpc_subnets=private_subnets,
            security_groups=[lambda_sg],
            environment=common_env,
            tracing=_lambda.Tracing.ACTIVE,
        )

        # 2. Text Extractor Lambda (1024 MB, 300s)
        self.text_extractor_fn = _lambda.Function(
            self, "TextExtractorFn",
            function_name=f"rag-text-extractor-{env_name}",
            runtime=runtime,
            handler="ingestion.text_extractor.handler.handler",
            code=lambda_code,
            memory_size=1024,
            timeout=cdk.Duration.seconds(300),
            vpc=vpc,
            vpc_subnets=private_subnets,
            security_groups=[lambda_sg],
            environment=common_env,
            tracing=_lambda.Tracing.ACTIVE,
        )

        # 3. Metadata Enricher Lambda (1024 MB, 120s)
        self.metadata_enricher_fn = _lambda.Function(
            self, "MetadataEnricherFn",
            function_name=f"rag-metadata-enricher-{env_name}",
            runtime=runtime,
            handler="ingestion.metadata_enricher.handler.handler",
            code=lambda_code,
            memory_size=1024,
            timeout=cdk.Duration.seconds(120),
            vpc=vpc,
            vpc_subnets=private_subnets,
            security_groups=[lambda_sg],
            environment=common_env,
            tracing=_lambda.Tracing.ACTIVE,
        )

        # 4. Chunker Lambda (512 MB, 60s)
        self.chunker_fn = _lambda.Function(
            self, "ChunkerFn",
            function_name=f"rag-chunker-{env_name}",
            runtime=runtime,
            handler="ingestion.chunker.handler.handler",
            code=lambda_code,
            memory_size=512,
            timeout=cdk.Duration.seconds(60),
            vpc=vpc,
            vpc_subnets=private_subnets,
            security_groups=[lambda_sg],
            environment=common_env,
            tracing=_lambda.Tracing.ACTIVE,
        )

        # 5. Embedding Generator Lambda (1024 MB, 300s)
        self.embedding_generator_fn = _lambda.Function(
            self, "EmbeddingGeneratorFn",
            function_name=f"rag-embedding-generator-{env_name}",
            runtime=runtime,
            handler="ingestion.embedding_generator.handler.handler",
            code=lambda_code,
            memory_size=1024,
            timeout=cdk.Duration.seconds(300),
            vpc=vpc,
            vpc_subnets=private_subnets,
            security_groups=[lambda_sg],
            environment=common_env,
            tracing=_lambda.Tracing.ACTIVE,
        )

        # 6. Vector Indexer Lambda (512 MB, 120s)
        self.vector_indexer_fn = _lambda.Function(
            self, "VectorIndexerFn",
            function_name=f"rag-vector-indexer-{env_name}",
            runtime=runtime,
            handler="ingestion.vector_indexer.handler.handler",
            code=lambda_code,
            memory_size=512,
            timeout=cdk.Duration.seconds(120),
            vpc=vpc,
            vpc_subnets=private_subnets,
            security_groups=[lambda_sg],
            environment=common_env,
            tracing=_lambda.Tracing.ACTIVE,
        )

        # --- Lambda Async Invocation DLQ Configuration (max 2 retries) ---
        for fn in [
            self.validator_fn,
            self.text_extractor_fn,
            self.metadata_enricher_fn,
            self.chunker_fn,
            self.embedding_generator_fn,
            self.vector_indexer_fn,
        ]:
            fn.configure_async_invoke(
                retry_attempts=2,
                on_failure=destinations.SqsDestination(self.lambda_async_dlq),
            )

        # --- IAM Permissions ---
        all_lambdas = [
            self.validator_fn,
            self.text_extractor_fn,
            self.metadata_enricher_fn,
            self.chunker_fn,
            self.embedding_generator_fn,
            self.vector_indexer_fn,
        ]

        # S3 + KMS permissions for all Lambdas (using imported bucket to avoid cyclic refs)
        for fn in all_lambdas:
            document_bucket.grant_read_write(fn)
            encryption_key.grant_encrypt_decrypt(fn)

        # Textract permissions for text extractor
        self.text_extractor_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "textract:DetectDocumentText",
                    "textract:AnalyzeDocument",
                    "textract:StartDocumentAnalysis",
                    "textract:GetDocumentAnalysis",
                ],
                resources=["*"],
            )
        )

        # Comprehend permissions for metadata enricher
        self.metadata_enricher_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "comprehend:DetectEntities",
                    "comprehend:DetectKeyPhrases",
                    "comprehend:DetectDominantLanguage",
                    "comprehend:DetectPiiEntities",
                ],
                resources=["*"],
            )
        )

        # Bedrock permissions for embedding generator
        self.embedding_generator_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=["bedrock:InvokeModel"],
                resources=["*"],
            )
        )

        # OpenSearch permissions for vector indexer
        opensearch_domain.grant_write(self.vector_indexer_fn)
        opensearch_domain.grant_read(self.vector_indexer_fn)

        # --- Failure Handler (move to failed/ + publish SNS + send to DLQ) ---
        fail_move_to_failed = sfn.Pass(
            self, "MoveToFailed",
            comment="Move document to failed/ prefix",
            parameters={
                "document_id.$": "$.document_id",
                "s3_bucket.$": "$.s3_bucket",
                "s3_key.$": "$.s3_key",
                "error.$": "$.error",
                "cause.$": "$.cause",
            },
        )

        fail_publish_sns = sfn_tasks.SnsPublish(
            self, "PublishFailureNotification",
            topic=alert_topic,
            message=sfn.TaskInput.from_json_path_at("$"),
            subject=f"RAG Ingestion Failure ({env_name})",
            result_path=sfn.JsonPath.DISCARD,
        )

        fail_send_to_dlq = sfn_tasks.SqsSendMessage(
            self, "SendToDLQ",
            queue=self.stepfunctions_dlq,
            message_body=sfn.TaskInput.from_json_path_at("$"),
            result_path=sfn.JsonPath.DISCARD,
        )

        fail_state = sfn.Fail(
            self, "FailedState",
            cause="Document processing failed after retries",
            error="ProcessingFailed",
        )

        failure_chain = fail_move_to_failed.next(fail_publish_sns).next(fail_send_to_dlq).next(fail_state)

        # --- Step Functions Task Definitions with Retry + Catch ---

        # 1. Validate step: 3 retries, backoff 2s → 4s → 8s
        validate_task = sfn_tasks.LambdaInvoke(
            self, "ValidateDocument",
            lambda_function=self.validator_fn,
            payload=sfn.TaskInput.from_json_path_at("$"),
            result_selector={
                "document_id.$": "$.Payload.document_id",
                "s3_bucket.$": "$.Payload.s3_bucket",
                "s3_key.$": "$.Payload.s3_key",
                "file_type.$": "$.Payload.file_type",
                "content_type.$": "$.Payload.content_type",
                "file_size.$": "$.Payload.file_size",
                "valid.$": "$.Payload.valid",
                "correlation_id.$": "$.Payload.correlation_id",
            },
            result_path="$",
        )
        validate_task.add_retry(
            errors=["States.ALL"],
            interval=cdk.Duration.seconds(2),
            max_attempts=3,
            backoff_rate=2.0,
            max_delay=cdk.Duration.seconds(8),
        )
        validate_task.add_catch(failure_chain, errors=["States.ALL"], result_path="$.error_info")

        # 2a. Extract via Textract: 3 retries, backoff 5s → 10s → 20s
        extract_textract_task = sfn_tasks.LambdaInvoke(
            self, "ExtractViaTextract",
            lambda_function=self.text_extractor_fn,
            payload=sfn.TaskInput.from_json_path_at("$"),
            result_selector={
                "document_id.$": "$.Payload.document_id",
                "s3_bucket.$": "$.Payload.s3_bucket",
                "s3_key.$": "$.Payload.s3_key",
                "file_type.$": "$.Payload.file_type",
                "content_type.$": "$.Payload.content_type",
                "text.$": "$.Payload.text",
                "pages.$": "$.Payload.pages",
                "extraction_method.$": "$.Payload.extraction_method",
                "confidence.$": "$.Payload.confidence",
                "valid.$": "$.Payload.valid",
                "output_key.$": "$.Payload.output_key",
                "correlation_id.$": "$.Payload.correlation_id",
            },
            result_path="$",
        )
        extract_textract_task.add_retry(
            errors=["States.ALL"],
            interval=cdk.Duration.seconds(5),
            max_attempts=3,
            backoff_rate=2.0,
            max_delay=cdk.Duration.seconds(20),
        )
        extract_textract_task.add_catch(failure_chain, errors=["States.ALL"], result_path="$.error_info")

        # 2b. Extract via Native: same retry config as Textract
        extract_native_task = sfn_tasks.LambdaInvoke(
            self, "ExtractViaNative",
            lambda_function=self.text_extractor_fn,
            payload=sfn.TaskInput.from_json_path_at("$"),
            result_selector={
                "document_id.$": "$.Payload.document_id",
                "s3_bucket.$": "$.Payload.s3_bucket",
                "s3_key.$": "$.Payload.s3_key",
                "file_type.$": "$.Payload.file_type",
                "content_type.$": "$.Payload.content_type",
                "text.$": "$.Payload.text",
                "pages.$": "$.Payload.pages",
                "extraction_method.$": "$.Payload.extraction_method",
                "confidence.$": "$.Payload.confidence",
                "valid.$": "$.Payload.valid",
                "output_key.$": "$.Payload.output_key",
                "correlation_id.$": "$.Payload.correlation_id",
            },
            result_path="$",
        )
        extract_native_task.add_retry(
            errors=["States.ALL"],
            interval=cdk.Duration.seconds(5),
            max_attempts=3,
            backoff_rate=2.0,
            max_delay=cdk.Duration.seconds(20),
        )
        extract_native_task.add_catch(failure_chain, errors=["States.ALL"], result_path="$.error_info")

        # 3. Choice state: route by document type (textract vs native)
        choose_extractor = sfn.Choice(
            self, "ChooseExtractor",
            comment="Route to Textract or Native extraction based on content_type",
        )
        choose_extractor.when(
            sfn.Condition.string_equals("$.content_type", "textract"),
            extract_textract_task,
        )
        choose_extractor.otherwise(extract_native_task)

        # 4. Quality Check: verify extraction was valid
        quality_check = sfn.Choice(
            self, "QualityCheck",
            comment="Check if extraction quality passed threshold",
        )

        quality_fail = sfn.Pass(
            self, "QualityCheckFailed",
            parameters={
                "document_id.$": "$.document_id",
                "s3_bucket.$": "$.s3_bucket",
                "s3_key.$": "$.s3_key",
                "error": "Extraction quality below threshold",
                "cause": "Text extraction did not meet minimum quality requirements",
            },
        )
        quality_fail.next(failure_chain)

        # 5. Extract Metadata: 3 retries, backoff 2s → 4s → 8s
        extract_metadata_task = sfn_tasks.LambdaInvoke(
            self, "ExtractMetadata",
            lambda_function=self.metadata_enricher_fn,
            payload=sfn.TaskInput.from_json_path_at("$"),
            result_selector={
                "document_id.$": "$.Payload.document_id",
                "s3_bucket.$": "$.Payload.s3_bucket",
                "s3_key.$": "$.s3_key",
                "file_type.$": "$.file_type",
                "content_type.$": "$.content_type",
                "text.$": "$.text",
                "output_key.$": "$.Payload.output_key",
                "entities.$": "$.Payload.entities",
                "key_phrases.$": "$.Payload.key_phrases",
                "language.$": "$.Payload.language",
                "correlation_id.$": "$.Payload.correlation_id",
            },
            result_path="$",
        )
        extract_metadata_task.add_retry(
            errors=["States.ALL"],
            interval=cdk.Duration.seconds(2),
            max_attempts=3,
            backoff_rate=2.0,
            max_delay=cdk.Duration.seconds(8),
        )
        extract_metadata_task.add_catch(failure_chain, errors=["States.ALL"], result_path="$.error_info")

        # 6. Chunk Text: 2 retries, backoff 2s → 4s
        chunk_task = sfn_tasks.LambdaInvoke(
            self, "ChunkText",
            lambda_function=self.chunker_fn,
            payload=sfn.TaskInput.from_json_path_at("$"),
            result_selector={
                "document_id.$": "$.Payload.document_id",
                "s3_bucket.$": "$.Payload.s3_bucket",
                "chunk_count.$": "$.Payload.chunk_count",
                "output_key.$": "$.Payload.output_key",
                "correlation_id.$": "$.Payload.correlation_id",
            },
            result_path="$",
        )
        chunk_task.add_retry(
            errors=["States.ALL"],
            interval=cdk.Duration.seconds(2),
            max_attempts=2,
            backoff_rate=2.0,
            max_delay=cdk.Duration.seconds(4),
        )
        chunk_task.add_catch(failure_chain, errors=["States.ALL"], result_path="$.error_info")

        # 7. Generate Embeddings: 5 retries, backoff 2s → 4s → 8s → 16s → 32s
        embed_task = sfn_tasks.LambdaInvoke(
            self, "GenerateEmbeddings",
            lambda_function=self.embedding_generator_fn,
            payload=sfn.TaskInput.from_object({
                "document_id.$": "$.document_id",
                "s3_bucket.$": "$.s3_bucket",
                "chunks_key.$": "$.output_key",
                "correlation_id.$": "$.correlation_id",
            }),
            result_selector={
                "document_id.$": "$.Payload.document_id",
                "s3_bucket.$": "$.Payload.s3_bucket",
                "embedding_count.$": "$.Payload.embedding_count",
                "output_key.$": "$.Payload.output_key",
                "correlation_id.$": "$.Payload.correlation_id",
            },
            result_path="$",
        )
        embed_task.add_retry(
            errors=["States.ALL"],
            interval=cdk.Duration.seconds(2),
            max_attempts=5,
            backoff_rate=2.0,
            max_delay=cdk.Duration.seconds(32),
        )
        embed_task.add_catch(failure_chain, errors=["States.ALL"], result_path="$.error_info")

        # 8. Index Vectors: 3 retries, backoff 2s → 4s → 8s
        index_task = sfn_tasks.LambdaInvoke(
            self, "IndexVectors",
            lambda_function=self.vector_indexer_fn,
            payload=sfn.TaskInput.from_object({
                "document_id.$": "$.document_id",
                "s3_bucket.$": "$.s3_bucket",
                "embeddings_key.$": "$.output_key",
                "correlation_id.$": "$.correlation_id",
            }),
            result_selector={
                "document_id.$": "$.Payload.document_id",
                "indexed_count.$": "$.Payload.indexed_count",
                "error_count.$": "$.Payload.error_count",
                "correlation_id.$": "$.Payload.correlation_id",
            },
            result_path="$",
        )
        index_task.add_retry(
            errors=["States.ALL"],
            interval=cdk.Duration.seconds(2),
            max_attempts=3,
            backoff_rate=2.0,
            max_delay=cdk.Duration.seconds(8),
        )
        index_task.add_catch(failure_chain, errors=["States.ALL"], result_path="$.error_info")

        # 9. Success state
        success_state = sfn.Succeed(
            self, "ProcessingSuccess",
            comment="Document processing completed successfully",
        )

        # --- Wire the state machine chain ---
        # Validate → Choice → Extract (Textract or Native) → Quality Check →
        # Extract Metadata → Chunk → Generate Embeddings → Index → Success

        extract_textract_task.next(quality_check)
        extract_native_task.next(quality_check)

        quality_check.when(
            sfn.Condition.boolean_equals("$.valid", True),
            extract_metadata_task,
        )
        quality_check.otherwise(quality_fail)

        extract_metadata_task.next(chunk_task)
        chunk_task.next(embed_task)
        embed_task.next(index_task)
        index_task.next(success_state)

        definition = validate_task.next(choose_extractor)

        # --- State Machine ---
        self.state_machine = sfn.StateMachine(
            self, "IngestionStateMachine",
            state_machine_name=f"rag-ingestion-pipeline-{env_name}",
            definition_body=sfn.DefinitionBody.from_chainable(definition),
            timeout=cdk.Duration.minutes(30),
            tracing_enabled=True,
        )

        # Grant state machine permission to invoke all Lambdas
        for fn in all_lambdas:
            fn.grant_invoke(self.state_machine)

        # Grant state machine permission to publish to SNS
        alert_topic.grant_publish(self.state_machine)

        # Grant state machine permission to send to Step Functions DLQ
        self.stepfunctions_dlq.grant_send_messages(self.state_machine)

        # --- EventBridge Rule (created here to avoid cross-stack cyclic refs) ---
        self.upload_rule = events.Rule(
            self, "RawUploadRule",
            rule_name=f"rag-ingestion-trigger-{env_name}",
            description="Trigger ingestion state machine on document upload to raw/ prefix",
            event_pattern=events.EventPattern(
                source=["aws.s3"],
                detail_type=["Object Created"],
                detail={
                    "bucket": {"name": [document_bucket_name]},
                    "object": {"key": [{"prefix": "raw/"}]},
                },
            ),
        )
        self.upload_rule.add_target(
            targets.SfnStateMachine(
                self.state_machine,
                dead_letter_queue=self.eventbridge_dlq,
                retry_attempts=2,
            )
        )

        # --- Outputs ---
        cdk.CfnOutput(
            self, "StateMachineArn",
            value=self.state_machine.state_machine_arn,
            description="Ingestion pipeline Step Functions state machine ARN",
        )
        cdk.CfnOutput(
            self, "StateMachineName",
            value=self.state_machine.state_machine_name,
            description="Ingestion pipeline state machine name",
        )
        cdk.CfnOutput(
            self, "EventBridgeDLQUrl",
            value=self.eventbridge_dlq.queue_url,
            description="SQS DLQ URL for EventBridge delivery failures",
        )
        cdk.CfnOutput(
            self, "LambdaAsyncDLQUrl",
            value=self.lambda_async_dlq.queue_url,
            description="SQS DLQ URL for Lambda async invocation failures",
        )
        cdk.CfnOutput(
            self, "StepFunctionsDLQUrl",
            value=self.stepfunctions_dlq.queue_url,
            description="SQS DLQ URL for Step Functions task failures",
        )
