"""MonitoringStack: SNS alerts, CloudWatch dashboard, alarms, and X-Ray tracing configuration."""
import aws_cdk as cdk
from aws_cdk import (
    aws_cloudwatch as cloudwatch,
    aws_cloudwatch_actions as cw_actions,
    aws_opensearchservice as opensearch,
    aws_sns as sns,
)
from constructs import Construct


class MonitoringStack(cdk.Stack):
    """CloudWatch dashboard, alarms, SNS alerts, and X-Ray tracing for the RAG service."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        env_name: str = "dev",
        opensearch_domain: opensearch.IDomain | None = None,
        **kwargs,
    ):
        super().__init__(scope, construct_id, **kwargs)

        self.env_name = env_name

        # --- SNS Topic for Alerts ---
        self.alert_topic = sns.Topic(
            self,
            "RagAlertTopic",
            topic_name=f"rag-alerts-{env_name}",
            display_name=f"RAG Service Alerts ({env_name})",
        )

        # --- Custom Metrics References (namespace: RAG/Application) ---
        namespace = "RAG/Application"

        query_latency_metric = cloudwatch.Metric(
            namespace=namespace,
            metric_name="QueryLatency",
            statistic="p95",
            period=cdk.Duration.minutes(5),
            dimensions_map={"Environment": env_name},
        )

        cache_hit_rate_metric = cloudwatch.Metric(
            namespace=namespace,
            metric_name="CacheHitRate",
            statistic="Average",
            period=cdk.Duration.minutes(5),
            dimensions_map={"Environment": env_name},
        )

        error_rate_metric = cloudwatch.Metric(
            namespace=namespace,
            metric_name="ErrorRate",
            statistic="Average",
            period=cdk.Duration.minutes(5),
            dimensions_map={"Environment": env_name},
        )

        embedding_time_metric = cloudwatch.Metric(
            namespace=namespace,
            metric_name="EmbeddingGenerationTime",
            statistic="Average",
            period=cdk.Duration.minutes(5),
            dimensions_map={"Environment": env_name},
        )

        processing_time_metric = cloudwatch.Metric(
            namespace=namespace,
            metric_name="DocumentProcessingTime",
            statistic="Average",
            period=cdk.Duration.minutes(5),
            dimensions_map={"Environment": env_name},
        )

        llm_response_time_metric = cloudwatch.Metric(
            namespace=namespace,
            metric_name="LLMResponseTime",
            statistic="Average",
            period=cdk.Duration.minutes(5),
            dimensions_map={"Environment": env_name},
        )

        # --- CloudWatch Dashboard ---
        dashboard = cloudwatch.Dashboard(
            self,
            "RagDashboard",
            dashboard_name=f"rag-dashboard-{env_name}",
        )

        # Query Latency widget
        dashboard.add_widgets(
            cloudwatch.GraphWidget(
                title="Query Latency (ms)",
                left=[
                    query_latency_metric,
                    cloudwatch.Metric(
                        namespace=namespace,
                        metric_name="QueryLatency",
                        statistic="p50",
                        period=cdk.Duration.minutes(5),
                        dimensions_map={"Environment": env_name},
                    ),
                    cloudwatch.Metric(
                        namespace=namespace,
                        metric_name="QueryLatency",
                        statistic="p99",
                        period=cdk.Duration.minutes(5),
                        dimensions_map={"Environment": env_name},
                    ),
                ],
                width=12,
            ),
            # Cache Hit Rate widget
            cloudwatch.GraphWidget(
                title="Cache Hit Rate (%)",
                left=[cache_hit_rate_metric],
                width=12,
            ),
        )

        # Error Rates and Processing Times
        dashboard.add_widgets(
            cloudwatch.GraphWidget(
                title="Error Rates (%)",
                left=[error_rate_metric],
                width=12,
            ),
            cloudwatch.GraphWidget(
                title="Processing Times (ms)",
                left=[
                    embedding_time_metric,
                    processing_time_metric,
                    llm_response_time_metric,
                ],
                width=12,
            ),
        )

        # --- CloudWatch Alarms ---
        sns_action = cw_actions.SnsAction(self.alert_topic)

        # HighQueryLatency: p95 > 10s for 5 minutes
        high_latency_alarm = cloudwatch.Alarm(
            self,
            "HighQueryLatency",
            alarm_name=f"rag-high-query-latency-{env_name}",
            alarm_description="Query latency p95 exceeds 10 seconds for 5 minutes",
            metric=query_latency_metric,
            threshold=10_000,  # 10s in milliseconds
            evaluation_periods=1,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
        )
        high_latency_alarm.add_alarm_action(sns_action)
        high_latency_alarm.add_ok_action(sns_action)

        # HighErrorRate: > 5% for 5 minutes
        high_error_alarm = cloudwatch.Alarm(
            self,
            "HighErrorRate",
            alarm_name=f"rag-high-error-rate-{env_name}",
            alarm_description="Error rate exceeds 5% for 5 minutes",
            metric=error_rate_metric,
            threshold=5,  # 5%
            evaluation_periods=1,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
        )
        high_error_alarm.add_alarm_action(sns_action)
        high_error_alarm.add_ok_action(sns_action)

        # StepFunctionFailures: > 0 for 5 minutes
        sfn_failures_alarm = cloudwatch.Alarm(
            self,
            "StepFunctionFailures",
            alarm_name=f"rag-sfn-failures-{env_name}",
            alarm_description="Step Functions execution failures detected",
            metric=cloudwatch.Metric(
                namespace="AWS/States",
                metric_name="ExecutionsFailed",
                statistic="Sum",
                period=cdk.Duration.minutes(5),
            ),
            threshold=0,
            evaluation_periods=1,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
        )
        sfn_failures_alarm.add_alarm_action(sns_action)
        sfn_failures_alarm.add_ok_action(sns_action)

        # OpenSearchClusterRed: >= 1 for 1 minute
        if opensearch_domain is not None:
            os_red_alarm = cloudwatch.Alarm(
                self,
                "OpenSearchClusterRed",
                alarm_name=f"rag-opensearch-red-{env_name}",
                alarm_description="OpenSearch cluster status is RED",
                metric=cloudwatch.Metric(
                    namespace="AWS/ES",
                    metric_name="ClusterStatus.red",
                    statistic="Maximum",
                    period=cdk.Duration.minutes(1),
                    dimensions_map={
                        "DomainName": opensearch_domain.domain_name,
                        "ClientId": cdk.Aws.ACCOUNT_ID,
                    },
                ),
                threshold=1,
                evaluation_periods=1,
                comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
                treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
            )
            os_red_alarm.add_alarm_action(sns_action)
            os_red_alarm.add_ok_action(sns_action)

        # CacheMemoryHigh: > 90%
        cache_memory_alarm = cloudwatch.Alarm(
            self,
            "CacheMemoryHigh",
            alarm_name=f"rag-cache-memory-high-{env_name}",
            alarm_description="ElastiCache memory usage exceeds 90%",
            metric=cloudwatch.Metric(
                namespace="AWS/ElastiCache",
                metric_name="DatabaseMemoryUsagePercentage",
                statistic="Maximum",
                period=cdk.Duration.minutes(5),
            ),
            threshold=90,
            evaluation_periods=1,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
        )
        cache_memory_alarm.add_alarm_action(sns_action)
        cache_memory_alarm.add_ok_action(sns_action)

        # --- X-Ray Tracing Configuration ---
        # X-Ray sampling rule: 5% in production, 100% in dev
        sampling_rate = 0.05 if env_name == "prod" else 1.0

        self.xray_sampling_rule = cdk.CfnResource(
            self,
            "XRaySamplingRule",
            type="AWS::XRay::SamplingRule",
            properties={
                "SamplingRule": {
                    "RuleName": f"rag-sampling-{env_name}",
                    "ResourceARN": "*",
                    "Priority": 1000,
                    "FixedRate": sampling_rate,
                    "ReservoirSize": 1,
                    "ServiceName": f"rag-service-{env_name}",
                    "ServiceType": "*",
                    "Host": "*",
                    "HTTPMethod": "*",
                    "URLPath": "*",
                    "Version": 1,
                },
            },
        )

        # --- Outputs ---
        cdk.CfnOutput(
            self,
            "AlertTopicArn",
            value=self.alert_topic.topic_arn,
            description="SNS topic ARN for RAG service alerts",
        )
        cdk.CfnOutput(
            self,
            "DashboardName",
            value=f"rag-dashboard-{env_name}",
            description="CloudWatch dashboard name",
        )
        cdk.CfnOutput(
            self,
            "XRaySamplingRate",
            value=str(sampling_rate),
            description="X-Ray trace sampling rate",
        )
