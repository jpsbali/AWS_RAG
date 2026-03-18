"""ApiStack: REST API Gateway, WebSocket API, Cognito auth, WAF, rate limiting, CORS, evaluation triggers."""
import aws_cdk as cdk
from aws_cdk import (
    aws_apigateway as apigw,
    aws_apigatewayv2 as apigwv2,
    aws_cognito as cognito,
    aws_ec2 as ec2,
    aws_events as events,
    aws_events_targets as targets,
    aws_iam as iam,
    aws_kms as kms,
    aws_lambda as _lambda,
    aws_logs as logs,
    aws_opensearchservice as opensearch,
    aws_s3 as s3,
    aws_sns as sns,
    aws_wafv2 as wafv2,
)
from constructs import Construct


class ApiStack(cdk.Stack):
    """REST and WebSocket API Gateway with Cognito auth, WAF, rate limiting, and CORS."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        vpc: ec2.IVpc,
        lambda_sg: ec2.ISecurityGroup,
        document_bucket: s3.IBucket,
        encryption_key: kms.IKey,
        user_pool: cognito.IUserPool,
        web_acl_arn: str,
        opensearch_domain: opensearch.IDomain,
        opensearch_endpoint: str,
        cache_cluster: object,
        alert_topic: sns.ITopic,
        env_name: str = "dev",
        **kwargs,
    ):
        super().__init__(scope, construct_id, **kwargs)

        self.env_name = env_name

        # --- Common Lambda configuration ---
        private_subnets = ec2.SubnetSelection(
            subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
        )
        lambda_code = _lambda.Code.from_asset("lambdas")
        runtime = _lambda.Runtime.PYTHON_3_11

        common_env = {
            "DOCUMENT_BUCKET": document_bucket.bucket_name,
            "OPENSEARCH_ENDPOINT": opensearch_endpoint,
            "REDIS_HOST": getattr(cache_cluster, "redis_endpoint", ""),
            "REDIS_PORT": getattr(cache_cluster, "redis_port", "6379"),
            "ENV_NAME": env_name,
        }

        # ---------------------------------------------------------------
        # Lambda Functions
        # ---------------------------------------------------------------

        # Query Handler (30s timeout, 1024 MB)
        self.query_handler_fn = _lambda.Function(
            self, "QueryHandlerFn",
            function_name=f"rag-query-handler-{env_name}",
            runtime=runtime,
            handler="query.query_handler.handler.handler",
            code=lambda_code,
            memory_size=1024,
            timeout=cdk.Duration.seconds(30),
            vpc=vpc,
            vpc_subnets=private_subnets,
            security_groups=[lambda_sg],
            environment=common_env,
            tracing=_lambda.Tracing.ACTIVE,
        )

        # Ingest Trigger (10s timeout, 512 MB)
        self.ingest_trigger_fn = _lambda.Function(
            self, "IngestTriggerFn",
            function_name=f"rag-ingest-trigger-{env_name}",
            runtime=runtime,
            handler="query.ingest_trigger.handler.handler",
            code=lambda_code,
            memory_size=512,
            timeout=cdk.Duration.seconds(10),
            vpc=vpc,
            vpc_subnets=private_subnets,
            security_groups=[lambda_sg],
            environment=common_env,
            tracing=_lambda.Tracing.ACTIVE,
        )

        # Document Handler (10s timeout, 512 MB)
        self.document_handler_fn = _lambda.Function(
            self, "DocumentHandlerFn",
            function_name=f"rag-document-handler-{env_name}",
            runtime=runtime,
            handler="query.document_handler.handler.handler",
            code=lambda_code,
            memory_size=512,
            timeout=cdk.Duration.seconds(10),
            vpc=vpc,
            vpc_subnets=private_subnets,
            security_groups=[lambda_sg],
            environment=common_env,
            tracing=_lambda.Tracing.ACTIVE,
        )

        # Health Handler (5s timeout, 256 MB)
        self.health_handler_fn = _lambda.Function(
            self, "HealthHandlerFn",
            function_name=f"rag-health-handler-{env_name}",
            runtime=runtime,
            handler="query.health_handler.handler.handler",
            code=lambda_code,
            memory_size=256,
            timeout=cdk.Duration.seconds(5),
            vpc=vpc,
            vpc_subnets=private_subnets,
            security_groups=[lambda_sg],
            environment=common_env,
            tracing=_lambda.Tracing.ACTIVE,
        )

        # Metrics Handler (10s timeout, 512 MB)
        self.metrics_handler_fn = _lambda.Function(
            self, "MetricsHandlerFn",
            function_name=f"rag-metrics-handler-{env_name}",
            runtime=runtime,
            handler="query.metrics_handler.handler.handler",
            code=lambda_code,
            memory_size=512,
            timeout=cdk.Duration.seconds(10),
            vpc=vpc,
            vpc_subnets=private_subnets,
            security_groups=[lambda_sg],
            environment=common_env,
            tracing=_lambda.Tracing.ACTIVE,
        )

        # Evaluation Lambda (300s timeout, 1024 MB)
        self.evaluation_fn = _lambda.Function(
            self, "EvaluationFn",
            function_name=f"rag-evaluator-{env_name}",
            runtime=runtime,
            handler="evaluation.evaluator.handler.handler",
            code=lambda_code,
            memory_size=1024,
            timeout=cdk.Duration.seconds(300),
            vpc=vpc,
            vpc_subnets=private_subnets,
            security_groups=[lambda_sg],
            environment={
                "DOCUMENT_BUCKET": document_bucket.bucket_name,
                "ENV_NAME": env_name,
            },
            tracing=_lambda.Tracing.ACTIVE,
        )

        # ---------------------------------------------------------------
        # IAM Permissions
        # ---------------------------------------------------------------
        all_api_lambdas = [
            self.query_handler_fn,
            self.ingest_trigger_fn,
            self.document_handler_fn,
            self.health_handler_fn,
            self.metrics_handler_fn,
        ]

        for fn in all_api_lambdas:
            document_bucket.grant_read_write(fn)
            encryption_key.grant_encrypt_decrypt(fn)

        # Query handler needs Bedrock + OpenSearch
        self.query_handler_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
                resources=["*"],
            )
        )
        opensearch_domain.grant_read(self.query_handler_fn)

        # Metrics handler needs CloudWatch read
        self.metrics_handler_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=["cloudwatch:GetMetricData", "cloudwatch:ListMetrics"],
                resources=["*"],
            )
        )

        # Evaluation Lambda needs S3 read/write, CloudWatch PutMetricData, Bedrock InvokeModel
        document_bucket.grant_read_write(self.evaluation_fn)
        encryption_key.grant_encrypt_decrypt(self.evaluation_fn)
        self.evaluation_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=["cloudwatch:PutMetricData"],
                resources=["*"],
            )
        )
        self.evaluation_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=["bedrock:InvokeModel"],
                resources=["*"],
            )
        )

        # ---------------------------------------------------------------
        # REST API Gateway
        # ---------------------------------------------------------------

        # Access log group
        access_log_group = logs.LogGroup(
            self, "ApiAccessLogs",
            log_group_name=f"/aws/apigateway/rag-api-{env_name}",
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=cdk.RemovalPolicy.DESTROY,
        )

        self.rest_api = apigw.RestApi(
            self, "RagRestApi",
            rest_api_name=f"rag-api-{env_name}",
            description=f"RAG Service REST API ({env_name})",
            deploy_options=apigw.StageOptions(
                stage_name=env_name,
                tracing_enabled=True,
                logging_level=apigw.MethodLoggingLevel.INFO,
                access_log_destination=apigw.LogGroupLogDestination(access_log_group),
                access_log_format=apigw.AccessLogFormat.json_with_standard_fields(
                    caller=True,
                    http_method=True,
                    ip=True,
                    protocol=True,
                    request_time=True,
                    resource_path=True,
                    response_length=True,
                    status=True,
                    user=True,
                ),
                throttling_rate_limit=1000,
                throttling_burst_limit=5000,
            ),
            default_cors_preflight_options=apigw.CorsOptions(
                allow_origins=["http://localhost:8501", "https://*.streamlit.app"],
                allow_methods=apigw.Cors.ALL_METHODS,
                allow_headers=[
                    "Content-Type",
                    "Authorization",
                    "X-Amz-Date",
                    "X-Api-Key",
                    "X-Amz-Security-Token",
                ],
                allow_credentials=True,
            ),
        )

        # ---------------------------------------------------------------
        # Cognito Authorizer
        # ---------------------------------------------------------------
        cognito_authorizer = apigw.CognitoUserPoolsAuthorizer(
            self, "CognitoAuthorizer",
            authorizer_name=f"rag-cognito-auth-{env_name}",
            cognito_user_pools=[user_pool],
        )

        # ---------------------------------------------------------------
        # JSON Schema Request Validators
        # ---------------------------------------------------------------
        request_validator = apigw.RequestValidator(
            self, "BodyValidator",
            rest_api=self.rest_api,
            request_validator_name="validate-body",
            validate_request_body=True,
            validate_request_parameters=False,
        )

        # POST /query request model
        query_request_model = apigw.Model(
            self, "QueryRequestModel",
            rest_api=self.rest_api,
            content_type="application/json",
            model_name="QueryRequest",
            schema=apigw.JsonSchema(
                type=apigw.JsonSchemaType.OBJECT,
                required=["query"],
                properties={
                    "query": apigw.JsonSchema(type=apigw.JsonSchemaType.STRING, min_length=1),
                    "filters": apigw.JsonSchema(type=apigw.JsonSchemaType.OBJECT),
                    "k": apigw.JsonSchema(type=apigw.JsonSchemaType.INTEGER, minimum=1, maximum=100),
                    "rerank": apigw.JsonSchema(type=apigw.JsonSchemaType.BOOLEAN),
                    "stream": apigw.JsonSchema(type=apigw.JsonSchemaType.BOOLEAN),
                },
            ),
        )

        # POST /ingest request model
        ingest_request_model = apigw.Model(
            self, "IngestRequestModel",
            rest_api=self.rest_api,
            content_type="application/json",
            model_name="IngestRequest",
            schema=apigw.JsonSchema(
                type=apigw.JsonSchemaType.OBJECT,
                required=["file_name"],
                properties={
                    "file_name": apigw.JsonSchema(type=apigw.JsonSchemaType.STRING, min_length=1),
                    "content_type": apigw.JsonSchema(type=apigw.JsonSchemaType.STRING),
                    "metadata": apigw.JsonSchema(type=apigw.JsonSchemaType.OBJECT),
                },
            ),
        )

        # POST /evaluate request model
        evaluate_request_model = apigw.Model(
            self, "EvaluateRequestModel",
            rest_api=self.rest_api,
            content_type="application/json",
            model_name="EvaluateRequest",
            schema=apigw.JsonSchema(
                type=apigw.JsonSchemaType.OBJECT,
                required=["dataset_s3_key"],
                properties={
                    "dataset_s3_key": apigw.JsonSchema(type=apigw.JsonSchemaType.STRING, min_length=1),
                    "bucket": apigw.JsonSchema(type=apigw.JsonSchemaType.STRING),
                },
            ),
        )

        # ---------------------------------------------------------------
        # REST API Resources and Methods
        # ---------------------------------------------------------------

        # POST /query (30s timeout)
        query_resource = self.rest_api.root.add_resource("query")
        query_resource.add_method(
            "POST",
            apigw.LambdaIntegration(
                self.query_handler_fn,
                timeout=cdk.Duration.seconds(30),
            ),
            authorization_type=apigw.AuthorizationType.COGNITO,
            authorizer=cognito_authorizer,
            request_validator=request_validator,
            request_models={"application/json": query_request_model},
        )

        # POST /ingest (10s timeout)
        ingest_resource = self.rest_api.root.add_resource("ingest")
        ingest_resource.add_method(
            "POST",
            apigw.LambdaIntegration(
                self.ingest_trigger_fn,
                timeout=cdk.Duration.seconds(10),
            ),
            authorization_type=apigw.AuthorizationType.COGNITO,
            authorizer=cognito_authorizer,
            request_validator=request_validator,
            request_models={"application/json": ingest_request_model},
        )

        # GET /document/{id} (10s timeout)
        document_resource = self.rest_api.root.add_resource("document")
        document_id_resource = document_resource.add_resource("{id}")
        document_id_resource.add_method(
            "GET",
            apigw.LambdaIntegration(
                self.document_handler_fn,
                timeout=cdk.Duration.seconds(10),
            ),
            authorization_type=apigw.AuthorizationType.COGNITO,
            authorizer=cognito_authorizer,
        )

        # GET /health (5s timeout)
        health_resource = self.rest_api.root.add_resource("health")
        health_resource.add_method(
            "GET",
            apigw.LambdaIntegration(
                self.health_handler_fn,
                timeout=cdk.Duration.seconds(5),
            ),
            authorization_type=apigw.AuthorizationType.COGNITO,
            authorizer=cognito_authorizer,
        )

        # GET /metrics (10s timeout)
        metrics_resource = self.rest_api.root.add_resource("metrics")
        metrics_resource.add_method(
            "GET",
            apigw.LambdaIntegration(
                self.metrics_handler_fn,
                timeout=cdk.Duration.seconds(10),
            ),
            authorization_type=apigw.AuthorizationType.COGNITO,
            authorizer=cognito_authorizer,
        )

        # POST /evaluate (30s API GW timeout, Lambda runs up to 300s)
        evaluate_resource = self.rest_api.root.add_resource("evaluate")
        evaluate_resource.add_method(
            "POST",
            apigw.LambdaIntegration(
                self.evaluation_fn,
                timeout=cdk.Duration.seconds(29),
            ),
            authorization_type=apigw.AuthorizationType.COGNITO,
            authorizer=cognito_authorizer,
            request_validator=request_validator,
            request_models={"application/json": evaluate_request_model},
        )

        # ---------------------------------------------------------------
        # EventBridge Scheduled Rule — weekly evaluation
        # ---------------------------------------------------------------
        self.evaluation_schedule_rule = events.Rule(
            self, "EvaluationScheduleRule",
            rule_name=f"rag-evaluation-weekly-{env_name}",
            description="Trigger RAGAS evaluation pipeline weekly",
            schedule=events.Schedule.rate(cdk.Duration.days(7)),
        )
        self.evaluation_schedule_rule.add_target(
            targets.LambdaFunction(
                self.evaluation_fn,
                event=events.RuleTargetInput.from_object({
                    "dataset_s3_key": "evaluations/dataset.json",
                }),
            )
        )

        # ---------------------------------------------------------------
        # WebSocket API
        # ---------------------------------------------------------------

        # WebSocket connect handler (reuse health handler for auth check)
        self.ws_connect_fn = _lambda.Function(
            self, "WsConnectFn",
            function_name=f"rag-ws-connect-{env_name}",
            runtime=runtime,
            handler="query.health_handler.handler.handler",
            code=lambda_code,
            memory_size=256,
            timeout=cdk.Duration.seconds(10),
            vpc=vpc,
            vpc_subnets=private_subnets,
            security_groups=[lambda_sg],
            environment=common_env,
            tracing=_lambda.Tracing.ACTIVE,
        )

        # WebSocket disconnect handler
        self.ws_disconnect_fn = _lambda.Function(
            self, "WsDisconnectFn",
            function_name=f"rag-ws-disconnect-{env_name}",
            runtime=runtime,
            handler="query.health_handler.handler.handler",
            code=lambda_code,
            memory_size=256,
            timeout=cdk.Duration.seconds(10),
            vpc=vpc,
            vpc_subnets=private_subnets,
            security_groups=[lambda_sg],
            environment=common_env,
            tracing=_lambda.Tracing.ACTIVE,
        )

        self.websocket_api = apigwv2.CfnApi(
            self, "RagWebSocketApi",
            name=f"rag-ws-api-{env_name}",
            protocol_type="WEBSOCKET",
            route_selection_expression="$request.body.action",
        )

        # WebSocket stage
        ws_log_group = logs.LogGroup(
            self, "WsAccessLogs",
            log_group_name=f"/aws/apigateway/rag-ws-api-{env_name}",
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=cdk.RemovalPolicy.DESTROY,
        )

        ws_stage = apigwv2.CfnStage(
            self, "WsStage",
            api_id=self.websocket_api.ref,
            stage_name=env_name,
            auto_deploy=True,
            default_route_settings=apigwv2.CfnStage.RouteSettingsProperty(
                throttling_rate_limit=1000,
                throttling_burst_limit=5000,
            ),
            access_log_settings=apigwv2.CfnStage.AccessLogSettingsProperty(
                destination_arn=ws_log_group.log_group_arn,
                format='{"requestId":"$context.requestId","ip":"$context.identity.sourceIp","caller":"$context.identity.caller","user":"$context.identity.user","requestTime":"$context.requestTime","routeKey":"$context.routeKey","status":"$context.status"}',
            ),
        )

        # --- WebSocket Integrations ---

        # $connect integration
        connect_integration = apigwv2.CfnIntegration(
            self, "WsConnectIntegration",
            api_id=self.websocket_api.ref,
            integration_type="AWS_PROXY",
            integration_uri=f"arn:aws:apigateway:{cdk.Aws.REGION}:lambda:path/2015-03-31/functions/{self.ws_connect_fn.function_arn}/invocations",
        )

        apigwv2.CfnRoute(
            self, "WsConnectRoute",
            api_id=self.websocket_api.ref,
            route_key="$connect",
            authorization_type="NONE",
            target=f"integrations/{connect_integration.ref}",
        )

        # $disconnect integration
        disconnect_integration = apigwv2.CfnIntegration(
            self, "WsDisconnectIntegration",
            api_id=self.websocket_api.ref,
            integration_type="AWS_PROXY",
            integration_uri=f"arn:aws:apigateway:{cdk.Aws.REGION}:lambda:path/2015-03-31/functions/{self.ws_disconnect_fn.function_arn}/invocations",
        )

        apigwv2.CfnRoute(
            self, "WsDisconnectRoute",
            api_id=self.websocket_api.ref,
            route_key="$disconnect",
            target=f"integrations/{disconnect_integration.ref}",
        )

        # query route integration (reuse query handler)
        query_ws_integration = apigwv2.CfnIntegration(
            self, "WsQueryIntegration",
            api_id=self.websocket_api.ref,
            integration_type="AWS_PROXY",
            integration_uri=f"arn:aws:apigateway:{cdk.Aws.REGION}:lambda:path/2015-03-31/functions/{self.query_handler_fn.function_arn}/invocations",
        )

        apigwv2.CfnRoute(
            self, "WsQueryRoute",
            api_id=self.websocket_api.ref,
            route_key="query",
            target=f"integrations/{query_ws_integration.ref}",
        )

        # Grant API Gateway permission to invoke WebSocket Lambda functions
        self.ws_connect_fn.grant_invoke(
            iam.ServicePrincipal("apigateway.amazonaws.com")
        )
        self.ws_disconnect_fn.grant_invoke(
            iam.ServicePrincipal("apigateway.amazonaws.com")
        )
        self.query_handler_fn.grant_invoke(
            iam.ServicePrincipal("apigateway.amazonaws.com")
        )

        # ---------------------------------------------------------------
        # WAF WebACL Association
        # ---------------------------------------------------------------
        wafv2.CfnWebACLAssociation(
            self, "RestApiWafAssociation",
            resource_arn=f"arn:aws:apigateway:{cdk.Aws.REGION}::/restapis/{self.rest_api.rest_api_id}/stages/{env_name}",
            web_acl_arn=web_acl_arn,
        )

        # ---------------------------------------------------------------
        # Outputs
        # ---------------------------------------------------------------
        self.rest_api_url = self.rest_api.url

        cdk.CfnOutput(
            self, "RestApiUrl",
            value=self.rest_api.url,
            description="REST API Gateway URL",
        )
        cdk.CfnOutput(
            self, "RestApiId",
            value=self.rest_api.rest_api_id,
            description="REST API Gateway ID",
        )
        cdk.CfnOutput(
            self, "WebSocketApiId",
            value=self.websocket_api.ref,
            description="WebSocket API Gateway ID",
        )
        cdk.CfnOutput(
            self, "WebSocketApiEndpoint",
            value=f"wss://{self.websocket_api.ref}.execute-api.{cdk.Aws.REGION}.amazonaws.com/{env_name}",
            description="WebSocket API endpoint URL",
        )
