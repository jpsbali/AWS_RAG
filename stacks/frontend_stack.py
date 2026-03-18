"""FrontendStack: App Runner service for the Streamlit frontend application."""
import aws_cdk as cdk
from aws_cdk import (
    aws_apprunner as apprunner,
    aws_cognito as cognito,
    aws_ecr_assets as ecr_assets,
    aws_iam as iam,
)
from constructs import Construct


class FrontendStack(cdk.Stack):
    """Deploy the Streamlit frontend on AWS App Runner with auto-scaling."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        rest_api_url: str,
        websocket_api_id: str,
        user_pool: cognito.IUserPool,
        user_pool_client_id: str,
        env_name: str = "dev",
        **kwargs,
    ):
        super().__init__(scope, construct_id, **kwargs)

        self.env_name = env_name

        # Construct the WebSocket endpoint URL
        ws_endpoint = (
            f"wss://{websocket_api_id}.execute-api."
            f"{cdk.Aws.REGION}.amazonaws.com/{env_name}"
        )

        # Cognito hosted UI domain
        cognito_domain = f"rag-service-{env_name}.auth.{cdk.Aws.REGION}.amazoncognito.com"

        # Build Docker image from frontend/ directory
        image_asset = ecr_assets.DockerImageAsset(
            self,
            "FrontendImage",
            directory="frontend",
        )

        # IAM role for App Runner to pull from ECR
        access_role = iam.Role(
            self,
            "AppRunnerAccessRole",
            assumed_by=iam.ServicePrincipal("build.apprunner.amazonaws.com"),
            inline_policies={
                "ecr-pull": iam.PolicyDocument(
                    statements=[
                        iam.PolicyStatement(
                            actions=[
                                "ecr:GetDownloadUrlForLayer",
                                "ecr:BatchGetImage",
                                "ecr:BatchCheckLayerAvailability",
                                "ecr:GetAuthorizationToken",
                                "ecr:DescribeImages",
                            ],
                            resources=["*"],
                        ),
                    ]
                )
            },
        )

        # Auto-scaling configuration: 1–5 instances, 100 concurrent requests
        auto_scaling_config = apprunner.CfnAutoScalingConfiguration(
            self,
            "FrontendAutoScaling",
            auto_scaling_configuration_name=f"rag-frontend-scaling-{env_name}",
            min_size=1,
            max_size=5,
            max_concurrency=100,
        )

        # App Runner service (L1 CfnService)
        self.service = apprunner.CfnService(
            self,
            "FrontendService",
            service_name=f"rag-frontend-{env_name}",
            source_configuration=apprunner.CfnService.SourceConfigurationProperty(
                authentication_configuration=apprunner.CfnService.AuthenticationConfigurationProperty(
                    access_role_arn=access_role.role_arn,
                ),
                auto_deployments_enabled=False,
                image_repository=apprunner.CfnService.ImageRepositoryProperty(
                    image_identifier=image_asset.image_uri,
                    image_repository_type="ECR",
                    image_configuration=apprunner.CfnService.ImageConfigurationProperty(
                        port="8501",
                        runtime_environment_variables=[
                            apprunner.CfnService.KeyValuePairProperty(
                                name="API_ENDPOINT", value=rest_api_url,
                            ),
                            apprunner.CfnService.KeyValuePairProperty(
                                name="WS_ENDPOINT", value=ws_endpoint,
                            ),
                            apprunner.CfnService.KeyValuePairProperty(
                                name="COGNITO_DOMAIN", value=cognito_domain,
                            ),
                            apprunner.CfnService.KeyValuePairProperty(
                                name="COGNITO_CLIENT_ID", value=user_pool_client_id,
                            ),
                            apprunner.CfnService.KeyValuePairProperty(
                                name="COGNITO_REDIRECT_URI",
                                value=f"https://placeholder.{cdk.Aws.REGION}.awsapprunner.com",
                            ),
                        ],
                    ),
                ),
            ),
            auto_scaling_configuration_arn=auto_scaling_config.attr_auto_scaling_configuration_arn,
            health_check_configuration=apprunner.CfnService.HealthCheckConfigurationProperty(
                protocol="HTTP",
                path="/_stcore/health",
                interval=10,
                timeout=5,
                healthy_threshold=1,
                unhealthy_threshold=5,
            ),
        )

        # Ensure service waits for the scaling config and access role
        self.service.add_dependency(auto_scaling_config)
        self.service.node.add_dependency(access_role)

        # --- Outputs ---
        cdk.CfnOutput(
            self,
            "FrontendServiceUrl",
            value=cdk.Fn.get_att(self.service.logical_id, "ServiceUrl").to_string(),
            description="App Runner frontend service URL",
        )
        cdk.CfnOutput(
            self,
            "FrontendServiceArn",
            value=self.service.attr_service_arn,
            description="App Runner frontend service ARN",
        )
