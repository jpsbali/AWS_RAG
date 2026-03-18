"""NetworkStack: VPC, subnets, security groups, and VPC endpoints."""
import aws_cdk as cdk
from aws_cdk import (
    aws_ec2 as ec2,
)
from constructs import Construct


class NetworkStack(cdk.Stack):
    """VPC with public/private subnets, security groups, and VPC endpoints for AWS services."""

    def __init__(self, scope: Construct, construct_id: str, env_name: str = "dev", **kwargs):
        super().__init__(scope, construct_id, **kwargs)

        self.env_name = env_name

        # --- VPC ---
        self.vpc = ec2.Vpc(
            self,
            "RagVpc",
            vpc_name=f"rag-vpc-{env_name}",
            max_azs=2,
            nat_gateways=1 if env_name == "dev" else 2,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="Public",
                    subnet_type=ec2.SubnetType.PUBLIC,
                    cidr_mask=24,
                ),
                ec2.SubnetConfiguration(
                    name="Private",
                    subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                    cidr_mask=24,
                ),
            ],
        )

        # --- Security Groups ---
        self.lambda_sg = ec2.SecurityGroup(
            self,
            "LambdaSg",
            vpc=self.vpc,
            security_group_name=f"rag-lambda-sg-{env_name}",
            description="Security group for Lambda functions",
            allow_all_outbound=True,
        )

        self.opensearch_sg = ec2.SecurityGroup(
            self,
            "OpenSearchSg",
            vpc=self.vpc,
            security_group_name=f"rag-opensearch-sg-{env_name}",
            description="Security group for OpenSearch domain",
            allow_all_outbound=False,
        )

        self.elasticache_sg = ec2.SecurityGroup(
            self,
            "ElastiCacheSg",
            vpc=self.vpc,
            security_group_name=f"rag-elasticache-sg-{env_name}",
            description="Security group for ElastiCache Redis",
            allow_all_outbound=False,
        )

        # Allow Lambda → OpenSearch (HTTPS 443)
        self.opensearch_sg.add_ingress_rule(
            peer=self.lambda_sg,
            connection=ec2.Port.tcp(443),
            description="Lambda to OpenSearch HTTPS",
        )

        # Allow Lambda → ElastiCache (Redis 6379)
        self.elasticache_sg.add_ingress_rule(
            peer=self.lambda_sg,
            connection=ec2.Port.tcp(6379),
            description="Lambda to ElastiCache Redis",
        )

        # --- VPC Endpoints (Gateway) ---
        self.vpc.add_gateway_endpoint(
            "S3Endpoint",
            service=ec2.GatewayVpcEndpointAwsService.S3,
        )

        # --- VPC Endpoints (Interface) ---
        interface_services = {
            "BedrockEndpoint": ec2.InterfaceVpcEndpointAwsService.BEDROCK_RUNTIME,
            "ComprehendEndpoint": ec2.InterfaceVpcEndpointAwsService("comprehend"),
            "TextractEndpoint": ec2.InterfaceVpcEndpointAwsService("textract"),
            "StepFunctionsEndpoint": ec2.InterfaceVpcEndpointAwsService.STEP_FUNCTIONS,
            "CloudWatchLogsEndpoint": ec2.InterfaceVpcEndpointAwsService.CLOUDWATCH_LOGS,
            "SecretsManagerEndpoint": ec2.InterfaceVpcEndpointAwsService.SECRETS_MANAGER,
        }

        for endpoint_id, service in interface_services.items():
            self.vpc.add_interface_endpoint(
                endpoint_id,
                service=service,
                private_dns_enabled=True,
                security_groups=[self.lambda_sg],
                subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS),
            )

        # --- Outputs ---
        cdk.CfnOutput(self, "VpcId", value=self.vpc.vpc_id)
        cdk.CfnOutput(self, "LambdaSgId", value=self.lambda_sg.security_group_id)
        cdk.CfnOutput(self, "OpenSearchSgId", value=self.opensearch_sg.security_group_id)
        cdk.CfnOutput(self, "ElastiCacheSgId", value=self.elasticache_sg.security_group_id)
