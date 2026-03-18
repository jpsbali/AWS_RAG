"""PipelineStack: CI/CD pipeline using CodePipeline and CodeBuild for automated build, test, and deploy."""
import aws_cdk as cdk
from aws_cdk import (
    aws_codebuild as codebuild,
    aws_codecommit as codecommit,
    aws_codepipeline as codepipeline,
    aws_codepipeline_actions as codepipeline_actions,
    aws_iam as iam,
    aws_sns as sns,
    aws_sns_subscriptions as sns_subs,
)
from constructs import Construct


class PipelineStack(cdk.Stack):
    """CI/CD pipeline: Source → Build → Deploy Dev → Integration Tests → Manual Approval → Deploy Prod."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        env_name: str = "dev",
        repository_name: str = "aws-native-rag-service",
        branch: str = "main",
        codestar_connection_arn: str | None = None,
        github_owner: str | None = None,
        github_repo: str | None = None,
        notification_email: str | None = None,
        **kwargs,
    ):
        super().__init__(scope, construct_id, **kwargs)

        self.env_name = env_name

        # --- Source artifacts ---
        source_output = codepipeline.Artifact("SourceOutput")
        build_output = codepipeline.Artifact("BuildOutput")

        # --- Source stage action ---
        # Use GitHub via CodeStar connection if configured, otherwise CodeCommit
        if codestar_connection_arn and github_owner and github_repo:
            source_action = codepipeline_actions.CodeStarConnectionsSourceAction(
                action_name="GitHub_Source",
                connection_arn=codestar_connection_arn,
                owner=github_owner,
                repo=github_repo,
                branch=branch,
                output=source_output,
                trigger_on_push=True,
            )
        else:
            repo = codecommit.Repository(
                self,
                "CodeCommitRepo",
                repository_name=repository_name,
                description="AWS-Native RAG Service repository",
            )
            source_action = codepipeline_actions.CodeCommitSourceAction(
                action_name="CodeCommit_Source",
                repository=repo,
                branch=branch,
                output=source_output,
                trigger=codepipeline_actions.CodeCommitTrigger.EVENTS,
            )

        # --- Build project (lint, unit tests, property tests, cdk synth) ---
        build_project = codebuild.PipelineProject(
            self,
            "BuildProject",
            project_name=f"rag-build-{env_name}",
            description="Lint, unit tests, property tests, and CDK synth",
            environment=codebuild.BuildEnvironment(
                build_image=codebuild.LinuxBuildImage.STANDARD_7_0,
                compute_type=codebuild.ComputeType.MEDIUM,
            ),
            build_spec=codebuild.BuildSpec.from_object(
                {
                    "version": "0.2",
                    "phases": {
                        "install": {
                            "runtime-versions": {"python": "3.11"},
                            "commands": [
                                "curl -LsSf https://astral.sh/uv/install.sh | sh",
                                'export PATH="$HOME/.local/bin:$PATH"',
                                "uv sync",
                            ],
                        },
                        "pre_build": {
                            "commands": [
                                'echo "Linting..."',
                                "uv run black --check lambdas/ stacks/ tests/",
                                "uv run flake8 lambdas/ stacks/ tests/",
                            ],
                        },
                        "build": {
                            "commands": [
                                'echo "Running unit and property tests..."',
                                "uv run pytest tests/unit/ tests/property/ -v --cov=lambdas --cov-report=xml",
                                'echo "Synthesizing CDK stacks..."',
                                "uv run cdk synth --context env=dev --output cdk.out",
                            ],
                        },
                        "post_build": {
                            "commands": ['echo "Build complete"'],
                        },
                    },
                    "artifacts": {
                        "files": ["**/*"],
                        "name": "BuildArtifact",
                    },
                    "reports": {
                        "coverage": {
                            "files": ["coverage.xml"],
                            "file-format": "COBERTURAXML",
                        },
                    },
                }
            ),
            timeout=cdk.Duration.minutes(30),
        )

        # Grant CDK synth permissions (CloudFormation describe, S3 for assets)
        build_project.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "cloudformation:DescribeStacks",
                    "cloudformation:GetTemplate",
                    "ssm:GetParameter",
                    "ssm:GetParameters",
                ],
                resources=["*"],
            )
        )

        build_action = codepipeline_actions.CodeBuildAction(
            action_name="Build",
            project=build_project,
            input=source_output,
            outputs=[build_output],
        )

        # --- Deploy Dev action (CDK deploy with env=dev) ---
        deploy_dev_project = codebuild.PipelineProject(
            self,
            "DeployDevProject",
            project_name=f"rag-deploy-dev",
            description="CDK deploy to dev environment",
            environment=codebuild.BuildEnvironment(
                build_image=codebuild.LinuxBuildImage.STANDARD_7_0,
                compute_type=codebuild.ComputeType.MEDIUM,
            ),
            build_spec=codebuild.BuildSpec.from_object(
                {
                    "version": "0.2",
                    "phases": {
                        "install": {
                            "runtime-versions": {"python": "3.11"},
                            "commands": [
                                "curl -LsSf https://astral.sh/uv/install.sh | sh",
                                'export PATH="$HOME/.local/bin:$PATH"',
                                "uv sync",
                            ],
                        },
                        "build": {
                            "commands": [
                                'echo "Deploying to dev environment..."',
                                "uv run cdk deploy --all --context env=dev --require-approval never",
                            ],
                        },
                    },
                }
            ),
            timeout=cdk.Duration.minutes(60),
        )

        # Grant CDK deploy permissions
        deploy_dev_project.add_to_role_policy(
            iam.PolicyStatement(
                actions=["sts:AssumeRole"],
                resources=[f"arn:aws:iam::{cdk.Aws.ACCOUNT_ID}:role/cdk-*"],
            )
        )
        deploy_dev_project.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "cloudformation:*",
                    "s3:*",
                    "iam:PassRole",
                    "ssm:GetParameter",
                    "ssm:GetParameters",
                ],
                resources=["*"],
            )
        )

        deploy_dev_action = codepipeline_actions.CodeBuildAction(
            action_name="Deploy_Dev",
            project=deploy_dev_project,
            input=build_output,
        )

        # --- Integration Tests action (run against dev environment) ---
        integration_test_project = codebuild.PipelineProject(
            self,
            "IntegrationTestProject",
            project_name=f"rag-integration-tests-{env_name}",
            description="Run integration tests against dev environment",
            environment=codebuild.BuildEnvironment(
                build_image=codebuild.LinuxBuildImage.STANDARD_7_0,
                compute_type=codebuild.ComputeType.MEDIUM,
            ),
            build_spec=codebuild.BuildSpec.from_object(
                {
                    "version": "0.2",
                    "phases": {
                        "install": {
                            "runtime-versions": {"python": "3.11"},
                            "commands": [
                                "curl -LsSf https://astral.sh/uv/install.sh | sh",
                                'export PATH="$HOME/.local/bin:$PATH"',
                                "uv sync",
                            ],
                        },
                        "build": {
                            "commands": [
                                'echo "Running integration tests against dev..."',
                                "uv run pytest tests/integration/ -v --tb=short",
                            ],
                        },
                    },
                }
            ),
            timeout=cdk.Duration.minutes(30),
        )

        integration_test_action = codepipeline_actions.CodeBuildAction(
            action_name="Integration_Tests",
            project=integration_test_project,
            input=build_output,
        )

        # --- Manual Approval action ---
        approval_topic = sns.Topic(
            self,
            "ApprovalTopic",
            topic_name=f"rag-pipeline-approval-{env_name}",
            display_name=f"RAG Pipeline Approval ({env_name})",
        )

        if notification_email:
            approval_topic.add_subscription(
                sns_subs.EmailSubscription(notification_email)
            )

        manual_approval_action = codepipeline_actions.ManualApprovalAction(
            action_name="Approve_Prod_Deploy",
            notification_topic=approval_topic,
            additional_information="Review the dev deployment and approve promotion to production.",
        )

        # --- Deploy Prod action (CDK deploy with env=prod) ---
        deploy_prod_project = codebuild.PipelineProject(
            self,
            "DeployProdProject",
            project_name=f"rag-deploy-prod",
            description="CDK deploy to production environment",
            environment=codebuild.BuildEnvironment(
                build_image=codebuild.LinuxBuildImage.STANDARD_7_0,
                compute_type=codebuild.ComputeType.MEDIUM,
            ),
            build_spec=codebuild.BuildSpec.from_object(
                {
                    "version": "0.2",
                    "phases": {
                        "install": {
                            "runtime-versions": {"python": "3.11"},
                            "commands": [
                                "curl -LsSf https://astral.sh/uv/install.sh | sh",
                                'export PATH="$HOME/.local/bin:$PATH"',
                                "uv sync",
                            ],
                        },
                        "build": {
                            "commands": [
                                'echo "Deploying to production environment..."',
                                "uv run cdk deploy --all --context env=prod --require-approval never",
                            ],
                        },
                    },
                }
            ),
            timeout=cdk.Duration.minutes(60),
        )

        # Grant CDK deploy permissions for prod
        deploy_prod_project.add_to_role_policy(
            iam.PolicyStatement(
                actions=["sts:AssumeRole"],
                resources=[f"arn:aws:iam::{cdk.Aws.ACCOUNT_ID}:role/cdk-*"],
            )
        )
        deploy_prod_project.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "cloudformation:*",
                    "s3:*",
                    "iam:PassRole",
                    "ssm:GetParameter",
                    "ssm:GetParameters",
                ],
                resources=["*"],
            )
        )

        deploy_prod_action = codepipeline_actions.CodeBuildAction(
            action_name="Deploy_Prod",
            project=deploy_prod_project,
            input=build_output,
        )

        # --- CodePipeline ---
        self.pipeline = codepipeline.Pipeline(
            self,
            "RagPipeline",
            pipeline_name=f"rag-pipeline-{env_name}",
            restart_execution_on_update=True,
            stages=[
                codepipeline.StageProps(
                    stage_name="Source",
                    actions=[source_action],
                ),
                codepipeline.StageProps(
                    stage_name="Build",
                    actions=[build_action],
                ),
                codepipeline.StageProps(
                    stage_name="Deploy_Dev",
                    actions=[deploy_dev_action],
                ),
                codepipeline.StageProps(
                    stage_name="Integration_Tests",
                    actions=[integration_test_action],
                ),
                codepipeline.StageProps(
                    stage_name="Manual_Approval",
                    actions=[manual_approval_action],
                ),
                codepipeline.StageProps(
                    stage_name="Deploy_Prod",
                    actions=[deploy_prod_action],
                ),
            ],
        )

        # --- Outputs ---
        cdk.CfnOutput(
            self,
            "PipelineName",
            value=self.pipeline.pipeline_name,
            description="CodePipeline name",
        )
        cdk.CfnOutput(
            self,
            "PipelineArn",
            value=self.pipeline.pipeline_arn,
            description="CodePipeline ARN",
        )
        cdk.CfnOutput(
            self,
            "ApprovalTopicArn",
            value=approval_topic.topic_arn,
            description="SNS topic ARN for pipeline approval notifications",
        )
