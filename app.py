#!/usr/bin/env python3
"""CDK app entry point for the AWS-Native RAG Service."""
import aws_cdk as cdk

from stacks.network_stack import NetworkStack
from stacks.storage_stack import StorageStack
from stacks.security_stack import SecurityStack

from stacks.search_stack import SearchStack
from stacks.cache_stack import CacheStack
from stacks.monitoring_stack import MonitoringStack
from stacks.processing_stack import ProcessingStack
from stacks.api_stack import ApiStack
from stacks.frontend_stack import FrontendStack

from stacks.pipeline_stack import PipelineStack

app = cdk.App()

env_name = app.node.try_get_context("env") or "dev"

# --- Core infrastructure stacks ---
network = NetworkStack(app, f"RagNetworkStack-{env_name}", env_name=env_name)

security = SecurityStack(app, f"RagSecurityStack-{env_name}", env_name=env_name)

storage = StorageStack(
    app,
    f"RagStorageStack-{env_name}",
    encryption_key=security.encryption_key,
    env_name=env_name,
)

search = SearchStack(
    app,
    f"RagSearchStack-{env_name}",
    vpc=network.vpc,
    opensearch_sg=network.opensearch_sg,
    encryption_key=security.encryption_key,
    opensearch_secret=security.opensearch_secret,
    env_name=env_name,
)

cache = CacheStack(
    app,
    f"RagCacheStack-{env_name}",
    vpc=network.vpc,
    elasticache_sg=network.elasticache_sg,
    encryption_key=security.encryption_key,
    env_name=env_name,
)

monitoring = MonitoringStack(
    app,
    f"RagMonitoringStack-{env_name}",
    env_name=env_name,
    opensearch_domain=search.domain,
)

processing = ProcessingStack(
    app,
    f"RagProcessingStack-{env_name}",
    vpc=network.vpc,
    lambda_sg=network.lambda_sg,
    document_bucket_name=storage.document_bucket.bucket_name,
    document_bucket_arn=storage.document_bucket.bucket_arn,
    encryption_key=security.encryption_key,
    opensearch_domain=search.domain,
    opensearch_endpoint=search.domain.domain_endpoint,
    cache_cluster=cache,
    alert_topic=monitoring.alert_topic,
    env_name=env_name,
)

api = ApiStack(
    app,
    f"RagApiStack-{env_name}",
    vpc=network.vpc,
    lambda_sg=network.lambda_sg,
    document_bucket=storage.document_bucket,
    encryption_key=security.encryption_key,
    user_pool=security.user_pool,
    web_acl_arn=security.web_acl.attr_arn,
    opensearch_domain=search.domain,
    opensearch_endpoint=search.domain.domain_endpoint,
    cache_cluster=cache,
    alert_topic=monitoring.alert_topic,
    env_name=env_name,
)

pipeline = PipelineStack(
    app,
    f"RagPipelineStack-{env_name}",
    env_name=env_name,
    repository_name="aws-native-rag-service",
    branch="main",
    codestar_connection_arn=app.node.try_get_context("codestar_connection_arn"),
    github_owner=app.node.try_get_context("github_owner"),
    github_repo=app.node.try_get_context("github_repo"),
    notification_email=app.node.try_get_context("notification_email"),
)

frontend = FrontendStack(
    app,
    f"RagFrontendStack-{env_name}",
    rest_api_url=api.rest_api_url,
    websocket_api_id=api.websocket_api.ref,
    user_pool=security.user_pool,
    user_pool_client_id=security.user_pool_client.user_pool_client_id,
    env_name=env_name,
)

app.synth()
