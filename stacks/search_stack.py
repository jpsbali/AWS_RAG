"""SearchStack: OpenSearch domain with k-NN vector search, KMS encryption, and VPC deployment."""
import json

import aws_cdk as cdk
from aws_cdk import (
    aws_ec2 as ec2,
    aws_kms as kms,
    aws_opensearchservice as opensearch,
    aws_secretsmanager as secretsmanager,
)
from constructs import Construct


class SearchStack(cdk.Stack):
    """OpenSearch Service domain with HNSW k-NN, multi-AZ, KMS encryption, and index template."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        vpc: ec2.IVpc,
        opensearch_sg: ec2.ISecurityGroup,
        encryption_key: kms.IKey,
        opensearch_secret: secretsmanager.ISecret,
        env_name: str = "dev",
        **kwargs,
    ):
        super().__init__(scope, construct_id, **kwargs)

        self.env_name = env_name

        # --- OpenSearch Domain ---
        self.domain = opensearch.Domain(
            self,
            "RagOpenSearchDomain",
            domain_name=f"rag-search-{env_name}",
            version=opensearch.EngineVersion.OPENSEARCH_2_11,
            # Data nodes: 3x r6g.xlarge with 500GB gp3
            capacity=opensearch.CapacityConfig(
                data_node_instance_type="r6g.xlarge.search",
                data_nodes=3,
                master_node_instance_type="r6g.large.search",
                master_nodes=3,
            ),
            ebs=opensearch.EbsOptions(
                volume_size=500,
                volume_type=ec2.EbsDeviceVolumeType.GP3,
            ),
            # Multi-AZ deployment (match VPC AZ count: 2)
            zone_awareness=opensearch.ZoneAwarenessConfig(
                availability_zone_count=2,
            ),
            # VPC deployment in private subnets
            vpc=vpc,
            vpc_subnets=[ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS)],
            security_groups=[opensearch_sg],
            # KMS encryption at rest
            encryption_at_rest=opensearch.EncryptionAtRestOptions(
                enabled=True,
                kms_key=encryption_key,
            ),
            # TLS in transit (node-to-node)
            node_to_node_encryption=True,
            enforce_https=True,
            tls_security_policy=opensearch.TLSSecurityPolicy.TLS_1_2,
            # Fine-grained access control with master user from Secrets Manager
            fine_grained_access_control=opensearch.AdvancedSecurityOptions(
                master_user_name="admin",
                master_user_password=opensearch_secret.secret_value_from_json("password"),
            ),
            # Logging
            logging=opensearch.LoggingOptions(
                slow_search_log_enabled=True,
                slow_index_log_enabled=True,
                app_log_enabled=True,
            ),
            removal_policy=cdk.RemovalPolicy.RETAIN if env_name == "prod" else cdk.RemovalPolicy.DESTROY,
        )

        # --- Index template with knn_vector mapping ---
        self.index_template = _build_index_template()

        # --- Index alias configuration ---
        self.index_alias = "rag-chunks"
        self.index_name = f"rag-chunks-v1-{env_name}"

        # --- Outputs ---
        cdk.CfnOutput(self, "OpenSearchDomainEndpoint", value=self.domain.domain_endpoint)
        cdk.CfnOutput(self, "OpenSearchDomainArn", value=self.domain.domain_arn)
        cdk.CfnOutput(
            self,
            "IndexTemplate",
            value=json.dumps(self.index_template, indent=2),
            description="OpenSearch index template with knn_vector mapping",
        )


def _build_index_template() -> dict:
    """Build the OpenSearch index template with knn_vector mapping (1024 dims, HNSW)."""
    return {
        "mappings": {
            "properties": {
                "chunk_id": {"type": "keyword"},
                "document_id": {"type": "keyword"},
                "content": {"type": "text", "analyzer": "standard"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": 1024,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "nmslib",
                        "parameters": {
                            "ef_construction": 512,
                            "m": 16,
                        },
                    },
                },
                "metadata": {
                    "properties": {
                        "document_id": {"type": "keyword"},
                        "source": {"type": "keyword"},
                        "timestamp": {"type": "date"},
                        "category": {"type": "keyword"},
                        "entities": {"type": "keyword"},
                        "key_phrases": {"type": "keyword"},
                        "section_title": {"type": "keyword"},
                        "page_number": {"type": "integer"},
                        "language": {"type": "keyword"},
                    }
                },
            }
        },
        "settings": {
            "index": {
                "knn": True,
                "number_of_shards": 3,
                "number_of_replicas": 2,
                "refresh_interval": "1s",
            }
        },
    }
