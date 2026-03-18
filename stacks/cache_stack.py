"""CacheStack: ElastiCache Redis cluster with encryption, multi-AZ, and VPC deployment."""
import aws_cdk as cdk
from aws_cdk import (
    aws_ec2 as ec2,
    aws_elasticache as elasticache,
    aws_kms as kms,
)
from constructs import Construct


class CacheStack(cdk.Stack):
    """ElastiCache Redis 7.x cluster with KMS encryption, TLS, and multi-AZ failover."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        vpc: ec2.IVpc,
        elasticache_sg: ec2.ISecurityGroup,
        encryption_key: kms.IKey,
        env_name: str = "dev",
        **kwargs,
    ):
        super().__init__(scope, construct_id, **kwargs)

        self.env_name = env_name

        # Select private subnets for Redis deployment
        private_subnets = vpc.select_subnets(
            subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS
        )

        # --- Subnet Group ---
        self.subnet_group = elasticache.CfnSubnetGroup(
            self,
            "RagRedisSubnetGroup",
            description=f"Subnet group for RAG Redis cluster ({env_name})",
            subnet_ids=private_subnets.subnet_ids,
            cache_subnet_group_name=f"rag-redis-subnet-{env_name}",
        )

        # --- ElastiCache Redis Replication Group ---
        self.redis_cluster = elasticache.CfnReplicationGroup(
            self,
            "RagRedisCluster",
            replication_group_description=f"RAG service Redis cache ({env_name})",
            engine="redis",
            engine_version="7.1",
            cache_node_type="cache.r6g.large",
            # Cluster mode: 2 shards, 1 replica per shard
            num_node_groups=2,
            replicas_per_node_group=1,
            # Multi-AZ with automatic failover
            multi_az_enabled=True,
            automatic_failover_enabled=True,
            # Encryption at rest with KMS
            at_rest_encryption_enabled=True,
            kms_key_id=encryption_key.key_id,
            # Encryption in transit (TLS)
            transit_encryption_enabled=True,
            # VPC deployment
            cache_subnet_group_name=self.subnet_group.cache_subnet_group_name,
            security_group_ids=[elasticache_sg.security_group_id],
            # Port
            port=6379,
        )

        self.redis_cluster.add_dependency(self.subnet_group)

        # --- Exposed properties for downstream consumers ---
        self.redis_endpoint = self.redis_cluster.attr_configuration_end_point_address
        self.redis_port = self.redis_cluster.attr_configuration_end_point_port

        # --- Outputs ---
        cdk.CfnOutput(
            self,
            "RedisEndpoint",
            value=self.redis_endpoint,
            description="Redis cluster configuration endpoint address",
        )
        cdk.CfnOutput(
            self,
            "RedisPort",
            value=self.redis_port,
            description="Redis cluster configuration endpoint port",
        )
