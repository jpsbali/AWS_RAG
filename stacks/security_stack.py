"""SecurityStack: KMS key, Cognito user pool, WAF WebACL, and Secrets Manager."""
import aws_cdk as cdk
from aws_cdk import (
    aws_cognito as cognito,
    aws_kms as kms,
    aws_secretsmanager as secretsmanager,
    aws_wafv2 as wafv2,
)
from constructs import Construct


class SecurityStack(cdk.Stack):
    """KMS encryption key, Cognito auth, WAF protection, and Secrets Manager."""

    def __init__(self, scope: Construct, construct_id: str, env_name: str = "dev", **kwargs):
        super().__init__(scope, construct_id, **kwargs)

        self.env_name = env_name

        # --- KMS Customer-Managed Key ---
        self.encryption_key = kms.Key(
            self,
            "RagEncryptionKey",
            alias=f"alias/rag-encryption-key-{env_name}",
            description="Customer-managed key for RAG service encryption at rest",
            enable_key_rotation=True,
            removal_policy=cdk.RemovalPolicy.RETAIN if env_name == "prod" else cdk.RemovalPolicy.DESTROY,
        )

        # --- Cognito User Pool ---
        self.user_pool = cognito.UserPool(
            self,
            "RagUserPool",
            user_pool_name=f"rag-user-pool-{env_name}",
            self_sign_up_enabled=False,
            sign_in_aliases=cognito.SignInAliases(email=True),
            auto_verify=cognito.AutoVerifiedAttrs(email=True),
            password_policy=cognito.PasswordPolicy(
                min_length=12,
                require_uppercase=True,
                require_digits=True,
                require_symbols=True,
            ),
            mfa=cognito.Mfa.OPTIONAL,
            mfa_second_factor=cognito.MfaSecondFactor(sms=False, otp=True),
            account_recovery=cognito.AccountRecovery.EMAIL_ONLY,
            removal_policy=cdk.RemovalPolicy.RETAIN if env_name == "prod" else cdk.RemovalPolicy.DESTROY,
        )

        # Cognito App Client
        self.user_pool_client = self.user_pool.add_client(
            "RagAppClient",
            user_pool_client_name=f"rag-app-client-{env_name}",
            auth_flows=cognito.AuthFlow(
                user_password=True,
                user_srp=True,
            ),
            o_auth=cognito.OAuthSettings(
                flows=cognito.OAuthFlows(authorization_code_grant=True),
                scopes=[cognito.OAuthScope.OPENID, cognito.OAuthScope.EMAIL, cognito.OAuthScope.PROFILE],
                callback_urls=["http://localhost:8501/callback"],  # Streamlit default
                logout_urls=["http://localhost:8501/"],
            ),
            generate_secret=True,
            id_token_validity=cdk.Duration.hours(1),
            access_token_validity=cdk.Duration.hours(1),
            refresh_token_validity=cdk.Duration.days(30),
        )

        # Cognito Domain for hosted UI
        self.user_pool.add_domain(
            "RagCognitoDomain",
            cognito_domain=cognito.CognitoDomainOptions(
                domain_prefix=f"rag-service-{env_name}",
            ),
        )

        # --- WAF WebACL ---
        self.web_acl = wafv2.CfnWebACL(
            self,
            "RagWebAcl",
            name=f"rag-web-acl-{env_name}",
            scope="REGIONAL",
            default_action=wafv2.CfnWebACL.DefaultActionProperty(allow={}),
            visibility_config=wafv2.CfnWebACL.VisibilityConfigProperty(
                cloud_watch_metrics_enabled=True,
                metric_name=f"rag-waf-{env_name}",
                sampled_requests_enabled=True,
            ),
            rules=[
                # SQL Injection protection
                wafv2.CfnWebACL.RuleProperty(
                    name="AWSManagedRulesSQLiRuleSet",
                    priority=1,
                    override_action=wafv2.CfnWebACL.OverrideActionProperty(none={}),
                    statement=wafv2.CfnWebACL.StatementProperty(
                        managed_rule_group_statement=wafv2.CfnWebACL.ManagedRuleGroupStatementProperty(
                            vendor_name="AWS",
                            name="AWSManagedRulesSQLiRuleSet",
                        )
                    ),
                    visibility_config=wafv2.CfnWebACL.VisibilityConfigProperty(
                        cloud_watch_metrics_enabled=True,
                        metric_name="SQLiRuleSet",
                        sampled_requests_enabled=True,
                    ),
                ),
                # XSS protection
                wafv2.CfnWebACL.RuleProperty(
                    name="AWSManagedRulesCommonRuleSet",
                    priority=2,
                    override_action=wafv2.CfnWebACL.OverrideActionProperty(none={}),
                    statement=wafv2.CfnWebACL.StatementProperty(
                        managed_rule_group_statement=wafv2.CfnWebACL.ManagedRuleGroupStatementProperty(
                            vendor_name="AWS",
                            name="AWSManagedRulesCommonRuleSet",
                        )
                    ),
                    visibility_config=wafv2.CfnWebACL.VisibilityConfigProperty(
                        cloud_watch_metrics_enabled=True,
                        metric_name="CommonRuleSet",
                        sampled_requests_enabled=True,
                    ),
                ),
                # Rate-based rule: 2000 requests per 5 minutes per IP
                wafv2.CfnWebACL.RuleProperty(
                    name="RateLimitRule",
                    priority=3,
                    action=wafv2.CfnWebACL.RuleActionProperty(block={}),
                    statement=wafv2.CfnWebACL.StatementProperty(
                        rate_based_statement=wafv2.CfnWebACL.RateBasedStatementProperty(
                            limit=2000,
                            aggregate_key_type="IP",
                        )
                    ),
                    visibility_config=wafv2.CfnWebACL.VisibilityConfigProperty(
                        cloud_watch_metrics_enabled=True,
                        metric_name="RateLimitRule",
                        sampled_requests_enabled=True,
                    ),
                ),
            ],
        )

        # --- Secrets Manager: OpenSearch credentials ---
        self.opensearch_secret = secretsmanager.Secret(
            self,
            "OpenSearchSecret",
            secret_name=f"rag/opensearch-credentials-{env_name}",
            description="OpenSearch domain master user credentials",
            encryption_key=self.encryption_key,
            generate_secret_string=secretsmanager.SecretStringGenerator(
                secret_string_template='{"username": "admin"}',
                generate_string_key="password",
                exclude_punctuation=False,
                password_length=32,
            ),
        )

        # --- Outputs ---
        cdk.CfnOutput(self, "KmsKeyArn", value=self.encryption_key.key_arn)
        cdk.CfnOutput(self, "UserPoolId", value=self.user_pool.user_pool_id)
        cdk.CfnOutput(self, "UserPoolClientId", value=self.user_pool_client.user_pool_client_id)
        cdk.CfnOutput(self, "WebAclArn", value=self.web_acl.attr_arn)
        cdk.CfnOutput(self, "OpenSearchSecretArn", value=self.opensearch_secret.secret_arn)
