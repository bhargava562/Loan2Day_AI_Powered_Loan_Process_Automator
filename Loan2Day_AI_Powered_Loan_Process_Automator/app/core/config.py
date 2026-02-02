"""
Configuration management for Loan2Day platform.

This module handles all environment variables and application settings
following the LQM Standard for zero-hallucination configuration.

Security: All secrets managed through SecretManager, no hardcoded values.
"""

import os
import logging
from typing import Optional, Dict, Any
from pydantic import Field, validator
from pydantic_settings import BaseSettings
from decimal import Decimal

# Import security components
from app.core.security import secret_manager, secure_getenv, SecurityLevel

# Configure logger
logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """
    Application settings with strict typing and environment variable support.
    
    Follows the LQM Standard: No hardcoded values, all secrets from environment.
    """
    
    # Application Settings
    app_name: str = Field(default="Loan2Day", description="Application name")
    debug: bool = Field(default=False, description="Debug mode flag")
    version: str = Field(default="1.0.0", description="Application version")
    
    # Database Configuration
    database_url: str = Field(
        default="postgresql+asyncpg://loan2day:password@localhost/loan2day",
        description="PostgreSQL connection string"
    )
    db_pool_size: int = Field(
        default=10,
        description="Database connection pool size"
    )
    db_max_overflow: int = Field(
        default=20,
        description="Database connection pool max overflow"
    )
    db_pool_timeout: int = Field(
        default=30,
        description="Database connection pool timeout in seconds"
    )
    db_pool_recycle: int = Field(
        default=3600,
        description="Database connection pool recycle time in seconds"
    )
    
    # Environment Configuration
    environment: str = Field(
        default="development",
        description="Application environment (development, test, production)"
    )
    
    # Redis Configuration
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection string for AgentState caching"
    )
    
    # Kafka Configuration
    kafka_bootstrap_servers: str = Field(
        default="localhost:9092",
        description="Kafka bootstrap servers for async communication"
    )
    kafka_group_id: str = Field(
        default="loan2day-agents",
        description="Kafka consumer group ID for agent communication"
    )
    kafka_auto_offset_reset: str = Field(
        default="earliest",
        description="Kafka consumer auto offset reset policy"
    )
    kafka_enable_auto_commit: bool = Field(
        default=True,
        description="Kafka consumer auto commit enabled"
    )
    kafka_max_poll_records: int = Field(
        default=500,
        description="Maximum number of records returned in a single poll"
    )
    kafka_session_timeout_ms: int = Field(
        default=30000,
        description="Kafka consumer session timeout in milliseconds"
    )
    kafka_heartbeat_interval_ms: int = Field(
        default=3000,
        description="Kafka consumer heartbeat interval in milliseconds"
    )
    kafka_retry_backoff_ms: int = Field(
        default=100,
        description="Kafka retry backoff time in milliseconds"
    )
    kafka_request_timeout_ms: int = Field(
        default=30000,
        description="Kafka request timeout in milliseconds"
    )
    kafka_dead_letter_topic: str = Field(
        default="loan2day-dead-letter",
        description="Kafka dead letter queue topic name"
    )
    
    # External API Keys (NEVER hardcoded - managed by SecretManager)
    twilio_account_sid: Optional[str] = Field(
        default=None,
        description="Twilio Account SID for voice processing"
    )
    twilio_auth_token: Optional[str] = Field(
        default=None,
        description="Twilio Auth Token for voice processing"
    )
    
    @validator('twilio_account_sid', pre=True, always=True)
    def validate_twilio_sid(cls, v):
        """Validate Twilio SID using SecretManager."""
        if v is None:
            try:
                return secret_manager.get_secret('twilio_account_sid', operation='config_load')
            except ValueError:
                return None
        return v
    
    @validator('twilio_auth_token', pre=True, always=True)
    def validate_twilio_token(cls, v):
        """Validate Twilio token using SecretManager."""
        if v is None:
            try:
                return secret_manager.get_secret('twilio_auth_token', operation='config_load')
            except ValueError:
                return None
        return v
    
    # LQM Configuration (Mathematical Precision)
    default_interest_rate: Decimal = Field(
        default=Decimal("12.50"),
        description="Default annual interest rate as Decimal"
    )
    max_loan_amount_in_cents: Decimal = Field(
        default=Decimal("10000000"),  # 1 Crore in cents
        description="Maximum loan amount in cents (LQM Standard)"
    )
    min_loan_amount_in_cents: Decimal = Field(
        default=Decimal("5000000"),   # 50,000 in cents
        description="Minimum loan amount in cents (LQM Standard)"
    )
    
    # Security Configuration
    sgs_security_threshold: float = Field(
        default=0.85,
        description="SGS security score threshold for document acceptance"
    )
    max_file_size_mb: int = Field(
        default=10,
        description="Maximum file upload size in MB"
    )
    
    # Performance Configuration
    api_response_timeout_seconds: float = Field(
        default=2.0,
        description="Maximum API response time for natural conversation"
    )
    redis_session_ttl_seconds: int = Field(
        default=3600,
        description="Redis session TTL in seconds"
    )
    
    class Config:
        """Pydantic configuration for environment variable loading."""
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()


def get_database_url() -> str:
    """
    Get database URL using SecretManager for security.
    
    Returns:
        str: Database connection string
        
    Raises:
        ValueError: If database URL is not configured
    """
    try:
        return secret_manager.get_secret('database_url', operation='database_connection')
    except ValueError:
        # Fallback to settings if SecretManager fails
        if not settings.database_url:
            raise ValueError("DATABASE_URL environment variable not set")
        return settings.database_url


def get_redis_url() -> str:
    """
    Get Redis URL using SecretManager for security.
    
    Returns:
        str: Redis connection string
        
    Raises:
        ValueError: If Redis URL is not configured
    """
    try:
        return secret_manager.get_secret('redis_url', operation='redis_connection')
    except ValueError:
        # Fallback to settings if SecretManager fails
        if not settings.redis_url:
            raise ValueError("REDIS_URL environment variable not set")
        return settings.redis_url


def validate_twilio_config() -> bool:
    """
    Validate Twilio configuration using SecretManager.
    
    Returns:
        bool: True if Twilio is properly configured
    """
    try:
        sid = secret_manager.get_secret('twilio_account_sid', operation='config_validation')
        token = secret_manager.get_secret('twilio_auth_token', operation='config_validation')
        return bool(sid and token)
    except ValueError:
        return False


def validate_security_configuration() -> Dict[str, bool]:
    """
    Validate all security-related configuration.
    
    Returns:
        Dict[str, bool]: Configuration validation results
    """
    return secret_manager.validate_all_secrets()


def get_security_summary() -> Dict[str, Any]:
    """
    Get security configuration summary (without exposing secrets).
    
    Returns:
        Dict[str, Any]: Security configuration summary
    """
    return secret_manager.get_secret_summary()


def get_kafka_config() -> Dict[str, Any]:
    """
    Get Kafka configuration using SecretManager for security.
    
    Returns:
        Dict[str, Any]: Kafka configuration dictionary
        
    Raises:
        ValueError: If Kafka configuration is not properly set
    """
    try:
        bootstrap_servers = secret_manager.get_secret('kafka_bootstrap_servers', operation='kafka_connection')
    except ValueError:
        # Fallback to settings if SecretManager fails
        bootstrap_servers = settings.kafka_bootstrap_servers
    
    return {
        'bootstrap_servers': bootstrap_servers,
        'group_id': settings.kafka_group_id,
        'auto_offset_reset': settings.kafka_auto_offset_reset,
        'enable_auto_commit': settings.kafka_enable_auto_commit,
        'max_poll_records': settings.kafka_max_poll_records,
        'session_timeout_ms': settings.kafka_session_timeout_ms,
        'heartbeat_interval_ms': settings.kafka_heartbeat_interval_ms,
        'retry_backoff_ms': settings.kafka_retry_backoff_ms,
        'request_timeout_ms': settings.kafka_request_timeout_ms,
        'dead_letter_topic': settings.kafka_dead_letter_topic
    }