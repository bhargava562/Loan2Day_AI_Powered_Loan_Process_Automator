"""
Configuration management for Loan2Day platform.

This module handles all environment variables and application settings
following the LQM Standard for zero-hallucination configuration.
"""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from decimal import Decimal


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
    
    # External API Keys (NEVER hardcoded)
    twilio_account_sid: Optional[str] = Field(
        default=None,
        description="Twilio Account SID for voice processing"
    )
    twilio_auth_token: Optional[str] = Field(
        default=None,
        description="Twilio Auth Token for voice processing"
    )
    
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
    Get database URL with proper error handling.
    
    Returns:
        str: Database connection string
        
    Raises:
        ValueError: If database URL is not configured
    """
    if not settings.database_url:
        raise ValueError("DATABASE_URL environment variable not set")
    return settings.database_url


def get_redis_url() -> str:
    """
    Get Redis URL with proper error handling.
    
    Returns:
        str: Redis connection string
        
    Raises:
        ValueError: If Redis URL is not configured
    """
    if not settings.redis_url:
        raise ValueError("REDIS_URL environment variable not set")
    return settings.redis_url


def validate_twilio_config() -> bool:
    """
    Validate Twilio configuration for voice processing.
    
    Returns:
        bool: True if Twilio is properly configured
    """
    return bool(settings.twilio_account_sid and settings.twilio_auth_token)