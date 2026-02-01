"""
Comprehensive Error Handling System for Loan2Day Platform

This module implements a robust error handling framework with:
- Structured error responses with proper HTTP status codes
- Circuit breaker pattern for external services
- Graceful degradation for agent failures
- Detailed logging for debugging
- Error classification and recovery strategies

Key Features:
- ErrorResponse standardization across all endpoints
- CircuitBreaker for external service protection
- AgentFailureHandler for graceful degradation
- ErrorLogger with structured context
- RetryHandler with exponential backoff

Architecture: Centralized error handling with component-specific handlers
Security: No sensitive data exposure in error responses
Performance: Fast error responses with proper status codes

Author: Lead AI Architect, Loan2Day Fintech Platform
"""

import logging
import time
import asyncio
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
import traceback
import uuid

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Configure logger
logger = logging.getLogger(__name__)

# Error Classification

class ErrorSeverity(str, Enum):
    """Error severity levels for proper handling and alerting."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class ErrorCategory(str, Enum):
    """Error categories for classification and handling."""
    VALIDATION = "VALIDATION"
    AUTHENTICATION = "AUTHENTICATION"
    AUTHORIZATION = "AUTHORIZATION"
    BUSINESS_LOGIC = "BUSINESS_LOGIC"
    EXTERNAL_SERVICE = "EXTERNAL_SERVICE"
    DATABASE = "DATABASE"
    SYSTEM = "SYSTEM"
    SECURITY = "SECURITY"
    PERFORMANCE = "PERFORMANCE"

class ErrorCode(str, Enum):
    """Standardized error codes for consistent error handling."""
    # Validation Errors (400)
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INVALID_INPUT = "INVALID_INPUT"
    MISSING_REQUIRED_FIELD = "MISSING_REQUIRED_FIELD"
    INVALID_FORMAT = "INVALID_FORMAT"
    
    # Authentication/Authorization Errors (401/403)
    AUTHENTICATION_FAILED = "AUTHENTICATION_FAILED"
    AUTHORIZATION_FAILED = "AUTHORIZATION_FAILED"
    SESSION_EXPIRED = "SESSION_EXPIRED"
    INVALID_TOKEN = "INVALID_TOKEN"
    
    # Business Logic Errors (422)
    BUSINESS_RULE_VIOLATION = "BUSINESS_RULE_VIOLATION"
    INSUFFICIENT_FUNDS = "INSUFFICIENT_FUNDS"
    LOAN_ELIGIBILITY_FAILED = "LOAN_ELIGIBILITY_FAILED"
    KYC_VERIFICATION_FAILED = "KYC_VERIFICATION_FAILED"
    
    # External Service Errors (502/503)
    EXTERNAL_SERVICE_UNAVAILABLE = "EXTERNAL_SERVICE_UNAVAILABLE"
    EXTERNAL_SERVICE_TIMEOUT = "EXTERNAL_SERVICE_TIMEOUT"
    EXTERNAL_SERVICE_ERROR = "EXTERNAL_SERVICE_ERROR"
    CIRCUIT_BREAKER_OPEN = "CIRCUIT_BREAKER_OPEN"
    
    # Database Errors (500)
    DATABASE_CONNECTION_ERROR = "DATABASE_CONNECTION_ERROR"
    DATABASE_QUERY_ERROR = "DATABASE_QUERY_ERROR"
    DATABASE_TIMEOUT = "DATABASE_TIMEOUT"
    
    # System Errors (500)
    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"
    AGENT_ORCHESTRATION_FAILED = "AGENT_ORCHESTRATION_FAILED"
    PROCESSING_ERROR = "PROCESSING_ERROR"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    
    # Security Errors (403/429)
    SECURITY_SCAN_FAILED = "SECURITY_SCAN_FAILED"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    SUSPICIOUS_ACTIVITY = "SUSPICIOUS_ACTIVITY"
    FILE_SECURITY_VIOLATION = "FILE_SECURITY_VIOLATION"

# Structured Error Response Models

class ErrorResponse(BaseModel):
    """
    Standardized error response model for consistent API responses.
    
    Provides structured error information without exposing sensitive data.
    """
    
    error_code: ErrorCode = Field(
        ...,
        description="Standardized error code for programmatic handling"
    )
    error_message: str = Field(
        ...,
        description="Human-readable error description"
    )
    error_details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional error context (sanitized)"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Error occurrence timestamp"
    )
    trace_id: str = Field(
        ...,
        descripti