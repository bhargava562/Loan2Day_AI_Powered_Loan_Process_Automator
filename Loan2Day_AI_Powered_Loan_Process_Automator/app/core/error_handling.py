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
        description="Request trace identifier for debugging"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session identifier if available"
    )
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

# Circuit Breaker Implementation

@dataclass
class CircuitBreakerState:
    """Circuit breaker state tracking."""
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    state: str = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    success_count: int = 0

class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""
    pass

class CircuitBreaker:
    """
    Circuit breaker pattern implementation for external service protection.
    
    Prevents cascading failures by temporarily blocking calls to failing
    external services and allowing graceful degradation.
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        success_threshold: int = 3
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.state = CircuitBreakerState()
        
        logger.info(
            f"Circuit breaker initialized: {name} - "
            f"Failure threshold: {failure_threshold}, "
            f"Recovery timeout: {recovery_timeout}s"
        )
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if self.state.state != "OPEN":
            return False
        
        if not self.state.last_failure_time:
            return True
        
        time_since_failure = (datetime.now() - self.state.last_failure_time).total_seconds()
        return time_since_failure >= self.recovery_timeout
    
    def _record_success(self):
        """Record successful operation."""
        if self.state.state == "HALF_OPEN":
            self.state.success_count += 1
            if self.state.success_count >= self.success_threshold:
                self.state.state = "CLOSED"
                self.state.failure_count = 0
                self.state.success_count = 0
                logger.info(f"Circuit breaker {self.name} reset to CLOSED")
        elif self.state.state == "CLOSED":
            self.state.failure_count = 0
    
    def _record_failure(self):
        """Record failed operation."""
        self.state.failure_count += 1
        self.state.last_failure_time = datetime.now()
        
        if self.state.failure_count >= self.failure_threshold:
            self.state.state = "OPEN"
            self.state.success_count = 0
            logger.warning(
                f"Circuit breaker {self.name} opened after "
                f"{self.state.failure_count} failures"
            )
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Any: Function result
            
        Raises:
            CircuitBreakerError: If circuit breaker is open
        """
        # Check if we should attempt reset
        if self._should_attempt_reset():
            self.state.state = "HALF_OPEN"
            self.state.success_count = 0
            logger.info(f"Circuit breaker {self.name} attempting reset (HALF_OPEN)")
        
        # Block calls if circuit is open
        if self.state.state == "OPEN":
            raise CircuitBreakerError(
                f"Circuit breaker {self.name} is OPEN. "
                f"Last failure: {self.state.last_failure_time}"
            )
        
        try:
            # Execute function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            self._record_success()
            return result
            
        except Exception as e:
            self._record_failure()
            raise e
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        return {
            "name": self.name,
            "state": self.state.state,
            "failure_count": self.state.failure_count,
            "success_count": self.state.success_count,
            "last_failure_time": self.state.last_failure_time.isoformat() if self.state.last_failure_time else None,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout
        }

# Retry Handler with Exponential Backoff

class RetryHandler:
    """
    Retry handler with exponential backoff for transient failures.
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
    
    async def execute_with_retry(
        self,
        func: Callable,
        *args,
        retryable_exceptions: tuple = (Exception,),
        **kwargs
    ) -> Any:
        """
        Execute function with retry logic.
        
        Args:
            func: Function to execute
            *args: Function arguments
            retryable_exceptions: Exceptions that should trigger retry
            **kwargs: Function keyword arguments
            
        Returns:
            Any: Function result
            
        Raises:
            Exception: Last exception if all retries failed
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
                    
            except retryable_exceptions as e:
                last_exception = e
                
                if attempt == self.max_retries:
                    logger.error(
                        f"All {self.max_retries} retry attempts failed for {func.__name__}: {str(e)}"
                    )
                    break
                
                # Calculate delay with exponential backoff
                delay = min(
                    self.base_delay * (self.backoff_factor ** attempt),
                    self.max_delay
                )
                
                logger.warning(
                    f"Attempt {attempt + 1} failed for {func.__name__}: {str(e)}. "
                    f"Retrying in {delay:.2f}s..."
                )
                
                await asyncio.sleep(delay)
        
        raise last_exception

# Agent Failure Handler

class AgentFailureHandler:
    """
    Handles graceful degradation for agent failures.
    """
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_handler = RetryHandler()
    
    def get_circuit_breaker(self, agent_name: str) -> CircuitBreaker:
        """Get or create circuit breaker for agent."""
        if agent_name not in self.circuit_breakers:
            self.circuit_breakers[agent_name] = CircuitBreaker(
                name=f"agent_{agent_name}",
                failure_threshold=3,
                recovery_timeout=30
            )
        return self.circuit_breakers[agent_name]
    
    async def execute_agent_task(
        self,
        agent_name: str,
        task_func: Callable,
        fallback_func: Optional[Callable] = None,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute agent task with circuit breaker and fallback.
        
        Args:
            agent_name: Name of the agent
            task_func: Primary task function
            fallback_func: Fallback function if primary fails
            *args: Task arguments
            **kwargs: Task keyword arguments
            
        Returns:
            Dict[str, Any]: Task result with metadata
        """
        circuit_breaker = self.get_circuit_breaker(agent_name)
        
        try:
            # Try primary function with circuit breaker
            result = await circuit_breaker.call(task_func, *args, **kwargs)
            
            return {
                "success": True,
                "result": result,
                "agent": agent_name,
                "execution_path": "primary",
                "circuit_breaker_state": circuit_breaker.get_state()
            }
            
        except CircuitBreakerError as e:
            logger.warning(f"Circuit breaker open for {agent_name}: {str(e)}")
            
            # Try fallback if available
            if fallback_func:
                try:
                    fallback_result = await self._execute_fallback(
                        fallback_func, *args, **kwargs
                    )
                    
                    return {
                        "success": True,
                        "result": fallback_result,
                        "agent": agent_name,
                        "execution_path": "fallback",
                        "circuit_breaker_state": circuit_breaker.get_state(),
                        "fallback_reason": "circuit_breaker_open"
                    }
                    
                except Exception as fallback_error:
                    logger.error(f"Fallback failed for {agent_name}: {str(fallback_error)}")
            
            # Return graceful degradation response
            return {
                "success": False,
                "error": str(e),
                "agent": agent_name,
                "execution_path": "degraded",
                "circuit_breaker_state": circuit_breaker.get_state(),
                "degradation_reason": "circuit_breaker_open_no_fallback"
            }
            
        except Exception as e:
            logger.error(f"Agent task failed for {agent_name}: {str(e)}")
            
            # Try fallback for other exceptions
            if fallback_func:
                try:
                    fallback_result = await self._execute_fallback(
                        fallback_func, *args, **kwargs
                    )
                    
                    return {
                        "success": True,
                        "result": fallback_result,
                        "agent": agent_name,
                        "execution_path": "fallback",
                        "circuit_breaker_state": circuit_breaker.get_state(),
                        "fallback_reason": "primary_task_failed"
                    }
                    
                except Exception as fallback_error:
                    logger.error(f"Fallback failed for {agent_name}: {str(fallback_error)}")
            
            return {
                "success": False,
                "error": str(e),
                "agent": agent_name,
                "execution_path": "failed",
                "circuit_breaker_state": circuit_breaker.get_state()
            }
    
    async def _execute_fallback(self, fallback_func: Callable, *args, **kwargs) -> Any:
        """Execute fallback function with retry."""
        return await self.retry_handler.execute_with_retry(
            fallback_func,
            *args,
            **kwargs
        )
    
    def get_all_circuit_breaker_states(self) -> Dict[str, Dict[str, Any]]:
        """Get states of all circuit breakers."""
        return {
            name: cb.get_state()
            for name, cb in self.circuit_breakers.items()
        }

# Error Logger with Structured Context

class ErrorLogger:
    """
    Structured error logging with context enrichment.
    """
    
    def __init__(self, logger_name: str = __name__):
        self.logger = logging.getLogger(logger_name)
    
    def log_error(
        self,
        error: Exception,
        context: Dict[str, Any],
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.SYSTEM
    ):
        """
        Log error with structured context.
        
        Args:
            error: Exception that occurred
            context: Additional context information
            severity: Error severity level
            category: Error category
        """
        error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "severity": severity.value,
            "category": category.value,
            "timestamp": datetime.now().isoformat(),
            "context": context
        }
        
        # Add stack trace for high severity errors
        if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            error_info["stack_trace"] = traceback.format_exc()
        
        # Log at appropriate level based on severity
        if severity == ErrorSeverity.CRITICAL:
            self.logger.critical("Critical error occurred", extra=error_info)
        elif severity == ErrorSeverity.HIGH:
            self.logger.error("High severity error occurred", extra=error_info)
        elif severity == ErrorSeverity.MEDIUM:
            self.logger.warning("Medium severity error occurred", extra=error_info)
        else:
            self.logger.info("Low severity error occurred", extra=error_info)
    
    def log_security_event(
        self,
        event_type: str,
        user_id: Optional[str],
        details: Dict[str, Any],
        severity: ErrorSeverity = ErrorSeverity.HIGH
    ):
        """
        Log security-related events.
        
        Args:
            event_type: Type of security event
            user_id: User identifier if available
            details: Event details
            severity: Event severity
        """
        security_event = {
            "event_type": "SECURITY_EVENT",
            "security_event_type": event_type,
            "user_id": user_id,
            "severity": severity.value,
            "timestamp": datetime.now().isoformat(),
            "details": details
        }
        
        self.logger.warning("Security event detected", extra=security_event)

# Global Error Handler Instances

# Initialize global instances
agent_failure_handler = AgentFailureHandler()
error_logger = ErrorLogger()

# Circuit breakers for external services
external_service_circuit_breakers = {
    "mock_bank": CircuitBreaker("mock_bank_api", failure_threshold=5, recovery_timeout=60),
    "sgs_module": CircuitBreaker("sgs_security_scan", failure_threshold=3, recovery_timeout=30),
    "redis_cache": CircuitBreaker("redis_cache", failure_threshold=5, recovery_timeout=30),
    "database": CircuitBreaker("database", failure_threshold=3, recovery_timeout=60)
}

# Utility Functions

def create_error_response(
    error_code: ErrorCode,
    error_message: str,
    error_details: Optional[Dict[str, Any]] = None,
    trace_id: Optional[str] = None,
    session_id: Optional[str] = None
) -> ErrorResponse:
    """
    Create standardized error response.
    
    Args:
        error_code: Standardized error code
        error_message: Human-readable error message
        error_details: Additional error context
        trace_id: Request trace identifier
        session_id: Session identifier if available
        
    Returns:
        ErrorResponse: Standardized error response
    """
    return ErrorResponse(
        error_code=error_code,
        error_message=error_message,
        error_details=error_details or {},
        timestamp=datetime.now(),
        trace_id=trace_id or f"trace_{uuid.uuid4().hex[:8]}",
        session_id=session_id
    )

def get_http_status_for_error_code(error_code: ErrorCode) -> int:
    """
    Get appropriate HTTP status code for error code.
    
    Args:
        error_code: Error code
        
    Returns:
        int: HTTP status code
    """
    status_mapping = {
        # 400 Bad Request
        ErrorCode.VALIDATION_ERROR: 400,
        ErrorCode.INVALID_INPUT: 400,
        ErrorCode.MISSING_REQUIRED_FIELD: 400,
        ErrorCode.INVALID_FORMAT: 400,
        
        # 401 Unauthorized
        ErrorCode.AUTHENTICATION_FAILED: 401,
        ErrorCode.SESSION_EXPIRED: 401,
        ErrorCode.INVALID_TOKEN: 401,
        
        # 403 Forbidden
        ErrorCode.AUTHORIZATION_FAILED: 403,
        ErrorCode.SECURITY_SCAN_FAILED: 403,
        ErrorCode.SUSPICIOUS_ACTIVITY: 403,
        ErrorCode.FILE_SECURITY_VIOLATION: 403,
        
        # 422 Unprocessable Entity
        ErrorCode.BUSINESS_RULE_VIOLATION: 422,
        ErrorCode.INSUFFICIENT_FUNDS: 422,
        ErrorCode.LOAN_ELIGIBILITY_FAILED: 422,
        ErrorCode.KYC_VERIFICATION_FAILED: 422,
        
        # 429 Too Many Requests
        ErrorCode.RATE_LIMIT_EXCEEDED: 429,
        
        # 500 Internal Server Error
        ErrorCode.INTERNAL_SERVER_ERROR: 500,
        ErrorCode.AGENT_ORCHESTRATION_FAILED: 500,
        ErrorCode.PROCESSING_ERROR: 500,
        ErrorCode.CONFIGURATION_ERROR: 500,
        ErrorCode.DATABASE_CONNECTION_ERROR: 500,
        ErrorCode.DATABASE_QUERY_ERROR: 500,
        ErrorCode.DATABASE_TIMEOUT: 500,
        
        # 502 Bad Gateway
        ErrorCode.EXTERNAL_SERVICE_ERROR: 502,
        
        # 503 Service Unavailable
        ErrorCode.EXTERNAL_SERVICE_UNAVAILABLE: 503,
        ErrorCode.EXTERNAL_SERVICE_TIMEOUT: 503,
        ErrorCode.CIRCUIT_BREAKER_OPEN: 503
    }
    
    return status_mapping.get(error_code, 500)

async def handle_external_service_call(
    service_name: str,
    func: Callable,
    *args,
    **kwargs
) -> Any:
    """
    Handle external service call with circuit breaker protection.
    
    Args:
        service_name: Name of the external service
        func: Function to call
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Any: Service response
        
    Raises:
        HTTPException: If service is unavailable or fails
    """
    if service_name not in external_service_circuit_breakers:
        # Create circuit breaker for unknown service
        external_service_circuit_breakers[service_name] = CircuitBreaker(
            service_name, failure_threshold=5, recovery_timeout=60
        )
    
    circuit_breaker = external_service_circuit_breakers[service_name]
    
    try:
        return await circuit_breaker.call(func, *args, **kwargs)
        
    except CircuitBreakerError as e:
        error_logger.log_error(
            e,
            {"service": service_name, "circuit_breaker_state": circuit_breaker.get_state()},
            ErrorSeverity.HIGH,
            ErrorCategory.EXTERNAL_SERVICE
        )
        
        raise HTTPException(
            status_code=503,
            detail=create_error_response(
                ErrorCode.CIRCUIT_BREAKER_OPEN,
                f"Service {service_name} is temporarily unavailable",
                {"service": service_name, "retry_after": circuit_breaker.recovery_timeout}
            ).dict()
        )
    
    except Exception as e:
        error_logger.log_error(
            e,
            {"service": service_name},
            ErrorSeverity.HIGH,
            ErrorCategory.EXTERNAL_SERVICE
        )
        
        raise HTTPException(
            status_code=502,
            detail=create_error_response(
                ErrorCode.EXTERNAL_SERVICE_ERROR,
                f"External service {service_name} error: {str(e)}",
                {"service": service_name}
            ).dict()
        )

# Health Check Functions

def get_system_health() -> Dict[str, Any]:
    """
    Get comprehensive system health status.
    
    Returns:
        Dict[str, Any]: System health information
    """
    return {
        "timestamp": datetime.now().isoformat(),
        "circuit_breakers": {
            name: cb.get_state()
            for name, cb in external_service_circuit_breakers.items()
        },
        "agent_circuit_breakers": agent_failure_handler.get_all_circuit_breaker_states(),
        "error_counts": {
            # This would be implemented with proper metrics collection
            "total_errors_last_hour": 0,
            "critical_errors_last_hour": 0,
            "circuit_breaker_trips_last_hour": 0
        }
    }

# Export all components
__all__ = [
    # Enums
    'ErrorSeverity',
    'ErrorCategory', 
    'ErrorCode',
    
    # Models
    'ErrorResponse',
    
    # Circuit Breaker
    'CircuitBreaker',
    'CircuitBreakerError',
    
    # Handlers
    'RetryHandler',
    'AgentFailureHandler',
    'ErrorLogger',
    
    # Global Instances
    'agent_failure_handler',
    'error_logger',
    'external_service_circuit_breakers',
    
    # Utility Functions
    'create_error_response',
    'get_http_status_for_error_code',
    'handle_external_service_call',
    'get_system_health'
]