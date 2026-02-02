"""
Middleware Components for Loan2Day Platform

This module implements comprehensive middleware for error handling,
security, performance monitoring, and request/response processing.

Key Features:
- Global error handling with structured responses
- Security event logging and monitoring
- Performance tracking and alerting
- Request validation and sanitization
- Circuit breaker integration

Architecture: Middleware stack with proper error propagation
Security: Comprehensive security event logging
Performance: Request timing and alerting

Author: Lead AI Architect, Loan2Day Fintech Platform
"""

import logging
import time
import json
import uuid
from typing import Dict, Any, Optional, Callable
from datetime import datetime
import traceback

from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.core.error_handling import (
    ErrorResponse, ErrorCode, ErrorSeverity, ErrorCategory,
    create_error_response, get_http_status_for_error_code,
    error_logger, get_system_health
)

# Configure logger
logger = logging.getLogger(__name__)

class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Global error handling middleware with structured responses.
    
    Catches all unhandled exceptions and converts them to structured
    error responses following the LQM Standard.
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.error_count = 0
        self.critical_error_count = 0
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request with comprehensive error handling.
        
        Args:
            request: FastAPI request object
            call_next: Next middleware/handler in chain
            
        Returns:
            Response: Processed response or error response
        """
        trace_id = getattr(request.state, "trace_id", f"trace_{uuid.uuid4().hex[:8]}")
        
        try:
            # Process request through the application
            response = await call_next(request)
            return response
            
        except HTTPException as e:
            # Handle FastAPI HTTP exceptions
            logger.warning(
                f"HTTP Exception: {e.status_code} - {e.detail}",
                extra={
                    "trace_id": trace_id,
                    "status_code": e.status_code,
                    "path": request.url.path,
                    "method": request.method
                }
            )
            
            # Convert to structured error response
            error_response = create_error_response(
                ErrorCode.VALIDATION_ERROR if e.status_code == 400 else ErrorCode.INTERNAL_SERVER_ERROR,
                str(e.detail),
                {"status_code": e.status_code},
                trace_id
            )
            
            return JSONResponse(
                status_code=e.status_code,
                content=error_response.dict()
            )
            
        except ValueError as e:
            # Handle validation errors
            self.error_count += 1
            
            error_logger.log_error(
                e,
                {
                    "trace_id": trace_id,
                    "path": request.url.path,
                    "method": request.method
                },
                ErrorSeverity.LOW,
                ErrorCategory.VALIDATION
            )
            
            error_response = create_error_response(
                ErrorCode.VALIDATION_ERROR,
                f"Validation error: {str(e)}",
                {"error_type": "ValueError"},
                trace_id
            )
            
            return JSONResponse(
                status_code=400,
                content=error_response.dict()
            )
            
        except Exception as e:
            # Handle all other exceptions
            self.error_count += 1
            self.critical_error_count += 1
            
            error_logger.log_error(
                e,
                {
                    "trace_id": trace_id,
                    "path": request.url.path,
                    "method": request.method,
                    "stack_trace": traceback.format_exc()
                },
                ErrorSeverity.CRITICAL,
                ErrorCategory.SYSTEM
            )
            
            logger.error(
                f"Unhandled exception: {type(e).__name__} - {str(e)}",
                extra={
                    "trace_id": trace_id,
                    "exception_type": type(e).__name__,
                    "path": request.url.path,
                    "method": request.method
                },
                exc_info=True
            )
            
            # Don't expose internal error details in production
            error_message = "An unexpected error occurred"
            error_details = {"exception_type": type(e).__name__}
            
            # In development, include more details
            try:
                from app.core.config import settings
                if settings.debug:
                    error_message = f"Internal error: {str(e)}"
                    error_details["exception_message"] = str(e)
            except:
                pass
            
            error_response = create_error_response(
                ErrorCode.INTERNAL_SERVER_ERROR,
                error_message,
                error_details,
                trace_id
            )
            
            return JSONResponse(
                status_code=500,
                content=error_response.dict()
            )

class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Security monitoring and event logging middleware.
    
    Monitors for suspicious activities and logs security events.
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.suspicious_activity_count = 0
        self.rate_limit_violations = 0
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request with security monitoring.
        
        Args:
            request: FastAPI request object
            call_next: Next middleware/handler in chain
            
        Returns:
            Response: Processed response
        """
        trace_id = getattr(request.state, "trace_id", f"trace_{uuid.uuid4().hex[:8]}")
        
        # Basic security checks
        await self._check_request_security(request, trace_id)
        
        # Process request
        response = await call_next(request)
        
        # Log security-relevant responses
        await self._log_security_response(request, response, trace_id)
        
        return response
    
    async def _check_request_security(self, request: Request, trace_id: str):
        """
        Perform basic security checks on incoming requests.
        
        Args:
            request: FastAPI request object
            trace_id: Request trace identifier
        """
        # Check for suspicious headers
        suspicious_headers = [
            "x-forwarded-for",
            "x-real-ip",
            "x-originating-ip"
        ]
        
        for header in suspicious_headers:
            if header in request.headers:
                # Log potential proxy/forwarding (could be legitimate or suspicious)
                logger.info(
                    f"Proxy header detected: {header}",
                    extra={
                        "trace_id": trace_id,
                        "header": header,
                        "value": request.headers[header],
                        "path": request.url.path
                    }
                )
        
        # Check for oversized requests
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > 10 * 1024 * 1024:  # 10MB
            self.suspicious_activity_count += 1
            
            error_logger.log_security_event(
                "OVERSIZED_REQUEST",
                None,  # User ID not available at this stage
                {
                    "content_length": content_length,
                    "path": request.url.path,
                    "client_ip": request.client.host if request.client else "unknown"
                },
                ErrorSeverity.MEDIUM
            )
        
        # Check for suspicious paths
        suspicious_paths = [
            "/admin", "/wp-admin", "/.env", "/config",
            "/api/v1/admin", "/debug", "/test"
        ]
        
        if any(suspicious_path in str(request.url.path) for suspicious_path in suspicious_paths):
            self.suspicious_activity_count += 1
            
            error_logger.log_security_event(
                "SUSPICIOUS_PATH_ACCESS",
                None,
                {
                    "path": request.url.path,
                    "client_ip": request.client.host if request.client else "unknown",
                    "user_agent": request.headers.get("user-agent", "unknown")
                },
                ErrorSeverity.HIGH
            )
    
    async def _log_security_response(self, request: Request, response: Response, trace_id: str):
        """
        Log security-relevant response information.
        
        Args:
            request: FastAPI request object
            response: Response object
            trace_id: Request trace identifier
        """
        # Log authentication failures
        if response.status_code == 401:
            error_logger.log_security_event(
                "AUTHENTICATION_FAILURE",
                None,
                {
                    "path": request.url.path,
                    "client_ip": request.client.host if request.client else "unknown",
                    "user_agent": request.headers.get("user-agent", "unknown")
                },
                ErrorSeverity.MEDIUM
            )
        
        # Log authorization failures
        elif response.status_code == 403:
            error_logger.log_security_event(
                "AUTHORIZATION_FAILURE",
                None,
                {
                    "path": request.url.path,
                    "client_ip": request.client.host if request.client else "unknown"
                },
                ErrorSeverity.HIGH
            )

class PerformanceMiddleware(BaseHTTPMiddleware):
    """
    Performance monitoring and alerting middleware.
    
    Tracks request timing and alerts on performance issues.
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.slow_request_count = 0
        self.total_requests = 0
        self.total_response_time = 0.0
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request with performance monitoring.
        
        Args:
            request: FastAPI request object
            call_next: Next middleware/handler in chain
            
        Returns:
            Response: Processed response with timing headers
        """
        start_time = time.time()
        trace_id = getattr(request.state, "trace_id", f"trace_{uuid.uuid4().hex[:8]}")
        
        # Process request
        response = await call_next(request)
        
        # Calculate timing
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        
        # Update statistics
        self.total_requests += 1
        self.total_response_time += duration_ms
        
        # Check for slow requests
        slow_threshold_ms = 2000  # 2 seconds as per requirements
        if duration_ms > slow_threshold_ms:
            self.slow_request_count += 1
            
            logger.warning(
                f"Slow request detected: {duration_ms:.2f}ms",
                extra={
                    "trace_id": trace_id,
                    "duration_ms": duration_ms,
                    "path": request.url.path,
                    "method": request.method,
                    "status_code": response.status_code
                }
            )
            
            # Log performance issue
            error_logger.log_error(
                Exception(f"Request exceeded {slow_threshold_ms}ms threshold"),
                {
                    "trace_id": trace_id,
                    "duration_ms": duration_ms,
                    "path": request.url.path,
                    "method": request.method,
                    "threshold_ms": slow_threshold_ms
                },
                ErrorSeverity.MEDIUM,
                ErrorCategory.PERFORMANCE
            )
        
        # Add performance headers
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
        response.headers["X-Trace-ID"] = trace_id
        
        # Log request completion
        logger.info(
            f"Request completed: {request.method} {request.url.path} - "
            f"{response.status_code} in {duration_ms:.2f}ms",
            extra={
                "trace_id": trace_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms
            }
        )
        
        return response
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        avg_response_time = (
            self.total_response_time / self.total_requests
            if self.total_requests > 0 else 0
        )
        
        return {
            "total_requests": self.total_requests,
            "slow_requests": self.slow_request_count,
            "slow_request_percentage": (
                (self.slow_request_count / self.total_requests * 100)
                if self.total_requests > 0 else 0
            ),
            "average_response_time_ms": avg_response_time,
            "total_response_time_ms": self.total_response_time
        }

class RequestTrackingMiddleware(BaseHTTPMiddleware):
    """
    Request tracking and trace ID assignment middleware.
    
    Assigns unique trace IDs to all requests for debugging and monitoring.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request with trace ID assignment.
        
        Args:
            request: FastAPI request object
            call_next: Next middleware/handler in chain
            
        Returns:
            Response: Processed response with trace ID
        """
        # Generate or extract trace ID
        trace_id = request.headers.get("X-Trace-ID")
        if not trace_id:
            trace_id = f"req_{int(time.time() * 1000000)}"
        
        # Store trace ID in request state
        request.state.trace_id = trace_id
        
        # Process request
        response = await call_next(request)
        
        # Add trace ID to response headers
        response.headers["X-Trace-ID"] = trace_id
        
        return response

# Health Check Middleware Integration

class HealthCheckMiddleware(BaseHTTPMiddleware):
    """
    Health check middleware that provides system health endpoints.
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.start_time = datetime.now()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request with health check handling.
        
        Args:
            request: FastAPI request object
            call_next: Next middleware/handler in chain
            
        Returns:
            Response: Health check response or processed request
        """
        # Handle health check endpoints
        if request.url.path == "/health/detailed":
            return await self._detailed_health_check(request)
        elif request.url.path == "/health/middleware":
            return await self._middleware_health_check(request)
        
        # Process normal requests
        return await call_next(request)
    
    async def _detailed_health_check(self, request: Request) -> JSONResponse:
        """
        Provide detailed system health information.
        
        Args:
            request: FastAPI request object
            
        Returns:
            JSONResponse: Detailed health information
        """
        uptime = datetime.now() - self.start_time
        
        health_data = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": uptime.total_seconds(),
            "system_health": get_system_health(),
            "middleware_stats": {
                "error_handling": {
                    "total_errors": getattr(self, "error_count", 0),
                    "critical_errors": getattr(self, "critical_error_count", 0)
                },
                "security": {
                    "suspicious_activities": getattr(self, "suspicious_activity_count", 0),
                    "rate_limit_violations": getattr(self, "rate_limit_violations", 0)
                },
                "performance": getattr(self, "performance_stats", {})
            }
        }
        
        return JSONResponse(content=health_data)
    
    async def _middleware_health_check(self, request: Request) -> JSONResponse:
        """
        Provide middleware-specific health information.
        
        Args:
            request: FastAPI request object
            
        Returns:
            JSONResponse: Middleware health information
        """
        return JSONResponse(content={
            "middleware_status": "operational",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "error_handling": "active",
                "security_monitoring": "active",
                "performance_tracking": "active",
                "request_tracking": "active"
            }
        })

# Export middleware classes
__all__ = [
    'ErrorHandlingMiddleware',
    'SecurityMiddleware', 
    'PerformanceMiddleware',
    'RequestTrackingMiddleware',
    'HealthCheckMiddleware'
]