"""
Main FastAPI application for Loan2Day Agentic AI Fintech Platform.

This module implements the API Gateway following the LQM Standard with:
- Zero-hallucination configuration management
- Structured logging (NO print statements)
- Async-first architecture
- Proper error handling and validation
"""

import logging
import sys
from contextlib import asynccontextmanager
from typing import Dict, Any
from datetime import datetime

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.middleware import (
    ErrorHandlingMiddleware,
    PerformanceMiddleware,
    RequestTrackingMiddleware,
    HealthCheckMiddleware
)
from app.core.security_middleware import create_security_middleware_stack
from app.core.error_handling import (
    ErrorResponse, ErrorCode, create_error_response,
    get_system_health, external_service_circuit_breakers
)
from app.core.security import audit_security_practices, security_event_logger
from app.core.dependencies import dependency_container, lifespan_manager


# Configure structured logging (LQM Standard: NO print statements)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("loan2day.log")
    ]
)

logger = logging.getLogger(__name__)


class HealthCheckResponse(BaseModel):
    """
    Health check response model with strict typing.
    
    Follows LQM Standard for structured API responses.
    """
    status: str = Field(description="Service health status")
    timestamp: datetime = Field(description="Current server timestamp")
    version: str = Field(description="Application version")
    environment: str = Field(description="Deployment environment")
    services: Dict[str, str] = Field(description="Dependent service status")


class ErrorResponse(BaseModel):
    """
    Standardized error response model.
    
    Ensures consistent error handling across all endpoints.
    """
    error_code: str = Field(description="Unique error identifier")
    error_message: str = Field(description="Human-readable error description")
    error_details: Dict[str, Any] = Field(description="Additional error context")
    timestamp: datetime = Field(description="Error occurrence timestamp")
    trace_id: str = Field(description="Request trace identifier")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    DEPRECATED: Application lifespan manager.
    
    This function is replaced by the DependencyContainer lifespan_manager.
    Kept for reference but not used.
    """
    # This is now handled by dependency_container.lifespan_manager
    yield


# Initialize FastAPI application with proper configuration
app = FastAPI(
    title="Loan2Day Agentic AI Fintech Platform",
    description="Master-Worker Agent architecture for intelligent loan processing",
    version=settings.version,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    lifespan=lifespan_manager
)

# Add comprehensive middleware stack
app.add_middleware(RequestTrackingMiddleware)
app.add_middleware(PerformanceMiddleware)
app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(HealthCheckMiddleware)

# Add security middleware stack (includes validation, rate limiting, audit)
create_security_middleware_stack(
    app,
    enable_rate_limiting=True,
    requests_per_minute=60,
    enable_validation=True,
    enable_audit=True
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Add trusted host middleware for security
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "testserver", "*.loan2day.com"]
)


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """
    Request/response logging middleware.
    
    Logs all API requests with timing information for monitoring.
    Follows LQM Standard: structured logging only.
    
    Note: This middleware works in conjunction with the comprehensive
    middleware stack for complete request processing.
    """
    # The trace ID is now set by RequestTrackingMiddleware
    trace_id = getattr(request.state, "trace_id", "unknown")
    
    logger.info(
        f"🔄 Request started: {request.method} {request.url.path}",
        extra={
            "trace_id": trace_id,
            "method": request.method,
            "path": request.url.path,
            "client_ip": request.client.host if request.client else "unknown"
        }
    )
    
    response = await call_next(request)
    
    # Performance tracking is now handled by PerformanceMiddleware
    # This middleware just logs the completion
    logger.info(
        f"✅ Request completed: {response.status_code}",
        extra={
            "trace_id": trace_id,
            "status_code": response.status_code
        }
    )
    
    return response


@app.get("/health", response_model=HealthCheckResponse)
async def health_check() -> HealthCheckResponse:
    """
    Health check endpoint for monitoring and load balancer probes.
    
    Returns:
        HealthCheckResponse: Current system health status
        
    This endpoint validates:
    - Application startup status
    - Database connectivity (TODO)
    - Redis connectivity (TODO)
    - Kafka connectivity (TODO)
    """
    logger.info("🏥 Health check requested")
    
    # Get circuit breaker states for service health
    circuit_breaker_states = {
        name: cb.get_state()
        for name, cb in external_service_circuit_breakers.items()
    }
    
    # Determine service health based on circuit breaker states
    services_status = {}
    for service_name, cb_state in circuit_breaker_states.items():
        if cb_state["state"] == "CLOSED":
            services_status[service_name] = "healthy"
        elif cb_state["state"] == "HALF_OPEN":
            services_status[service_name] = "recovering"
        else:  # OPEN
            services_status[service_name] = "unhealthy"
    
    # Add LQM and SGS module health (always healthy for now)
    services_status.update({
        "lqm_module": "healthy",
        "sgs_module": "healthy",
        "dependency_container": "healthy" if dependency_container._initialized else "unhealthy"
    })
    
    return HealthCheckResponse(
        status="healthy",
        timestamp=datetime.now(),
        version=settings.version,
        environment="development" if settings.debug else "production",
        services=services_status
    )


@app.get("/health/system")
async def system_health_check() -> Dict[str, Any]:
    """
    Comprehensive system health check with detailed information.
    
    Returns:
        Dict[str, Any]: Detailed system health status
    """
    logger.info("🔍 System health check requested")
    
    try:
        system_health = get_system_health()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": settings.version,
            "environment": "development" if settings.debug else "production",
            "detailed_health": system_health,
            "uptime_info": {
                "process_start_time": datetime.now().isoformat(),  # This would be actual start time
                "current_time": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"System health check failed: {str(e)}")
        
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "version": settings.version
        }


@app.get("/health/security")
async def security_audit() -> Dict[str, Any]:
    """
    Security audit endpoint for compliance checking.
    
    Returns:
        Dict[str, Any]: Security audit report
    """
    logger.info("🔒 Security audit requested")
    
    try:
        # Generate security audit report
        audit_report = audit_security_practices(".")
        
        # Get recent security events
        recent_events = security_event_logger.get_security_events(hours=24)
        
        return {
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "audit_report": audit_report,
            "recent_security_events": len(recent_events),
            "security_score": audit_report["summary"]["overall_score"],
            "recommendations": audit_report["recommendations"]
        }
        
    except Exception as e:
        logger.error(f"Security audit failed: {str(e)}")
        
        return {
            "status": "failed",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }


@app.get("/")
async def root() -> Dict[str, str]:
    """
    Root endpoint with basic platform information.
    
    Returns:
        Dict[str, str]: Platform welcome message and version
    """
    logger.info("🏠 Root endpoint accessed")
    
    return {
        "message": "Welcome to Loan2Day Agentic AI Fintech Platform",
        "version": settings.version,
        "documentation": "/docs" if settings.debug else "Contact administrator",
        "architecture": "Master-Worker Agent Pattern with LangGraph Orchestration"
    }


# Import and include route modules
from app.api.routes import chat, upload, plan_b, documents
app.include_router(chat.router, prefix="/v1/chat", tags=["chat"])
app.include_router(upload.router, prefix="/v1/upload", tags=["upload"])
app.include_router(plan_b.router, prefix="/v1/loan", tags=["plan-b"])
app.include_router(documents.router, prefix="/v1/documents", tags=["documents"])
# TODO: Add voice route when Twilio integration is implemented
# app.include_router(voice.router, prefix="/webhook", tags=["voice"])


if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Starting Loan2Day platform in development mode")
    
    uvicorn.run(
        "app.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level="info"
    )