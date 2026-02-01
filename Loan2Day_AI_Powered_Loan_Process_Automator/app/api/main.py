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
    Application lifespan manager for startup/shutdown tasks.
    
    Handles:
    - Database connection initialization
    - Redis connection setup
    - Kafka producer initialization
    - Graceful shutdown procedures
    """
    logger.info("🚀 Loan2Day platform starting up...")
    
    # TODO: Initialize database connections
    # TODO: Initialize Redis connection
    # TODO: Initialize Kafka producer
    
    logger.info("✅ Loan2Day platform ready to serve requests")
    
    yield
    
    logger.info("🛑 Loan2Day platform shutting down...")
    
    # TODO: Close database connections
    # TODO: Close Redis connection
    # TODO: Close Kafka producer
    
    logger.info("✅ Loan2Day platform shutdown complete")


# Initialize FastAPI application with proper configuration
app = FastAPI(
    title="Loan2Day Agentic AI Fintech Platform",
    description="Master-Worker Agent architecture for intelligent loan processing",
    version=settings.version,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    lifespan=lifespan
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
    """
    start_time = datetime.now()
    
    # Generate trace ID for request tracking
    trace_id = f"req_{int(start_time.timestamp() * 1000000)}"
    request.state.trace_id = trace_id
    
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
    
    end_time = datetime.now()
    duration_ms = (end_time - start_time).total_seconds() * 1000
    
    logger.info(
        f"✅ Request completed: {response.status_code} in {duration_ms:.2f}ms",
        extra={
            "trace_id": trace_id,
            "status_code": response.status_code,
            "duration_ms": duration_ms
        }
    )
    
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Global HTTP exception handler.
    
    Ensures consistent error response format across all endpoints.
    """
    trace_id = getattr(request.state, "trace_id", "unknown")
    
    error_response = ErrorResponse(
        error_code=f"HTTP_{exc.status_code}",
        error_message=exc.detail,
        error_details={"status_code": exc.status_code},
        timestamp=datetime.now(),
        trace_id=trace_id
    )
    
    logger.error(
        f"❌ HTTP Exception: {exc.status_code} - {exc.detail}",
        extra={
            "trace_id": trace_id,
            "status_code": exc.status_code,
            "detail": exc.detail
        }
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for unhandled errors.
    
    Provides graceful degradation and proper error logging.
    """
    trace_id = getattr(request.state, "trace_id", "unknown")
    
    error_response = ErrorResponse(
        error_code="INTERNAL_SERVER_ERROR",
        error_message="An unexpected error occurred",
        error_details={"exception_type": type(exc).__name__},
        timestamp=datetime.now(),
        trace_id=trace_id
    )
    
    logger.error(
        f"💥 Unhandled Exception: {type(exc).__name__} - {str(exc)}",
        extra={
            "trace_id": trace_id,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc)
        },
        exc_info=True
    )
    
    return JSONResponse(
        status_code=500,
        content=error_response.dict()
    )


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
    
    # TODO: Add actual service health checks
    services_status = {
        "database": "healthy",  # TODO: Implement actual DB health check
        "redis": "healthy",     # TODO: Implement actual Redis health check
        "kafka": "healthy",     # TODO: Implement actual Kafka health check
        "sgs_module": "healthy", # TODO: Implement SGS health check
        "lqm_module": "healthy"  # TODO: Implement LQM health check
    }
    
    return HealthCheckResponse(
        status="healthy",
        timestamp=datetime.now(),
        version=settings.version,
        environment="development" if settings.debug else "production",
        services=services_status
    )


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