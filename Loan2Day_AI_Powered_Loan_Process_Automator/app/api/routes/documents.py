"""
Document API Routes - PDF Generation and Secure Downloads

This module implements secure document generation and download endpoints
for sanction letters and other loan-related documents. It provides real-time
PDF generation upon loan approval with secure download links and proper
file cleanup mechanisms.

Key Features:
- POST /v1/documents/generate-sanction: Generate sanction letter PDF
- GET /v1/documents/download/{token}: Secure document download
- Secure token-based download links with expiration
- Real-time PDF generation upon loan approval
- Proper file cleanup and security measures

Architecture: Routes -> Services -> Repositories pattern
Security: Token-based secure downloads with expiration
Performance: Real-time generation with caching

Author: Lead AI Architect, Loan2Day Fintech Platform
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import hashlib
import secrets
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, validator

# Import core modules
from app.services.pdf_service import PDFService
from app.services.session_service import SessionService
from app.core.dependencies import get_pdf_service, get_session_service
from app.models.pydantic_models import AgentState
from app.core.config import settings

# Configure logger
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Remove global service initialization - now handled by dependency injection

# In-memory token store (in production, use Redis or database)
download_tokens: Dict[str, Dict[str, Any]] = {}

class DocumentError(Exception):
    """Base exception for document API errors."""
    pass

class GenerationError(DocumentError):
    """Raised when document generation fails."""
    pass

class TokenError(DocumentError):
    """Raised when download token is invalid or expired."""
    pass

# Request/Response Models

class SanctionGenerationRequest(BaseModel):
    """
    Request model for sanction letter generation.
    """
    
    session_id: str = Field(
        ...,
        min_length=1,
        description="Session identifier"
    )
    user_id: str = Field(
        ...,
        min_length=1,
        description="User identifier for authorization"
    )
    loan_id: Optional[str] = Field(
        default=None,
        description="Custom loan ID (optional, will be generated if not provided)"
    )
    include_processing_fee: bool = Field(
        default=True,
        description="Whether to include processing fee in the document"
    )
    include_insurance: bool = Field(
        default=False,
        description="Whether to include insurance premium"
    )
    
    @validator('session_id', 'user_id')
    def validate_ids(cls, v):
        """Validate IDs are not empty."""
        if not v or not v.strip():
            raise ValueError("ID cannot be empty")
        return v.strip()

class SanctionGenerationResponse(BaseModel):
    """
    Response model for sanction letter generation.
    """
    
    document_id: str = Field(
        ...,
        description="Unique document identifier"
    )
    loan_id: str = Field(
        ...,
        description="Loan reference number"
    )
    filename: str = Field(
        ...,
        description="Generated PDF filename"
    )
    file_size_bytes: int = Field(
        ...,
        description="File size in bytes"
    )
    download_token: str = Field(
        ...,
        description="Secure download token"
    )
    download_url: str = Field(
        ...,
        description="Secure download URL"
    )
    expires_at: datetime = Field(
        ...,
        description="Download link expiration time"
    )
    generated_at: datetime = Field(
        default_factory=datetime.now,
        description="Document generation timestamp"
    )

class DownloadInfo(BaseModel):
    """
    Download information response.
    """
    
    filename: str = Field(
        ...,
        description="Original filename"
    )
    file_size_bytes: int = Field(
        ...,
        description="File size in bytes"
    )
    content_type: str = Field(
        default="application/pdf",
        description="MIME content type"
    )
    generated_at: datetime = Field(
        ...,
        description="When the document was generated"
    )
    expires_at: datetime = Field(
        ...,
        description="When the download link expires"
    )

# Utility Functions

def generate_secure_token() -> str:
    """Generate cryptographically secure download token."""
    return secrets.token_urlsafe(32)

def create_download_token(
    file_path: str,
    filename: str,
    user_id: str,
    expires_in_hours: int = 24
) -> str:
    """
    Create secure download token with expiration.
    
    Args:
        file_path: Path to the file
        filename: Original filename
        user_id: User identifier for authorization
        expires_in_hours: Token expiration in hours
        
    Returns:
        str: Secure download token
    """
    token = generate_secure_token()
    expires_at = datetime.now() + timedelta(hours=expires_in_hours)
    
    download_tokens[token] = {
        "file_path": file_path,
        "filename": filename,
        "user_id": user_id,
        "created_at": datetime.now(),
        "expires_at": expires_at,
        "download_count": 0,
        "max_downloads": 5  # Allow up to 5 downloads
    }
    
    logger.info(f"Download token created: {token[:8]}... for user: {user_id}")
    return token

def validate_download_token(token: str, user_id: str) -> Dict[str, Any]:
    """
    Validate download token and return file information.
    
    Args:
        token: Download token
        user_id: User identifier for authorization
        
    Returns:
        Dict[str, Any]: Token information
        
    Raises:
        TokenError: If token is invalid or expired
    """
    if token not in download_tokens:
        raise TokenError("Invalid download token")
    
    token_info = download_tokens[token]
    
    # Check expiration
    if datetime.now() > token_info["expires_at"]:
        # Clean up expired token
        del download_tokens[token]
        raise TokenError("Download token has expired")
    
    # Check user authorization
    if token_info["user_id"] != user_id:
        raise TokenError("Unauthorized access to download token")
    
    # Check download limit
    if token_info["download_count"] >= token_info["max_downloads"]:
        raise TokenError("Download limit exceeded for this token")
    
    # Check if file still exists
    if not os.path.exists(token_info["file_path"]):
        # Clean up token for missing file
        del download_tokens[token]
        raise TokenError("File no longer available")
    
    return token_info

def cleanup_expired_tokens():
    """Clean up expired download tokens."""
    current_time = datetime.now()
    expired_tokens = [
        token for token, info in download_tokens.items()
        if current_time > info["expires_at"]
    ]
    
    for token in expired_tokens:
        del download_tokens[token]
    
    if expired_tokens:
        logger.info(f"Cleaned up {len(expired_tokens)} expired download tokens")

# Dependency Functions

async def get_pdf_service_dep() -> PDFService:
    """Dependency to get PDF service instance."""
    return await get_pdf_service()

async def get_session_service_dep() -> SessionService:
    """Dependency to get session service instance."""
    return await get_session_service()

# Route Handlers

@router.post(
    "/generate-sanction",
    response_model=SanctionGenerationResponse,
    summary="Generate sanction letter PDF",
    description="""
    Generate legally binding PDF sanction letter upon loan approval.
    
    This endpoint creates a professional sanction letter with verified data
    from the AgentState, including all required fields and regulatory
    disclosures. The generated PDF is made available through a secure
    download link with expiration.
    
    **Features:**
    - Real-time PDF generation using ReportLab
    - Verified data from AgentState (name, loan amount, EMI, interest rate)
    - Regulatory disclosures and legal terms
    - Secure download links with 24-hour expiration
    - Professional formatting with company branding
    
    **Requirements:**
    - Session must have completed loan approval
    - AgentState must contain user profile, loan request, and EMI calculation
    - User must be authorized for the session
    
    **Security:**
    - Token-based secure downloads
    - User authorization validation
    - File cleanup after expiration
    """,
    responses={
        200: {
            "description": "Sanction letter generated successfully",
            "content": {
                "application/json": {
                    "example": {
                        "document_id": "DOC_20240301_123456_LN20240301123456USER",
                        "loan_id": "LN20240301123456USER",
                        "filename": "sanction_letter_LN20240301123456USER_20240301_123456.pdf",
                        "file_size_bytes": 245760,
                        "download_token": "abc123def456ghi789jkl012mno345pqr678stu901vwx234yz",
                        "download_url": "/v1/documents/download/abc123def456ghi789jkl012mno345pqr678stu901vwx234yz",
                        "expires_at": "2024-03-02T12:34:56.789Z",
                        "generated_at": "2024-03-01T12:34:56.789Z"
                    }
                }
            }
        },
        400: {
            "description": "Invalid request or missing required data",
            "content": {
                "application/json": {
                    "example": {
                        "error_code": "MISSING_REQUIRED_DATA",
                        "error_message": "EMI calculation is required for sanction letter",
                        "error_details": {"missing_field": "emi_calculation"},
                        "timestamp": "2024-03-01T12:34:56.789Z",
                        "trace_id": "req_1234567890"
                    }
                }
            }
        },
        404: {
            "description": "Session not found or unauthorized",
            "content": {
                "application/json": {
                    "example": {
                        "error_code": "SESSION_NOT_FOUND",
                        "error_message": "Session not found or unauthorized",
                        "error_details": {"session_id": "sess_invalid"},
                        "timestamp": "2024-03-01T12:34:56.789Z",
                        "trace_id": "req_1234567890"
                    }
                }
            }
        }
    }
)
async def generate_sanction_letter(
    request: Request,
    generation_request: SanctionGenerationRequest,
    pdf_service: PDFService = Depends(get_pdf_service_dep),
    session_service: SessionService = Depends(get_session_service_dep)
) -> SanctionGenerationResponse:
    """
    Generate sanction letter PDF with secure download link.
    
    Args:
        request: FastAPI request object for tracing
        generation_request: Sanction generation request
        pdf_service: PDF generation service
        session_service: Session management service
        
    Returns:
        SanctionGenerationResponse: Generation result with download link
        
    Raises:
        HTTPException: For validation errors or generation failures
    """
    start_time = datetime.now()
    trace_id = getattr(request.state, "trace_id", "unknown")
    
    logger.info(
        f"Sanction letter generation started - Session: {generation_request.session_id}, "
        f"User: {generation_request.user_id}, Trace: {trace_id}"
    )
    
    try:
        # Clean up expired tokens first
        cleanup_expired_tokens()
        
        # Get and validate session
        agent_state = await session_service.get_session(
            generation_request.session_id,
            generation_request.user_id
        )
        
        if not agent_state:
            raise HTTPException(
                status_code=404,
                detail={
                    "error_code": "SESSION_NOT_FOUND",
                    "error_message": "Session not found or unauthorized",
                    "error_details": {"session_id": generation_request.session_id},
                    "timestamp": datetime.now().isoformat(),
                    "trace_id": trace_id
                }
            )
        
        # Validate loan approval status
        if not agent_state.emi_calculation:
            raise HTTPException(
                status_code=400,
                detail={
                    "error_code": "MISSING_REQUIRED_DATA",
                    "error_message": "EMI calculation is required for sanction letter",
                    "error_details": {"missing_field": "emi_calculation"},
                    "timestamp": datetime.now().isoformat(),
                    "trace_id": trace_id
                }
            )
        
        if not agent_state.user_profile:
            raise HTTPException(
                status_code=400,
                detail={
                    "error_code": "MISSING_REQUIRED_DATA",
                    "error_message": "User profile is required for sanction letter",
                    "error_details": {"missing_field": "user_profile"},
                    "timestamp": datetime.now().isoformat(),
                    "trace_id": trace_id
                }
            )
        
        # Generate loan ID if not provided
        loan_id = generation_request.loan_id
        if not loan_id:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            loan_id = f"LN{timestamp}{generation_request.user_id[-4:].upper()}"
        
        # Generate PDF
        logger.info(f"Generating PDF for loan: {loan_id}, Trace: {trace_id}")
        
        pdf_path = pdf_service.generate_from_agent_state(agent_state, loan_id)
        
        # Get PDF file information
        pdf_info = pdf_service.get_pdf_info(pdf_path)
        
        if "error" in pdf_info:
            raise GenerationError(f"Failed to get PDF info: {pdf_info['error']}")
        
        # Create secure download token
        download_token = create_download_token(
            pdf_path,
            pdf_info["filename"],
            generation_request.user_id,
            expires_in_hours=24
        )
        
        # Generate document ID
        document_id = f"DOC_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{loan_id}"
        
        # Create download URL
        download_url = f"/v1/documents/download/{download_token}"
        
        # Calculate processing time
        end_time = datetime.now()
        processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
        
        # Update agent state with document information
        try:
            agent_state.conversation_context["sanction_letter"] = {
                "document_id": document_id,
                "loan_id": loan_id,
                "generated_at": end_time.isoformat(),
                "download_token": download_token,
                "expires_at": (end_time + timedelta(hours=24)).isoformat()
            }
            await session_service.update_session(agent_state)
        except Exception as e:
            logger.warning(f"Failed to update session with document info: {str(e)}")
            # Don't fail the request if session update fails
        
        # Prepare response
        response = SanctionGenerationResponse(
            document_id=document_id,
            loan_id=loan_id,
            filename=pdf_info["filename"],
            file_size_bytes=pdf_info["file_size_bytes"],
            download_token=download_token,
            download_url=download_url,
            expires_at=datetime.now() + timedelta(hours=24),
            generated_at=end_time
        )
        
        logger.info(
            f"Sanction letter generated successfully - "
            f"Document: {document_id}, Loan: {loan_id}, "
            f"Size: {pdf_info['file_size_bytes']} bytes, "
            f"Time: {processing_time_ms}ms, Trace: {trace_id}"
        )
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    
    except Exception as e:
        logger.error(
            f"Sanction letter generation failed: {str(e)}, Trace: {trace_id}",
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": "GENERATION_ERROR",
                "error_message": "Failed to generate sanction letter",
                "error_details": {
                    "component": "pdf_generation",
                    "retry_suggested": True
                },
                "timestamp": datetime.now().isoformat(),
                "trace_id": trace_id
            }
        )

@router.get(
    "/download/{token}",
    summary="Secure document download",
    description="""
    Download document using secure token.
    
    This endpoint provides secure access to generated documents using
    time-limited tokens. Tokens expire after 24 hours and have a
    maximum download limit for security.
    
    **Security Features:**
    - Token-based authentication
    - User authorization validation
    - Download count limits
    - Automatic token expiration
    - File existence validation
    """,
    responses={
        200: {
            "description": "File download successful",
            "content": {
                "application/pdf": {
                    "example": "Binary PDF content"
                }
            }
        },
        403: {
            "description": "Invalid or expired token",
            "content": {
                "application/json": {
                    "example": {
                        "error_code": "INVALID_TOKEN",
                        "error_message": "Download token has expired",
                        "timestamp": "2024-03-01T12:34:56.789Z"
                    }
                }
            }
        },
        404: {
            "description": "File not found",
            "content": {
                "application/json": {
                    "example": {
                        "error_code": "FILE_NOT_FOUND",
                        "error_message": "File no longer available",
                        "timestamp": "2024-03-01T12:34:56.789Z"
                    }
                }
            }
        }
    }
)
async def download_document(
    token: str,
    user_id: str,
    request: Request
) -> FileResponse:
    """
    Download document using secure token.
    
    Args:
        token: Secure download token
        user_id: User identifier for authorization
        request: FastAPI request object
        
    Returns:
        FileResponse: PDF file download
        
    Raises:
        HTTPException: For invalid tokens or missing files
    """
    trace_id = getattr(request.state, "trace_id", "unknown")
    
    logger.info(f"Document download requested - Token: {token[:8]}..., User: {user_id}, Trace: {trace_id}")
    
    try:
        # Clean up expired tokens
        cleanup_expired_tokens()
        
        # Validate token
        token_info = validate_download_token(token, user_id)
        
        # Increment download count
        download_tokens[token]["download_count"] += 1
        
        # Get file information
        file_path = token_info["file_path"]
        filename = token_info["filename"]
        
        logger.info(
            f"Document download authorized - File: {filename}, "
            f"Downloads: {token_info['download_count']}/{token_info['max_downloads']}, "
            f"Trace: {trace_id}"
        )
        
        # Return file response
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
        
    except TokenError as e:
        logger.warning(f"Token validation failed: {str(e)}, Trace: {trace_id}")
        raise HTTPException(
            status_code=403,
            detail={
                "error_code": "INVALID_TOKEN",
                "error_message": str(e),
                "timestamp": datetime.now().isoformat(),
                "trace_id": trace_id
            }
        )
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}, Trace: {trace_id}")
        raise HTTPException(
            status_code=404,
            detail={
                "error_code": "FILE_NOT_FOUND",
                "error_message": "File no longer available",
                "timestamp": datetime.now().isoformat(),
                "trace_id": trace_id
            }
        )
    
    except Exception as e:
        logger.error(f"Document download failed: {str(e)}, Trace: {trace_id}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": "DOWNLOAD_ERROR",
                "error_message": "Failed to download document",
                "timestamp": datetime.now().isoformat(),
                "trace_id": trace_id
            }
        )

@router.get(
    "/info/{token}",
    response_model=DownloadInfo,
    summary="Get download information",
    description="Get information about a document without downloading it"
)
async def get_download_info(
    token: str,
    user_id: str,
    request: Request
) -> DownloadInfo:
    """
    Get information about a downloadable document.
    
    Args:
        token: Secure download token
        user_id: User identifier for authorization
        request: FastAPI request object
        
    Returns:
        DownloadInfo: Document information
        
    Raises:
        HTTPException: For invalid tokens
    """
    trace_id = getattr(request.state, "trace_id", "unknown")
    
    try:
        # Clean up expired tokens
        cleanup_expired_tokens()
        
        # Validate token (without incrementing download count)
        token_info = validate_download_token(token, user_id)
        
        # Get file stats
        file_path = Path(token_info["file_path"])
        stat = file_path.stat()
        
        return DownloadInfo(
            filename=token_info["filename"],
            file_size_bytes=stat.st_size,
            content_type="application/pdf",
            generated_at=token_info["created_at"],
            expires_at=token_info["expires_at"]
        )
        
    except TokenError as e:
        logger.warning(f"Token validation failed: {str(e)}, Trace: {trace_id}")
        raise HTTPException(
            status_code=403,
            detail={
                "error_code": "INVALID_TOKEN",
                "error_message": str(e),
                "timestamp": datetime.now().isoformat(),
                "trace_id": trace_id
            }
        )
    
    except Exception as e:
        logger.error(f"Get download info failed: {str(e)}, Trace: {trace_id}")
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": "INFO_ERROR",
                "error_message": "Failed to get download information",
                "timestamp": datetime.now().isoformat(),
                "trace_id": trace_id
            }
        )

# Export router
__all__ = ['router']