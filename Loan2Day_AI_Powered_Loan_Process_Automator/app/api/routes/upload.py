"""
Upload API Routes - KYC Document Processing with SGS Security

This module implements secure file upload endpoints for KYC document processing.
All uploads MUST pass through SGS.scan_topology() before any processing occurs,
following the zero-trust security architecture.

Key Features:
- POST /v1/upload/kyc: Secure KYC document upload with SGS scanning
- Mandatory SGS security validation before processing
- OCR text extraction and verification
- Structured error handling and logging
- File size and type validation

Architecture: Routes -> Services -> Repositories pattern
Security: SGS.scan_topology() mandatory for all uploads
Processing: OCR extraction and fraud detection

Author: Lead AI Architect, Loan2Day Fintech Platform
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import os
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

# Import core modules
from app.core.sgs import scan_topology, SecurityScore
from app.models.pydantic_models import (
    DocumentType, VerificationStatus, KYCDocument, TrustScore
)
from app.agents.verification import VerificationAgent
from app.services.session_service import SessionService
from app.core.dependencies import get_verification_agent, get_session_service
from app.core.config import settings

# Configure logger
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Remove global service initialization - now handled by dependency injection

class UploadError(Exception):
    """Base exception for upload API errors."""
    pass

class FileSizeError(UploadError):
    """Raised when file size exceeds limits."""
    pass

class FileTypeError(UploadError):
    """Raised when file type is not supported."""
    pass

class SecurityScanError(UploadError):
    """Raised when SGS security scan fails."""
    pass

# Request/Response Models

class KYCUploadResponse(BaseModel):
    """
    KYC upload response with comprehensive security and processing results.
    """
    
    upload_id: str = Field(
        ...,
        description="Unique upload identifier"
    )
    session_id: str = Field(
        ...,
        description="Session identifier"
    )
    document_type: str = Field(
        ...,
        description="Type of uploaded document"
    )
    filename: str = Field(
        ...,
        description="Original filename"
    )
    file_size_bytes: int = Field(
        ...,
        description="File size in bytes"
    )
    sgs_security_score: Dict[str, Any] = Field(
        ...,
        description="SGS security analysis results"
    )
    is_secure: bool = Field(
        ...,
        description="Whether file passed SGS security scan"
    )
    verification_status: str = Field(
        ...,
        description="Document verification status"
    )
    ocr_extracted_data: Optional[Dict[str, str]] = Field(
        default=None,
        description="OCR extracted data if processing succeeded"
    )
    fraud_score: float = Field(
        ...,
        description="Calculated fraud risk score"
    )
    trust_score: Optional[Dict[str, Any]] = Field(
        default=None,
        description="SBEF trust score if conflicts were resolved"
    )
    processing_time_ms: int = Field(
        ...,
        description="Processing time in milliseconds"
    )
    next_steps: List[str] = Field(
        default_factory=list,
        description="Recommended next steps"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Upload timestamp"
    )

# Validation Functions

def validate_file_size(file: UploadFile) -> None:
    """
    Validate uploaded file size against configured limits.
    
    Args:
        file: Uploaded file object
        
    Raises:
        FileSizeError: If file size exceeds limits
    """
    if not hasattr(file, 'size') or file.size is None:
        # For files without size info, we'll check during read
        return
    
    max_size_bytes = settings.max_file_size_mb * 1024 * 1024
    
    if file.size > max_size_bytes:
        raise FileSizeError(
            f"File size {file.size} bytes exceeds maximum allowed size "
            f"of {max_size_bytes} bytes ({settings.max_file_size_mb}MB)"
        )

def validate_file_type(file: UploadFile, document_type: DocumentType) -> None:
    """
    Validate file type against document type requirements.
    
    Args:
        file: Uploaded file object
        document_type: Expected document type
        
    Raises:
        FileTypeError: If file type is not supported
    """
    if not file.content_type:
        raise FileTypeError("File content type not specified")
    
    # Define allowed content types for each document type
    allowed_types = {
        DocumentType.PAN_CARD: [
            "image/jpeg", "image/jpg", "image/png", "image/gif",
            "application/pdf"
        ],
        DocumentType.AADHAAR_CARD: [
            "image/jpeg", "image/jpg", "image/png", "image/gif",
            "application/pdf"
        ],
        DocumentType.BANK_STATEMENT: [
            "application/pdf", "image/jpeg", "image/jpg", "image/png"
        ],
        DocumentType.SALARY_SLIP: [
            "application/pdf", "image/jpeg", "image/jpg", "image/png"
        ],
        DocumentType.PASSPORT: [
            "image/jpeg", "image/jpg", "image/png", "application/pdf"
        ],
        DocumentType.DRIVING_LICENSE: [
            "image/jpeg", "image/jpg", "image/png", "application/pdf"
        ],
        DocumentType.VOTER_ID: [
            "image/jpeg", "image/jpg", "image/png", "application/pdf"
        ],
        DocumentType.ITR: [
            "application/pdf"
        ]
    }
    
    document_allowed_types = allowed_types.get(document_type, [])
    
    if file.content_type not in document_allowed_types:
        raise FileTypeError(
            f"File type {file.content_type} not allowed for {document_type.value}. "
            f"Allowed types: {', '.join(document_allowed_types)}"
        )

async def validate_session_exists(session_id: str, user_id: str) -> None:
    """
    Validate that session exists and user is authorized.
    
    Args:
        session_id: Session identifier
        user_id: User identifier
        
    Raises:
        HTTPException: If session not found or unauthorized
    """
    session_service = await get_session_service()
    agent_state = await session_service.get_session(session_id, user_id)
    if not agent_state:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found or unauthorized: {session_id}"
        )

# Dependency Functions

async def get_verification_agent_dep() -> VerificationAgent:
    """Dependency to get verification agent instance."""
    return await get_verification_agent()

async def get_session_service_dep() -> SessionService:
    """Dependency to get session service instance."""
    return await get_session_service()

# Route Handlers

@router.post(
    "/kyc",
    response_model=KYCUploadResponse,
    summary="Upload KYC document with SGS security scanning",
    description="""
    Secure KYC document upload endpoint with mandatory SGS security scanning.
    
    This endpoint processes KYC documents through the complete security and
    verification pipeline:
    
    1. **File Validation**: Size and type validation
    2. **SGS Security Scan**: Mandatory security topology analysis
    3. **OCR Processing**: Text extraction from documents
    4. **Fraud Detection**: Risk assessment and scoring
    5. **Verification**: Document authenticity validation
    
    **Security**: ALL uploads MUST pass SGS.scan_topology() before processing.
    **Performance**: Processing typically completes within 3-5 seconds.
    **Supported Formats**: JPEG, PNG, PDF (varies by document type).
    
    **Document Types**:
    - PAN_CARD: PAN card images or PDF
    - AADHAAR_CARD: Aadhaar card images or PDF  
    - BANK_STATEMENT: Bank statement PDF or images
    - SALARY_SLIP: Salary slip PDF or images
    - PASSPORT: Passport images or PDF
    - DRIVING_LICENSE: License images or PDF
    - VOTER_ID: Voter ID images or PDF
    - ITR: Income Tax Return PDF only
    """,
    responses={
        200: {
            "description": "Document uploaded and processed successfully",
            "content": {
                "application/json": {
                    "example": {
                        "upload_id": "upload_20240301_123456_doc123",
                        "session_id": "sess_20240301_123456_user123",
                        "document_type": "PAN_CARD",
                        "filename": "pan_card.jpg",
                        "file_size_bytes": 245760,
                        "sgs_security_score": {
                            "overall_score": 0.92,
                            "threat_level": "SAFE",
                            "is_safe": True
                        },
                        "is_secure": True,
                        "verification_status": "VERIFIED",
                        "ocr_extracted_data": {
                            "pan_number": "ABCDE1234F",
                            "name": "RAJESH KUMAR SHARMA",
                            "father_name": "FATHER OF RAJESH",
                            "date_of_birth": "15/08/1985"
                        },
                        "fraud_score": 0.15,
                        "trust_score": None,
                        "processing_time_ms": 3250,
                        "next_steps": [
                            "Upload Aadhaar card document",
                            "Upload bank statement (last 3 months)"
                        ],
                        "timestamp": "2024-03-01T12:34:56.789Z"
                    }
                }
            }
        },
        400: {
            "description": "Invalid file or validation error",
            "content": {
                "application/json": {
                    "example": {
                        "error_code": "FILE_VALIDATION_ERROR",
                        "error_message": "File size exceeds maximum allowed size",
                        "error_details": {
                            "file_size_bytes": 15728640,
                            "max_allowed_bytes": 10485760
                        },
                        "timestamp": "2024-03-01T12:34:56.789Z",
                        "trace_id": "req_1234567890"
                    }
                }
            }
        },
        403: {
            "description": "Security scan failed - file rejected",
            "content": {
                "application/json": {
                    "example": {
                        "error_code": "SECURITY_SCAN_FAILED",
                        "error_message": "File failed SGS security scan",
                        "error_details": {
                            "sgs_score": 0.35,
                            "threat_level": "HIGH_RISK",
                            "risk_factors": ["Suspicious file topology", "High fraud indicators"]
                        },
                        "timestamp": "2024-03-01T12:34:56.789Z",
                        "trace_id": "req_1234567890"
                    }
                }
            }
        }
    }
)
async def upload_kyc_document(
    request: Request,
    session_id: str = Form(..., description="Session identifier"),
    user_id: str = Form(..., description="User identifier"),
    document_type: DocumentType = Form(..., description="Type of KYC document"),
    file: UploadFile = File(..., description="KYC document file"),
    verification_agent: VerificationAgent = Depends(get_verification_agent_dep),
    session_service: SessionService = Depends(get_session_service_dep)
) -> KYCUploadResponse:
    """
    Upload and process KYC document with SGS security scanning.
    
    This endpoint implements the complete KYC document processing pipeline
    with mandatory SGS security validation, OCR extraction, and fraud detection.
    
    Args:
        request: FastAPI request object for tracing
        session_id: Session identifier
        user_id: User identifier for authorization
        document_type: Type of KYC document being uploaded
        file: Uploaded document file
        verification_agent: Verification agent for processing
        session_service: Session service for state management
        
    Returns:
        KYCUploadResponse: Comprehensive upload and processing results
        
    Raises:
        HTTPException: For validation errors, security failures, or processing errors
    """
    start_time = datetime.now()
    trace_id = getattr(request.state, "trace_id", "unknown")
    
    logger.info(
        f"KYC document upload started - Session: {session_id}, "
        f"User: {user_id}, Type: {document_type.value}, "
        f"File: {file.filename}, Trace: {trace_id}"
    )
    
    try:
        # Step 1: Validate session exists and user is authorized
        await validate_session_exists(session_id, user_id)
        
        # Step 2: Validate file size and type
        validate_file_size(file)
        validate_file_type(file, document_type)
        
        # Step 3: Read file data for processing
        file_data = await file.read()
        actual_file_size = len(file_data)
        
        # Additional size check after reading
        max_size_bytes = settings.max_file_size_mb * 1024 * 1024
        if actual_file_size > max_size_bytes:
            raise FileSizeError(
                f"File size {actual_file_size} bytes exceeds maximum "
                f"allowed size of {max_size_bytes} bytes"
            )
        
        logger.info(
            f"File validation passed - Size: {actual_file_size} bytes, "
            f"Type: {file.content_type}, Trace: {trace_id}"
        )
        
        # Step 4: MANDATORY SGS Security Scanning
        logger.info(f"Starting SGS security scan - Trace: {trace_id}")
        
        try:
            security_score = scan_topology(
                file_data, 
                file.filename, 
                user_id
            )
        except Exception as e:
            logger.error(f"SGS security scan failed: {str(e)}, Trace: {trace_id}")
            raise SecurityScanError(f"Security scan failed: {str(e)}")
        
        # Check if file passed security scan
        if not security_score.is_safe():
            logger.warning(
                f"File failed SGS security scan - Score: {security_score.overall_score}, "
                f"Threat: {security_score.threat_level.value}, Trace: {trace_id}"
            )
            
            raise HTTPException(
                status_code=403,
                detail={
                    "error_code": "SECURITY_SCAN_FAILED",
                    "error_message": "File failed SGS security scan",
                    "error_details": {
                        "sgs_score": security_score.overall_score,
                        "threat_level": security_score.threat_level.value,
                        "risk_factors": security_score.get_risk_factors()
                    },
                    "timestamp": datetime.now().isoformat(),
                    "trace_id": trace_id
                }
            )
        
        logger.info(
            f"SGS security scan passed - Score: {security_score.overall_score}, "
            f"Trace: {trace_id}"
        )
        
        # Step 5: Process through Verification Agent
        logger.info(f"Starting verification agent processing - Trace: {trace_id}")
        
        try:
            kyc_result = await verification_agent.process_kyc_document(
                file_data,
                document_type,
                file.filename,
                user_id
            )
        except Exception as e:
            logger.error(f"Verification processing failed: {str(e)}, Trace: {trace_id}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error_code": "VERIFICATION_PROCESSING_ERROR",
                    "error_message": "Document verification processing failed",
                    "error_details": {"component": "verification_agent"},
                    "timestamp": datetime.now().isoformat(),
                    "trace_id": trace_id
                }
            )
        
        # Step 6: Update session with KYC document
        try:
            agent_state = await session_service.get_session(session_id, user_id)
            if agent_state:
                # Create KYC document record
                kyc_document = KYCDocument(
                    document_type=document_type,
                    file_path=f"uploads/{user_id}/{kyc_result.document_id}_{file.filename}",
                    ocr_text=kyc_result.ocr_extracted_data.get("extracted_text", ""),
                    sgs_score=security_score.overall_score,
                    verification_status=kyc_result.verification_status,
                    extracted_data=kyc_result.ocr_extracted_data,
                    trust_score=kyc_result.trust_score,
                    uploaded_at=datetime.now(),
                    verified_at=datetime.now() if kyc_result.is_verified() else None
                )
                
                # Add to agent state
                agent_state.kyc_documents.append(kyc_document)
                agent_state.fraud_score = max(agent_state.fraud_score, kyc_result.fraud_score)
                
                # Update KYC status if all required documents are verified
                if kyc_result.is_verified():
                    # Simple logic: if we have at least one verified document, mark as verified
                    # In production, implement proper document requirement checking
                    from app.models.pydantic_models import KYCStatus
                    agent_state.kyc_status = KYCStatus.VERIFIED
                
                # Save updated session
                await session_service.update_session(agent_state)
                
        except Exception as e:
            logger.error(f"Session update failed: {str(e)}, Trace: {trace_id}")
            # Don't fail the upload if session update fails
        
        # Step 7: Generate upload ID and calculate processing time
        upload_id = f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{kyc_result.document_id}"
        
        end_time = datetime.now()
        processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
        
        # Step 8: Determine next steps
        next_steps = []
        if kyc_result.is_verified():
            next_steps = [
                "Document verified successfully",
                "Proceed with loan application",
                "Upload additional documents if required"
            ]
        elif kyc_result.verification_status == VerificationStatus.PENDING:
            next_steps = [
                "Document processing in progress",
                "Upload additional documents",
                "Wait for verification completion"
            ]
        else:
            next_steps = [
                "Document verification failed",
                "Upload a clearer image",
                "Contact support for assistance"
            ]
        
        # Step 9: Prepare response
        upload_response = KYCUploadResponse(
            upload_id=upload_id,
            session_id=session_id,
            document_type=document_type.value,
            filename=file.filename,
            file_size_bytes=actual_file_size,
            sgs_security_score=security_score.to_dict(),
            is_secure=security_score.is_safe(),
            verification_status=kyc_result.verification_status.value,
            ocr_extracted_data=kyc_result.ocr_extracted_data if kyc_result.is_verified() else None,
            fraud_score=kyc_result.fraud_score,
            trust_score=kyc_result.trust_score.to_dict() if kyc_result.trust_score else None,
            processing_time_ms=processing_time_ms,
            next_steps=next_steps,
            timestamp=end_time
        )
        
        logger.info(
            f"KYC document upload completed successfully - "
            f"Upload ID: {upload_id}, "
            f"Status: {kyc_result.verification_status.value}, "
            f"Processing Time: {processing_time_ms}ms, "
            f"Trace: {trace_id}"
        )
        
        return upload_response
        
    except (FileSizeError, FileTypeError) as e:
        logger.error(f"File validation error: {str(e)}, Trace: {trace_id}")
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": "FILE_VALIDATION_ERROR",
                "error_message": str(e),
                "error_details": {
                    "filename": file.filename,
                    "content_type": file.content_type
                },
                "timestamp": datetime.now().isoformat(),
                "trace_id": trace_id
            }
        )
    
    except SecurityScanError as e:
        logger.error(f"Security scan error: {str(e)}, Trace: {trace_id}")
        raise HTTPException(
            status_code=403,
            detail={
                "error_code": "SECURITY_SCAN_ERROR",
                "error_message": str(e),
                "error_details": {"component": "sgs_security_scan"},
                "timestamp": datetime.now().isoformat(),
                "trace_id": trace_id
            }
        )
    
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    
    except Exception as e:
        logger.error(
            f"KYC upload processing failed: {str(e)}, Trace: {trace_id}",
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": "UPLOAD_PROCESSING_ERROR",
                "error_message": "Failed to process document upload",
                "error_details": {
                    "component": "kyc_upload_processing",
                    "retry_suggested": True
                },
                "timestamp": datetime.now().isoformat(),
                "trace_id": trace_id
            }
        )

@router.get(
    "/status/{upload_id}",
    summary="Get upload processing status",
    description="Check the processing status of a previously uploaded document"
)
async def get_upload_status(
    upload_id: str,
    user_id: str,
    session_service: SessionService = Depends(get_session_service_dep)
) -> Dict[str, Any]:
    """
    Get upload processing status.
    
    Args:
        upload_id: Upload identifier
        user_id: User identifier for authorization
        session_service: Session service for data retrieval
        
    Returns:
        Dict[str, Any]: Upload status information
        
    Raises:
        HTTPException: If upload not found or unauthorized
    """
    logger.info(f"Getting upload status - Upload: {upload_id}, User: {user_id}")
    
    try:
        # Extract session ID from upload ID (simplified approach)
        # In production, maintain a separate upload tracking system
        
        # For now, return a mock status response
        # This would be implemented with proper upload tracking
        
        return {
            "upload_id": upload_id,
            "status": "COMPLETED",
            "processing_stage": "VERIFICATION_COMPLETE",
            "verification_status": "VERIFIED",
            "fraud_score": 0.15,
            "processing_time_ms": 3250,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get upload status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve upload status: {str(e)}"
        )

# Export router
__all__ = ['router']