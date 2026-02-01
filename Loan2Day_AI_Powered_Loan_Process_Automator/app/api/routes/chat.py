"""
Chat API Routes - Main Conversation Endpoint

This module implements the primary chat endpoint for Master Agent orchestration.
It handles user conversations through the complete LangGraph workflow with
proper validation, error handling, and response formatting.

Key Features:
- POST /v1/chat/message: Main conversation endpoint
- Master Agent orchestration through LangGraph
- Session state management with Redis persistence
- Structured error handling and logging
- Response latency under 2 seconds for natural conversation

Architecture: Routes -> Services -> Repositories pattern
Security: Input validation using Pydantic V2
Performance: Async-first with proper timeout handling

Author: Lead AI Architect, Loan2Day Fintech Platform
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import uuid

from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

# Import core modules
from app.models.pydantic_models import (
    ChatMessageRequest, ChatMessageResponse, AgentState, 
    AgentStep, KYCStatus, UserProfile, LoanRequest
)
from app.agents.master import MasterAgent
from app.services.session_service import SessionService
from app.core.config import settings

# Configure logger
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Initialize services (will be dependency injected in production)
master_agent = MasterAgent()
session_service = SessionService()

class ChatError(Exception):
    """Base exception for chat API errors."""
    pass

class SessionNotFoundError(ChatError):
    """Raised when session is not found."""
    pass

class ValidationError(ChatError):
    """Raised when request validation fails."""
    pass

# Request/Response Models

class ChatRequest(BaseModel):
    """
    Chat request model with comprehensive validation.
    
    Validates user input and session information before processing
    through Master Agent orchestration.
    """
    
    session_id: Optional[str] = Field(
        default=None,
        description="Existing session ID (optional for new sessions)"
    )
    user_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="User identifier"
    )
    message: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="User's message content"
    )
    message_type: str = Field(
        default="text",
        description="Type of message (text, voice, etc.)"
    )
    language: Optional[str] = Field(
        default="en",
        description="Message language (en, ta, tanglish)"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional message metadata"
    )
    
    @validator('message')
    def validate_message_content(cls, v):
        """Validate message content is not empty or just whitespace."""
        if not v or not v.strip():
            raise ValueError("Message cannot be empty")
        return v.strip()
    
    @validator('user_id')
    def validate_user_id_format(cls, v):
        """Validate user ID format."""
        if not v or not v.strip():
            raise ValueError("User ID cannot be empty")
        # Basic format validation (alphanumeric with underscores/hyphens)
        import re
        if not re.match(r'^[a-zA-Z0-9_-]+$', v.strip()):
            raise ValueError("User ID contains invalid characters")
        return v.strip()

class ChatResponse(BaseModel):
    """
    Chat response model with comprehensive agent information.
    
    Returns structured response from Master Agent orchestration
    with session state and next action guidance.
    """
    
    session_id: str = Field(
        ...,
        description="Session identifier"
    )
    agent_response: str = Field(
        ...,
        description="Agent's response message"
    )
    current_step: str = Field(
        ...,
        description="Current agent state machine step"
    )
    next_actions: List[str] = Field(
        default_factory=list,
        description="Suggested next actions for the user"
    )
    requires_input: bool = Field(
        default=True,
        description="Whether agent is waiting for user input"
    )
    user_intent: Optional[str] = Field(
        default=None,
        description="Detected user intent"
    )
    sentiment_detected: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Detected sentiment in user message"
    )
    emi_calculation: Optional[Dict[str, Any]] = Field(
        default=None,
        description="EMI calculation if available"
    )
    plan_b_triggered: bool = Field(
        default=False,
        description="Whether Plan B logic was triggered"
    )
    processing_time_ms: int = Field(
        ...,
        description="Processing time in milliseconds"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Response timestamp"
    )

# Dependency Functions

async def get_session_service() -> SessionService:
    """Dependency to get session service instance."""
    return session_service

async def get_master_agent() -> MasterAgent:
    """Dependency to get master agent instance."""
    return master_agent

async def validate_request_size(request: Request):
    """Validate request size to prevent abuse."""
    content_length = request.headers.get('content-length')
    if content_length and int(content_length) > 1024 * 1024:  # 1MB limit
        raise HTTPException(
            status_code=413,
            detail="Request too large"
        )

# Route Handlers

@router.post(
    "/message",
    response_model=ChatResponse,
    summary="Process chat message through Master Agent",
    description="""
    Main conversation endpoint for Loan2Day platform.
    
    This endpoint processes user messages through the complete Master Agent
    orchestration using LangGraph workflow. It handles:
    
    - Intent classification and routing
    - Worker Agent coordination (Sales, Verification, Underwriting)
    - Session state management with Redis persistence
    - Empathetic response generation
    - Plan B logic for rejection recovery
    
    **Performance**: Response latency guaranteed under 2 seconds.
    **Security**: All inputs validated using Pydantic V2.
    **Architecture**: Master-Worker Agent pattern with LangGraph orchestration.
    """,
    responses={
        200: {
            "description": "Successful chat response",
            "content": {
                "application/json": {
                    "example": {
                        "session_id": "sess_20240301_123456_user123",
                        "agent_response": "Hello! I'd be happy to help you with your loan requirements. What amount are you looking to borrow?",
                        "current_step": "GREETING",
                        "next_actions": [
                            "Provide loan amount and tenure requirements",
                            "Share your monthly income details",
                            "Ask any questions about our loan products"
                        ],
                        "requires_input": True,
                        "user_intent": "LOAN_INQUIRY",
                        "sentiment_detected": {
                            "emotion": "neutral",
                            "polarity": 0.1,
                            "confidence": 0.8
                        },
                        "emi_calculation": None,
                        "plan_b_triggered": False,
                        "processing_time_ms": 1250,
                        "timestamp": "2024-03-01T12:34:56.789Z"
                    }
                }
            }
        },
        400: {
            "description": "Invalid request format or validation error",
            "content": {
                "application/json": {
                    "example": {
                        "error_code": "VALIDATION_ERROR",
                        "error_message": "Message cannot be empty",
                        "error_details": {"field": "message", "issue": "empty_value"},
                        "timestamp": "2024-03-01T12:34:56.789Z",
                        "trace_id": "req_1234567890"
                    }
                }
            }
        },
        500: {
            "description": "Internal server error during processing",
            "content": {
                "application/json": {
                    "example": {
                        "error_code": "PROCESSING_ERROR",
                        "error_message": "Agent orchestration failed",
                        "error_details": {"component": "master_agent", "retry_suggested": True},
                        "timestamp": "2024-03-01T12:34:56.789Z",
                        "trace_id": "req_1234567890"
                    }
                }
            }
        }
    }
)
async def process_chat_message(
    chat_request: ChatRequest,
    request: Request,
    session_service: SessionService = Depends(get_session_service),
    master_agent: MasterAgent = Depends(get_master_agent),
    _: None = Depends(validate_request_size)
) -> ChatResponse:
    """
    Process chat message through Master Agent orchestration.
    
    This endpoint serves as the main entry point for user conversations,
    orchestrating the complete loan processing workflow through LangGraph
    state machine with proper error handling and performance monitoring.
    
    Args:
        chat_request: Validated chat request with user message
        request: FastAPI request object for tracing
        session_service: Session management service
        master_agent: Master Agent for orchestration
        
    Returns:
        ChatResponse: Structured response with agent message and state
        
    Raises:
        HTTPException: For validation errors or processing failures
    """
    start_time = datetime.now()
    trace_id = getattr(request.state, "trace_id", "unknown")
    
    logger.info(
        f"Processing chat message - User: {chat_request.user_id}, "
        f"Session: {chat_request.session_id}, Trace: {trace_id}"
    )
    
    try:
        # Get or create session
        if chat_request.session_id:
            # Retrieve existing session
            agent_state = await session_service.get_session(
                chat_request.session_id, chat_request.user_id
            )
            if not agent_state:
                raise SessionNotFoundError(f"Session not found: {chat_request.session_id}")
        else:
            # Create new session
            session_id = f"sess_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{chat_request.user_id}"
            agent_state = await session_service.create_session(
                session_id, chat_request.user_id
            )
        
        # Process through Master Agent orchestration
        orchestration_result = await master_agent.process_user_request(
            agent_state.session_id,
            chat_request.user_id,
            chat_request.message,
            agent_state
        )
        
        # Update session with orchestration results
        updated_agent_state = await session_service.deserialize_agent_state(
            orchestration_result["updated_agent_state"]
        )
        await session_service.update_session(updated_agent_state)
        
        # Calculate processing time
        end_time = datetime.now()
        processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
        
        # Check performance requirement (< 2 seconds)
        if processing_time_ms > 2000:
            logger.warning(
                f"Response latency exceeded 2s: {processing_time_ms}ms - "
                f"Session: {agent_state.session_id}, Trace: {trace_id}"
            )
        
        # Extract sentiment information if available
        sentiment_detected = None
        if orchestration_result.get("processing_results", {}).get("sales_result"):
            sales_result = orchestration_result["processing_results"]["sales_result"]
            if "sentiment_analysis" in sales_result:
                sentiment_detected = sales_result["sentiment_analysis"]
        
        # Extract EMI calculation if available
        emi_calculation = None
        if orchestration_result.get("processing_results", {}).get("underwriting_result"):
            underwriting_result = orchestration_result["processing_results"]["underwriting_result"]
            if "emi_calculation" in underwriting_result:
                emi_calculation = underwriting_result["emi_calculation"]
        
        # Check if Plan B was triggered
        plan_b_triggered = False
        if orchestration_result.get("processing_results", {}).get("sales_result"):
            sales_result = orchestration_result["processing_results"]["sales_result"]
            plan_b_triggered = sales_result.get("triggered", False)
        
        # Prepare response
        chat_response = ChatResponse(
            session_id=agent_state.session_id,
            agent_response=orchestration_result["agent_response"],
            current_step=orchestration_result["current_step"],
            next_actions=orchestration_result["next_actions"],
            requires_input=orchestration_result["requires_input"],
            user_intent=orchestration_result.get("user_intent"),
            sentiment_detected=sentiment_detected,
            emi_calculation=emi_calculation,
            plan_b_triggered=plan_b_triggered,
            processing_time_ms=processing_time_ms,
            timestamp=end_time
        )
        
        logger.info(
            f"Chat message processed successfully - "
            f"Session: {agent_state.session_id}, "
            f"Step: {orchestration_result['current_step']}, "
            f"Time: {processing_time_ms}ms, "
            f"Trace: {trace_id}"
        )
        
        return chat_response
        
    except SessionNotFoundError as e:
        logger.error(f"Session not found: {str(e)}, Trace: {trace_id}")
        raise HTTPException(
            status_code=404,
            detail={
                "error_code": "SESSION_NOT_FOUND",
                "error_message": str(e),
                "error_details": {"session_id": chat_request.session_id},
                "trace_id": trace_id
            }
        )
    
    except ValidationError as e:
        logger.error(f"Validation error: {str(e)}, Trace: {trace_id}")
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": "VALIDATION_ERROR",
                "error_message": str(e),
                "error_details": {"request_data": chat_request.dict()},
                "trace_id": trace_id
            }
        )
    
    except Exception as e:
        logger.error(
            f"Chat processing failed: {str(e)}, Trace: {trace_id}",
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": "PROCESSING_ERROR",
                "error_message": "Failed to process chat message",
                "error_details": {
                    "component": "master_agent_orchestration",
                    "retry_suggested": True
                },
                "trace_id": trace_id
            }
        )

@router.get(
    "/session/{session_id}",
    summary="Get session information",
    description="Retrieve current session state and conversation history"
)
async def get_session_info(
    session_id: str,
    user_id: str,
    session_service: SessionService = Depends(get_session_service)
) -> Dict[str, Any]:
    """
    Get session information and current state.
    
    Args:
        session_id: Session identifier
        user_id: User identifier for authorization
        session_service: Session management service
        
    Returns:
        Dict[str, Any]: Session information and state
        
    Raises:
        HTTPException: If session not found or unauthorized
    """
    logger.info(f"Getting session info - Session: {session_id}, User: {user_id}")
    
    try:
        agent_state = await session_service.get_session(session_id, user_id)
        
        if not agent_state:
            raise HTTPException(
                status_code=404,
                detail=f"Session not found: {session_id}"
            )
        
        # Prepare session information
        session_info = {
            "session_id": agent_state.session_id,
            "user_id": agent_state.user_id,
            "current_step": agent_state.current_step.value,
            "kyc_status": agent_state.kyc_status.value,
            "fraud_score": agent_state.fraud_score,
            "created_at": agent_state.created_at.isoformat(),
            "updated_at": agent_state.updated_at.isoformat(),
            "conversation_context": agent_state.conversation_context,
            "loan_details": {k: str(v) for k, v in agent_state.loan_details.items()},
            "has_user_profile": agent_state.user_profile is not None,
            "has_loan_request": agent_state.loan_request is not None,
            "has_emi_calculation": agent_state.emi_calculation is not None,
            "kyc_documents_count": len(agent_state.kyc_documents),
            "sentiment_history_count": len(agent_state.sentiment_history),
            "plan_b_offers_count": len(agent_state.plan_b_offers)
        }
        
        logger.info(f"Session info retrieved - Session: {session_id}")
        return session_info
        
    except Exception as e:
        logger.error(f"Failed to get session info: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve session information: {str(e)}"
        )

@router.delete(
    "/session/{session_id}",
    summary="Delete session",
    description="Delete session and clear all associated data"
)
async def delete_session(
    session_id: str,
    user_id: str,
    session_service: SessionService = Depends(get_session_service)
) -> Dict[str, str]:
    """
    Delete session and clear all data.
    
    Args:
        session_id: Session identifier
        user_id: User identifier for authorization
        session_service: Session management service
        
    Returns:
        Dict[str, str]: Deletion confirmation
        
    Raises:
        HTTPException: If session not found or deletion fails
    """
    logger.info(f"Deleting session - Session: {session_id}, User: {user_id}")
    
    try:
        success = await session_service.delete_session(session_id, user_id)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Session not found: {session_id}"
            )
        
        logger.info(f"Session deleted successfully - Session: {session_id}")
        return {
            "message": "Session deleted successfully",
            "session_id": session_id,
            "deleted_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to delete session: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete session: {str(e)}"
        )

# Export router
__all__ = ['router']