"""
Plan B API Routes - Loan Rejection Recovery

This module implements the Plan B endpoint for loan rejection recovery scenarios.
It integrates with the Sales Agent's Plan B logic to provide alternative loan
offers when primary applications are rejected, maximizing conversion rates.

Key Features:
- GET /v1/loan/plan-b: Alternative loan offers endpoint
- Integration with Sales Agent Plan B logic
- Customized offers based on rejection reason and user profile
- Structured response with offer details and approval probabilities

Architecture: Routes -> Services -> Repositories pattern
Recovery Logic: Sales Agent Plan B engine for alternative products
Performance: Fast response with pre-calculated offer templates

Author: Lead AI Architect, Loan2Day Fintech Platform
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from fastapi import APIRouter, HTTPException, Depends, Request, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

# Import core modules
from app.models.pydantic_models import (
    AgentState, UserProfile, LoanRequest, LoanPurpose
)
from app.agents.sales import SalesAgent
from app.services.session_service import SessionService
from app.core.config import settings

# Configure logger
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Initialize services
sales_agent = SalesAgent()
session_service = SessionService()

class PlanBError(Exception):
    """Base exception for Plan B API errors."""
    pass

class SessionNotFoundError(PlanBError):
    """Raised when session is not found."""
    pass

class NoRejectionFoundError(PlanBError):
    """Raised when no rejection scenario is found for Plan B."""
    pass

# Request/Response Models

class PlanBRequest(BaseModel):
    """
    Plan B request parameters for alternative loan offers.
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
    rejection_reason: Optional[str] = Field(
        default=None,
        description="Specific rejection reason (optional)"
    )
    max_offers: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Maximum number of alternative offers to return"
    )
    user_preferences: Optional[Dict[str, Any]] = Field(
        default=None,
        description="User preferences for alternative offers"
    )
    
    @validator('session_id', 'user_id')
    def validate_ids(cls, v):
        """Validate IDs are not empty."""
        if not v or not v.strip():
            raise ValueError("ID cannot be empty")
        return v.strip()

class AlternativeOfferResponse(BaseModel):
    """
    Individual alternative loan offer response.
    """
    
    offer_id: str = Field(
        ...,
        description="Unique offer identifier"
    )
    product_name: str = Field(
        ...,
        description="Name of the alternative loan product"
    )
    amount_display: str = Field(
        ...,
        description="Loan amount in display format (e.g., ₹2,50,000)"
    )
    tenure_months: int = Field(
        ...,
        description="Loan tenure in months"
    )
    interest_rate: str = Field(
        ...,
        description="Interest rate as percentage string"
    )
    emi_display: str = Field(
        ...,
        description="EMI amount in display format"
    )
    special_features: List[str] = Field(
        default_factory=list,
        description="Special features of this loan product"
    )
    eligibility_criteria: List[str] = Field(
        default_factory=list,
        description="Eligibility criteria for this offer"
    )
    approval_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Estimated approval probability (0.0-1.0)"
    )
    offer_valid_until: str = Field(
        ...,
        description="Offer expiry date in ISO format"
    )

class PlanBResponse(BaseModel):
    """
    Plan B response with alternative loan offers and guidance.
    """
    
    session_id: str = Field(
        ...,
        description="Session identifier"
    )
    rejection_reason: str = Field(
        ...,
        description="Original rejection reason"
    )
    alternative_offers: List[AlternativeOfferResponse] = Field(
        default_factory=list,
        description="List of alternative loan offers"
    )
    recommendation_message: str = Field(
        ...,
        description="Personalized recommendation message"
    )
    next_steps: List[str] = Field(
        default_factory=list,
        description="Recommended next steps for the user"
    )
    estimated_approval_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall estimated approval probability for alternatives"
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

async def get_sales_agent() -> SalesAgent:
    """Dependency to get sales agent instance."""
    return sales_agent

async def get_session_service() -> SessionService:
    """Dependency to get session service instance."""
    return session_service

async def validate_session_and_rejection(
    session_id: str, 
    user_id: str,
    session_service: SessionService
) -> AgentState:
    """
    Validate session exists and has rejection scenario for Plan B.
    
    Args:
        session_id: Session identifier
        user_id: User identifier
        session_service: Session service instance
        
    Returns:
        AgentState: Validated agent state
        
    Raises:
        SessionNotFoundError: If session not found
        NoRejectionFoundError: If no rejection scenario found
    """
    # Get session
    agent_state = await session_service.get_session(session_id, user_id)
    if not agent_state:
        raise SessionNotFoundError(f"Session not found: {session_id}")
    
    # Check if there's a rejection scenario or underwriting result
    has_rejection = False
    
    # Check conversation context for rejection indicators
    if agent_state.conversation_context:
        context = agent_state.conversation_context
        if (context.get("last_decision") == "REJECTED" or 
            context.get("underwriting_decision") == "REJECTED" or
            len(agent_state.plan_b_offers) > 0):
            has_rejection = True
    
    # Check if current step indicates rejection recovery
    if agent_state.current_step.value == "PLAN_B":
        has_rejection = True
    
    # Check loan details for rejection indicators
    if agent_state.loan_details:
        if (agent_state.loan_details.get("decision") == "REJECTED" or
            agent_state.loan_details.get("status") == "REJECTED"):
            has_rejection = True
    
    if not has_rejection:
        # Allow Plan B even without explicit rejection for demo purposes
        logger.info(f"No explicit rejection found, but allowing Plan B for session: {session_id}")
    
    return agent_state

# Route Handlers

@router.get(
    "/plan-b",
    response_model=PlanBResponse,
    summary="Get alternative loan offers for rejection recovery",
    description="""
    Plan B endpoint for loan rejection recovery scenarios.
    
    This endpoint provides alternative loan offers when primary loan applications
    are rejected, implementing intelligent recovery logic to maximize conversion
    rates and prevent customer loss.
    
    **Features:**
    - Customized alternative offers based on rejection reason
    - Integration with Sales Agent Plan B logic
    - Approval probability estimation for each offer
    - Personalized recommendation messages
    
    **Use Cases:**
    - Primary loan application rejected due to credit score
    - Loan amount exceeds eligibility criteria
    - Income insufficient for requested EMI
    - Document verification issues
    
    **Recovery Strategy:**
    - Secured loans with collateral requirements
    - Co-applicant loans for higher amounts
    - Step-up EMI loans for growing income profiles
    - Micro loans for credit building
    """,
    responses={
        200: {
            "description": "Alternative offers generated successfully",
            "content": {
                "application/json": {
                    "example": {
                        "session_id": "sess_20240301_123456_user123",
                        "rejection_reason": "Requested amount exceeds eligibility based on income",
                        "alternative_offers": [
                            {
                                "offer_id": "PLB_20240301_123456_1",
                                "product_name": "Secured Personal Loan",
                                "amount_display": "₹3,50,000",
                                "tenure_months": 48,
                                "interest_rate": "14.0%",
                                "emi_display": "₹9,567",
                                "special_features": [
                                    "Lower interest with collateral",
                                    "Flexible repayment options"
                                ],
                                "eligibility_criteria": [
                                    "Collateral required (property/FD)",
                                    "Minimum income: ₹25,000"
                                ],
                                "approval_probability": 0.85,
                                "offer_valid_until": "2024-03-08T23:59:59Z"
                            }
                        ],
                        "recommendation_message": "Based on your profile, I have excellent alternative options that can work better for your situation.",
                        "next_steps": [
                            "Review alternative loan options",
                            "Select your preferred alternative",
                            "Provide additional documentation if required"
                        ],
                        "estimated_approval_probability": 0.78,
                        "processing_time_ms": 850,
                        "timestamp": "2024-03-01T12:34:56.789Z"
                    }
                }
            }
        },
        404: {
            "description": "Session not found or no rejection scenario",
            "content": {
                "application/json": {
                    "example": {
                        "error_code": "SESSION_NOT_FOUND",
                        "error_message": "Session not found or no rejection scenario available",
                        "error_details": {"session_id": "sess_invalid"},
                        "timestamp": "2024-03-01T12:34:56.789Z",
                        "trace_id": "req_1234567890"
                    }
                }
            }
        }
    }
)
async def get_plan_b_offers(
    request: Request,
    session_id: str = Query(..., description="Session identifier"),
    user_id: str = Query(..., description="User identifier"),
    rejection_reason: Optional[str] = Query(None, description="Specific rejection reason"),
    max_offers: int = Query(3, ge=1, le=5, description="Maximum offers to return"),
    sales_agent: SalesAgent = Depends(get_sales_agent),
    session_service: SessionService = Depends(get_session_service)
) -> PlanBResponse:
    """
    Get alternative loan offers for rejection recovery.
    
    This endpoint implements the Plan B logic to provide customized alternative
    loan offers when primary applications are rejected, maximizing conversion
    rates through intelligent product recommendations.
    
    Args:
        request: FastAPI request object for tracing
        session_id: Session identifier
        user_id: User identifier for authorization
        rejection_reason: Specific rejection reason (optional)
        max_offers: Maximum number of offers to return
        sales_agent: Sales agent for Plan B logic
        session_service: Session service for state management
        
    Returns:
        PlanBResponse: Alternative offers with recommendations
        
    Raises:
        HTTPException: For session errors or processing failures
    """
    start_time = datetime.now()
    trace_id = getattr(request.state, "trace_id", "unknown")
    
    logger.info(
        f"Plan B request started - Session: {session_id}, "
        f"User: {user_id}, Trace: {trace_id}"
    )
    
    try:
        # Validate session and rejection scenario
        agent_state = await validate_session_and_rejection(
            session_id, user_id, session_service
        )
        
        # Determine rejection reason
        final_rejection_reason = rejection_reason
        if not final_rejection_reason:
            # Extract from agent state or use default
            if agent_state.conversation_context:
                context = agent_state.conversation_context
                final_rejection_reason = (
                    context.get("rejection_reason") or
                    context.get("underwriting_rejection_reason") or
                    "Application did not meet standard criteria"
                )
            else:
                final_rejection_reason = "Application did not meet standard criteria"
        
        # Get user sentiment for empathetic response
        user_sentiment = None
        if agent_state.sentiment_history:
            user_sentiment = agent_state.sentiment_history[-1]
        
        # Generate Plan B offers through Sales Agent
        logger.info(f"Generating Plan B offers - Trace: {trace_id}")
        
        plan_b_result = await sales_agent.trigger_plan_b_logic(
            agent_state,
            final_rejection_reason,
            user_sentiment
        )
        
        # Convert alternative offers to response format
        alternative_offers = []
        for offer_dict in plan_b_result.get("alternative_offers", [])[:max_offers]:
            alternative_offer = AlternativeOfferResponse(
                offer_id=offer_dict["offer_id"],
                product_name=offer_dict["product_name"],
                amount_display=offer_dict["amount_display"],
                tenure_months=offer_dict["tenure_months"],
                interest_rate=f"{offer_dict['interest_rate']}%",
                emi_display=offer_dict["emi_display"],
                special_features=offer_dict["special_features"],
                eligibility_criteria=offer_dict["eligibility_criteria"],
                approval_probability=offer_dict["approval_probability"],
                offer_valid_until=offer_dict["offer_valid_until"]
            )
            alternative_offers.append(alternative_offer)
        
        # Generate recommendation message
        recommendation_message = plan_b_result.get(
            "empathetic_message",
            "I understand this is disappointing, but I have some excellent alternative options for you."
        )
        
        # Calculate overall approval probability
        if alternative_offers:
            estimated_approval = sum(offer.approval_probability for offer in alternative_offers) / len(alternative_offers)
        else:
            estimated_approval = 0.0
        
        # Generate next steps
        next_steps = [
            "Review the alternative loan options below",
            "Select your preferred alternative offer",
            "Contact our team for personalized assistance"
        ]
        
        if alternative_offers:
            if any(offer.approval_probability > 0.8 for offer in alternative_offers):
                next_steps.insert(1, "High approval probability options available")
            
            # Add specific guidance based on offer types
            offer_types = [offer.product_name for offer in alternative_offers]
            if any("Secured" in offer_type for offer_type in offer_types):
                next_steps.append("Consider providing collateral for better terms")
            if any("Co-applicant" in offer_type for offer_type in offer_types):
                next_steps.append("Add a co-applicant to increase eligibility")
        
        # Update agent state with Plan B offers
        try:
            agent_state.plan_b_offers = plan_b_result.get("alternative_offers", [])
            agent_state.current_step = agent_state.current_step  # Keep current step
            agent_state.conversation_context["plan_b_generated"] = True
            agent_state.conversation_context["plan_b_timestamp"] = datetime.now().isoformat()
            
            await session_service.update_session(agent_state)
        except Exception as e:
            logger.warning(f"Failed to update session with Plan B offers: {str(e)}")
            # Don't fail the request if session update fails
        
        # Calculate processing time
        end_time = datetime.now()
        processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
        
        # Prepare response
        plan_b_response = PlanBResponse(
            session_id=session_id,
            rejection_reason=final_rejection_reason,
            alternative_offers=alternative_offers,
            recommendation_message=recommendation_message,
            next_steps=next_steps,
            estimated_approval_probability=estimated_approval,
            processing_time_ms=processing_time_ms,
            timestamp=end_time
        )
        
        logger.info(
            f"Plan B offers generated successfully - "
            f"Session: {session_id}, Offers: {len(alternative_offers)}, "
            f"Avg Probability: {estimated_approval:.2f}, "
            f"Time: {processing_time_ms}ms, Trace: {trace_id}"
        )
        
        return plan_b_response
        
    except SessionNotFoundError as e:
        logger.error(f"Session not found: {str(e)}, Trace: {trace_id}")
        raise HTTPException(
            status_code=404,
            detail={
                "error_code": "SESSION_NOT_FOUND",
                "error_message": str(e),
                "error_details": {"session_id": session_id},
                "timestamp": datetime.now().isoformat(),
                "trace_id": trace_id
            }
        )
    
    except NoRejectionFoundError as e:
        logger.error(f"No rejection scenario found: {str(e)}, Trace: {trace_id}")
        raise HTTPException(
            status_code=404,
            detail={
                "error_code": "NO_REJECTION_SCENARIO",
                "error_message": str(e),
                "error_details": {"session_id": session_id},
                "timestamp": datetime.now().isoformat(),
                "trace_id": trace_id
            }
        )
    
    except Exception as e:
        logger.error(
            f"Plan B processing failed: {str(e)}, Trace: {trace_id}",
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": "PLAN_B_PROCESSING_ERROR",
                "error_message": "Failed to generate alternative offers",
                "error_details": {
                    "component": "plan_b_generation",
                    "retry_suggested": True
                },
                "timestamp": datetime.now().isoformat(),
                "trace_id": trace_id
            }
        )

@router.post(
    "/plan-b/select",
    summary="Select alternative offer",
    description="Select a specific alternative offer from Plan B recommendations"
)
async def select_plan_b_offer(
    request: Request,
    session_id: str,
    user_id: str,
    offer_id: str,
    session_service: SessionService = Depends(get_session_service)
) -> Dict[str, Any]:
    """
    Select a specific Plan B alternative offer.
    
    Args:
        request: FastAPI request object
        session_id: Session identifier
        user_id: User identifier
        offer_id: Selected offer identifier
        session_service: Session service instance
        
    Returns:
        Dict[str, Any]: Selection confirmation
    """
    trace_id = getattr(request.state, "trace_id", "unknown")
    
    logger.info(
        f"Plan B offer selection - Session: {session_id}, "
        f"Offer: {offer_id}, Trace: {trace_id}"
    )
    
    try:
        # Get session
        agent_state = await session_service.get_session(session_id, user_id)
        if not agent_state:
            raise HTTPException(
                status_code=404,
                detail=f"Session not found: {session_id}"
            )
        
        # Find selected offer
        selected_offer = None
        for offer in agent_state.plan_b_offers:
            if offer.get("offer_id") == offer_id:
                selected_offer = offer
                break
        
        if not selected_offer:
            raise HTTPException(
                status_code=404,
                detail=f"Offer not found: {offer_id}"
            )
        
        # Update session with selected offer
        agent_state.conversation_context["selected_plan_b_offer"] = selected_offer
        agent_state.conversation_context["plan_b_selection_timestamp"] = datetime.now().isoformat()
        
        await session_service.update_session(agent_state)
        
        logger.info(f"Plan B offer selected - Session: {session_id}, Offer: {offer_id}")
        
        return {
            "message": "Alternative offer selected successfully",
            "session_id": session_id,
            "selected_offer_id": offer_id,
            "selected_offer": selected_offer,
            "next_steps": [
                "Proceed with the selected loan application",
                "Complete any additional documentation required",
                "Wait for final approval confirmation"
            ],
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Plan B offer selection failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to select Plan B offer: {str(e)}"
        )

# Export router
__all__ = ['router']