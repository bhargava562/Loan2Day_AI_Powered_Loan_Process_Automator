"""
Pydantic Models for Loan2Day Agentic AI Fintech Platform

This module defines all Pydantic models used throughout the Loan2Day system.
All monetary fields strictly use decimal.Decimal following the LQM Standard
to ensure zero-hallucination mathematics and financial precision.

Key Features:
- AgentState: Central state tracking for Master-Worker Agent pattern
- UserProfile: User information with validated monetary fields
- LoanRequest: Loan application data with decimal precision
- EMICalculation: EMI calculation results from LQM module
- KYCDocument: KYC document processing and verification
- SentimentScore: Sentiment analysis results for empathetic responses

Author: Lead AI Architect, Loan2Day Fintech Platform
"""

from typing import Dict, List, Literal, Optional, Any, Union
from decimal import Decimal
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from enum import Enum
import logging

# Configure logger
logger = logging.getLogger(__name__)

# Import LQM validation functions
from app.core.lqm import validate_monetary_input, LQMError, FloatInputError

class ValidationError(Exception):
    """Custom validation error for Pydantic models."""
    pass

# Enums for type safety
class AgentStep(str, Enum):
    """Valid agent state machine steps."""
    GREETING = "GREETING"
    KYC = "KYC"
    NEGOTIATION = "NEGOTIATION"
    SANCTION = "SANCTION"
    PLAN_B = "PLAN_B"

class KYCStatus(str, Enum):
    """Valid KYC verification statuses."""
    PENDING = "PENDING"
    VERIFIED = "VERIFIED"
    REJECTED = "REJECTED"

class EmploymentType(str, Enum):
    """Valid employment types."""
    SALARIED = "SALARIED"
    SELF_EMPLOYED = "SELF_EMPLOYED"
    BUSINESS_OWNER = "BUSINESS_OWNER"
    FREELANCER = "FREELANCER"
    RETIRED = "RETIRED"
    UNEMPLOYED = "UNEMPLOYED"

class DocumentType(str, Enum):
    """Valid KYC document types."""
    PAN_CARD = "PAN_CARD"
    AADHAAR_CARD = "AADHAAR_CARD"
    PASSPORT = "PASSPORT"
    DRIVING_LICENSE = "DRIVING_LICENSE"
    VOTER_ID = "VOTER_ID"
    BANK_STATEMENT = "BANK_STATEMENT"
    SALARY_SLIP = "SALARY_SLIP"
    ITR = "ITR"

class VerificationStatus(str, Enum):
    """Valid document verification statuses."""
    PENDING = "PENDING"
    VERIFIED = "VERIFIED"
    REJECTED = "REJECTED"
    SUSPICIOUS = "SUSPICIOUS"

class LoanPurpose(str, Enum):
    """Valid loan purposes."""
    PERSONAL = "PERSONAL"
    HOME_IMPROVEMENT = "HOME_IMPROVEMENT"
    MEDICAL = "MEDICAL"
    EDUCATION = "EDUCATION"
    BUSINESS = "BUSINESS"
    DEBT_CONSOLIDATION = "DEBT_CONSOLIDATION"
    TRAVEL = "TRAVEL"
    WEDDING = "WEDDING"
    OTHER = "OTHER"

# Base model with common configuration
class BaseModelConfig(BaseModel):
    """Base model with common Pydantic configuration."""
    
    model_config = ConfigDict(
        # Enable validation on assignment
        validate_assignment=True,
        # Use enum values instead of names
        use_enum_values=True,
        # Allow population by field name or alias
        populate_by_name=True,
        # Custom serializers for Decimal
        arbitrary_types_allowed=True
    )

# Core Domain Models

class SentimentScore(BaseModelConfig):
    """
    Sentiment analysis results for empathetic response generation.
    
    Used by Sales Agent to understand user emotional state and adapt
    communication style accordingly. Supports Plan B logic activation
    based on negative sentiment patterns.
    """
    
    polarity: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Sentiment polarity from -1 (negative) to 1 (positive)"
    )
    subjectivity: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Sentiment subjectivity from 0 (objective) to 1 (subjective)"
    )
    emotion: str = Field(
        ...,
        description="Detected emotion (e.g., 'happy', 'frustrated', 'anxious')"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for sentiment analysis"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the sentiment was analyzed"
    )
    
    @field_validator('emotion')
    @classmethod
    def validate_emotion(cls, v):
        """Validate emotion is not empty."""
        if not v or not v.strip():
            raise ValueError("Emotion cannot be empty")
        return v.strip().lower()

class TrustScore(BaseModelConfig):
    """
    SBEF (Semantic-Bayesian Evidence Fusion) trust score for resolving
    data conflicts between user input and OCR document data.
    """
    
    user_input_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in user-provided data"
    )
    ocr_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in OCR-extracted data"
    )
    final_trust_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Final calculated trust score using SBEF algorithm"
    )
    resolution_method: str = Field(
        ...,
        description="Method used to resolve the conflict (e.g., 'user_priority', 'ocr_priority', 'manual_review')"
    )
    conflict_details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Details about the data conflict"
    )

class KYCDocument(BaseModelConfig):
    """
    KYC document processing and verification results.
    
    Handles document uploads through SGS (Spectral-Graph Sentinel) security
    scanning and OCR text extraction for verification purposes.
    """
    
    document_type: DocumentType = Field(
        ...,
        description="Type of KYC document"
    )
    file_path: str = Field(
        ...,
        min_length=1,
        description="Secure file path for uploaded document"
    )
    ocr_text: Optional[str] = Field(
        default=None,
        description="Extracted text from OCR processing"
    )
    sgs_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="SGS security score (0=suspicious, 1=authentic)"
    )
    verification_status: VerificationStatus = Field(
        default=VerificationStatus.PENDING,
        description="Current verification status"
    )
    extracted_data: Optional[Dict[str, str]] = Field(
        default=None,
        description="Structured data extracted from document"
    )
    trust_score: Optional[TrustScore] = Field(
        default=None,
        description="Trust score if data conflicts were resolved"
    )
    uploaded_at: datetime = Field(
        default_factory=datetime.now,
        description="When the document was uploaded"
    )
    verified_at: Optional[datetime] = Field(
        default=None,
        description="When the document was verified"
    )
    
    @field_validator('file_path')
    @classmethod
    def validate_file_path(cls, v):
        """Validate file path is not empty and has reasonable format."""
        if not v or not v.strip():
            raise ValueError("File path cannot be empty")
        # Basic security check - no path traversal
        if '..' in v or v.startswith('/'):
            raise ValueError("Invalid file path format")
        return v.strip()

class EMICalculation(BaseModelConfig):
    """
    EMI calculation results from LQM module with zero-hallucination mathematics.
    
    All monetary values are stored as decimal.Decimal with exactly 2 decimal
    places to ensure financial precision and regulatory compliance.
    """
    
    principal_in_cents: Decimal = Field(
        ...,
        description="Loan principal amount in cents (Decimal type)"
    )
    rate_per_annum: Decimal = Field(
        ...,
        description="Annual interest rate as percentage (Decimal type)"
    )
    tenure_months: int = Field(
        ...,
        gt=0,
        le=360,  # Maximum 30 years
        description="Loan tenure in months"
    )
    emi_in_cents: Decimal = Field(
        ...,
        description="Calculated EMI amount in cents (Decimal type)"
    )
    total_interest_in_cents: Decimal = Field(
        ...,
        description="Total interest payable in cents (Decimal type)"
    )
    total_amount_in_cents: Decimal = Field(
        ...,
        description="Total amount payable in cents (Decimal type)"
    )
    calculated_at: datetime = Field(
        default_factory=datetime.now,
        description="When the EMI was calculated"
    )
    
    @field_validator('principal_in_cents', 'emi_in_cents', 'total_interest_in_cents', 'total_amount_in_cents')
    @classmethod
    def validate_monetary_fields(cls, v, info):
        """Validate all monetary fields use Decimal and reject float inputs."""
        try:
            # Use LQM validation to ensure Decimal type and precision
            validated_value = validate_monetary_input(v, info.field_name)
            return validated_value
        except (LQMError, FloatInputError) as e:
            raise ValueError(f"Invalid monetary value for {info.field_name}: {str(e)}")
    
    @field_validator('rate_per_annum')
    @classmethod
    def validate_interest_rate(cls, v):
        """Validate interest rate is reasonable."""
        try:
            validated_rate = validate_monetary_input(v, "rate_per_annum")
            # Check reasonable bounds (0% to 50% annual)
            if validated_rate < Decimal('0.00') or validated_rate > Decimal('50.00'):
                raise ValueError("Interest rate must be between 0% and 50% per annum")
            return validated_rate
        except (LQMError, FloatInputError) as e:
            raise ValueError(f"Invalid interest rate: {str(e)}")
    
    @model_validator(mode='after')
    def validate_calculation_consistency(self):
        """Validate that EMI calculation results are mathematically consistent."""
        principal = self.principal_in_cents
        emi = self.emi_in_cents
        total_interest = self.total_interest_in_cents
        total_amount = self.total_amount_in_cents
        tenure = self.tenure_months
        
        if all([principal, emi, total_interest, total_amount, tenure]):
            # Validate total_amount = principal + total_interest
            expected_total = principal + total_interest
            if abs(total_amount - expected_total) > Decimal('0.01'):  # Allow 1 cent tolerance
                raise ValueError(
                    f"Inconsistent calculation: total_amount ({total_amount}) != "
                    f"principal + interest ({expected_total})"
                )
            
            # Validate EMI * tenure is close to total_amount (allowing for rounding)
            emi_total = emi * Decimal(str(tenure))
            if abs(emi_total - total_amount) > Decimal('1.00'):  # Allow 1 rupee tolerance
                logger.warning(
                    f"EMI calculation rounding difference: "
                    f"EMI * tenure = {emi_total}, total_amount = {total_amount}"
                )
        
        return self

class UserProfile(BaseModelConfig):
    """
    User profile information with validated monetary fields.
    
    Stores user demographic and financial information required for
    loan processing and risk assessment. All income fields use
    decimal.Decimal following the LQM Standard.
    """
    
    user_id: str = Field(
        ...,
        min_length=1,
        description="Unique user identifier"
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="User's full name"
    )
    phone: str = Field(
        ...,
        pattern=r'^\+?[1-9]\d{1,14}$',  # E.164 format
        description="User's phone number in E.164 format"
    )
    email: str = Field(
        ...,
        pattern=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
        description="User's email address"
    )
    income_in_cents: Decimal = Field(
        ...,
        description="User's monthly income in cents (Decimal type)"
    )
    employment_type: EmploymentType = Field(
        ...,
        description="Type of employment"
    )
    credit_score: Optional[int] = Field(
        default=None,
        ge=300,
        le=900,
        description="CIBIL credit score (300-900 range)"
    )
    age: Optional[int] = Field(
        default=None,
        ge=18,
        le=100,
        description="User's age in years"
    )
    city: Optional[str] = Field(
        default=None,
        max_length=100,
        description="User's city of residence"
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="When the profile was created"
    )
    updated_at: datetime = Field(
        default_factory=datetime.now,
        description="When the profile was last updated"
    )
    
    @field_validator('income_in_cents')
    @classmethod
    def validate_income(cls, v):
        """Validate income uses Decimal type and is positive."""
        try:
            validated_income = validate_monetary_input(v, "income_in_cents")
            if validated_income <= Decimal('0.00'):
                raise ValueError("Income must be positive")
            return validated_income
        except (LQMError, FloatInputError) as e:
            raise ValueError(f"Invalid income value: {str(e)}")
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        """Validate name is not empty and contains valid characters."""
        if not v or not v.strip():
            raise ValueError("Name cannot be empty")
        # Allow letters, spaces, dots, and common name characters
        import re
        if not re.match(r'^[a-zA-Z\s\.\-\']+$', v.strip()):
            raise ValueError("Name contains invalid characters")
        return v.strip()
    
    @field_validator('phone')
    @classmethod
    def validate_phone_format(cls, v):
        """Additional phone validation."""
        # Remove any spaces or dashes for validation
        cleaned_phone = v.replace(' ', '').replace('-', '')
        if len(cleaned_phone) < 10:
            raise ValueError("Phone number too short")
        return cleaned_phone

class LoanRequest(BaseModelConfig):
    """
    Loan application request with decimal precision for monetary fields.
    
    Captures loan application details including amount, tenure, and purpose.
    All monetary fields strictly use decimal.Decimal for precision.
    """
    
    amount_in_cents: Decimal = Field(
        ...,
        description="Requested loan amount in cents (Decimal type)"
    )
    tenure_months: int = Field(
        ...,
        gt=0,
        le=360,  # Maximum 30 years
        description="Requested loan tenure in months"
    )
    purpose: LoanPurpose = Field(
        ...,
        description="Purpose of the loan"
    )
    requested_rate: Optional[Decimal] = Field(
        default=None,
        description="User's requested interest rate (optional)"
    )
    monthly_income_in_cents: Optional[Decimal] = Field(
        default=None,
        description="User's declared monthly income in cents"
    )
    existing_emi_in_cents: Optional[Decimal] = Field(
        default=Decimal('0.00'),
        description="Existing EMI obligations in cents"
    )
    collateral_offered: Optional[bool] = Field(
        default=False,
        description="Whether collateral is offered"
    )
    co_applicant_required: Optional[bool] = Field(
        default=False,
        description="Whether a co-applicant is required"
    )
    requested_at: datetime = Field(
        default_factory=datetime.now,
        description="When the loan was requested"
    )
    
    @field_validator('amount_in_cents')
    @classmethod
    def validate_amount(cls, v):
        """Validate loan amount uses Decimal type and is within bounds."""
        try:
            validated_amount = validate_monetary_input(v, "amount_in_cents")
            # Check reasonable bounds (₹50,000 to ₹50 Lakh for broader testing)
            min_amount = Decimal('5000000')   # ₹50,000 in cents
            max_amount = Decimal('5000000000') # ₹50 Lakh in cents
            
            if validated_amount < min_amount:
                raise ValueError(f"Loan amount must be at least ₹{min_amount/100:,.2f}")
            if validated_amount > max_amount:
                raise ValueError(f"Loan amount cannot exceed ₹{max_amount/100:,.2f}")
            
            return validated_amount
        except (LQMError, FloatInputError) as e:
            raise ValueError(f"Invalid loan amount: {str(e)}")
    
    @field_validator('requested_rate')
    @classmethod
    def validate_requested_rate(cls, v):
        """Validate requested interest rate if provided."""
        if v is not None:
            try:
                validated_rate = validate_monetary_input(v, "requested_rate")
                if validated_rate < Decimal('0.00') or validated_rate > Decimal('50.00'):
                    raise ValueError("Requested rate must be between 0% and 50% per annum")
                return validated_rate
            except (LQMError, FloatInputError) as e:
                raise ValueError(f"Invalid requested rate: {str(e)}")
        return v
    
    @field_validator('monthly_income_in_cents', 'existing_emi_in_cents')
    @classmethod
    def validate_optional_monetary_fields(cls, v, info):
        """Validate optional monetary fields."""
        if v is not None:
            try:
                validated_value = validate_monetary_input(v, info.field_name)
                if validated_value < Decimal('0.00'):
                    raise ValueError(f"{info.field_name} cannot be negative")
                return validated_value
            except (LQMError, FloatInputError) as e:
                raise ValueError(f"Invalid {info.field_name}: {str(e)}")
        return v

class AgentState(BaseModelConfig):
    """
    Central state tracking for Master-Worker Agent pattern.
    
    This is the core state object that tracks user sessions, loan details,
    and processing status throughout the agent orchestration workflow.
    All monetary values use decimal.Decimal following the LQM Standard.
    """
    
    session_id: str = Field(
        ...,
        min_length=1,
        description="Unique session identifier"
    )
    user_id: str = Field(
        ...,
        min_length=1,
        description="Unique user identifier"
    )
    current_step: AgentStep = Field(
        default=AgentStep.GREETING,
        description="Current step in the agent state machine"
    )
    loan_details: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Loan calculation results using Decimal types"
    )
    kyc_status: KYCStatus = Field(
        default=KYCStatus.PENDING,
        description="Current KYC verification status"
    )
    fraud_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Calculated fraud risk score (0=low risk, 1=high risk)"
    )
    sentiment_history: List[SentimentScore] = Field(
        default_factory=list,
        description="History of sentiment analysis results"
    )
    trust_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="SBEF trust score for data conflict resolution"
    )
    user_profile: Optional[UserProfile] = Field(
        default=None,
        description="User profile information"
    )
    loan_request: Optional[LoanRequest] = Field(
        default=None,
        description="Current loan request details"
    )
    emi_calculation: Optional[EMICalculation] = Field(
        default=None,
        description="Latest EMI calculation results"
    )
    kyc_documents: List[KYCDocument] = Field(
        default_factory=list,
        description="Uploaded and processed KYC documents"
    )
    plan_b_offers: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Alternative loan offers from Plan B logic"
    )
    conversation_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Conversation context and history"
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="When the session was created"
    )
    updated_at: datetime = Field(
        default_factory=datetime.now,
        description="When the session was last updated"
    )
    
    @field_validator('loan_details')
    @classmethod
    def validate_loan_details_decimal(cls, v):
        """Validate that all loan details use Decimal types."""
        if not isinstance(v, dict):
            return v
        
        for key, value in v.items():
            if isinstance(value, (int, float, str)):
                try:
                    # Convert to Decimal and validate
                    v[key] = validate_monetary_input(value, f"loan_details.{key}")
                except (LQMError, FloatInputError) as e:
                    raise ValueError(f"Invalid monetary value in loan_details.{key}: {str(e)}")
            elif not isinstance(value, Decimal):
                raise ValueError(f"loan_details.{key} must be a Decimal type, got {type(value)}")
        
        return v
    
    @field_validator('sentiment_history')
    @classmethod
    def validate_sentiment_history_limit(cls, v):
        """Limit sentiment history to last 50 entries for performance."""
        if len(v) > 50:
            return v[-50:]  # Keep only the last 50 entries
        return v
    
    @model_validator(mode='after')
    def validate_state_consistency(self):
        """Validate state machine consistency and data relationships."""
        current_step = self.current_step
        kyc_status = self.kyc_status
        loan_request = self.loan_request
        emi_calculation = self.emi_calculation
        
        # Validate state transitions
        if current_step == AgentStep.NEGOTIATION:
            if kyc_status != KYCStatus.VERIFIED:
                logger.warning(f"State inconsistency: NEGOTIATION step with KYC status {kyc_status}")
        
        if current_step == AgentStep.SANCTION:
            if not loan_request or not emi_calculation:
                raise ValueError("SANCTION step requires both loan_request and emi_calculation")
        
        # Note: Do not update timestamp here to avoid recursion
        # Timestamp updates should be handled explicitly when needed
        
        return self

# API Request/Response Models

class ChatMessageRequest(BaseModelConfig):
    """
    Chat message request for the main conversation endpoint.
    
    Handles user input processing for the Master Agent orchestration.
    """
    
    session_id: Optional[str] = Field(
        default=None,
        description="Existing session ID (optional for new sessions)"
    )
    user_id: str = Field(
        ...,
        min_length=1,
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

class ChatMessageResponse(BaseModelConfig):
    """
    Chat message response from the agent system.
    
    Returns agent response with updated session state.
    """
    
    session_id: str = Field(
        ...,
        description="Session identifier"
    )
    agent_response: str = Field(
        ...,
        description="Agent's response message"
    )
    current_step: AgentStep = Field(
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
    sentiment_detected: Optional[SentimentScore] = Field(
        default=None,
        description="Detected sentiment in user message"
    )
    emi_calculation: Optional[EMICalculation] = Field(
        default=None,
        description="EMI calculation if available"
    )
    plan_b_triggered: bool = Field(
        default=False,
        description="Whether Plan B logic was triggered"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Response timestamp"
    )

class FileUploadRequest(BaseModelConfig):
    """
    File upload request for KYC document processing.
    
    Handles document uploads through SGS security scanning.
    """
    
    user_id: str = Field(
        ...,
        min_length=1,
        description="User identifier"
    )
    document_type: DocumentType = Field(
        ...,
        description="Type of KYC document being uploaded"
    )
    filename: str = Field(
        ...,
        min_length=1,
        description="Original filename"
    )
    file_size_bytes: int = Field(
        ...,
        gt=0,
        description="File size in bytes"
    )

class FileUploadResponse(BaseModelConfig):
    """
    File upload response with security scan results.
    
    Returns SGS security analysis and processing status.
    """
    
    upload_id: str = Field(
        ...,
        description="Unique upload identifier"
    )
    sgs_score: float = Field(
        ...,
        description="SGS security score"
    )
    is_safe: bool = Field(
        ...,
        description="Whether file passed security scan"
    )
    verification_status: VerificationStatus = Field(
        ...,
        description="Document verification status"
    )
    extracted_data: Optional[Dict[str, str]] = Field(
        default=None,
        description="OCR extracted data if available"
    )
    risk_factors: List[str] = Field(
        default_factory=list,
        description="Identified security risk factors"
    )
    processing_time_ms: int = Field(
        ...,
        description="Processing time in milliseconds"
    )

class PlanBRequest(BaseModelConfig):
    """
    Plan B request for alternative loan offers.
    
    Triggered when primary loan application is rejected.
    """
    
    session_id: str = Field(
        ...,
        description="Session identifier"
    )
    rejection_reason: str = Field(
        ...,
        description="Reason for primary loan rejection"
    )
    user_preferences: Optional[Dict[str, Any]] = Field(
        default=None,
        description="User preferences for alternative offers"
    )

class PlanBResponse(BaseModelConfig):
    """
    Plan B response with alternative loan offers.
    
    Returns customized alternative loan products.
    """
    
    session_id: str = Field(
        ...,
        description="Session identifier"
    )
    alternative_offers: List[Dict[str, Any]] = Field(
        ...,
        description="List of alternative loan offers"
    )
    recommendation_reason: str = Field(
        ...,
        description="Explanation for the recommendations"
    )
    estimated_approval_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Estimated approval probability for alternatives"
    )

# Export all models
__all__ = [
    # Enums
    'AgentStep',
    'KYCStatus', 
    'EmploymentType',
    'DocumentType',
    'VerificationStatus',
    'LoanPurpose',
    
    # Core Models
    'SentimentScore',
    'TrustScore',
    'KYCDocument',
    'EMICalculation',
    'UserProfile',
    'LoanRequest',
    'AgentState',
    
    # API Models
    'ChatMessageRequest',
    'ChatMessageResponse',
    'FileUploadRequest',
    'FileUploadResponse',
    'PlanBRequest',
    'PlanBResponse',
    
    # Base
    'BaseModelConfig',
    'ValidationError'
]