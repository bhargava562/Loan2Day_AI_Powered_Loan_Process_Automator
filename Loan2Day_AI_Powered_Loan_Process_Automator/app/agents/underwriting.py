"""
Underwriting Agent - Risk Assessment and EMI Calculations

This agent handles loan underwriting decisions using the LQM (Logic Quantization Module)
for zero-hallucination mathematics. All monetary calculations strictly use decimal.Decimal
to ensure financial precision and regulatory compliance.

Key Responsibilities:
- EMI calculation using reducing balance formula via LQM
- Risk assessment based on credit score, income, and banking data
- Loan eligibility determination with clear reasoning
- Integration with mock banking APIs for income verification
- Storage of all results using Decimal types in loan_details

Architecture: Worker Agent in Master-Worker pattern
Security: All calculations validated through LQM Standard
Precision: Decimal arithmetic with 2 decimal place validation

Author: Lead AI Architect, Loan2Day Fintech Platform
"""

from typing import Dict, List, Optional, Any, Union
from decimal import Decimal
from datetime import datetime
import logging
from enum import Enum

# Import core modules
from app.core.lqm import calculate_emi, validate_monetary_input, LQMError, EMICalculation
from app.core.mock_bank import get_mock_banking_api, CreditScoreResult, IncomeVerificationResult
from app.models.pydantic_models import AgentState, UserProfile, LoanRequest

# Configure logger
logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Risk assessment levels for loan applications."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class LoanDecision(Enum):
    """Loan decision outcomes."""
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    CONDITIONAL = "CONDITIONAL"
    MANUAL_REVIEW = "MANUAL_REVIEW"

class UnderwritingError(Exception):
    """Base exception for underwriting agent errors."""
    pass

class InsufficientDataError(UnderwritingError):
    """Raised when insufficient data is available for underwriting."""
    pass

class RiskAssessmentError(UnderwritingError):
    """Raised when risk assessment fails."""
    pass

class RiskAssessment:
    """
    Comprehensive risk assessment result with detailed scoring.
    
    All monetary values use decimal.Decimal following LQM Standard.
    """
    
    def __init__(
        self,
        overall_risk_level: RiskLevel,
        risk_score: float,
        credit_risk_score: float,
        income_risk_score: float,
        debt_to_income_ratio: Decimal,
        loan_to_income_ratio: Decimal,
        affordability_score: float,
        risk_factors: List[str],
        mitigating_factors: List[str],
        recommended_decision: LoanDecision,
        max_eligible_amount_in_cents: Decimal,
        recommended_rate: Decimal,
        assessment_timestamp: datetime
    ):
        self.overall_risk_level = overall_risk_level
        self.risk_score = risk_score
        self.credit_risk_score = credit_risk_score
        self.income_risk_score = income_risk_score
        self.debt_to_income_ratio = debt_to_income_ratio
        self.loan_to_income_ratio = loan_to_income_ratio
        self.affordability_score = affordability_score
        self.risk_factors = risk_factors
        self.mitigating_factors = mitigating_factors
        self.recommended_decision = recommended_decision
        self.max_eligible_amount_in_cents = max_eligible_amount_in_cents
        self.recommended_rate = recommended_rate
        self.assessment_timestamp = assessment_timestamp
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert risk assessment to dictionary for AgentState storage."""
        return {
            "overall_risk_level": self.overall_risk_level.value,
            "risk_score": self.risk_score,
            "credit_risk_score": self.credit_risk_score,
            "income_risk_score": self.income_risk_score,
            "debt_to_income_ratio": str(self.debt_to_income_ratio),
            "loan_to_income_ratio": str(self.loan_to_income_ratio),
            "affordability_score": self.affordability_score,
            "risk_factors": self.risk_factors,
            "mitigating_factors": self.mitigating_factors,
            "recommended_decision": self.recommended_decision.value,
            "max_eligible_amount_in_cents": str(self.max_eligible_amount_in_cents),
            "recommended_rate": str(self.recommended_rate),
            "assessment_timestamp": self.assessment_timestamp.isoformat()
        }

class UnderwritingAgent:
    """
    Underwriting Agent - The Accountant of Loan2Day.
    
    This agent performs comprehensive risk assessment and EMI calculations
    using the LQM module to ensure zero-hallucination mathematics. All
    monetary operations strictly use decimal.Decimal for precision.
    
    The agent integrates with mock banking APIs to verify income and
    assess creditworthiness before making loan decisions.
    """
    
    def __init__(self):
        """Initialize the Underwriting Agent."""
        self.banking_api = get_mock_banking_api()
        logger.info("UnderwritingAgent initialized successfully")
    
    async def calculate_emi_for_loan(
        self,
        principal_in_cents: Union[Decimal, str, int],
        annual_rate: Union[Decimal, str, int],
        tenure_months: int
    ) -> EMICalculation:
        """
        Calculate EMI using LQM module with zero-hallucination mathematics.
        
        This method serves as the primary interface for EMI calculations
        in the Loan2Day system, ensuring all monetary operations use
        decimal.Decimal for precision and regulatory compliance.
        
        Args:
            principal_in_cents: Loan principal amount in cents
            annual_rate: Annual interest rate as percentage
            tenure_months: Loan tenure in months
            
        Returns:
            EMICalculation: Complete EMI calculation results
            
        Raises:
            LQMError: If LQM validation fails
            UnderwritingError: If calculation parameters are invalid
        """
        logger.info(
            f"Calculating EMI - Principal: {principal_in_cents}, "
            f"Rate: {annual_rate}%, Tenure: {tenure_months} months"
        )
        
        try:
            # Use LQM module for zero-hallucination mathematics
            emi_result = calculate_emi(principal_in_cents, annual_rate, tenure_months)
            
            logger.info(
                f"EMI calculation completed - EMI: {emi_result.emi_in_cents}, "
                f"Total Interest: {emi_result.total_interest_in_cents}"
            )
            
            return emi_result
            
        except LQMError as e:
            logger.error(f"LQM calculation error: {str(e)}")
            raise UnderwritingError(f"EMI calculation failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in EMI calculation: {str(e)}")
            raise UnderwritingError(f"EMI calculation failed: {str(e)}")
    
    async def assess_credit_risk(
        self,
        user_profile: UserProfile,
        credit_score_result: Optional[CreditScoreResult] = None
    ) -> float:
        """
        Assess credit risk based on user profile and credit bureau data.
        
        Args:
            user_profile: User profile information
            credit_score_result: Credit bureau lookup result (optional)
            
        Returns:
            float: Credit risk score (0.0 = low risk, 1.0 = high risk)
        """
        logger.info(f"Assessing credit risk for user: {user_profile.user_id}")
        
        # Get credit score from profile or bureau data
        if credit_score_result:
            credit_score = credit_score_result.credit_score
        elif user_profile.credit_score:
            credit_score = user_profile.credit_score
        else:
            # Fetch from mock banking API
            try:
                # Extract PAN from user_id for demo (in production, use actual PAN)
                mock_pan = f"ABCDE1234{user_profile.user_id[-1]}"
                credit_result = self.banking_api.lookup_credit_score(mock_pan)
                credit_score = credit_result.credit_score
            except Exception as e:
                logger.warning(f"Could not fetch credit score: {str(e)}")
                credit_score = 650  # Default moderate score
        
        # Calculate credit risk score (inverted - higher credit score = lower risk)
        if credit_score >= 750:
            credit_risk = 0.1  # Excellent credit
        elif credit_score >= 650:
            credit_risk = 0.3  # Good credit
        elif credit_score >= 550:
            credit_risk = 0.6  # Fair credit
        else:
            credit_risk = 0.9  # Poor credit
        
        logger.info(f"Credit risk assessment: Score={credit_score}, Risk={credit_risk}")
        return credit_risk
    
    async def assess_income_risk(
        self,
        user_profile: UserProfile,
        loan_request: LoanRequest,
        income_verification: Optional[IncomeVerificationResult] = None
    ) -> float:
        """
        Assess income-related risk factors.
        
        Args:
            user_profile: User profile information
            loan_request: Loan request details
            income_verification: Income verification result (optional)
            
        Returns:
            float: Income risk score (0.0 = low risk, 1.0 = high risk)
        """
        logger.info(f"Assessing income risk for user: {user_profile.user_id}")
        
        # Use verified income if available, otherwise use declared income
        if income_verification:
            monthly_income = income_verification.verified_monthly_income_in_cents
            income_stability = income_verification.income_stability_score
        else:
            monthly_income = user_profile.income_in_cents
            income_stability = 0.7  # Default moderate stability
        
        # Calculate loan-to-income ratio (LQM Standard: Decimal arithmetic)
        monthly_income_decimal = validate_monetary_input(monthly_income, "monthly_income")
        loan_amount_decimal = validate_monetary_input(loan_request.amount_in_cents, "loan_amount")
        
        # Annual income for ratio calculation
        annual_income = monthly_income_decimal * Decimal('12')
        loan_to_income_ratio = loan_amount_decimal / annual_income
        
        # Risk assessment based on loan-to-income ratio
        if loan_to_income_ratio <= Decimal('2.0'):
            ratio_risk = 0.1  # Very safe
        elif loan_to_income_ratio <= Decimal('4.0'):
            ratio_risk = 0.3  # Moderate
        elif loan_to_income_ratio <= Decimal('6.0'):
            ratio_risk = 0.6  # High
        else:
            ratio_risk = 0.9  # Very high
        
        # Adjust for income stability
        stability_adjustment = (1.0 - income_stability) * 0.3
        income_risk = min(1.0, ratio_risk + stability_adjustment)
        
        logger.info(
            f"Income risk assessment: Ratio={loan_to_income_ratio}, "
            f"Stability={income_stability}, Risk={income_risk}"
        )
        
        return income_risk
    
    async def calculate_affordability(
        self,
        user_profile: UserProfile,
        loan_request: LoanRequest,
        emi_calculation: EMICalculation
    ) -> float:
        """
        Calculate loan affordability based on income and existing obligations.
        
        Args:
            user_profile: User profile information
            loan_request: Loan request details
            emi_calculation: EMI calculation result
            
        Returns:
            float: Affordability score (0.0 = unaffordable, 1.0 = highly affordable)
        """
        logger.info(f"Calculating affordability for user: {user_profile.user_id}")
        
        # Get monthly income (LQM Standard: Decimal arithmetic)
        monthly_income = validate_monetary_input(user_profile.income_in_cents, "monthly_income")
        proposed_emi = validate_monetary_input(emi_calculation.emi_in_cents, "proposed_emi")
        
        # Get existing EMI obligations
        existing_emi = Decimal('0.00')
        if loan_request.existing_emi_in_cents:
            existing_emi = validate_monetary_input(loan_request.existing_emi_in_cents, "existing_emi")
        
        # Calculate total EMI burden
        total_emi = proposed_emi + existing_emi
        
        # Calculate EMI-to-income ratio
        emi_to_income_ratio = total_emi / monthly_income
        
        # Affordability scoring (conservative approach)
        if emi_to_income_ratio <= Decimal('0.30'):  # ≤30% of income
            affordability = 1.0  # Highly affordable
        elif emi_to_income_ratio <= Decimal('0.40'):  # ≤40% of income
            affordability = 0.8  # Good affordability
        elif emi_to_income_ratio <= Decimal('0.50'):  # ≤50% of income
            affordability = 0.5  # Moderate affordability
        elif emi_to_income_ratio <= Decimal('0.60'):  # ≤60% of income
            affordability = 0.3  # Poor affordability
        else:
            affordability = 0.1  # Unaffordable
        
        logger.info(
            f"Affordability assessment: EMI Ratio={emi_to_income_ratio}, "
            f"Score={affordability}"
        )
        
        return affordability
    
    async def perform_risk_assessment(
        self,
        user_profile: UserProfile,
        loan_request: LoanRequest
    ) -> RiskAssessment:
        """
        Perform comprehensive risk assessment for loan application.
        
        This method integrates credit risk, income risk, and affordability
        analysis to provide a holistic risk assessment with clear reasoning.
        
        Args:
            user_profile: User profile information
            loan_request: Loan request details
            
        Returns:
            RiskAssessment: Comprehensive risk assessment result
            
        Raises:
            RiskAssessmentError: If risk assessment fails
            InsufficientDataError: If required data is missing
        """
        logger.info(
            f"Starting comprehensive risk assessment - User: {user_profile.user_id}, "
            f"Loan Amount: {loan_request.amount_in_cents}"
        )
        
        try:
            # Validate required data
            if not user_profile.income_in_cents or user_profile.income_in_cents <= 0:
                raise InsufficientDataError("Valid income information required")
            
            # Calculate EMI for affordability assessment
            # Use market rate if no rate specified (12% default)
            interest_rate = loan_request.requested_rate or Decimal('12.00')
            emi_calculation = await self.calculate_emi_for_loan(
                loan_request.amount_in_cents,
                interest_rate,
                loan_request.tenure_months
            )
            
            # Perform individual risk assessments
            credit_risk = await self.assess_credit_risk(user_profile)
            income_risk = await self.assess_income_risk(user_profile, loan_request)
            affordability = await self.calculate_affordability(user_profile, loan_request, emi_calculation)
            
            # Calculate overall risk score (weighted combination)
            overall_risk_score = (
                credit_risk * 0.4 +      # 40% weight on credit history
                income_risk * 0.35 +     # 35% weight on income stability
                (1.0 - affordability) * 0.25  # 25% weight on affordability (inverted)
            )
            
            # Determine risk level
            if overall_risk_score <= 0.3:
                risk_level = RiskLevel.LOW
            elif overall_risk_score <= 0.5:
                risk_level = RiskLevel.MEDIUM
            elif overall_risk_score <= 0.7:
                risk_level = RiskLevel.HIGH
            else:
                risk_level = RiskLevel.CRITICAL
            
            # Calculate debt-to-income and loan-to-income ratios (LQM Standard)
            monthly_income = validate_monetary_input(user_profile.income_in_cents, "monthly_income")
            loan_amount = validate_monetary_input(loan_request.amount_in_cents, "loan_amount")
            existing_emi = validate_monetary_input(
                loan_request.existing_emi_in_cents or 0, "existing_emi"
            )
            
            debt_to_income_ratio = (existing_emi + emi_calculation.emi_in_cents) / monthly_income
            loan_to_income_ratio = loan_amount / (monthly_income * Decimal('12'))
            
            # Identify risk factors
            risk_factors = []
            if credit_risk > 0.6:
                risk_factors.append("Poor credit history")
            if income_risk > 0.6:
                risk_factors.append("High income risk or instability")
            if affordability < 0.5:
                risk_factors.append("Poor loan affordability")
            if debt_to_income_ratio > Decimal('0.5'):
                risk_factors.append("High debt-to-income ratio")
            if loan_to_income_ratio > Decimal('5.0'):
                risk_factors.append("Excessive loan amount relative to income")
            
            # Identify mitigating factors
            mitigating_factors = []
            if credit_risk < 0.3:
                mitigating_factors.append("Excellent credit history")
            if affordability > 0.8:
                mitigating_factors.append("Strong affordability profile")
            if user_profile.employment_type.value == "SALARIED":
                mitigating_factors.append("Stable salaried employment")
            if debt_to_income_ratio < Decimal('0.3'):
                mitigating_factors.append("Low existing debt burden")
            
            # Make loan decision recommendation
            if risk_level == RiskLevel.LOW and affordability > 0.7:
                decision = LoanDecision.APPROVED
                max_eligible = loan_amount
                recommended_rate = Decimal('10.50')  # Best rate
            elif risk_level == RiskLevel.MEDIUM and affordability > 0.5:
                decision = LoanDecision.APPROVED
                max_eligible = min(loan_amount, monthly_income * Decimal('36'))  # 3x annual income
                recommended_rate = Decimal('12.00')  # Standard rate
            elif risk_level == RiskLevel.HIGH and affordability > 0.3:
                decision = LoanDecision.CONDITIONAL
                max_eligible = min(loan_amount, monthly_income * Decimal('24'))  # 2x annual income
                recommended_rate = Decimal('15.00')  # Higher rate
            else:
                decision = LoanDecision.REJECTED
                max_eligible = Decimal('0.00')
                recommended_rate = Decimal('18.00')  # Highest rate (for reference)
            
            # Create risk assessment result
            risk_assessment = RiskAssessment(
                overall_risk_level=risk_level,
                risk_score=overall_risk_score,
                credit_risk_score=credit_risk,
                income_risk_score=income_risk,
                debt_to_income_ratio=debt_to_income_ratio,
                loan_to_income_ratio=loan_to_income_ratio,
                affordability_score=affordability,
                risk_factors=risk_factors,
                mitigating_factors=mitigating_factors,
                recommended_decision=decision,
                max_eligible_amount_in_cents=max_eligible,
                recommended_rate=recommended_rate,
                assessment_timestamp=datetime.now()
            )
            
            logger.info(
                f"Risk assessment completed - User: {user_profile.user_id}, "
                f"Risk Level: {risk_level.value}, Decision: {decision.value}, "
                f"Max Eligible: {max_eligible}"
            )
            
            return risk_assessment
            
        except (LQMError, InsufficientDataError) as e:
            logger.error(f"Risk assessment failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in risk assessment: {str(e)}")
            raise RiskAssessmentError(f"Risk assessment failed: {str(e)}")
    
    async def validate_loan_terms(
        self,
        loan_request: LoanRequest,
        risk_assessment: RiskAssessment
    ) -> Dict[str, Any]:
        """
        Validate proposed loan terms against risk assessment results.
        
        Args:
            loan_request: Loan request details
            risk_assessment: Risk assessment result
            
        Returns:
            Dict[str, Any]: Validation result with recommendations
        """
        logger.info(f"Validating loan terms for amount: {loan_request.amount_in_cents}")
        
        validation_result = {
            "is_valid": False,
            "validation_errors": [],
            "recommendations": [],
            "adjusted_terms": {}
        }
        
        # Validate loan amount against maximum eligible
        requested_amount = validate_monetary_input(loan_request.amount_in_cents, "requested_amount")
        max_eligible = risk_assessment.max_eligible_amount_in_cents
        
        if requested_amount > max_eligible:
            validation_result["validation_errors"].append(
                f"Requested amount exceeds maximum eligible amount of ₹{max_eligible/100:,.2f}"
            )
            validation_result["adjusted_terms"]["amount_in_cents"] = str(max_eligible)
        
        # Validate interest rate
        if loan_request.requested_rate:
            requested_rate = validate_monetary_input(loan_request.requested_rate, "requested_rate")
            if requested_rate < risk_assessment.recommended_rate:
                validation_result["validation_errors"].append(
                    f"Requested rate too low. Minimum rate: {risk_assessment.recommended_rate}%"
                )
                validation_result["adjusted_terms"]["rate"] = str(risk_assessment.recommended_rate)
        
        # Add recommendations based on risk assessment
        if risk_assessment.recommended_decision == LoanDecision.APPROVED:
            validation_result["is_valid"] = len(validation_result["validation_errors"]) == 0
            validation_result["recommendations"].append("Loan approved with standard terms")
        elif risk_assessment.recommended_decision == LoanDecision.CONDITIONAL:
            validation_result["recommendations"].append("Conditional approval with adjusted terms")
            validation_result["recommendations"].append("Consider co-applicant or collateral")
        else:
            validation_result["recommendations"].append("Loan application rejected")
            validation_result["recommendations"].append("Consider Plan B alternatives")
        
        logger.info(f"Loan terms validation completed: Valid={validation_result['is_valid']}")
        return validation_result
    
    async def process_underwriting_request(
        self,
        agent_state: AgentState
    ) -> Dict[str, Any]:
        """
        Process complete underwriting request and update agent state.
        
        This is the main entry point for underwriting processing in the
        Master-Worker agent pattern. It performs comprehensive risk assessment
        and updates the agent state with results.
        
        Args:
            agent_state: Current agent state with user and loan information
            
        Returns:
            Dict[str, Any]: Underwriting result with updated loan details
            
        Raises:
            UnderwritingError: If underwriting processing fails
            InsufficientDataError: If required data is missing
        """
        logger.info(f"Processing underwriting request for session: {agent_state.session_id}")
        
        try:
            # Validate required data in agent state
            if not agent_state.user_profile:
                raise InsufficientDataError("User profile required for underwriting")
            
            if not agent_state.loan_request:
                raise InsufficientDataError("Loan request required for underwriting")
            
            # Perform risk assessment
            risk_assessment = await self.perform_risk_assessment(
                agent_state.user_profile,
                agent_state.loan_request
            )
            
            # Calculate EMI with recommended terms
            final_rate = risk_assessment.recommended_rate
            emi_calculation = await self.calculate_emi_for_loan(
                risk_assessment.max_eligible_amount_in_cents,
                final_rate,
                agent_state.loan_request.tenure_months
            )
            
            # Validate loan terms
            validation_result = await self.validate_loan_terms(
                agent_state.loan_request,
                risk_assessment
            )
            
            # Prepare underwriting result (LQM Standard: Decimal values)
            underwriting_result = {
                "decision": risk_assessment.recommended_decision.value,
                "risk_assessment": risk_assessment.to_dict(),
                "emi_calculation": emi_calculation.to_dict(),
                "validation_result": validation_result,
                "approved_amount_in_cents": str(risk_assessment.max_eligible_amount_in_cents),
                "approved_rate": str(risk_assessment.recommended_rate),
                "approved_emi_in_cents": str(emi_calculation.emi_in_cents),
                "processing_timestamp": datetime.now().isoformat()
            }
            
            logger.info(
                f"Underwriting completed - Session: {agent_state.session_id}, "
                f"Decision: {risk_assessment.recommended_decision.value}, "
                f"Amount: {risk_assessment.max_eligible_amount_in_cents}"
            )
            
            return underwriting_result
            
        except (UnderwritingError, InsufficientDataError) as e:
            logger.error(f"Underwriting processing failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in underwriting processing: {str(e)}")
            raise UnderwritingError(f"Underwriting processing failed: {str(e)}")

# Export main classes and functions
__all__ = [
    'UnderwritingAgent',
    'RiskAssessment',
    'RiskLevel',
    'LoanDecision',
    'UnderwritingError',
    'InsufficientDataError',
    'RiskAssessmentError'
]