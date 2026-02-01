"""
Loan Application Repository for Loan2Day Agentic AI Fintech Platform

This module implements loan application-specific database operations following
the Routes → Services → Repositories pattern. All monetary operations use
decimal.Decimal following the LQM Standard for zero-hallucination mathematics.

Key Features:
- Async SQLAlchemy operations for loan application management
- Risk score tracking with Decimal precision
- Decision workflow and status management
- EMI calculation storage with LQM Standard compliance
- Comprehensive loan application analytics

Author: Lead AI Architect, Loan2Day Fintech Platform
"""

from typing import List, Optional, Dict, Any
from decimal import Decimal
from uuid import UUID
from datetime import datetime, timedelta
import logging

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, desc, asc
from sqlalchemy.exc import SQLAlchemyError

from app.repositories.base_repository import BaseRepository, NotFoundError, ValidationError
from app.models.db_models import LoanApplication, User
from app.models.pydantic_models import LoanRequest

# Configure logger
logger = logging.getLogger(__name__)

class LoanApplicationRepository(BaseRepository[LoanApplication, LoanRequest, LoanRequest]):
    """
    Repository for loan application-specific database operations.
    
    Handles loan application lifecycle management with proper validation
    for financial data using the LQM Standard (Decimal types).
    """
    
    def __init__(self):
        """Initialize loan application repository."""
        super().__init__(LoanApplication)
        logger.info("LoanApplicationRepository initialized")
    
    async def get_by_user_id(
        self, 
        db: AsyncSession, 
        user_id: UUID,
        limit: int = 50
    ) -> List[LoanApplication]:
        """
        Get loan applications by user ID.
        
        Args:
            db: Database session
            user_id: User's UUID
            limit: Maximum number of applications to return
            
        Returns:
            List[LoanApplication]: User's loan applications
        """
        logger.debug(f"Getting loan applications for user: {user_id}")
        
        try:
            result = await db.execute(
                select(LoanApplication)
                .where(LoanApplication.user_id == user_id)
                .order_by(desc(LoanApplication.created_at))
                .limit(limit)
            )
            applications = result.scalars().all()
            
            logger.debug(f"Found {len(applications)} loan applications for user: {user_id}")
            return applications
            
        except SQLAlchemyError as e:
            logger.error(f"Database error getting loan applications by user: {str(e)}")
            raise
    
    async def get_by_status(
        self,
        db: AsyncSession,
        status: str,
        limit: int = 100
    ) -> List[LoanApplication]:
        """
        Get loan applications by status.
        
        Args:
            db: Database session
            status: Application status to filter by
            limit: Maximum number of applications to return
            
        Returns:
            List[LoanApplication]: Applications with specified status
        """
        logger.debug(f"Getting loan applications by status: {status}")
        
        try:
            result = await db.execute(
                select(LoanApplication)
                .where(LoanApplication.status == status)
                .order_by(desc(LoanApplication.created_at))
                .limit(limit)
            )
            applications = result.scalars().all()
            
            logger.debug(f"Found {len(applications)} applications with status: {status}")
            return applications
            
        except SQLAlchemyError as e:
            logger.error(f"Database error getting applications by status: {str(e)}")
            raise
    
    async def get_by_decision(
        self,
        db: AsyncSession,
        decision: str,
        limit: int = 100
    ) -> List[LoanApplication]:
        """
        Get loan applications by decision.
        
        Args:
            db: Database session
            decision: Decision to filter by (APPROVED, REJECTED, etc.)
            limit: Maximum number of applications to return
            
        Returns:
            List[LoanApplication]: Applications with specified decision
        """
        logger.debug(f"Getting loan applications by decision: {decision}")
        
        try:
            result = await db.execute(
                select(LoanApplication)
                .where(LoanApplication.decision == decision)
                .order_by(desc(LoanApplication.decision_date))
                .limit(limit)
            )
            applications = result.scalars().all()
            
            logger.debug(f"Found {len(applications)} applications with decision: {decision}")
            return applications
            
        except SQLAlchemyError as e:
            logger.error(f"Database error getting applications by decision: {str(e)}")
            raise
    
    async def get_by_amount_range(
        self,
        db: AsyncSession,
        min_amount_in_cents: Decimal,
        max_amount_in_cents: Decimal,
        limit: int = 100
    ) -> List[LoanApplication]:
        """
        Get loan applications within amount range (LQM Standard: Decimal precision).
        
        Args:
            db: Database session
            min_amount_in_cents: Minimum loan amount in cents (Decimal)
            max_amount_in_cents: Maximum loan amount in cents (Decimal)
            limit: Maximum number of applications to return
            
        Returns:
            List[LoanApplication]: Applications within amount range
        """
        logger.debug(f"Getting applications by amount range: {min_amount_in_cents} - {max_amount_in_cents}")
        
        try:
            result = await db.execute(
                select(LoanApplication)
                .where(
                    and_(
                        LoanApplication.amount_in_cents >= min_amount_in_cents,
                        LoanApplication.amount_in_cents <= max_amount_in_cents
                    )
                )
                .order_by(desc(LoanApplication.amount_in_cents))
                .limit(limit)
            )
            applications = result.scalars().all()
            
            logger.debug(f"Found {len(applications)} applications in amount range")
            return applications
            
        except SQLAlchemyError as e:
            logger.error(f"Database error getting applications by amount range: {str(e)}")
            raise
    
    async def get_by_risk_score_range(
        self,
        db: AsyncSession,
        min_risk_score: Decimal,
        max_risk_score: Decimal,
        limit: int = 100
    ) -> List[LoanApplication]:
        """
        Get loan applications within risk score range (LQM Standard: Decimal precision).
        
        Args:
            db: Database session
            min_risk_score: Minimum risk score (Decimal)
            max_risk_score: Maximum risk score (Decimal)
            limit: Maximum number of applications to return
            
        Returns:
            List[LoanApplication]: Applications within risk score range
        """
        logger.debug(f"Getting applications by risk score range: {min_risk_score} - {max_risk_score}")
        
        try:
            result = await db.execute(
                select(LoanApplication)
                .where(
                    and_(
                        LoanApplication.overall_risk_score >= min_risk_score,
                        LoanApplication.overall_risk_score <= max_risk_score,
                        LoanApplication.overall_risk_score.isnot(None)
                    )
                )
                .order_by(asc(LoanApplication.overall_risk_score))
                .limit(limit)
            )
            applications = result.scalars().all()
            
            logger.debug(f"Found {len(applications)} applications in risk score range")
            return applications
            
        except SQLAlchemyError as e:
            logger.error(f"Database error getting applications by risk score range: {str(e)}")
            raise
    
    async def update_decision(
        self,
        db: AsyncSession,
        application_id: UUID,
        decision: str,
        decision_reason: Optional[str] = None,
        approved_amount_in_cents: Optional[Decimal] = None,
        interest_rate: Optional[Decimal] = None,
        emi_in_cents: Optional[Decimal] = None
    ) -> LoanApplication:
        """
        Update loan application decision with LQM Standard compliance.
        
        Args:
            db: Database session
            application_id: Application UUID
            decision: Decision (APPROVED, REJECTED, CONDITIONAL)
            decision_reason: Reason for the decision
            approved_amount_in_cents: Approved amount in cents (Decimal)
            interest_rate: Approved interest rate (Decimal)
            emi_in_cents: Calculated EMI in cents (Decimal)
            
        Returns:
            LoanApplication: Updated application entity
            
        Raises:
            NotFoundError: If application not found
            ValidationError: If monetary values are invalid
        """
        logger.info(f"Updating decision for application {application_id}: {decision}")
        
        try:
            # Get application
            application = await self.get(db, application_id)
            if not application:
                raise NotFoundError(f"Loan application not found with ID: {application_id}")
            
            # Validate monetary fields if provided
            if approved_amount_in_cents is not None:
                if not isinstance(approved_amount_in_cents, Decimal):
                    raise ValidationError("approved_amount_in_cents must be Decimal type")
                if approved_amount_in_cents <= Decimal('0'):
                    raise ValidationError("approved_amount_in_cents must be positive")
            
            if interest_rate is not None:
                if not isinstance(interest_rate, Decimal):
                    raise ValidationError("interest_rate must be Decimal type")
                if not (Decimal('0') <= interest_rate <= Decimal('50')):
                    raise ValidationError("interest_rate must be between 0% and 50%")
            
            if emi_in_cents is not None:
                if not isinstance(emi_in_cents, Decimal):
                    raise ValidationError("emi_in_cents must be Decimal type")
                if emi_in_cents <= Decimal('0'):
                    raise ValidationError("emi_in_cents must be positive")
            
            # Update decision fields
            application.decision = decision
            application.decision_date = datetime.now()
            application.decision_reason = decision_reason
            
            # Update approved amounts if provided
            if approved_amount_in_cents is not None:
                application.approved_amount_in_cents = approved_amount_in_cents
            if interest_rate is not None:
                application.interest_rate = interest_rate
            if emi_in_cents is not None:
                application.emi_in_cents = emi_in_cents
            
            # Update status based on decision
            if decision == "APPROVED":
                application.status = "APPROVED"
            elif decision == "REJECTED":
                application.status = "REJECTED"
            elif decision == "CONDITIONAL":
                application.status = "CONDITIONAL_APPROVAL"
            
            await db.commit()
            await db.refresh(application)
            
            logger.info(f"Successfully updated decision for application {application_id}")
            return application
            
        except SQLAlchemyError as e:
            await db.rollback()
            logger.error(f"Database error updating application decision: {str(e)}")
            raise
    
    async def update_risk_scores(
        self,
        db: AsyncSession,
        application_id: UUID,
        fraud_score: Optional[Decimal] = None,
        credit_risk_score: Optional[Decimal] = None,
        income_risk_score: Optional[Decimal] = None,
        overall_risk_score: Optional[Decimal] = None
    ) -> LoanApplication:
        """
        Update risk scores with LQM Standard compliance.
        
        Args:
            db: Database session
            application_id: Application UUID
            fraud_score: Fraud risk score (Decimal, 0.0-1.0)
            credit_risk_score: Credit risk score (Decimal, 0.0-1.0)
            income_risk_score: Income risk score (Decimal, 0.0-1.0)
            overall_risk_score: Overall risk score (Decimal, 0.0-1.0)
            
        Returns:
            LoanApplication: Updated application entity
            
        Raises:
            NotFoundError: If application not found
            ValidationError: If risk scores are invalid
        """
        logger.info(f"Updating risk scores for application {application_id}")
        
        try:
            # Get application
            application = await self.get(db, application_id)
            if not application:
                raise NotFoundError(f"Loan application not found with ID: {application_id}")
            
            # Validate risk scores (must be Decimal between 0.0 and 1.0)
            risk_scores = {
                "fraud_score": fraud_score,
                "credit_risk_score": credit_risk_score,
                "income_risk_score": income_risk_score,
                "overall_risk_score": overall_risk_score
            }
            
            for score_name, score_value in risk_scores.items():
                if score_value is not None:
                    if not isinstance(score_value, Decimal):
                        raise ValidationError(f"{score_name} must be Decimal type")
                    if not (Decimal('0') <= score_value <= Decimal('1')):
                        raise ValidationError(f"{score_name} must be between 0.0 and 1.0")
            
            # Update risk scores
            if fraud_score is not None:
                application.fraud_score = fraud_score
            if credit_risk_score is not None:
                application.credit_risk_score = credit_risk_score
            if income_risk_score is not None:
                application.income_risk_score = income_risk_score
            if overall_risk_score is not None:
                application.overall_risk_score = overall_risk_score
            
            await db.commit()
            await db.refresh(application)
            
            logger.info(f"Successfully updated risk scores for application {application_id}")
            return application
            
        except SQLAlchemyError as e:
            await db.rollback()
            logger.error(f"Database error updating risk scores: {str(e)}")
            raise
    
    async def mark_disbursed(
        self,
        db: AsyncSession,
        application_id: UUID,
        disbursed_amount_in_cents: Decimal,
        loan_id: Optional[str] = None
    ) -> LoanApplication:
        """
        Mark loan application as disbursed.
        
        Args:
            db: Database session
            application_id: Application UUID
            disbursed_amount_in_cents: Actual disbursed amount in cents (Decimal)
            loan_id: External loan reference number
            
        Returns:
            LoanApplication: Updated application entity
            
        Raises:
            NotFoundError: If application not found
            ValidationError: If disbursed amount is invalid
        """
        logger.info(f"Marking application {application_id} as disbursed")
        
        try:
            # Get application
            application = await self.get(db, application_id)
            if not application:
                raise NotFoundError(f"Loan application not found with ID: {application_id}")
            
            # Validate disbursed amount
            if not isinstance(disbursed_amount_in_cents, Decimal):
                raise ValidationError("disbursed_amount_in_cents must be Decimal type")
            if disbursed_amount_in_cents <= Decimal('0'):
                raise ValidationError("disbursed_amount_in_cents must be positive")
            
            # Update disbursement fields
            application.disbursed_at = datetime.now()
            application.disbursed_amount_in_cents = disbursed_amount_in_cents
            application.status = "DISBURSED"
            
            if loan_id:
                application.loan_id = loan_id
            
            await db.commit()
            await db.refresh(application)
            
            logger.info(f"Successfully marked application {application_id} as disbursed")
            return application
            
        except SQLAlchemyError as e:
            await db.rollback()
            logger.error(f"Database error marking application as disbursed: {str(e)}")
            raise
    
    async def get_applications_for_review(
        self,
        db: AsyncSession,
        limit: int = 50
    ) -> List[LoanApplication]:
        """
        Get loan applications pending review (no decision yet).
        
        Args:
            db: Database session
            limit: Maximum number of applications to return
            
        Returns:
            List[LoanApplication]: Applications pending review
        """
        logger.debug("Getting applications pending review")
        
        try:
            result = await db.execute(
                select(LoanApplication)
                .where(
                    and_(
                        LoanApplication.decision.is_(None),
                        LoanApplication.status.in_(["PENDING", "UNDER_REVIEW"])
                    )
                )
                .order_by(asc(LoanApplication.created_at))  # FIFO processing
                .limit(limit)
            )
            applications = result.scalars().all()
            
            logger.debug(f"Found {len(applications)} applications pending review")
            return applications
            
        except SQLAlchemyError as e:
            logger.error(f"Database error getting applications for review: {str(e)}")
            raise
    
    async def get_application_statistics(
        self,
        db: AsyncSession,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get loan application statistics for analytics.
        
        Args:
            db: Database session
            date_from: Start date for statistics (optional)
            date_to: End date for statistics (optional)
            
        Returns:
            Dict[str, Any]: Application statistics
        """
        logger.debug("Getting loan application statistics")
        
        try:
            # Base query with date filtering
            base_query = select(LoanApplication)
            if date_from:
                base_query = base_query.where(LoanApplication.created_at >= date_from)
            if date_to:
                base_query = base_query.where(LoanApplication.created_at <= date_to)
            
            # Total applications
            total_result = await db.execute(
                select(func.count(LoanApplication.id)).select_from(base_query.subquery())
            )
            total_applications = total_result.scalar()
            
            # Applications by status
            status_result = await db.execute(
                select(LoanApplication.status, func.count(LoanApplication.id))
                .select_from(base_query.subquery())
                .group_by(LoanApplication.status)
            )
            status_distribution = dict(status_result.all())
            
            # Applications by decision
            decision_result = await db.execute(
                select(LoanApplication.decision, func.count(LoanApplication.id))
                .where(LoanApplication.decision.isnot(None))
                .select_from(base_query.subquery())
                .group_by(LoanApplication.decision)
            )
            decision_distribution = dict(decision_result.all())
            
            # Amount statistics (LQM Standard: Decimal precision)
            amount_stats_result = await db.execute(
                select(
                    func.avg(LoanApplication.amount_in_cents),
                    func.min(LoanApplication.amount_in_cents),
                    func.max(LoanApplication.amount_in_cents),
                    func.sum(LoanApplication.amount_in_cents)
                )
                .select_from(base_query.subquery())
            )
            amount_stats = amount_stats_result.first()
            
            # Approval rate
            approved_count = decision_distribution.get("APPROVED", 0)
            total_decided = sum(decision_distribution.values())
            approval_rate = (approved_count / total_decided * 100) if total_decided > 0 else 0
            
            # Average risk scores
            risk_stats_result = await db.execute(
                select(
                    func.avg(LoanApplication.fraud_score),
                    func.avg(LoanApplication.credit_risk_score),
                    func.avg(LoanApplication.income_risk_score),
                    func.avg(LoanApplication.overall_risk_score)
                )
                .where(LoanApplication.overall_risk_score.isnot(None))
                .select_from(base_query.subquery())
            )
            risk_stats = risk_stats_result.first()
            
            statistics = {
                "total_applications": total_applications,
                "status_distribution": status_distribution,
                "decision_distribution": decision_distribution,
                "approval_rate_percent": round(approval_rate, 2),
                "amount_statistics": {
                    "average_in_cents": str(amount_stats[0]) if amount_stats[0] else "0",
                    "minimum_in_cents": str(amount_stats[1]) if amount_stats[1] else "0",
                    "maximum_in_cents": str(amount_stats[2]) if amount_stats[2] else "0",
                    "total_requested_in_cents": str(amount_stats[3]) if amount_stats[3] else "0"
                },
                "risk_score_averages": {
                    "fraud_score": float(risk_stats[0]) if risk_stats[0] else None,
                    "credit_risk_score": float(risk_stats[1]) if risk_stats[1] else None,
                    "income_risk_score": float(risk_stats[2]) if risk_stats[2] else None,
                    "overall_risk_score": float(risk_stats[3]) if risk_stats[3] else None
                }
            }
            
            logger.debug("Loan application statistics retrieved successfully")
            return statistics
            
        except SQLAlchemyError as e:
            logger.error(f"Database error getting application statistics: {str(e)}")
            raise
    
    async def search_applications(
        self,
        db: AsyncSession,
        search_term: str,
        limit: int = 50
    ) -> List[LoanApplication]:
        """
        Search loan applications by loan ID, user details, or processing notes.
        
        Args:
            db: Database session
            search_term: Term to search for
            limit: Maximum number of applications to return
            
        Returns:
            List[LoanApplication]: Applications matching search term
        """
        logger.debug(f"Searching loan applications with term: {search_term}")
        
        try:
            search_pattern = f"%{search_term}%"
            
            result = await db.execute(
                select(LoanApplication)
                .join(User, LoanApplication.user_id == User.id)
                .where(
                    or_(
                        LoanApplication.loan_id.ilike(search_pattern),
                        LoanApplication.processing_notes.ilike(search_pattern),
                        User.name.ilike(search_pattern),
                        User.phone.ilike(search_pattern),
                        User.email.ilike(search_pattern)
                    )
                )
                .order_by(desc(LoanApplication.created_at))
                .limit(limit)
            )
            applications = result.scalars().all()
            
            logger.debug(f"Found {len(applications)} applications matching search term")
            return applications
            
        except SQLAlchemyError as e:
            logger.error(f"Database error searching applications: {str(e)}")
            raise

# Export loan application repository
__all__ = ['LoanApplicationRepository']