"""
User Repository for Loan2Day Agentic AI Fintech Platform

This module implements user-specific database operations following the
Routes → Services → Repositories pattern. All monetary operations use
decimal.Decimal following the LQM Standard for zero-hallucination mathematics.

Key Features:
- Async SQLAlchemy operations for user management
- Phone and email uniqueness validation
- Credit score range validation (300-900)
- Income validation with Decimal precision
- Comprehensive user profile management

Author: Lead AI Architect, Loan2Day Fintech Platform
"""

from typing import List, Optional, Dict, Any
from decimal import Decimal
from uuid import UUID
import logging

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_
from sqlalchemy.exc import SQLAlchemyError

from app.repositories.base_repository import BaseRepository, NotFoundError, DuplicateError
from app.models.db_models import User
from app.models.pydantic_models import UserProfile

# Configure logger
logger = logging.getLogger(__name__)

class UserRepository(BaseRepository[User, UserProfile, UserProfile]):
    """
    Repository for user-specific database operations.
    
    Handles user profile management with proper validation for
    financial data using the LQM Standard (Decimal types).
    """
    
    def __init__(self):
        """Initialize user repository."""
        super().__init__(User)
        logger.info("UserRepository initialized")
    
    async def get_by_phone(
        self, 
        db: AsyncSession, 
        phone: str
    ) -> Optional[User]:
        """
        Get user by phone number.
        
        Args:
            db: Database session
            phone: User's phone number
            
        Returns:
            Optional[User]: User if found, None otherwise
        """
        logger.debug(f"Getting user by phone: {phone}")
        
        try:
            result = await db.execute(
                select(User).where(User.phone == phone)
            )
            user = result.scalar_one_or_none()
            
            if user:
                logger.debug(f"Found user with phone: {phone}")
            else:
                logger.debug(f"No user found with phone: {phone}")
            
            return user
            
        except SQLAlchemyError as e:
            logger.error(f"Database error getting user by phone: {str(e)}")
            raise
    
    async def get_by_email(
        self, 
        db: AsyncSession, 
        email: str
    ) -> Optional[User]:
        """
        Get user by email address.
        
        Args:
            db: Database session
            email: User's email address
            
        Returns:
            Optional[User]: User if found, None otherwise
        """
        logger.debug(f"Getting user by email: {email}")
        
        try:
            result = await db.execute(
                select(User).where(User.email == email)
            )
            user = result.scalar_one_or_none()
            
            if user:
                logger.debug(f"Found user with email: {email}")
            else:
                logger.debug(f"No user found with email: {email}")
            
            return user
            
        except SQLAlchemyError as e:
            logger.error(f"Database error getting user by email: {str(e)}")
            raise
    
    async def check_phone_exists(
        self, 
        db: AsyncSession, 
        phone: str,
        exclude_user_id: Optional[UUID] = None
    ) -> bool:
        """
        Check if phone number already exists.
        
        Args:
            db: Database session
            phone: Phone number to check
            exclude_user_id: User ID to exclude from check (for updates)
            
        Returns:
            bool: True if phone exists, False otherwise
        """
        logger.debug(f"Checking if phone exists: {phone}")
        
        try:
            query = select(User.id).where(User.phone == phone)
            
            if exclude_user_id:
                query = query.where(User.id != exclude_user_id)
            
            result = await db.execute(query.limit(1))
            exists = result.scalar() is not None
            
            logger.debug(f"Phone {phone} exists: {exists}")
            return exists
            
        except SQLAlchemyError as e:
            logger.error(f"Database error checking phone existence: {str(e)}")
            raise
    
    async def check_email_exists(
        self, 
        db: AsyncSession, 
        email: str,
        exclude_user_id: Optional[UUID] = None
    ) -> bool:
        """
        Check if email address already exists.
        
        Args:
            db: Database session
            email: Email address to check
            exclude_user_id: User ID to exclude from check (for updates)
            
        Returns:
            bool: True if email exists, False otherwise
        """
        logger.debug(f"Checking if email exists: {email}")
        
        try:
            query = select(User.id).where(User.email == email)
            
            if exclude_user_id:
                query = query.where(User.id != exclude_user_id)
            
            result = await db.execute(query.limit(1))
            exists = result.scalar() is not None
            
            logger.debug(f"Email {email} exists: {exists}")
            return exists
            
        except SQLAlchemyError as e:
            logger.error(f"Database error checking email existence: {str(e)}")
            raise
    
    async def get_users_by_income_range(
        self,
        db: AsyncSession,
        min_income_in_cents: Decimal,
        max_income_in_cents: Decimal,
        limit: int = 100
    ) -> List[User]:
        """
        Get users within income range (LQM Standard: Decimal precision).
        
        Args:
            db: Database session
            min_income_in_cents: Minimum income in cents (Decimal)
            max_income_in_cents: Maximum income in cents (Decimal)
            limit: Maximum number of users to return
            
        Returns:
            List[User]: Users within income range
        """
        logger.debug(f"Getting users by income range: {min_income_in_cents} - {max_income_in_cents}")
        
        try:
            result = await db.execute(
                select(User)
                .where(
                    and_(
                        User.income_in_cents >= min_income_in_cents,
                        User.income_in_cents <= max_income_in_cents
                    )
                )
                .order_by(User.income_in_cents.desc())
                .limit(limit)
            )
            users = result.scalars().all()
            
            logger.debug(f"Found {len(users)} users in income range")
            return users
            
        except SQLAlchemyError as e:
            logger.error(f"Database error getting users by income range: {str(e)}")
            raise
    
    async def get_users_by_credit_score_range(
        self,
        db: AsyncSession,
        min_credit_score: int,
        max_credit_score: int,
        limit: int = 100
    ) -> List[User]:
        """
        Get users within credit score range.
        
        Args:
            db: Database session
            min_credit_score: Minimum credit score (300-900)
            max_credit_score: Maximum credit score (300-900)
            limit: Maximum number of users to return
            
        Returns:
            List[User]: Users within credit score range
        """
        logger.debug(f"Getting users by credit score range: {min_credit_score} - {max_credit_score}")
        
        try:
            result = await db.execute(
                select(User)
                .where(
                    and_(
                        User.credit_score >= min_credit_score,
                        User.credit_score <= max_credit_score,
                        User.credit_score.isnot(None)
                    )
                )
                .order_by(User.credit_score.desc())
                .limit(limit)
            )
            users = result.scalars().all()
            
            logger.debug(f"Found {len(users)} users in credit score range")
            return users
            
        except SQLAlchemyError as e:
            logger.error(f"Database error getting users by credit score range: {str(e)}")
            raise
    
    async def get_users_by_employment_type(
        self,
        db: AsyncSession,
        employment_type: str,
        limit: int = 100
    ) -> List[User]:
        """
        Get users by employment type.
        
        Args:
            db: Database session
            employment_type: Employment type to filter by
            limit: Maximum number of users to return
            
        Returns:
            List[User]: Users with specified employment type
        """
        logger.debug(f"Getting users by employment type: {employment_type}")
        
        try:
            result = await db.execute(
                select(User)
                .where(User.employment_type == employment_type)
                .order_by(User.created_at.desc())
                .limit(limit)
            )
            users = result.scalars().all()
            
            logger.debug(f"Found {len(users)} users with employment type: {employment_type}")
            return users
            
        except SQLAlchemyError as e:
            logger.error(f"Database error getting users by employment type: {str(e)}")
            raise
    
    async def search_users(
        self,
        db: AsyncSession,
        search_term: str,
        limit: int = 50
    ) -> List[User]:
        """
        Search users by name, phone, or email.
        
        Args:
            db: Database session
            search_term: Term to search for
            limit: Maximum number of users to return
            
        Returns:
            List[User]: Users matching search term
        """
        logger.debug(f"Searching users with term: {search_term}")
        
        try:
            search_pattern = f"%{search_term}%"
            
            result = await db.execute(
                select(User)
                .where(
                    or_(
                        User.name.ilike(search_pattern),
                        User.phone.ilike(search_pattern),
                        User.email.ilike(search_pattern)
                    )
                )
                .order_by(User.name)
                .limit(limit)
            )
            users = result.scalars().all()
            
            logger.debug(f"Found {len(users)} users matching search term")
            return users
            
        except SQLAlchemyError as e:
            logger.error(f"Database error searching users: {str(e)}")
            raise
    
    async def update_credit_score(
        self,
        db: AsyncSession,
        user_id: UUID,
        credit_score: int
    ) -> User:
        """
        Update user's credit score.
        
        Args:
            db: Database session
            user_id: User's UUID
            credit_score: New credit score (300-900)
            
        Returns:
            User: Updated user entity
            
        Raises:
            NotFoundError: If user not found
            ValidationError: If credit score is invalid
        """
        logger.info(f"Updating credit score for user {user_id}: {credit_score}")
        
        # Validate credit score range
        if not (300 <= credit_score <= 900):
            raise ValueError(f"Credit score must be between 300 and 900, got: {credit_score}")
        
        try:
            # Get user
            user = await self.get(db, user_id)
            if not user:
                raise NotFoundError(f"User not found with ID: {user_id}")
            
            # Update credit score
            user.credit_score = credit_score
            
            await db.commit()
            await db.refresh(user)
            
            logger.info(f"Successfully updated credit score for user {user_id}")
            return user
            
        except SQLAlchemyError as e:
            await db.rollback()
            logger.error(f"Database error updating credit score: {str(e)}")
            raise
    
    async def get_user_statistics(
        self,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """
        Get user statistics for analytics.
        
        Args:
            db: Database session
            
        Returns:
            Dict[str, Any]: User statistics
        """
        logger.debug("Getting user statistics")
        
        try:
            # Total users
            total_users_result = await db.execute(
                select(User.id).count()
            )
            total_users = total_users_result.scalar()
            
            # Users by employment type
            employment_stats_result = await db.execute(
                select(User.employment_type, User.id.count())
                .group_by(User.employment_type)
            )
            employment_stats = dict(employment_stats_result.all())
            
            # Average income (LQM Standard: Decimal precision)
            avg_income_result = await db.execute(
                select(User.income_in_cents.avg())
            )
            avg_income = avg_income_result.scalar() or Decimal('0')
            
            # Credit score distribution
            credit_score_stats_result = await db.execute(
                select(User.credit_score.avg(), User.credit_score.min(), User.credit_score.max())
                .where(User.credit_score.isnot(None))
            )
            credit_stats = credit_score_stats_result.first()
            
            statistics = {
                "total_users": total_users,
                "employment_distribution": employment_stats,
                "average_income_in_cents": str(avg_income),  # Convert Decimal to string for JSON
                "credit_score_stats": {
                    "average": credit_stats[0] if credit_stats and credit_stats[0] else None,
                    "minimum": credit_stats[1] if credit_stats and credit_stats[1] else None,
                    "maximum": credit_stats[2] if credit_stats and credit_stats[2] else None
                }
            }
            
            logger.debug("User statistics retrieved successfully")
            return statistics
            
        except SQLAlchemyError as e:
            logger.error(f"Database error getting user statistics: {str(e)}")
            raise

# Export user repository
__all__ = ['UserRepository']