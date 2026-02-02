"""
Session Repository for Loan2Day Agentic AI Fintech Platform

This module implements session-specific database operations for AgentState
management following the Routes → Services → Repositories pattern.
Supports sub-millisecond state retrieval for active sessions.

Key Features:
- Async SQLAlchemy operations for session management
- JSONB storage for flexible AgentState serialization
- Session recovery and restoration capabilities
- Performance optimized for high-frequency state updates
- State machine step validation and tracking

Author: Lead AI Architect, Loan2Day Fintech Platform
"""

from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime, timedelta
import logging

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, and_, or_, func
from sqlalchemy.exc import SQLAlchemyError

from app.repositories.base_repository import BaseRepository, NotFoundError
from app.models.db_models import AgentSession
from app.models.pydantic_models import AgentState, AgentStep

# Configure logger
logger = logging.getLogger(__name__)

class SessionRepository(BaseRepository[AgentSession, AgentState, AgentState]):
    """
    Repository for agent session database operations.
    
    Handles AgentState persistence with JSONB storage for flexible
    state management and sub-millisecond retrieval performance.
    """
    
    def __init__(self):
        """Initialize session repository."""
        super().__init__(AgentSession)
        logger.info("SessionRepository initialized")
    
    async def get_by_session_id(
        self, 
        db: AsyncSession, 
        session_id: str
    ) -> Optional[AgentSession]:
        """
        Get session by session ID.
        
        Args:
            db: Database session
            session_id: Session ID string
            
        Returns:
            Optional[AgentSession]: Session if found, None otherwise
        """
        logger.debug(f"Getting session by ID: {session_id}")
        
        try:
            result = await db.execute(
                select(AgentSession).where(AgentSession.session_id == session_id)
            )
            session = result.scalar_one_or_none()
            
            if session:
                logger.debug(f"Found session: {session_id}")
                # Update last activity timestamp
                await self._update_last_activity(db, session)
            else:
                logger.debug(f"No session found: {session_id}")
            
            return session
            
        except SQLAlchemyError as e:
            logger.error(f"Database error getting session: {str(e)}")
            raise
    
    async def get_active_sessions_by_user(
        self, 
        db: AsyncSession, 
        user_id: UUID
    ) -> List[AgentSession]:
        """
        Get all active sessions for a user.
        
        Args:
            db: Database session
            user_id: User UUID
            
        Returns:
            List[AgentSession]: Active sessions for the user
        """
        logger.debug(f"Getting active sessions for user: {user_id}")
        
        try:
            result = await db.execute(
                select(AgentSession)
                .where(
                    and_(
                        AgentSession.user_id == user_id,
                        AgentSession.is_active == True
                    )
                )
                .order_by(AgentSession.last_activity_at.desc())
            )
            sessions = result.scalars().all()
            
            logger.debug(f"Found {len(sessions)} active sessions for user: {user_id}")
            return sessions
            
        except SQLAlchemyError as e:
            logger.error(f"Database error getting active sessions: {str(e)}")
            raise
    
    async def create_session(
        self,
        db: AsyncSession,
        user_id: UUID,
        initial_state_data: Dict[str, Any],
        session_id: Optional[str] = None
    ) -> AgentSession:
        """
        Create new agent session with initial state.
        
        Args:
            db: Database session
            user_id: User UUID
            initial_state_data: Initial AgentState data
            session_id: Optional session ID string (generated if not provided)
            
        Returns:
            AgentSession: Created session entity
        """
        logger.info(f"Creating new session for user: {user_id}")
        
        try:
            import uuid
            session = AgentSession(
                session_id=session_id or str(uuid.uuid4()),
                user_id=user_id,
                current_step=AgentStep.GREETING.value,
                state_data=initial_state_data,
                is_active=True,
                last_activity_at=datetime.now(),
                message_count=0,
                total_processing_time_ms=0
            )
            
            db.add(session)
            await db.commit()
            await db.refresh(session)
            
            logger.info(f"Successfully created session: {session.session_id}")
            return session
            
        except SQLAlchemyError as e:
            await db.rollback()
            logger.error(f"Database error creating session: {str(e)}")
            raise
    
    async def update_session_state(
        self,
        db: AsyncSession,
        session_id: str,
        state_data: Dict[str, Any],
        current_step: Optional[str] = None
    ) -> AgentSession:
        """
        Update session state data and current step.
        
        Args:
            db: Database session
            session_id: Session ID string
            state_data: Updated AgentState data
            current_step: New current step (optional)
            
        Returns:
            AgentSession: Updated session entity
            
        Raises:
            NotFoundError: If session not found
        """
        logger.debug(f"Updating session state: {session_id}")
        
        try:
            # Get existing session
            session = await self.get_by_session_id(db, session_id)
            if not session:
                raise NotFoundError(f"Session not found: {session_id}")
            
            # Update state data
            session.state_data = state_data
            session.last_activity_at = datetime.now()
            
            # Update current step if provided
            if current_step:
                session.current_step = current_step
            
            await db.commit()
            await db.refresh(session)
            
            logger.debug(f"Successfully updated session state: {session_id}")
            return session
            
        except SQLAlchemyError as e:
            await db.rollback()
            logger.error(f"Database error updating session state: {str(e)}")
            raise
    
    async def increment_message_count(
        self,
        db: AsyncSession,
        session_id: str
    ) -> AgentSession:
        """
        Increment message count for session.
        
        Args:
            db: Database session
            session_id: Session ID string
            
        Returns:
            AgentSession: Updated session entity
        """
        logger.debug(f"Incrementing message count for session: {session_id}")
        
        try:
            session = await self.get_by_session_id(db, session_id)
            if not session:
                raise NotFoundError(f"Session not found: {session_id}")
            
            session.message_count += 1
            session.last_activity_at = datetime.now()
            
            await db.commit()
            await db.refresh(session)
            
            return session
            
        except SQLAlchemyError as e:
            await db.rollback()
            logger.error(f"Database error incrementing message count: {str(e)}")
            raise
    
    async def increment_upload_count(
        self,
        db: AsyncSession,
        session_id: str
    ) -> AgentSession:
        """
        Increment upload count for session.
        
        Args:
            db: Database session
            session_id: Session ID string
            
        Returns:
            AgentSession: Updated session entity
        """
        logger.debug(f"Incrementing upload count for session: {session_id}")
        
        try:
            session = await self.get_by_session_id(db, session_id)
            if not session:
                raise NotFoundError(f"Session not found: {session_id}")
            
            # Note: Using total_processing_time_ms as a counter since there's no upload count field
            session.total_processing_time_ms += 1
            session.last_activity_at = datetime.now()
            
            await db.commit()
            await db.refresh(session)
            
            return session
            
        except SQLAlchemyError as e:
            await db.rollback()
            logger.error(f"Database error incrementing upload count: {str(e)}")
            raise
    
    async def trigger_plan_b(
        self,
        db: AsyncSession,
        session_id: str
    ) -> AgentSession:
        """
        Mark session as having Plan B logic triggered.
        
        Args:
            db: Database session
            session_id: Session ID string
            
        Returns:
            AgentSession: Updated session entity
        """
        logger.info(f"Triggering Plan B for session: {session_id}")
        
        try:
            session = await self.get_by_session_id(db, session_id)
            if not session:
                raise NotFoundError(f"Session not found: {session_id}")
            
            # Update current step to Plan B and mark as triggered
            session.current_step = AgentStep.PLAN_B.value
            session.last_activity_at = datetime.now()
            
            # Store Plan B trigger in state data
            if session.state_data is None:
                session.state_data = {}
            session.state_data['plan_b_triggered'] = True
            session.state_data['plan_b_triggered_at'] = datetime.now().isoformat()
            
            await db.commit()
            await db.refresh(session)
            
            logger.info(f"Successfully triggered Plan B for session: {session_id}")
            return session
            
        except SQLAlchemyError as e:
            await db.rollback()
            logger.error(f"Database error triggering Plan B: {str(e)}")
            raise
    
    async def deactivate_session(
        self,
        db: AsyncSession,
        session_id: str
    ) -> AgentSession:
        """
        Deactivate session (mark as inactive).
        
        Args:
            db: Database session
            session_id: Session ID string
            
        Returns:
            AgentSession: Updated session entity
        """
        logger.info(f"Deactivating session: {session_id}")
        
        try:
            session = await self.get_by_session_id(db, session_id)
            if not session:
                raise NotFoundError(f"Session not found: {session_id}")
            
            session.is_active = False
            session.last_activity_at = datetime.now()
            
            await db.commit()
            await db.refresh(session)
            
            logger.info(f"Successfully deactivated session: {session_id}")
            return session
            
        except SQLAlchemyError as e:
            await db.rollback()
            logger.error(f"Database error deactivating session: {str(e)}")
            raise
    
    async def get_sessions_by_step(
        self,
        db: AsyncSession,
        current_step: str,
        is_active: bool = True,
        limit: int = 100
    ) -> List[AgentSession]:
        """
        Get sessions by current step.
        
        Args:
            db: Database session
            current_step: Agent step to filter by
            is_active: Whether to include only active sessions
            limit: Maximum number of sessions to return
            
        Returns:
            List[AgentSession]: Sessions in the specified step
        """
        logger.debug(f"Getting sessions by step: {current_step}")
        
        try:
            query = select(AgentSession).where(AgentSession.current_step == current_step)
            
            if is_active:
                query = query.where(AgentSession.is_active == True)
            
            query = query.order_by(AgentSession.last_activity_at.desc()).limit(limit)
            
            result = await db.execute(query)
            sessions = result.scalars().all()
            
            logger.debug(f"Found {len(sessions)} sessions in step: {current_step}")
            return sessions
            
        except SQLAlchemyError as e:
            logger.error(f"Database error getting sessions by step: {str(e)}")
            raise
    
    async def cleanup_inactive_sessions(
        self,
        db: AsyncSession,
        inactive_threshold_hours: int = 24
    ) -> int:
        """
        Clean up sessions inactive for specified hours.
        
        Args:
            db: Database session
            inactive_threshold_hours: Hours of inactivity before cleanup
            
        Returns:
            int: Number of sessions cleaned up
        """
        logger.info(f"Cleaning up sessions inactive for {inactive_threshold_hours} hours")
        
        try:
            threshold_time = datetime.now() - timedelta(hours=inactive_threshold_hours)
            
            result = await db.execute(
                update(AgentSession)
                .where(
                    and_(
                        AgentSession.last_activity_at < threshold_time,
                        AgentSession.is_active == True
                    )
                )
                .values(is_active=False)
            )
            
            cleaned_count = result.rowcount
            await db.commit()
            
            logger.info(f"Cleaned up {cleaned_count} inactive sessions")
            return cleaned_count
            
        except SQLAlchemyError as e:
            await db.rollback()
            logger.error(f"Database error cleaning up sessions: {str(e)}")
            raise
    
    async def get_session_statistics(
        self,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """
        Get session statistics for analytics.
        
        Args:
            db: Database session
            
        Returns:
            Dict[str, Any]: Session statistics
        """
        logger.debug("Getting session statistics")
        
        try:
            # Total sessions
            total_sessions_result = await db.execute(
                select(func.count(AgentSession.session_id))
            )
            total_sessions = total_sessions_result.scalar()
            
            # Active sessions
            active_sessions_result = await db.execute(
                select(func.count(AgentSession.session_id))
                .where(AgentSession.is_active == True)
            )
            active_sessions = active_sessions_result.scalar()
            
            # Sessions by step
            step_stats_result = await db.execute(
                select(AgentSession.current_step, func.count(AgentSession.session_id))
                .where(AgentSession.is_active == True)
                .group_by(AgentSession.current_step)
            )
            step_stats = dict(step_stats_result.all())
            
            # Plan B statistics (check state_data for plan_b_triggered)
            plan_b_stats_result = await db.execute(
                select(func.count(AgentSession.session_id))
                .where(AgentSession.state_data.op('->>')('plan_b_triggered') == 'true')
            )
            plan_b_sessions = plan_b_stats_result.scalar()
            
            # Average message count
            avg_messages_result = await db.execute(
                select(func.avg(AgentSession.message_count))
                .where(AgentSession.is_active == False)  # Only completed sessions
            )
            avg_messages = avg_messages_result.scalar() or 0
            
            statistics = {
                "total_sessions": total_sessions,
                "active_sessions": active_sessions,
                "step_distribution": step_stats,
                "plan_b_triggered_sessions": plan_b_sessions,
                "average_messages_per_session": float(avg_messages),
                "plan_b_conversion_rate": (
                    (plan_b_sessions / total_sessions * 100) if total_sessions > 0 else 0
                )
            }
            
            logger.debug("Session statistics retrieved successfully")
            return statistics
            
        except SQLAlchemyError as e:
            logger.error(f"Database error getting session statistics: {str(e)}")
            raise
    
    async def _update_last_activity(
        self,
        db: AsyncSession,
        session: AgentSession
    ) -> None:
        """
        Update last activity timestamp for session.
        
        Args:
            db: Database session
            session: Session entity to update
        """
        try:
            session.last_activity_at = datetime.now()
            await db.commit()
        except SQLAlchemyError as e:
            logger.warning(f"Failed to update last activity: {str(e)}")
            # Don't raise - this is a non-critical operation

# Export session repository
__all__ = ['SessionRepository']