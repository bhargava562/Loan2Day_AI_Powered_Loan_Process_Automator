"""
Audit Log Repository for Loan2Day Agentic AI Fintech Platform

This module implements audit log-specific database operations following
the Routes → Services → Repositories pattern. Provides comprehensive
audit trail for compliance and security monitoring.

Key Features:
- Async SQLAlchemy operations for audit log management
- Event categorization and severity tracking
- Security incident logging and monitoring
- Compliance audit trail with proper indexing
- Performance optimized for high-volume logging

Author: Lead AI Architect, Loan2Day Fintech Platform
"""

from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime, timedelta
import logging

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, desc, asc
from sqlalchemy.exc import SQLAlchemyError

from app.repositories.base_repository import BaseRepository, ValidationError
from app.models.db_models import AuditLog

# Configure logger
logger = logging.getLogger(__name__)

class AuditLogRepository(BaseRepository[AuditLog, Dict[str, Any], Dict[str, Any]]):
    """
    Repository for audit log database operations.
    
    Handles comprehensive audit trail management for security monitoring,
    compliance tracking, and system event logging.
    """
    
    def __init__(self):
        """Initialize audit log repository."""
        super().__init__(AuditLog)
        logger.info("AuditLogRepository initialized")
    
    async def create_audit_entry(
        self,
        db: AsyncSession,
        event_type: str,
        event_category: str,
        message: str,
        severity: str = "INFO",
        user_id: Optional[UUID] = None,
        session_id: Optional[str] = None,
        event_data: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        request_id: Optional[str] = None,
        component: Optional[str] = None,
        hostname: Optional[str] = None
    ) -> AuditLog:
        """
        Create new audit log entry.
        
        Args:
            db: Database session
            event_type: Type of event (LOGIN, DOCUMENT_UPLOAD, etc.)
            event_category: Event category (SECURITY, BUSINESS, SYSTEM)
            message: Human-readable event description
            severity: Event severity (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            user_id: User associated with event (optional)
            session_id: Session associated with event (optional)
            event_data: Structured event data (optional)
            ip_address: Client IP address (optional)
            user_agent: Client user agent (optional)
            request_id: Request trace ID (optional)
            component: System component (optional)
            hostname: Server hostname (optional)
            
        Returns:
            AuditLog: Created audit log entry
            
        Raises:
            ValidationError: If required fields are invalid
        """
        logger.debug(f"Creating audit entry: {event_type} - {event_category}")
        
        # Validate required fields
        if not event_type or not event_type.strip():
            raise ValidationError("Event type cannot be empty")
        if not event_category or not event_category.strip():
            raise ValidationError("Event category cannot be empty")
        if not message or not message.strip():
            raise ValidationError("Message cannot be empty")
        
        # Validate severity
        valid_severities = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if severity not in valid_severities:
            raise ValidationError(f"Invalid severity: {severity}. Must be one of {valid_severities}")
        
        try:
            audit_entry = AuditLog(
                event_type=event_type.strip(),
                event_category=event_category.strip(),
                severity=severity,
                user_id=user_id,
                session_id=session_id,
                message=message.strip(),
                event_data=event_data or {},
                ip_address=ip_address,
                user_agent=user_agent,
                request_id=request_id,
                component=component,
                hostname=hostname
            )
            
            db.add(audit_entry)
            await db.commit()
            await db.refresh(audit_entry)
            
            logger.info(f"Created audit entry: {audit_entry.id} - {event_type}")
            return audit_entry
            
        except SQLAlchemyError as e:
            await db.rollback()
            logger.error(f"Database error creating audit entry: {str(e)}")
            raise
    
    async def get_by_event_type(
        self,
        db: AsyncSession,
        event_type: str,
        limit: int = 100,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None
    ) -> List[AuditLog]:
        """
        Get audit logs by event type.
        
        Args:
            db: Database session
            event_type: Event type to filter by
            limit: Maximum number of entries to return
            date_from: Start date for filtering (optional)
            date_to: End date for filtering (optional)
            
        Returns:
            List[AuditLog]: Audit logs matching criteria
        """
        logger.debug(f"Getting audit logs by event type: {event_type}")
        
        try:
            query = select(AuditLog).where(AuditLog.event_type == event_type)
            
            # Apply date filters
            if date_from:
                query = query.where(AuditLog.created_at >= date_from)
            if date_to:
                query = query.where(AuditLog.created_at <= date_to)
            
            query = query.order_by(desc(AuditLog.created_at)).limit(limit)
            
            result = await db.execute(query)
            logs = result.scalars().all()
            
            logger.debug(f"Found {len(logs)} audit logs for event type: {event_type}")
            return logs
            
        except SQLAlchemyError as e:
            logger.error(f"Database error getting logs by event type: {str(e)}")
            raise
    
    async def get_by_event_category(
        self,
        db: AsyncSession,
        event_category: str,
        limit: int = 100,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None
    ) -> List[AuditLog]:
        """
        Get audit logs by event category.
        
        Args:
            db: Database session
            event_category: Event category to filter by
            limit: Maximum number of entries to return
            date_from: Start date for filtering (optional)
            date_to: End date for filtering (optional)
            
        Returns:
            List[AuditLog]: Audit logs matching criteria
        """
        logger.debug(f"Getting audit logs by category: {event_category}")
        
        try:
            query = select(AuditLog).where(AuditLog.event_category == event_category)
            
            # Apply date filters
            if date_from:
                query = query.where(AuditLog.created_at >= date_from)
            if date_to:
                query = query.where(AuditLog.created_at <= date_to)
            
            query = query.order_by(desc(AuditLog.created_at)).limit(limit)
            
            result = await db.execute(query)
            logs = result.scalars().all()
            
            logger.debug(f"Found {len(logs)} audit logs for category: {event_category}")
            return logs
            
        except SQLAlchemyError as e:
            logger.error(f"Database error getting logs by category: {str(e)}")
            raise
    
    async def get_by_severity(
        self,
        db: AsyncSession,
        severity: str,
        limit: int = 100,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None
    ) -> List[AuditLog]:
        """
        Get audit logs by severity level.
        
        Args:
            db: Database session
            severity: Severity level to filter by
            limit: Maximum number of entries to return
            date_from: Start date for filtering (optional)
            date_to: End date for filtering (optional)
            
        Returns:
            List[AuditLog]: Audit logs matching criteria
        """
        logger.debug(f"Getting audit logs by severity: {severity}")
        
        try:
            query = select(AuditLog).where(AuditLog.severity == severity)
            
            # Apply date filters
            if date_from:
                query = query.where(AuditLog.created_at >= date_from)
            if date_to:
                query = query.where(AuditLog.created_at <= date_to)
            
            query = query.order_by(desc(AuditLog.created_at)).limit(limit)
            
            result = await db.execute(query)
            logs = result.scalars().all()
            
            logger.debug(f"Found {len(logs)} audit logs for severity: {severity}")
            return logs
            
        except SQLAlchemyError as e:
            logger.error(f"Database error getting logs by severity: {str(e)}")
            raise
    
    async def get_by_user_id(
        self,
        db: AsyncSession,
        user_id: UUID,
        limit: int = 100,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None
    ) -> List[AuditLog]:
        """
        Get audit logs for specific user.
        
        Args:
            db: Database session
            user_id: User UUID to filter by
            limit: Maximum number of entries to return
            date_from: Start date for filtering (optional)
            date_to: End date for filtering (optional)
            
        Returns:
            List[AuditLog]: Audit logs for the user
        """
        logger.debug(f"Getting audit logs for user: {user_id}")
        
        try:
            query = select(AuditLog).where(AuditLog.user_id == user_id)
            
            # Apply date filters
            if date_from:
                query = query.where(AuditLog.created_at >= date_from)
            if date_to:
                query = query.where(AuditLog.created_at <= date_to)
            
            query = query.order_by(desc(AuditLog.created_at)).limit(limit)
            
            result = await db.execute(query)
            logs = result.scalars().all()
            
            logger.debug(f"Found {len(logs)} audit logs for user: {user_id}")
            return logs
            
        except SQLAlchemyError as e:
            logger.error(f"Database error getting logs by user: {str(e)}")
            raise
    
    async def get_by_session_id(
        self,
        db: AsyncSession,
        session_id: str,
        limit: int = 100
    ) -> List[AuditLog]:
        """
        Get audit logs for specific session.
        
        Args:
            db: Database session
            session_id: Session ID to filter by
            limit: Maximum number of entries to return
            
        Returns:
            List[AuditLog]: Audit logs for the session
        """
        logger.debug(f"Getting audit logs for session: {session_id}")
        
        try:
            result = await db.execute(
                select(AuditLog)
                .where(AuditLog.session_id == session_id)
                .order_by(asc(AuditLog.created_at))  # Chronological order for session
                .limit(limit)
            )
            logs = result.scalars().all()
            
            logger.debug(f"Found {len(logs)} audit logs for session: {session_id}")
            return logs
            
        except SQLAlchemyError as e:
            logger.error(f"Database error getting logs by session: {str(e)}")
            raise
    
    async def get_security_events(
        self,
        db: AsyncSession,
        limit: int = 100,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        min_severity: str = "WARNING"
    ) -> List[AuditLog]:
        """
        Get security-related audit events.
        
        Args:
            db: Database session
            limit: Maximum number of entries to return
            date_from: Start date for filtering (optional)
            date_to: End date for filtering (optional)
            min_severity: Minimum severity level (WARNING, ERROR, CRITICAL)
            
        Returns:
            List[AuditLog]: Security audit logs
        """
        logger.debug(f"Getting security events with min severity: {min_severity}")
        
        # Define severity hierarchy
        severity_levels = {
            "DEBUG": 0,
            "INFO": 1,
            "WARNING": 2,
            "ERROR": 3,
            "CRITICAL": 4
        }
        
        min_level = severity_levels.get(min_severity, 2)
        valid_severities = [
            sev for sev, level in severity_levels.items() 
            if level >= min_level
        ]
        
        try:
            query = select(AuditLog).where(
                and_(
                    AuditLog.event_category == "SECURITY",
                    AuditLog.severity.in_(valid_severities)
                )
            )
            
            # Apply date filters
            if date_from:
                query = query.where(AuditLog.created_at >= date_from)
            if date_to:
                query = query.where(AuditLog.created_at <= date_to)
            
            query = query.order_by(desc(AuditLog.created_at)).limit(limit)
            
            result = await db.execute(query)
            logs = result.scalars().all()
            
            logger.debug(f"Found {len(logs)} security events")
            return logs
            
        except SQLAlchemyError as e:
            logger.error(f"Database error getting security events: {str(e)}")
            raise
    
    async def get_failed_operations(
        self,
        db: AsyncSession,
        limit: int = 100,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None
    ) -> List[AuditLog]:
        """
        Get failed operation audit logs (ERROR and CRITICAL severity).
        
        Args:
            db: Database session
            limit: Maximum number of entries to return
            date_from: Start date for filtering (optional)
            date_to: End date for filtering (optional)
            
        Returns:
            List[AuditLog]: Failed operation audit logs
        """
        logger.debug("Getting failed operation audit logs")
        
        try:
            query = select(AuditLog).where(
                AuditLog.severity.in_(["ERROR", "CRITICAL"])
            )
            
            # Apply date filters
            if date_from:
                query = query.where(AuditLog.created_at >= date_from)
            if date_to:
                query = query.where(AuditLog.created_at <= date_to)
            
            query = query.order_by(desc(AuditLog.created_at)).limit(limit)
            
            result = await db.execute(query)
            logs = result.scalars().all()
            
            logger.debug(f"Found {len(logs)} failed operation logs")
            return logs
            
        except SQLAlchemyError as e:
            logger.error(f"Database error getting failed operations: {str(e)}")
            raise
    
    async def search_audit_logs(
        self,
        db: AsyncSession,
        search_term: str,
        limit: int = 100,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None
    ) -> List[AuditLog]:
        """
        Search audit logs by message content or event data.
        
        Args:
            db: Database session
            search_term: Term to search for
            limit: Maximum number of entries to return
            date_from: Start date for filtering (optional)
            date_to: End date for filtering (optional)
            
        Returns:
            List[AuditLog]: Audit logs matching search term
        """
        logger.debug(f"Searching audit logs with term: {search_term}")
        
        try:
            search_pattern = f"%{search_term}%"
            
            query = select(AuditLog).where(
                or_(
                    AuditLog.message.ilike(search_pattern),
                    AuditLog.event_type.ilike(search_pattern),
                    AuditLog.component.ilike(search_pattern)
                )
            )
            
            # Apply date filters
            if date_from:
                query = query.where(AuditLog.created_at >= date_from)
            if date_to:
                query = query.where(AuditLog.created_at <= date_to)
            
            query = query.order_by(desc(AuditLog.created_at)).limit(limit)
            
            result = await db.execute(query)
            logs = result.scalars().all()
            
            logger.debug(f"Found {len(logs)} audit logs matching search term")
            return logs
            
        except SQLAlchemyError as e:
            logger.error(f"Database error searching audit logs: {str(e)}")
            raise
    
    async def get_audit_statistics(
        self,
        db: AsyncSession,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get audit log statistics for monitoring dashboard.
        
        Args:
            db: Database session
            date_from: Start date for statistics (optional)
            date_to: End date for statistics (optional)
            
        Returns:
            Dict[str, Any]: Audit statistics
        """
        logger.debug("Getting audit log statistics")
        
        try:
            # Base query with date filtering
            base_query = select(AuditLog)
            if date_from:
                base_query = base_query.where(AuditLog.created_at >= date_from)
            if date_to:
                base_query = base_query.where(AuditLog.created_at <= date_to)
            
            # Total entries
            total_result = await db.execute(
                select(func.count(AuditLog.id)).select_from(base_query.subquery())
            )
            total_entries = total_result.scalar()
            
            # Entries by severity
            severity_result = await db.execute(
                select(AuditLog.severity, func.count(AuditLog.id))
                .select_from(base_query.subquery())
                .group_by(AuditLog.severity)
            )
            severity_distribution = dict(severity_result.all())
            
            # Entries by category
            category_result = await db.execute(
                select(AuditLog.event_category, func.count(AuditLog.id))
                .select_from(base_query.subquery())
                .group_by(AuditLog.event_category)
            )
            category_distribution = dict(category_result.all())
            
            # Top event types
            event_type_result = await db.execute(
                select(AuditLog.event_type, func.count(AuditLog.id))
                .select_from(base_query.subquery())
                .group_by(AuditLog.event_type)
                .order_by(desc(func.count(AuditLog.id)))
                .limit(10)
            )
            top_event_types = dict(event_type_result.all())
            
            # Security incidents (WARNING and above in SECURITY category)
            security_incidents_result = await db.execute(
                select(func.count(AuditLog.id))
                .where(
                    and_(
                        AuditLog.event_category == "SECURITY",
                        AuditLog.severity.in_(["WARNING", "ERROR", "CRITICAL"])
                    )
                )
                .select_from(base_query.subquery())
            )
            security_incidents = security_incidents_result.scalar()
            
            # Error rate
            error_count = (
                severity_distribution.get("ERROR", 0) + 
                severity_distribution.get("CRITICAL", 0)
            )
            error_rate = (error_count / total_entries * 100) if total_entries > 0 else 0
            
            statistics = {
                "total_entries": total_entries,
                "severity_distribution": severity_distribution,
                "category_distribution": category_distribution,
                "top_event_types": top_event_types,
                "security_incidents": security_incidents,
                "error_rate_percent": round(error_rate, 2),
                "period": {
                    "from": date_from.isoformat() if date_from else None,
                    "to": date_to.isoformat() if date_to else None
                }
            }
            
            logger.debug("Audit log statistics retrieved successfully")
            return statistics
            
        except SQLAlchemyError as e:
            logger.error(f"Database error getting audit statistics: {str(e)}")
            raise
    
    async def cleanup_old_logs(
        self,
        db: AsyncSession,
        retention_days: int = 90,
        batch_size: int = 1000
    ) -> int:
        """
        Clean up old audit logs based on retention policy.
        
        Args:
            db: Database session
            retention_days: Number of days to retain logs
            batch_size: Number of logs to delete per batch
            
        Returns:
            int: Number of logs deleted
        """
        logger.info(f"Cleaning up audit logs older than {retention_days} days")
        
        try:
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            # Count logs to be deleted
            count_result = await db.execute(
                select(func.count(AuditLog.id))
                .where(AuditLog.created_at < cutoff_date)
            )
            total_to_delete = count_result.scalar()
            
            if total_to_delete == 0:
                logger.info("No old audit logs to clean up")
                return 0
            
            # Delete in batches to avoid long-running transactions
            deleted_count = 0
            while deleted_count < total_to_delete:
                # Get batch of IDs to delete
                ids_result = await db.execute(
                    select(AuditLog.id)
                    .where(AuditLog.created_at < cutoff_date)
                    .limit(batch_size)
                )
                ids_to_delete = [row[0] for row in ids_result.all()]
                
                if not ids_to_delete:
                    break
                
                # Delete batch
                delete_result = await db.execute(
                    AuditLog.__table__.delete()
                    .where(AuditLog.id.in_(ids_to_delete))
                )
                
                batch_deleted = delete_result.rowcount
                deleted_count += batch_deleted
                
                await db.commit()
                
                logger.debug(f"Deleted batch of {batch_deleted} audit logs")
            
            logger.info(f"Successfully cleaned up {deleted_count} old audit logs")
            return deleted_count
            
        except SQLAlchemyError as e:
            await db.rollback()
            logger.error(f"Database error cleaning up audit logs: {str(e)}")
            raise

# Export audit log repository
__all__ = ['AuditLogRepository']