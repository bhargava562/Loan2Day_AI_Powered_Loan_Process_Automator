"""
KYC Document Repository for Loan2Day Agentic AI Fintech Platform

This module implements KYC document-specific database operations following
the Routes → Services → Repositories pattern. Integrates with SGS security
scanning and SBEF trust scoring for document verification.

Key Features:
- Async SQLAlchemy operations for KYC document management
- SGS security score tracking with Decimal precision
- SBEF trust score calculation and storage
- OCR text extraction and structured data storage
- Document verification workflow management

Author: Lead AI Architect, Loan2Day Fintech Platform
"""

from typing import List, Optional, Dict, Any
from decimal import Decimal
from uuid import UUID
from datetime import datetime
import logging

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, desc, asc
from sqlalchemy.exc import SQLAlchemyError

from app.repositories.base_repository import BaseRepository, NotFoundError, ValidationError
from app.models.db_models import KYCDocument, User
from app.models.pydantic_models import KYCDocument as KYCDocumentModel

# Configure logger
logger = logging.getLogger(__name__)

class KYCDocumentRepository(BaseRepository[KYCDocument, KYCDocumentModel, KYCDocumentModel]):
    """
    Repository for KYC document-specific database operations.
    
    Handles document lifecycle management with SGS security integration
    and SBEF trust scoring for verification workflows.
    """
    
    def __init__(self):
        """Initialize KYC document repository."""
        super().__init__(KYCDocument)
        logger.info("KYCDocumentRepository initialized")
    
    async def get_by_user_id(
        self, 
        db: AsyncSession, 
        user_id: UUID,
        document_type: Optional[str] = None,
        limit: int = 50
    ) -> List[KYCDocument]:
        """
        Get KYC documents by user ID.
        
        Args:
            db: Database session
            user_id: User's UUID
            document_type: Optional document type filter
            limit: Maximum number of documents to return
            
        Returns:
            List[KYCDocument]: User's KYC documents
        """
        logger.debug(f"Getting KYC documents for user: {user_id}")
        
        try:
            query = select(KYCDocument).where(KYCDocument.user_id == user_id)
            
            if document_type:
                query = query.where(KYCDocument.document_type == document_type)
            
            query = query.order_by(desc(KYCDocument.created_at)).limit(limit)
            
            result = await db.execute(query)
            documents = result.scalars().all()
            
            logger.debug(f"Found {len(documents)} KYC documents for user: {user_id}")
            return documents
            
        except SQLAlchemyError as e:
            logger.error(f"Database error getting KYC documents by user: {str(e)}")
            raise
    
    async def get_by_verification_status(
        self,
        db: AsyncSession,
        status: str,
        limit: int = 100
    ) -> List[KYCDocument]:
        """
        Get KYC documents by verification status.
        
        Args:
            db: Database session
            status: Verification status to filter by
            limit: Maximum number of documents to return
            
        Returns:
            List[KYCDocument]: Documents with specified status
        """
        logger.debug(f"Getting KYC documents by status: {status}")
        
        try:
            result = await db.execute(
                select(KYCDocument)
                .where(KYCDocument.verification_status == status)
                .order_by(desc(KYCDocument.created_at))
                .limit(limit)
            )
            documents = result.scalars().all()
            
            logger.debug(f"Found {len(documents)} documents with status: {status}")
            return documents
            
        except SQLAlchemyError as e:
            logger.error(f"Database error getting documents by status: {str(e)}")
            raise
    
    async def get_by_document_type(
        self,
        db: AsyncSession,
        document_type: str,
        limit: int = 100
    ) -> List[KYCDocument]:
        """
        Get KYC documents by document type.
        
        Args:
            db: Database session
            document_type: Document type to filter by
            limit: Maximum number of documents to return
            
        Returns:
            List[KYCDocument]: Documents of specified type
        """
        logger.debug(f"Getting KYC documents by type: {document_type}")
        
        try:
            result = await db.execute(
                select(KYCDocument)
                .where(KYCDocument.document_type == document_type)
                .order_by(desc(KYCDocument.created_at))
                .limit(limit)
            )
            documents = result.scalars().all()
            
            logger.debug(f"Found {len(documents)} documents of type: {document_type}")
            return documents
            
        except SQLAlchemyError as e:
            logger.error(f"Database error getting documents by type: {str(e)}")
            raise
    
    async def get_by_sgs_score_range(
        self,
        db: AsyncSession,
        min_score: Decimal,
        max_score: Decimal,
        limit: int = 100
    ) -> List[KYCDocument]:
        """
        Get KYC documents within SGS score range (LQM Standard: Decimal precision).
        
        Args:
            db: Database session
            min_score: Minimum SGS security score (Decimal for precision)
            max_score: Maximum SGS security score (Decimal for precision)
            limit: Maximum number of documents to return
            
        Returns:
            List[KYCDocument]: Documents within SGS score range
            
        Raises:
            ValidationError: If score range is invalid
        """
        logger.debug(f"Getting KYC documents by SGS score range: {min_score} - {max_score}")
        
        # Validate score range (LQM Standard: Decimal validation)
        if min_score < Decimal('0') or max_score > Decimal('100'):
            raise ValidationError("SGS score must be between 0 and 100")
        if min_score > max_score:
            raise ValidationError("Minimum score cannot be greater than maximum score")
        
        try:
            result = await db.execute(
                select(KYCDocument)
                .where(
                    and_(
                        KYCDocument.sgs_security_score >= min_score,
                        KYCDocument.sgs_security_score <= max_score
                    )
                )
                .order_by(desc(KYCDocument.sgs_security_score))
                .limit(limit)
            )
            documents = result.scalars().all()
            
            logger.debug(f"Found {len(documents)} documents in SGS score range")
            return documents
            
        except SQLAlchemyError as e:
            logger.error(f"Database error getting documents by SGS score range: {str(e)}")
            raise
    
    async def get_by_sbef_trust_score_range(
        self,
        db: AsyncSession,
        min_trust_score: Decimal,
        max_trust_score: Decimal,
        limit: int = 100
    ) -> List[KYCDocument]:
        """
        Get KYC documents within SBEF trust score range (LQM Standard: Decimal precision).
        
        Args:
            db: Database session
            min_trust_score: Minimum SBEF trust score (Decimal for precision)
            max_trust_score: Maximum SBEF trust score (Decimal for precision)
            limit: Maximum number of documents to return
            
        Returns:
            List[KYCDocument]: Documents within SBEF trust score range
            
        Raises:
            ValidationError: If trust score range is invalid
        """
        logger.debug(f"Getting KYC documents by SBEF trust score range: {min_trust_score} - {max_trust_score}")
        
        # Validate trust score range (LQM Standard: Decimal validation)
        if min_trust_score < Decimal('0') or max_trust_score > Decimal('100'):
            raise ValidationError("SBEF trust score must be between 0 and 100")
        if min_trust_score > max_trust_score:
            raise ValidationError("Minimum trust score cannot be greater than maximum trust score")
        
        try:
            result = await db.execute(
                select(KYCDocument)
                .where(
                    and_(
                        KYCDocument.sbef_trust_score >= min_trust_score,
                        KYCDocument.sbef_trust_score <= max_trust_score
                    )
                )
                .order_by(desc(KYCDocument.sbef_trust_score))
                .limit(limit)
            )
            documents = result.scalars().all()
            
            logger.debug(f"Found {len(documents)} documents in SBEF trust score range")
            return documents
            
        except SQLAlchemyError as e:
            logger.error(f"Database error getting documents by SBEF trust score range: {str(e)}")
            raise
    
    async def update_verification_status(
        self,
        db: AsyncSession,
        document_id: UUID,
        new_status: str,
        verification_notes: Optional[str] = None
    ) -> KYCDocument:
        """
        Update document verification status with audit trail.
        
        Args:
            db: Database session
            document_id: Document UUID to update
            new_status: New verification status
            verification_notes: Optional verification notes
            
        Returns:
            KYCDocument: Updated document
            
        Raises:
            NotFoundError: If document not found
            ValidationError: If status is invalid
        """
        logger.debug(f"Updating verification status for document: {document_id}")
        
        # Validate status (following Agent State pattern)
        valid_statuses = ['pending', 'verified', 'rejected', 'requires_review', 'expired']
        if new_status not in valid_statuses:
            raise ValidationError(f"Invalid verification status: {new_status}")
        
        try:
            # Get existing document
            document = await self.get(db, document_id)
            if not document:
                raise NotFoundError(f"KYC document not found: {document_id}")
            
            # Update verification status and timestamp
            document.verification_status = new_status
            document.verification_timestamp = datetime.utcnow()
            
            if verification_notes:
                document.verification_notes = verification_notes
            
            await db.commit()
            await db.refresh(document)
            
            logger.info(f"Updated verification status for document {document_id} to: {new_status}")
            return document
            
        except SQLAlchemyError as e:
            await db.rollback()
            logger.error(f"Database error updating verification status: {str(e)}")
            raise
    
    async def update_sgs_security_score(
        self,
        db: AsyncSession,
        document_id: UUID,
        sgs_score: Decimal,
        sgs_scan_details: Optional[Dict[str, Any]] = None
    ) -> KYCDocument:
        """
        Update SGS security score with scan details (LQM Standard: Decimal precision).
        
        Args:
            db: Database session
            document_id: Document UUID to update
            sgs_score: SGS security score (Decimal for precision)
            sgs_scan_details: Optional SGS scan details JSON
            
        Returns:
            KYCDocument: Updated document
            
        Raises:
            NotFoundError: If document not found
            ValidationError: If SGS score is invalid
        """
        logger.debug(f"Updating SGS security score for document: {document_id}")
        
        # Validate SGS score (LQM Standard: Decimal validation)
        if sgs_score < Decimal('0') or sgs_score > Decimal('100'):
            raise ValidationError("SGS security score must be between 0 and 100")
        
        try:
            # Get existing document
            document = await self.get(db, document_id)
            if not document:
                raise NotFoundError(f"KYC document not found: {document_id}")
            
            # Update SGS security score and scan timestamp
            document.sgs_security_score = sgs_score
            document.sgs_scan_timestamp = datetime.utcnow()
            
            if sgs_scan_details:
                document.sgs_scan_details = sgs_scan_details
            
            await db.commit()
            await db.refresh(document)
            
            logger.info(f"Updated SGS security score for document {document_id} to: {sgs_score}")
            return document
            
        except SQLAlchemyError as e:
            await db.rollback()
            logger.error(f"Database error updating SGS security score: {str(e)}")
            raise
    
    async def update_sbef_trust_score(
        self,
        db: AsyncSession,
        document_id: UUID,
        sbef_trust_score: Decimal,
        sbef_analysis_details: Optional[Dict[str, Any]] = None
    ) -> KYCDocument:
        """
        Update SBEF trust score with analysis details (LQM Standard: Decimal precision).
        
        Args:
            db: Database session
            document_id: Document UUID to update
            sbef_trust_score: SBEF trust score (Decimal for precision)
            sbef_analysis_details: Optional SBEF analysis details JSON
            
        Returns:
            KYCDocument: Updated document
            
        Raises:
            NotFoundError: If document not found
            ValidationError: If SBEF trust score is invalid
        """
        logger.debug(f"Updating SBEF trust score for document: {document_id}")
        
        # Validate SBEF trust score (LQM Standard: Decimal validation)
        if sbef_trust_score < Decimal('0') or sbef_trust_score > Decimal('100'):
            raise ValidationError("SBEF trust score must be between 0 and 100")
        
        try:
            # Get existing document
            document = await self.get(db, document_id)
            if not document:
                raise NotFoundError(f"KYC document not found: {document_id}")
            
            # Update SBEF trust score and analysis timestamp
            document.sbef_trust_score = sbef_trust_score
            document.sbef_analysis_timestamp = datetime.utcnow()
            
            if sbef_analysis_details:
                document.sbef_analysis_details = sbef_analysis_details
            
            await db.commit()
            await db.refresh(document)
            
            logger.info(f"Updated SBEF trust score for document {document_id} to: {sbef_trust_score}")
            return document
            
        except SQLAlchemyError as e:
            await db.rollback()
            logger.error(f"Database error updating SBEF trust score: {str(e)}")
            raise
    
    async def update_ocr_extracted_data(
        self,
        db: AsyncSession,
        document_id: UUID,
        ocr_extracted_text: str,
        structured_data: Optional[Dict[str, Any]] = None
    ) -> KYCDocument:
        """
        Update OCR extracted text and structured data.
        
        Args:
            db: Database session
            document_id: Document UUID to update
            ocr_extracted_text: OCR extracted text content
            structured_data: Optional structured data extracted from OCR
            
        Returns:
            KYCDocument: Updated document
            
        Raises:
            NotFoundError: If document not found
            ValidationError: If OCR text is empty
        """
        logger.debug(f"Updating OCR extracted data for document: {document_id}")
        
        # Validate OCR text
        if not ocr_extracted_text or not ocr_extracted_text.strip():
            raise ValidationError("OCR extracted text cannot be empty")
        
        try:
            # Get existing document
            document = await self.get(db, document_id)
            if not document:
                raise NotFoundError(f"KYC document not found: {document_id}")
            
            # Update OCR data and processing timestamp
            document.ocr_extracted_text = ocr_extracted_text.strip()
            document.ocr_processing_timestamp = datetime.utcnow()
            
            if structured_data:
                document.structured_data = structured_data
            
            await db.commit()
            await db.refresh(document)
            
            logger.info(f"Updated OCR extracted data for document: {document_id}")
            return document
            
        except SQLAlchemyError as e:
            await db.rollback()
            logger.error(f"Database error updating OCR extracted data: {str(e)}")
            raise
    
    async def get_pending_verification_documents(
        self,
        db: AsyncSession,
        limit: int = 50
    ) -> List[KYCDocument]:
        """
        Get documents pending verification for Agent processing queue.
        
        Args:
            db: Database session
            limit: Maximum number of documents to return
            
        Returns:
            List[KYCDocument]: Documents pending verification
        """
        logger.debug("Getting documents pending verification")
        
        try:
            result = await db.execute(
                select(KYCDocument)
                .where(KYCDocument.verification_status == 'pending')
                .order_by(asc(KYCDocument.created_at))  # FIFO processing
                .limit(limit)
            )
            documents = result.scalars().all()
            
            logger.debug(f"Found {len(documents)} documents pending verification")
            return documents
            
        except SQLAlchemyError as e:
            logger.error(f"Database error getting pending verification documents: {str(e)}")
            raise
    
    async def get_high_risk_documents(
        self,
        db: AsyncSession,
        sgs_threshold: Decimal = Decimal('30'),
        sbef_threshold: Decimal = Decimal('40'),
        limit: int = 100
    ) -> List[KYCDocument]:
        """
        Get high-risk documents based on SGS and SBEF scores (LQM Standard: Decimal precision).
        
        Args:
            db: Database session
            sgs_threshold: SGS security score threshold (below = high risk)
            sbef_threshold: SBEF trust score threshold (below = high risk)
            limit: Maximum number of documents to return
            
        Returns:
            List[KYCDocument]: High-risk documents requiring review
        """
        logger.debug(f"Getting high-risk documents (SGS < {sgs_threshold}, SBEF < {sbef_threshold})")
        
        try:
            result = await db.execute(
                select(KYCDocument)
                .where(
                    or_(
                        KYCDocument.sgs_security_score < sgs_threshold,
                        KYCDocument.sbef_trust_score < sbef_threshold
                    )
                )
                .order_by(
                    asc(KYCDocument.sgs_security_score),
                    asc(KYCDocument.sbef_trust_score)
                )
                .limit(limit)
            )
            documents = result.scalars().all()
            
            logger.warning(f"Found {len(documents)} high-risk documents requiring review")
            return documents
            
        except SQLAlchemyError as e:
            logger.error(f"Database error getting high-risk documents: {str(e)}")
            raise
    
    async def get_documents_by_date_range(
        self,
        db: AsyncSession,
        start_date: datetime,
        end_date: datetime,
        limit: int = 1000
    ) -> List[KYCDocument]:
        """
        Get KYC documents within date range for reporting and analytics.
        
        Args:
            db: Database session
            start_date: Start date for range filter
            end_date: End date for range filter
            limit: Maximum number of documents to return
            
        Returns:
            List[KYCDocument]: Documents within date range
            
        Raises:
            ValidationError: If date range is invalid
        """
        logger.debug(f"Getting KYC documents by date range: {start_date} - {end_date}")
        
        # Validate date range
        if start_date > end_date:
            raise ValidationError("Start date cannot be after end date")
        
        try:
            result = await db.execute(
                select(KYCDocument)
                .where(
                    and_(
                        KYCDocument.created_at >= start_date,
                        KYCDocument.created_at <= end_date
                    )
                )
                .order_by(desc(KYCDocument.created_at))
                .limit(limit)
            )
            documents = result.scalars().all()
            
            logger.debug(f"Found {len(documents)} documents in date range")
            return documents
            
        except SQLAlchemyError as e:
            logger.error(f"Database error getting documents by date range: {str(e)}")
            raise
    
    async def get_verification_statistics(
        self,
        db: AsyncSession,
        user_id: Optional[UUID] = None
    ) -> Dict[str, int]:
        """
        Get verification statistics for dashboard analytics.
        
        Args:
            db: Database session
            user_id: Optional user ID to filter statistics
            
        Returns:
            Dict[str, int]: Verification status counts
        """
        logger.debug(f"Getting verification statistics for user: {user_id}")
        
        try:
            query = select(
                KYCDocument.verification_status,
                func.count(KYCDocument.id).label('count')
            )
            
            if user_id:
                query = query.where(KYCDocument.user_id == user_id)
            
            query = query.group_by(KYCDocument.verification_status)
            
            result = await db.execute(query)
            stats = {row.verification_status: row.count for row in result}
            
            logger.debug(f"Verification statistics: {stats}")
            return stats
            
        except SQLAlchemyError as e:
            logger.error(f"Database error getting verification statistics: {str(e)}")
            raise
    
    async def bulk_update_verification_status(
        self,
        db: AsyncSession,
        document_ids: List[UUID],
        new_status: str,
        verification_notes: Optional[str] = None
    ) -> List[KYCDocument]:
        """
        Bulk update verification status for multiple documents.
        
        Args:
            db: Database session
            document_ids: List of document UUIDs to update
            new_status: New verification status
            verification_notes: Optional verification notes
            
        Returns:
            List[KYCDocument]: Updated documents
            
        Raises:
            ValidationError: If status is invalid or document list is empty
        """
        logger.debug(f"Bulk updating verification status for {len(document_ids)} documents")
        
        # Validate inputs
        if not document_ids:
            raise ValidationError("Document IDs list cannot be empty")
        
        valid_statuses = ['pending', 'verified', 'rejected', 'requires_review', 'expired']
        if new_status not in valid_statuses:
            raise ValidationError(f"Invalid verification status: {new_status}")
        
        try:
            # Get all documents to update
            result = await db.execute(
                select(KYCDocument).where(KYCDocument.id.in_(document_ids))
            )
            documents = result.scalars().all()
            
            if len(documents) != len(document_ids):
                found_ids = {doc.id for doc in documents}
                missing_ids = set(document_ids) - found_ids
                raise NotFoundError(f"Documents not found: {missing_ids}")
            
            # Update all documents
            for document in documents:
                document.verification_status = new_status
                document.verification_timestamp = datetime.utcnow()
                if verification_notes:
                    document.verification_notes = verification_notes
            
            await db.commit()
            
            # Refresh all documents
            for document in documents:
                await db.refresh(document)
            
            logger.info(f"Bulk updated {len(documents)} documents to status: {new_status}")
            return documents
            
        except SQLAlchemyError as e:
            await db.rollback()
            logger.error(f"Database error in bulk update verification status: {str(e)}")
            raise