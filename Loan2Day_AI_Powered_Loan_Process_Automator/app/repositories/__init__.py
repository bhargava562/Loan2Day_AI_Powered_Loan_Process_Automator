"""
Repository layer for Loan2Day Agentic AI Fintech Platform

This module provides the repository pattern implementation for data access
following the Routes → Services → Repositories architecture. All repositories
use async SQLAlchemy with proper error handling and LQM Standard compliance.

Key Features:
- BaseRepository with common CRUD operations
- Specialized repositories for each domain entity
- Async database operations with connection pooling
- Proper error handling and logging
- LQM Standard compliance for monetary operations

Author: Lead AI Architect, Loan2Day Fintech Platform
"""

from app.repositories.base_repository import (
    BaseRepository,
    RepositoryError,
    NotFoundError,
    DuplicateError,
    ValidationError
)
from app.repositories.user_repository import UserRepository
from app.repositories.loan_application_repository import LoanApplicationRepository
from app.repositories.kyc_document_repository import KYCDocumentRepository
from app.repositories.session_repository import SessionRepository
from app.repositories.audit_log_repository import AuditLogRepository

# Export all repository classes and exceptions
__all__ = [
    # Base repository and exceptions
    'BaseRepository',
    'RepositoryError',
    'NotFoundError',
    'DuplicateError',
    'ValidationError',
    
    # Domain repositories
    'UserRepository',
    'LoanApplicationRepository',
    'KYCDocumentRepository',
    'SessionRepository',
    'AuditLogRepository'
]