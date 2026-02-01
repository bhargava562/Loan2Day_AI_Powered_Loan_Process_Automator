"""
Repository layer for Loan2Day Agentic AI Fintech Platform.

This module implements the repository pattern for data access,
following the Routes → Services → Repositories architectural pattern.
All repositories use async SQLAlchemy for optimal I/O performance.
"""

from .user_repository import UserRepository
from .loan_repository import LoanRepository
from .session_repository import SessionRepository
from .kyc_repository import KYCRepository

__all__ = [
    'UserRepository',
    'LoanRepository', 
    'SessionRepository',
    'KYCRepository'
]