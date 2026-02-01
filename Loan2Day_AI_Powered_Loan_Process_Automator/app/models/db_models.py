"""
SQLAlchemy Database Models for Loan2Day Platform

This module defines all database models using SQLAlchemy ORM with proper
indexing for performance optimization. All monetary fields use DECIMAL type
following the LQM Standard to ensure precision in financial calculations.

Key Features:
- Users table with proper indexing on phone and email
- Loan applications with comprehensive tracking
- Agent sessions for state persistence
- KYC documents with file path and OCR text storage
- Proper foreign key relationships and constraints
- Performance-optimized indexes for WHERE and JOIN operations

Architecture: Database layer in Routes -> Services -> Repositories pattern
Precision: DECIMAL types for all monetary values (LQM Standard)
Performance: Strategic indexing for query optimization

Author: Lead AI Architect, Loan2Day Fintech Platform
"""

from typing import Optional, Dict, Any
from datetime import datetime
from decimal import Decimal
import uuid

from sqlalchemy import (
    Column, String, Integer, DECIMAL, DateTime, Boolean, Text, JSON,
    ForeignKey, Index, CheckConstraint, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Session
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func

# Create declarative base
Base = declarative_base()

class TimestampMixin:
    """Mixin for created_at and updated_at timestamps."""
    
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="Record creation timestamp"
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        comment="Record last update timestamp"
    )

class User(Base, TimestampMixin):
    """
    User table for storing customer information.
    
    All monetary fields use DECIMAL type following LQM Standard.
    Proper indexing on phone and email for fast lookups.
    """
    
    __tablename__ = "users"
    
    # Primary key
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Unique user identifier"
    )
    
    # Personal information
    name = Column(
        String(255),
        nullable=False,
        comment="User's full name"
    )
    phone = Column(
        String(20),
        nullable=False,
        unique=True,
        comment="User's phone number in E.164 format"
    )
    email = Column(
        String(255),
        nullable=False,
        unique=True,
        comment="User's email address"
    )
    
    # Financial information (LQM Standard: DECIMAL for monetary values)
    income_in_cents = Column(
        DECIMAL(15, 2),
        nullable=False,
        comment="User's monthly income in cents (DECIMAL for precision)"
    )
    employment_type = Column(
        String(50),
        nullable=False,
        comment="Type of employment (SALARIED, SELF_EMPLOYED, etc.)"
    )
    credit_score = Column(
        Integer,
        nullable=True,
        comment="CIBIL credit score (300-900 range)"
    )
    
    # Additional information
    age = Column(
        Integer,
        nullable=True,
        comment="User's age in years"
    )
    city = Column(
        String(100),
        nullable=True,
        comment="User's city of residence"
    )
    
    # Status and verification
    is_active = Column(
        Boolean,
        default=True,
        nullable=False,
        comment="Whether user account is active"
    )
    kyc_verified = Column(
        Boolean,
        default=False,
        nullable=False,
        comment="Whether user has completed KYC verification"
    )
    
    # Relationships
    loan_applications = relationship(
        "LoanApplication",
        back_populates="user",
        cascade="all, delete-orphan"
    )
    agent_sessions = relationship(
        "AgentSession",
        back_populates="user",
        cascade="all, delete-orphan"
    )
    kyc_documents = relationship(
        "KYCDocument",
        back_populates="user",
        cascade="all, delete-orphan"
    )
    
    # Constraints
    __table_args__ = (
        CheckConstraint(
            "income_in_cents > 0",
            name="check_positive_income"
        ),
        CheckConstraint(
            "credit_score IS NULL OR (credit_score >= 300 AND credit_score <= 900)",
            name="check_credit_score_range"
        ),
        CheckConstraint(
            "age IS NULL OR (age >= 18 AND age <= 100)",
            name="check_age_range"
        ),
        Index("idx_users_phone", "phone"),
        Index("idx_users_email", "email"),
        Index("idx_users_credit_score", "credit_score"),
        Index("idx_users_employment_type", "employment_type"),
        Index("idx_users_city", "city"),
        Index("idx_users_created_at", "created_at"),
        {"comment": "User information with financial data and KYC status"}
    )
    
    def __repr__(self):
        return f"<User(id={self.id}, name='{self.name}', phone='{self.phone}')>"

class LoanApplication(Base, TimestampMixin):
    """
    Loan application table with comprehensive tracking.
    
    All monetary fields use DECIMAL type following LQM Standard.
    Includes risk assessment, decision tracking, and EMI calculations.
    """
    
    __tablename__ = "loan_applications"
    
    # Primary key
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Unique loan application identifier"
    )
    
    # Foreign key to user
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        comment="Reference to user who applied for loan"
    )
    
    # Loan details (LQM Standard: DECIMAL for monetary values)
    amount_in_cents = Column(
        DECIMAL(15, 2),
        nullable=False,
        comment="Requested loan amount in cents (DECIMAL for precision)"
    )
    approved_amount_in_cents = Column(
        DECIMAL(15, 2),
        nullable=True,
        comment="Approved loan amount in cents (may differ from requested)"
    )
    tenure_months = Column(
        Integer,
        nullable=False,
        comment="Loan tenure in months"
    )
    interest_rate = Column(
        DECIMAL(5, 2),
        nullable=True,
        comment="Approved interest rate as percentage"
    )
    emi_in_cents = Column(
        DECIMAL(15, 2),
        nullable=True,
        comment="Calculated EMI amount in cents"
    )
    
    # Loan purpose and type
    purpose = Column(
        String(100),
        nullable=False,
        comment="Purpose of the loan (PERSONAL, HOME_IMPROVEMENT, etc.)"
    )
    loan_type = Column(
        String(50),
        default="PERSONAL",
        nullable=False,
        comment="Type of loan product"
    )
    
    # Application status and decision
    status = Column(
        String(50),
        default="PENDING",
        nullable=False,
        comment="Current application status (PENDING, APPROVED, REJECTED, etc.)"
    )
    decision = Column(
        String(50),
        nullable=True,
        comment="Final underwriting decision"
    )
    decision_date = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="When the decision was made"
    )
    decision_reason = Column(
        Text,
        nullable=True,
        comment="Reason for approval/rejection decision"
    )
    
    # Risk assessment (LQM Standard: DECIMAL for scores)
    fraud_score = Column(
        DECIMAL(5, 4),
        nullable=True,
        comment="Calculated fraud risk score (0.0000-1.0000)"
    )
    credit_risk_score = Column(
        DECIMAL(5, 4),
        nullable=True,
        comment="Credit risk assessment score"
    )
    income_risk_score = Column(
        DECIMAL(5, 4),
        nullable=True,
        comment="Income stability risk score"
    )
    overall_risk_score = Column(
        DECIMAL(5, 4),
        nullable=True,
        comment="Overall risk assessment score"
    )
    
    # Processing information
    loan_id = Column(
        String(50),
        nullable=True,
        unique=True,
        comment="External loan reference number"
    )
    agent_session_id = Column(
        String(100),
        nullable=True,
        comment="Associated agent session ID"
    )
    processing_notes = Column(
        Text,
        nullable=True,
        comment="Internal processing notes"
    )
    
    # Disbursement information
    disbursed_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="When the loan was disbursed"
    )
    disbursed_amount_in_cents = Column(
        DECIMAL(15, 2),
        nullable=True,
        comment="Actual disbursed amount in cents"
    )
    
    # Relationships
    user = relationship("User", back_populates="loan_applications")
    
    # Constraints
    __table_args__ = (
        CheckConstraint(
            "amount_in_cents > 0",
            name="check_positive_amount"
        ),
        CheckConstraint(
            "approved_amount_in_cents IS NULL OR approved_amount_in_cents > 0",
            name="check_positive_approved_amount"
        ),
        CheckConstraint(
            "tenure_months > 0 AND tenure_months <= 360",
            name="check_tenure_range"
        ),
        CheckConstraint(
            "interest_rate IS NULL OR (interest_rate >= 0 AND interest_rate <= 50)",
            name="check_interest_rate_range"
        ),
        CheckConstraint(
            "fraud_score IS NULL OR (fraud_score >= 0 AND fraud_score <= 1)",
            name="check_fraud_score_range"
        ),
        Index("idx_loan_applications_user_id", "user_id"),
        Index("idx_loan_applications_status", "status"),
        Index("idx_loan_applications_decision", "decision"),
        Index("idx_loan_applications_loan_id", "loan_id"),
        Index("idx_loan_applications_created_at", "created_at"),
        Index("idx_loan_applications_decision_date", "decision_date"),
        Index("idx_loan_applications_purpose", "purpose"),
        Index("idx_loan_applications_fraud_score", "fraud_score"),
        {"comment": "Loan applications with risk assessment and decision tracking"}
    )
    
    def __repr__(self):
        return f"<LoanApplication(id={self.id}, user_id={self.user_id}, amount={self.amount_in_cents}, status='{self.status}')>"

class AgentSession(Base, TimestampMixin):
    """
    Agent session table for state persistence.
    
    Stores serialized AgentState data with proper indexing for fast retrieval.
    Supports the Master-Worker agent pattern with session recovery.
    """
    
    __tablename__ = "agent_sessions"
    
    # Primary key
    session_id = Column(
        String(100),
        primary_key=True,
        comment="Unique session identifier"
    )
    
    # Foreign key to user
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        comment="Reference to user who owns this session"
    )
    
    # Session state information
    current_step = Column(
        String(50),
        nullable=False,
        default="GREETING",
        comment="Current step in agent state machine"
    )
    kyc_status = Column(
        String(50),
        nullable=False,
        default="PENDING",
        comment="Current KYC verification status"
    )
    
    # State data (using JSONB for PostgreSQL performance)
    state_data = Column(
        JSONB,
        nullable=False,
        comment="Serialized AgentState data in JSON format"
    )
    
    # Session metadata
    is_active = Column(
        Boolean,
        default=True,
        nullable=False,
        comment="Whether session is currently active"
    )
    last_activity_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        comment="Last activity timestamp for session cleanup"
    )
    expires_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Session expiration timestamp"
    )
    
    # Processing statistics
    message_count = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Number of messages processed in this session"
    )
    total_processing_time_ms = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Total processing time in milliseconds"
    )
    
    # Relationships
    user = relationship("User", back_populates="agent_sessions")
    
    # Constraints
    __table_args__ = (
        Index("idx_agent_sessions_user_id", "user_id"),
        Index("idx_agent_sessions_current_step", "current_step"),
        Index("idx_agent_sessions_kyc_status", "kyc_status"),
        Index("idx_agent_sessions_is_active", "is_active"),
        Index("idx_agent_sessions_last_activity", "last_activity_at"),
        Index("idx_agent_sessions_expires_at", "expires_at"),
        Index("idx_agent_sessions_created_at", "created_at"),
        # Composite index for common queries
        Index("idx_agent_sessions_user_active", "user_id", "is_active"),
        {"comment": "Agent session state persistence with performance optimization"}
    )
    
    def __repr__(self):
        return f"<AgentSession(session_id='{self.session_id}', user_id={self.user_id}, step='{self.current_step}')>"

class KYCDocument(Base, TimestampMixin):
    """
    KYC document table with file storage and OCR text.
    
    Stores document metadata, SGS security scores, and extracted text
    for verification purposes. Includes trust scores from SBEF algorithm.
    """
    
    __tablename__ = "kyc_documents"
    
    # Primary key
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Unique document identifier"
    )
    
    # Foreign key to user
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        comment="Reference to user who uploaded document"
    )
    
    # Document information
    document_type = Column(
        String(50),
        nullable=False,
        comment="Type of KYC document (PAN_CARD, AADHAAR_CARD, etc.)"
    )
    original_filename = Column(
        String(255),
        nullable=False,
        comment="Original filename of uploaded document"
    )
    file_path = Column(
        String(500),
        nullable=False,
        comment="Secure file path for stored document"
    )
    file_size_bytes = Column(
        Integer,
        nullable=False,
        comment="File size in bytes"
    )
    content_type = Column(
        String(100),
        nullable=False,
        comment="MIME content type of the file"
    )
    
    # Security and verification
    sgs_score = Column(
        DECIMAL(5, 4),
        nullable=False,
        comment="SGS security score (0.0000-1.0000)"
    )
    verification_status = Column(
        String(50),
        nullable=False,
        default="PENDING",
        comment="Document verification status"
    )
    fraud_score = Column(
        DECIMAL(5, 4),
        nullable=True,
        comment="Document-specific fraud score"
    )
    
    # OCR and text extraction
    ocr_text = Column(
        Text,
        nullable=True,
        comment="Extracted text from OCR processing"
    )
    extracted_data = Column(
        JSONB,
        nullable=True,
        comment="Structured data extracted from document"
    )
    
    # SBEF trust scoring
    trust_score = Column(
        DECIMAL(5, 4),
        nullable=True,
        comment="SBEF trust score for data conflict resolution"
    )
    trust_score_details = Column(
        JSONB,
        nullable=True,
        comment="Detailed trust score calculation data"
    )
    
    # Processing information
    processed_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="When document processing was completed"
    )
    verified_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="When document was verified"
    )
    processing_notes = Column(
        Text,
        nullable=True,
        comment="Internal processing notes"
    )
    
    # Relationships
    user = relationship("User", back_populates="kyc_documents")
    
    # Constraints
    __table_args__ = (
        CheckConstraint(
            "file_size_bytes > 0",
            name="check_positive_file_size"
        ),
        CheckConstraint(
            "sgs_score >= 0 AND sgs_score <= 1",
            name="check_sgs_score_range"
        ),
        CheckConstraint(
            "fraud_score IS NULL OR (fraud_score >= 0 AND fraud_score <= 1)",
            name="check_fraud_score_range"
        ),
        CheckConstraint(
            "trust_score IS NULL OR (trust_score >= 0 AND trust_score <= 1)",
            name="check_trust_score_range"
        ),
        Index("idx_kyc_documents_user_id", "user_id"),
        Index("idx_kyc_documents_document_type", "document_type"),
        Index("idx_kyc_documents_verification_status", "verification_status"),
        Index("idx_kyc_documents_sgs_score", "sgs_score"),
        Index("idx_kyc_documents_fraud_score", "fraud_score"),
        Index("idx_kyc_documents_created_at", "created_at"),
        Index("idx_kyc_documents_processed_at", "processed_at"),
        # Composite index for common queries
        Index("idx_kyc_documents_user_type", "user_id", "document_type"),
        Index("idx_kyc_documents_user_status", "user_id", "verification_status"),
        {"comment": "KYC documents with security scores and OCR text extraction"}
    )
    
    def __repr__(self):
        return f"<KYCDocument(id={self.id}, user_id={self.user_id}, type='{self.document_type}', status='{self.verification_status}')>"

class AuditLog(Base, TimestampMixin):
    """
    Audit log table for tracking system events and security incidents.
    
    Provides comprehensive audit trail for compliance and security monitoring.
    """
    
    __tablename__ = "audit_logs"
    
    # Primary key
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Unique audit log entry identifier"
    )
    
    # Event information
    event_type = Column(
        String(100),
        nullable=False,
        comment="Type of event (LOGIN, DOCUMENT_UPLOAD, LOAN_DECISION, etc.)"
    )
    event_category = Column(
        String(50),
        nullable=False,
        comment="Event category (SECURITY, BUSINESS, SYSTEM, etc.)"
    )
    severity = Column(
        String(20),
        nullable=False,
        default="INFO",
        comment="Event severity (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    
    # User and session context
    user_id = Column(
        UUID(as_uuid=True),
        nullable=True,
        comment="User associated with the event (if applicable)"
    )
    session_id = Column(
        String(100),
        nullable=True,
        comment="Session associated with the event (if applicable)"
    )
    
    # Event details
    message = Column(
        Text,
        nullable=False,
        comment="Human-readable event description"
    )
    event_data = Column(
        JSONB,
        nullable=True,
        comment="Structured event data and context"
    )
    
    # Request context
    ip_address = Column(
        String(45),
        nullable=True,
        comment="Client IP address (IPv4 or IPv6)"
    )
    user_agent = Column(
        Text,
        nullable=True,
        comment="Client user agent string"
    )
    request_id = Column(
        String(100),
        nullable=True,
        comment="Request trace ID for correlation"
    )
    
    # System context
    component = Column(
        String(100),
        nullable=True,
        comment="System component that generated the event"
    )
    hostname = Column(
        String(255),
        nullable=True,
        comment="Server hostname where event occurred"
    )
    
    # Constraints
    __table_args__ = (
        Index("idx_audit_logs_event_type", "event_type"),
        Index("idx_audit_logs_event_category", "event_category"),
        Index("idx_audit_logs_severity", "severity"),
        Index("idx_audit_logs_user_id", "user_id"),
        Index("idx_audit_logs_session_id", "session_id"),
        Index("idx_audit_logs_created_at", "created_at"),
        Index("idx_audit_logs_ip_address", "ip_address"),
        Index("idx_audit_logs_component", "component"),
        # Composite indexes for common queries
        Index("idx_audit_logs_user_created", "user_id", "created_at"),
        Index("idx_audit_logs_type_created", "event_type", "created_at"),
        Index("idx_audit_logs_severity_created", "severity", "created_at"),
        {"comment": "Audit log for security and compliance monitoring"}
    )
    
    def __repr__(self):
        return f"<AuditLog(id={self.id}, event_type='{self.event_type}', severity='{self.severity}')>"

# Database utility functions

def create_all_tables(engine):
    """
    Create all database tables.
    
    Args:
        engine: SQLAlchemy engine instance
    """
    Base.metadata.create_all(bind=engine)

def drop_all_tables(engine):
    """
    Drop all database tables (use with caution).
    
    Args:
        engine: SQLAlchemy engine instance
    """
    Base.metadata.drop_all(bind=engine)

def get_table_info() -> Dict[str, Any]:
    """
    Get information about all defined tables.
    
    Returns:
        Dict[str, Any]: Table information summary
    """
    tables = {}
    
    for table_name, table in Base.metadata.tables.items():
        tables[table_name] = {
            "columns": len(table.columns),
            "indexes": len(table.indexes),
            "constraints": len(table.constraints),
            "foreign_keys": len([fk for col in table.columns for fk in col.foreign_keys]),
            "comment": table.comment
        }
    
    return {
        "total_tables": len(Base.metadata.tables),
        "tables": tables
    }

# Export all models and utilities
__all__ = [
    'Base',
    'TimestampMixin',
    'User',
    'LoanApplication',
    'AgentSession',
    'KYCDocument',
    'AuditLog',
    'create_all_tables',
    'drop_all_tables',
    'get_table_info'
]