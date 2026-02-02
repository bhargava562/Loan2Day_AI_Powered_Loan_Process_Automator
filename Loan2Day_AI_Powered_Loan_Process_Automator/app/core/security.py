"""
Security Practices Module for Loan2Day Platform

This module implements comprehensive security practices including:
- Environment variable validation and secret management
- Security event logging for sensitive operations
- Fail-fast validation using Pydantic V2
- API key and credential protection
- Security audit and compliance checks

Key Features:
- SecretManager for secure credential handling
- SecurityAuditor for code and runtime security checks
- SecurityEventLogger for audit trail
- ValidationSecurityWrapper for fail-fast validation
- ComplianceChecker for security policy enforcement

Architecture: Defense-in-depth security with multiple validation layers
Compliance: Follows LQM Standard and financial security best practices
Performance: Minimal overhead with efficient security checks

Author: Lead AI Architect, Loan2Day Fintech Platform
"""

import os
import re
import logging
import hashlib
import secrets
from typing import Dict, List, Optional, Any, Set, Union, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
import inspect
import ast

from pydantic import BaseModel, Field, validator, ValidationError
from pydantic_settings import BaseSettings

from app.core.error_handling import (
    ErrorLogger, 
    ErrorSeverity, 
    ErrorCategory,
    ErrorCode,
    create_error_response
)

# Configure logger
logger = logging.getLogger(__name__)

# Security Event Types

class SecurityEventType(str, Enum):
    """Security event types for audit logging."""
    API_KEY_ACCESS = "API_KEY_ACCESS"
    CREDENTIAL_VALIDATION = "CREDENTIAL_VALIDATION"
    FILE_UPLOAD_SCAN = "FILE_UPLOAD_SCAN"
    AUTHENTICATION_ATTEMPT = "AUTHENTICATION_ATTEMPT"
    AUTHORIZATION_CHECK = "AUTHORIZATION_CHECK"
    DATA_ACCESS = "DATA_ACCESS"
    CONFIGURATION_CHANGE = "CONFIGURATION_CHANGE"
    SECURITY_VIOLATION = "SECURITY_VIOLATION"
    AUDIT_LOG_ACCESS = "AUDIT_LOG_ACCESS"
    ENCRYPTION_OPERATION = "ENCRYPTION_OPERATION"

class SecurityLevel(str, Enum):
    """Security levels for operations and data."""
    PUBLIC = "PUBLIC"
    INTERNAL = "INTERNAL"
    CONFIDENTIAL = "CONFIDENTIAL"
    RESTRICTED = "RESTRICTED"
    TOP_SECRET = "TOP_SECRET"

class ValidationResult(BaseModel):
    """Result of security validation."""
    is_valid: bool = Field(..., description="Whether validation passed")
    error_code: Optional[ErrorCode] = Field(default=None, description="Error code if validation failed")
    error_message: Optional[str] = Field(default=None, description="Error message if validation failed")
    security_score: float = Field(default=1.0, description="Security confidence score (0-1)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional validation metadata")

# Secret Management

@dataclass
class SecretInfo:
    """Information about a managed secret."""
    name: str
    env_var: str
    required: bool = True
    description: str = ""
    security_level: SecurityLevel = SecurityLevel.CONFIDENTIAL
    last_accessed: Optional[datetime] = None
    access_count: int = 0

class SecretManager:
    """
    Secure secret and API key management.
    
    Ensures no hardcoded secrets and provides secure access to environment variables.
    """
    
    def __init__(self):
        self.secrets: Dict[str, SecretInfo] = {}
        self.error_logger = ErrorLogger("security.secret_manager")
        self._initialize_known_secrets()
    
    def _initialize_known_secrets(self):
        """Initialize known secrets configuration."""
        known_secrets = [
            SecretInfo(
                name="database_url",
                env_var="DATABASE_URL",
                required=True,
                description="PostgreSQL database connection string",
                security_level=SecurityLevel.CONFIDENTIAL
            ),
            SecretInfo(
                name="redis_url",
                env_var="REDIS_URL",
                required=True,
                description="Redis cache connection string",
                security_level=SecurityLevel.INTERNAL
            ),
            SecretInfo(
                name="twilio_account_sid",
                env_var="TWILIO_ACCOUNT_SID",
                required=False,
                description="Twilio Account SID for voice processing",
                security_level=SecurityLevel.CONFIDENTIAL
            ),
            SecretInfo(
                name="twilio_auth_token",
                env_var="TWILIO_AUTH_TOKEN",
                required=False,
                description="Twilio Auth Token for voice processing",
                security_level=SecurityLevel.RESTRICTED
            ),
            SecretInfo(
                name="jwt_secret_key",
                env_var="JWT_SECRET_KEY",
                required=False,
                description="JWT signing secret key",
                security_level=SecurityLevel.RESTRICTED
            ),
            SecretInfo(
                name="encryption_key",
                env_var="ENCRYPTION_KEY",
                required=False,
                description="Data encryption key",
                security_level=SecurityLevel.TOP_SECRET
            )
        ]
        
        for secret in known_secrets:
            self.secrets[secret.name] = secret
    
    def register_secret(
        self,
        name: str,
        env_var: str,
        required: bool = True,
        description: str = "",
        security_level: SecurityLevel = SecurityLevel.CONFIDENTIAL
    ):
        """
        Register a new secret for management.
        
        Args:
            name: Internal name for the secret
            env_var: Environment variable name
            required: Whether the secret is required
            description: Description of the secret
            security_level: Security classification level
        """
        self.secrets[name] = SecretInfo(
            name=name,
            env_var=env_var,
            required=required,
            description=description,
            security_level=security_level
        )
        
        logger.info(f"Registered secret: {name} -> {env_var}")
    
    def get_secret(
        self,
        name: str,
        user_id: Optional[str] = None,
        operation: str = "access"
    ) -> Optional[str]:
        """
        Securely retrieve a secret value.
        
        Args:
            name: Secret name
            user_id: User requesting the secret (for audit)
            operation: Operation being performed
            
        Returns:
            Optional[str]: Secret value or None if not found
            
        Raises:
            ValueError: If required secret is missing
        """
        if name not in self.secrets:
            raise ValueError(f"Unknown secret: {name}")
        
        secret_info = self.secrets[name]
        
        # Log security event
        self._log_secret_access(secret_info, user_id, operation)
        
        # Get value from environment
        value = os.getenv(secret_info.env_var)
        
        if value is None and secret_info.required:
            self.error_logger.log_security_event(
                event_type="MISSING_REQUIRED_SECRET",
                user_id=user_id,
                details={
                    "secret_name": name,
                    "env_var": secret_info.env_var,
                    "operation": operation
                },
                severity=ErrorSeverity.CRITICAL
            )
            raise ValueError(f"Required secret {name} not found in environment variable {secret_info.env_var}")
        
        # Update access tracking
        secret_info.last_accessed = datetime.now()
        secret_info.access_count += 1
        
        return value
    
    def _log_secret_access(
        self,
        secret_info: SecretInfo,
        user_id: Optional[str],
        operation: str
    ):
        """Log secret access for audit trail."""
        self.error_logger.log_security_event(
            event_type="SECRET_ACCESS",
            user_id=user_id,
            details={
                "secret_name": secret_info.name,
                "env_var": secret_info.env_var,
                "security_level": secret_info.security_level.value,
                "operation": operation,
                "access_count": secret_info.access_count
            },
            severity=ErrorSeverity.MEDIUM if secret_info.security_level in [
                SecurityLevel.RESTRICTED, SecurityLevel.TOP_SECRET
            ] else ErrorSeverity.LOW
        )
    
    def validate_all_secrets(self) -> Dict[str, bool]:
        """
        Validate all registered secrets are available.
        
        Returns:
            Dict[str, bool]: Secret validation results
        """
        results = {}
        
        for name, secret_info in self.secrets.items():
            value = os.getenv(secret_info.env_var)
            is_available = value is not None
            
            results[name] = is_available
            
            if not is_available and secret_info.required:
                logger.error(f"Required secret {name} not found in {secret_info.env_var}")
        
        return results
    
    def get_secret_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        Get summary of all secrets (without values).
        
        Returns:
            Dict[str, Dict[str, Any]]: Secret summary information
        """
        summary = {}
        
        for name, secret_info in self.secrets.items():
            is_set = os.getenv(secret_info.env_var) is not None
            
            summary[name] = {
                "env_var": secret_info.env_var,
                "required": secret_info.required,
                "description": secret_info.description,
                "security_level": secret_info.security_level.value,
                "is_set": is_set,
                "last_accessed": secret_info.last_accessed.isoformat() if secret_info.last_accessed else None,
                "access_count": secret_info.access_count
            }
        
        return summary

# Security Auditor

class SecurityAuditor:
    """
    Security auditor for code and runtime security checks.
    
    Performs static analysis and runtime validation to ensure security practices.
    """
    
    def __init__(self):
        self.error_logger = ErrorLogger("security.auditor")
        self.hardcoded_patterns = [
            # API Keys
            r'api_key\s*=\s*["\'][^"\']+["\']',
            r'apikey\s*=\s*["\'][^"\']+["\']',
            r'sk-[a-zA-Z0-9]{32,}',
            r'pk_[a-zA-Z0-9]{32,}',
            
            # Database URLs with credentials
            r'postgresql://[^:]+:[^@]+@[^/]+/[^"\']+',
            r'mysql://[^:]+:[^@]+@[^/]+/[^"\']+',
            r'mongodb://[^:]+:[^@]+@[^/]+/[^"\']+',
            
            # JWT Tokens
            r'eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+',
            
            # AWS Keys
            r'AKIA[0-9A-Z]{16}',
            r'aws_access_key_id\s*=\s*["\'][^"\']+["\']',
            r'aws_secret_access_key\s*=\s*["\'][^"\']+["\']',
            
            # Generic secrets
            r'password\s*=\s*["\'][^"\']{8,}["\']',
            r'secret\s*=\s*["\'][^"\']{16,}["\']',
            r'token\s*=\s*["\'][^"\']{16,}["\']'
        ]
    
    def scan_file_for_secrets(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Scan a file for potential hardcoded secrets.
        
        Args:
            file_path: Path to file to scan
            
        Returns:
            List[Dict[str, Any]]: List of potential security issues
        """
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
            
            for line_num, line in enumerate(lines, 1):
                for pattern in self.hardcoded_patterns:
                    matches = re.finditer(pattern, line, re.IGNORECASE)
                    for match in matches:
                        issues.append({
                            "file": str(file_path),
                            "line": line_num,
                            "column": match.start(),
                            "pattern": pattern,
                            "matched_text": match.group()[:50] + "..." if len(match.group()) > 50 else match.group(),
                            "severity": "HIGH",
                            "description": "Potential hardcoded secret detected"
                        })
        
        except Exception as e:
            logger.error(f"Error scanning file {file_path}: {str(e)}")
        
        return issues
    
    def scan_directory_for_secrets(self, directory: Path) -> Dict[str, List[Dict[str, Any]]]:
        """
        Scan directory for potential hardcoded secrets.
        
        Args:
            directory: Directory to scan
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: Issues by file
        """
        all_issues = {}
        
        # Python files to scan
        python_files = list(directory.rglob("*.py"))
        
        # Configuration files to scan
        config_files = list(directory.rglob("*.env*")) + list(directory.rglob("*.yaml")) + list(directory.rglob("*.yml"))
        
        all_files = python_files + config_files
        
        for file_path in all_files:
            # Skip virtual environment and cache directories
            if any(part in str(file_path) for part in ['venv', '__pycache__', '.git', 'node_modules']):
                continue
            
            issues = self.scan_file_for_secrets(file_path)
            if issues:
                all_issues[str(file_path)] = issues
        
        return all_issues
    
    def check_print_statements(self, directory: Path) -> List[Dict[str, Any]]:
        """
        Check for print statements that should be replaced with logging.
        
        Args:
            directory: Directory to scan
            
        Returns:
            List[Dict[str, Any]]: List of print statement violations
        """
        violations = []
        
        python_files = list(directory.rglob("*.py"))
        
        for file_path in python_files:
            # Skip virtual environment and test files
            if any(part in str(file_path) for part in ['venv', '__pycache__', '.git']):
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                
                for line_num, line in enumerate(lines, 1):
                    # Look for print statements (excluding comments)
                    stripped_line = line.strip()
                    if (stripped_line.startswith('print(') and 
                        not stripped_line.startswith('#') and
                        'test' not in str(file_path).lower()):  # Allow in test files
                        
                        violations.append({
                            "file": str(file_path),
                            "line": line_num,
                            "content": line.strip(),
                            "severity": "MEDIUM",
                            "description": "Print statement found - should use structured logging",
                            "recommendation": "Replace with logger.info(), logger.debug(), etc."
                        })
            
            except Exception as e:
                logger.error(f"Error checking print statements in {file_path}: {str(e)}")
        
        return violations
    
    def validate_environment_variables(self, required_vars: List[str]) -> ValidationResult:
        """
        Validate that required environment variables are set.
        
        Args:
            required_vars: List of required environment variable names
            
        Returns:
            ValidationResult: Validation result
        """
        missing_vars = []
        
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            return ValidationResult(
                is_valid=False,
                error_code=ErrorCode.CONFIGURATION_ERROR,
                error_message=f"Missing required environment variables: {', '.join(missing_vars)}",
                security_score=0.0,
                metadata={"missing_variables": missing_vars}
            )
        
        return ValidationResult(
            is_valid=True,
            security_score=1.0,
            metadata={"validated_variables": required_vars}
        )
    
    def generate_security_report(self, directory: Path) -> Dict[str, Any]:
        """
        Generate comprehensive security audit report.
        
        Args:
            directory: Directory to audit
            
        Returns:
            Dict[str, Any]: Security audit report
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "directory": str(directory),
            "summary": {
                "total_files_scanned": 0,
                "secret_violations": 0,
                "print_violations": 0,
                "overall_score": 0.0
            },
            "secret_scan_results": {},
            "print_statement_violations": [],
            "environment_validation": {},
            "recommendations": []
        }
        
        # Scan for secrets
        secret_issues = self.scan_directory_for_secrets(directory)
        report["secret_scan_results"] = secret_issues
        report["summary"]["secret_violations"] = sum(len(issues) for issues in secret_issues.values())
        
        # Check print statements
        print_violations = self.check_print_statements(directory)
        report["print_statement_violations"] = print_violations
        report["summary"]["print_violations"] = len(print_violations)
        
        # Count total files
        python_files = list(directory.rglob("*.py"))
        report["summary"]["total_files_scanned"] = len([f for f in python_files if 'venv' not in str(f)])
        
        # Calculate overall score
        total_violations = report["summary"]["secret_violations"] + report["summary"]["print_violations"]
        if total_violations == 0:
            report["summary"]["overall_score"] = 1.0
        else:
            # Deduct points for violations
            score = max(0.0, 1.0 - (total_violations * 0.1))
            report["summary"]["overall_score"] = score
        
        # Generate recommendations
        if report["summary"]["secret_violations"] > 0:
            report["recommendations"].append(
                "Move hardcoded secrets to environment variables and update .env.example"
            )
        
        if report["summary"]["print_violations"] > 0:
            report["recommendations"].append(
                "Replace print statements with structured logging using the configured logger"
            )
        
        if report["summary"]["overall_score"] < 0.8:
            report["recommendations"].append(
                "Address security violations to improve overall security score"
            )
        
        return report

# Security Event Logger

class SecurityEventLogger:
    """
    Specialized logger for security events and audit trail.
    """
    
    def __init__(self):
        self.error_logger = ErrorLogger("security.events")
        self.event_history: List[Dict[str, Any]] = []
    
    def log_security_event(
        self,
        event_type: SecurityEventType,
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        action: Optional[str] = None,
        result: str = "SUCCESS",
        details: Optional[Dict[str, Any]] = None,
        security_level: SecurityLevel = SecurityLevel.INTERNAL
    ):
        """
        Log a security event for audit trail.
        
        Args:
            event_type: Type of security event
            user_id: User involved in the event
            resource: Resource being accessed
            action: Action being performed
            result: Result of the action (SUCCESS, FAILURE, BLOCKED)
            details: Additional event details
            security_level: Security level of the event
        """
        event = {
            "event_id": secrets.token_hex(16),
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type.value,
            "user_id": user_id,
            "resource": resource,
            "action": action,
            "result": result,
            "security_level": security_level.value,
            "details": details or {},
            "source_ip": None,  # Would be populated from request context
            "user_agent": None  # Would be populated from request context
        }
        
        # Add to history (keep last 1000 events in memory)
        self.event_history.append(event)
        if len(self.event_history) > 1000:
            self.event_history.pop(0)
        
        # Log using structured logging
        severity = ErrorSeverity.HIGH if result == "FAILURE" else ErrorSeverity.MEDIUM
        
        self.error_logger.log_security_event(
            event_type=f"SECURITY_{event_type.value}",
            user_id=user_id,
            details=event,
            severity=severity
        )
    
    def get_security_events(
        self,
        user_id: Optional[str] = None,
        event_type: Optional[SecurityEventType] = None,
        hours: int = 24
    ) -> List[Dict[str, Any]]:
        """
        Retrieve security events from history.
        
        Args:
            user_id: Filter by user ID
            event_type: Filter by event type
            hours: Number of hours to look back
            
        Returns:
            List[Dict[str, Any]]: Filtered security events
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        filtered_events = []
        
        for event in self.event_history:
            event_time = datetime.fromisoformat(event["timestamp"])
            
            # Time filter
            if event_time < cutoff_time:
                continue
            
            # User filter
            if user_id and event.get("user_id") != user_id:
                continue
            
            # Event type filter
            if event_type and event.get("event_type") != event_type.value:
                continue
            
            filtered_events.append(event)
        
        return filtered_events

# Validation Security Wrapper

class ValidationSecurityWrapper:
    """
    Wrapper for Pydantic V2 validation with security enhancements.
    
    Implements fail-fast validation with security event logging.
    """
    
    def __init__(self):
        self.security_logger = SecurityEventLogger()
    
    def validate_with_security(
        self,
        model_class: type,
        data: Dict[str, Any],
        user_id: Optional[str] = None,
        operation: str = "validation"
    ) -> ValidationResult:
        """
        Validate data with security logging.
        
        Args:
            model_class: Pydantic model class
            data: Data to validate
            user_id: User performing validation
            operation: Operation being performed
            
        Returns:
            ValidationResult: Validation result with security metadata
        """
        try:
            # Attempt validation
            validated_model = model_class(**data)
            
            # Log successful validation
            self.security_logger.log_security_event(
                event_type=SecurityEventType.DATA_ACCESS,
                user_id=user_id,
                resource=model_class.__name__,
                action=f"validate_{operation}",
                result="SUCCESS",
                details={
                    "model_class": model_class.__name__,
                    "field_count": len(data),
                    "operation": operation
                }
            )
            
            return ValidationResult(
                is_valid=True,
                security_score=1.0,
                metadata={
                    "model_class": model_class.__name__,
                    "validated_fields": list(data.keys()),
                    "validation_time": datetime.now().isoformat()
                }
            )
        
        except ValidationError as e:
            # Log validation failure
            self.security_logger.log_security_event(
                event_type=SecurityEventType.DATA_ACCESS,
                user_id=user_id,
                resource=model_class.__name__,
                action=f"validate_{operation}",
                result="FAILURE",
                details={
                    "model_class": model_class.__name__,
                    "validation_errors": [str(error) for error in e.errors()],
                    "operation": operation
                }
            )
            
            return ValidationResult(
                is_valid=False,
                error_code=ErrorCode.VALIDATION_ERROR,
                error_message=f"Validation failed: {str(e)}",
                security_score=0.0,
                metadata={
                    "validation_errors": e.errors(),
                    "model_class": model_class.__name__
                }
            )
        
        except Exception as e:
            # Log unexpected error
            self.security_logger.log_security_event(
                event_type=SecurityEventType.SECURITY_VIOLATION,
                user_id=user_id,
                resource=model_class.__name__,
                action=f"validate_{operation}",
                result="FAILURE",
                details={
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "operation": operation
                }
            )
            
            return ValidationResult(
                is_valid=False,
                error_code=ErrorCode.PROCESSING_ERROR,
                error_message=f"Validation error: {str(e)}",
                security_score=0.0,
                metadata={
                    "error_type": type(e).__name__,
                    "unexpected_error": True
                }
            )

# Global Security Instances

# Initialize global security components
secret_manager = SecretManager()
security_auditor = SecurityAuditor()
security_event_logger = SecurityEventLogger()
validation_security_wrapper = ValidationSecurityWrapper()

# Utility Functions

def secure_getenv(
    env_var: str,
    default: Optional[str] = None,
    required: bool = False,
    user_id: Optional[str] = None
) -> Optional[str]:
    """
    Secure environment variable access with logging.
    
    Args:
        env_var: Environment variable name
        default: Default value if not found
        required: Whether the variable is required
        user_id: User requesting the variable
        
    Returns:
        Optional[str]: Environment variable value
        
    Raises:
        ValueError: If required variable is missing
    """
    value = os.getenv(env_var, default)
    
    # Log access
    security_event_logger.log_security_event(
        event_type=SecurityEventType.CONFIGURATION_CHANGE,
        user_id=user_id,
        resource=env_var,
        action="getenv",
        result="SUCCESS" if value is not None else "NOT_FOUND",
        details={
            "env_var": env_var,
            "has_default": default is not None,
            "required": required
        }
    )
    
    if required and value is None:
        raise ValueError(f"Required environment variable {env_var} not set")
    
    return value

def validate_no_hardcoded_secrets(code: str) -> List[str]:
    """
    Validate that code contains no hardcoded secrets.
    
    Args:
        code: Code to validate
        
    Returns:
        List[str]: List of potential security issues
    """
    issues = []
    
    # Check for common hardcoded patterns
    patterns = [
        (r'api_key\s*=\s*["\'][^"\']+["\']', "Hardcoded API key detected"),
        (r'password\s*=\s*["\'][^"\']{8,}["\']', "Hardcoded password detected"),
        (r'secret\s*=\s*["\'][^"\']{16,}["\']', "Hardcoded secret detected"),
        (r'token\s*=\s*["\'][^"\']{16,}["\']', "Hardcoded token detected")
    ]
    
    for pattern, message in patterns:
        if re.search(pattern, code, re.IGNORECASE):
            issues.append(message)
    
    return issues

def audit_security_practices(directory_path: str = ".") -> Dict[str, Any]:
    """
    Perform comprehensive security audit.
    
    Args:
        directory_path: Directory to audit
        
    Returns:
        Dict[str, Any]: Security audit report
    """
    directory = Path(directory_path)
    return security_auditor.generate_security_report(directory)

# Export all components
__all__ = [
    # Enums
    'SecurityEventType',
    'SecurityLevel',
    
    # Models
    'ValidationResult',
    'SecretInfo',
    
    # Classes
    'SecretManager',
    'SecurityAuditor',
    'SecurityEventLogger',
    'ValidationSecurityWrapper',
    
    # Global Instances
    'secret_manager',
    'security_auditor',
    'security_event_logger',
    'validation_security_wrapper',
    
    # Utility Functions
    'secure_getenv',
    'validate_no_hardcoded_secrets',
    'audit_security_practices'
]