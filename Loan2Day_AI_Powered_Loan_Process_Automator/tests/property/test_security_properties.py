"""
Property-based tests for Security Practices.

This test suite uses Hypothesis to verify universal properties of the security
infrastructure across all possible security scenarios through randomization.
Property tests ensure consistent security behavior regardless of input variations.

Test Coverage:
- Property 13: Security Practices Enforcement
- Environment variable security validation
- Hardcoded secret detection
- Structured logging enforcement
- Security event logging consistency
- API key and credential protection

Framework: Hypothesis
"""

import pytest
import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from hypothesis import given, strategies as st, assume, settings as hypothesis_settings
from hypothesis.stateful import RuleBasedStateMachine, rule, initialize, invariant

from app.core.security import (
    SecretManager,
    SecurityAuditor,
    SecurityEventLogger,
    ValidationSecurityWrapper,
    SecurityEventType,
    SecurityLevel,
    ValidationResult,
    SecretInfo,
    secure_getenv,
    validate_no_hardcoded_secrets,
    audit_security_practices
)
from app.core.config import settings, validate_security_configuration

# Test Strategies

secret_names = st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc')))
env_var_names = st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc'))).map(str.upper)
secret_values = st.text(min_size=8, max_size=100)
security_levels = st.sampled_from(list(SecurityLevel))
event_types = st.sampled_from(list(SecurityEventType))

user_ids = st.one_of(st.none(), st.uuids().map(str))
resource_names = st.text(min_size=1, max_size=100)
action_names = st.text(min_size=1, max_size=50)

# Code samples for security testing
safe_code_samples = st.sampled_from([
    "import os\napi_key = os.getenv('API_KEY')",
    "from app.core.security import secret_manager\ntoken = secret_manager.get_secret('token')",
    "logger.info('Processing request')",
    "config = {'database_url': os.environ.get('DATABASE_URL')}"
])

unsafe_code_samples = st.sampled_from([
    "api_key = 'sk-1234567890abcdef'",  # Matches api_key pattern
    "password = 'hardcoded_password_123'",  # Matches password pattern (8+ chars)
    "secret = 'my_secret_token_value_12345'",  # Matches secret pattern (16+ chars)
    "token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9abcdef'",  # Matches token pattern (16+ chars)
])

# Feature: loan2day, Property 13: Security Practices Enforcement
# **Property 13: Security Practices Enforcement**
# *For any* system component, there should be no hardcoded API keys (only environment variables), 
# no print statements (only structured logging), and all sensitive operations should generate audit log entries.
# **Validates: Requirements 8.1, 8.3, 8.5**

class TestSecurityPracticesProperties:
    """Property-based tests for security practices enforcement."""
    
    @given(
        secret_name=secret_names,
        env_var=env_var_names,
        secret_value=secret_values,
        security_level=security_levels,
        required=st.booleans()
    )
    def test_secret_manager_consistency(
        self,
        secret_name: str,
        env_var: str,
        secret_value: str,
        security_level: SecurityLevel,
        required: bool
    ):
        """
        Property: SecretManager should consistently manage secrets without exposing values.
        
        For any secret configuration, SecretManager should:
        1. Register secrets with proper metadata
        2. Retrieve secrets from environment variables only
        3. Never expose secret values in logs or errors
        4. Track access for audit purposes
        """
        assume(len(secret_name.strip()) > 0)
        assume(len(env_var.strip()) > 0)
        
        secret_manager = SecretManager()
        
        # Register the secret
        secret_manager.register_secret(
            name=secret_name,
            env_var=env_var,
            required=required,
            security_level=security_level
        )
        
        # Verify registration
        assert secret_name in secret_manager.secrets
        secret_info = secret_manager.secrets[secret_name]
        assert secret_info.name == secret_name
        assert secret_info.env_var == env_var
        assert secret_info.required == required
        assert secret_info.security_level == security_level
        
        # Test with environment variable set
        with patch.dict(os.environ, {env_var: secret_value}):
            retrieved_value = secret_manager.get_secret(secret_name)
            assert retrieved_value == secret_value
            
            # Verify access tracking
            assert secret_info.access_count > 0
            assert secret_info.last_accessed is not None
        
        # Test with environment variable not set
        with patch.dict(os.environ, {}, clear=True):
            if required:
                with pytest.raises(ValueError) as exc_info:
                    secret_manager.get_secret(secret_name)
                # Verify error message doesn't expose secret value
                assert secret_value not in str(exc_info.value)
            else:
                result = secret_manager.get_secret(secret_name)
                assert result is None
        
        # Verify secret summary doesn't expose values
        summary = secret_manager.get_secret_summary()
        assert secret_name in summary
        assert "value" not in summary[secret_name]
        assert summary[secret_name]["is_set"] == (env_var in os.environ)
    
    @given(code_sample=safe_code_samples)
    def test_safe_code_validation(self, code_sample: str):
        """
        Property: Safe code should pass security validation.
        
        For any code that follows security practices, validation should:
        1. Return no security issues
        2. Pass hardcoded secret detection
        3. Not trigger security violations
        """
        issues = validate_no_hardcoded_secrets(code_sample)
        
        # Safe code should have no security issues
        assert len(issues) == 0
    
    @given(code_sample=unsafe_code_samples)
    def test_unsafe_code_detection(self, code_sample: str):
        """
        Property: Unsafe code should be detected by security validation.
        
        For any code with security violations, validation should:
        1. Detect hardcoded secrets
        2. Identify security issues
        3. Provide clear violation descriptions
        """
        issues = validate_no_hardcoded_secrets(code_sample)
        
        # Unsafe code should trigger security issues
        assert len(issues) > 0
        
        # Each issue should have a descriptive message
        for issue in issues:
            assert isinstance(issue, str)
            assert len(issue) > 0
    
    @given(
        event_type=event_types,
        user_id=user_ids,
        resource=resource_names,
        action=action_names,
        security_level=security_levels
    )
    def test_security_event_logging_consistency(
        self,
        event_type: SecurityEventType,
        user_id: Optional[str],
        resource: str,
        action: str,
        security_level: SecurityLevel
    ):
        """
        Property: Security event logging should be consistent and complete.
        
        For any security event, logging should:
        1. Generate unique event IDs
        2. Include all required metadata
        3. Maintain event history
        4. Support filtering and retrieval
        """
        assume(len(resource.strip()) > 0)
        assume(len(action.strip()) > 0)
        
        security_logger = SecurityEventLogger()
        initial_event_count = len(security_logger.event_history)
        
        # Log security event
        security_logger.log_security_event(
            event_type=event_type,
            user_id=user_id,
            resource=resource,
            action=action,
            security_level=security_level
        )
        
        # Verify event was logged
        assert len(security_logger.event_history) == initial_event_count + 1
        
        # Verify event structure
        latest_event = security_logger.event_history[-1]
        assert "event_id" in latest_event
        assert "timestamp" in latest_event
        assert latest_event["event_type"] == event_type.value
        assert latest_event["user_id"] == user_id
        assert latest_event["resource"] == resource
        assert latest_event["action"] == action
        assert latest_event["security_level"] == security_level.value
        
        # Verify event ID is unique
        event_ids = [event["event_id"] for event in security_logger.event_history]
        assert len(set(event_ids)) == len(event_ids)
        
        # Test event retrieval
        if user_id:
            user_events = security_logger.get_security_events(user_id=user_id)
            assert any(event["user_id"] == user_id for event in user_events)
        
        type_events = security_logger.get_security_events(event_type=event_type)
        assert any(event["event_type"] == event_type.value for event in type_events)
    
    @given(
        env_vars=st.dictionaries(
            keys=env_var_names,
            values=secret_values,
            min_size=1,
            max_size=5
        )
    )
    def test_environment_variable_validation(self, env_vars: Dict[str, str]):
        """
        Property: Environment variable validation should be consistent.
        
        For any set of environment variables, validation should:
        1. Correctly identify missing required variables
        2. Report validation status accurately
        3. Provide helpful error messages
        """
        required_vars = list(env_vars.keys())
        
        # Test with all variables set
        with patch.dict(os.environ, env_vars):
            validation_result = validate_security_configuration()
            
            # All registered secrets should be validated
            for var_name in required_vars:
                # Check if this variable corresponds to a known secret
                secret_found = False
                for secret_name, secret_info in SecretManager().secrets.items():
                    if secret_info.env_var == var_name:
                        assert validation_result.get(secret_name, False)
                        secret_found = True
                        break
        
        # Test with variables missing
        with patch.dict(os.environ, {}, clear=True):
            validation_result = validate_security_configuration()
            
            # Required secrets should be marked as missing
            for secret_name, secret_info in SecretManager().secrets.items():
                if secret_info.required:
                    assert not validation_result.get(secret_name, True)
    
    def test_secure_getenv_logging(self):
        """
        Property: secure_getenv should log all access attempts.
        
        For any environment variable access, secure_getenv should:
        1. Log the access attempt
        2. Include user context if provided
        3. Handle missing variables appropriately
        4. Never expose variable values in logs
        """
        with patch('app.core.security.security_event_logger') as mock_logger:
            # Test successful access
            with patch.dict(os.environ, {'TEST_VAR': 'test_value'}):
                result = secure_getenv('TEST_VAR', user_id='test_user')
                assert result == 'test_value'
                
                # Verify logging was called
                mock_logger.log_security_event.assert_called()
                call_args = mock_logger.log_security_event.call_args
                
                assert call_args[1]['event_type'] == SecurityEventType.CONFIGURATION_CHANGE
                assert call_args[1]['user_id'] == 'test_user'
                assert call_args[1]['resource'] == 'TEST_VAR'
                assert call_args[1]['result'] == 'SUCCESS'
            
            # Test missing variable
            mock_logger.reset_mock()
            with patch.dict(os.environ, {}, clear=True):
                result = secure_getenv('MISSING_VAR', user_id='test_user')
                assert result is None
                
                # Verify logging was called for missing variable
                mock_logger.log_security_event.assert_called()
                call_args = mock_logger.log_security_event.call_args
                assert call_args[1]['result'] == 'NOT_FOUND'

class TestSecurityAuditorProperties:
    """Property-based tests for security auditor functionality."""
    
    def setup_method(self):
        """Set up test environment with temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.auditor = SecurityAuditor()
    
    def teardown_method(self):
        """Clean up test environment."""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @given(
        file_contents=st.lists(
            st.one_of(safe_code_samples, unsafe_code_samples),
            min_size=1,
            max_size=10
        )
    )
    def test_file_scanning_consistency(self, file_contents: List[str]):
        """
        Property: File scanning should consistently detect security issues.
        
        For any file content, scanning should:
        1. Detect all hardcoded secrets
        2. Report accurate line numbers
        3. Provide consistent results across multiple scans
        4. Handle various file types appropriately
        """
        # Create test files
        test_files = []
        for i, content in enumerate(file_contents):
            file_path = Path(self.temp_dir) / f"test_file_{i}.py"
            with open(file_path, 'w') as f:
                f.write(content)
            test_files.append(file_path)
        
        # Scan each file
        all_issues = {}
        for file_path in test_files:
            issues = self.auditor.scan_file_for_secrets(file_path)
            all_issues[str(file_path)] = issues
        
        # Verify consistency - scanning same file multiple times should give same results
        for file_path in test_files:
            issues1 = self.auditor.scan_file_for_secrets(file_path)
            issues2 = self.auditor.scan_file_for_secrets(file_path)
            assert len(issues1) == len(issues2)
        
        # Verify issue structure
        for file_path, issues in all_issues.items():
            for issue in issues:
                assert "file" in issue
                assert "line" in issue
                assert "pattern" in issue
                assert "severity" in issue
                assert "description" in issue
                assert issue["line"] > 0
                assert issue["severity"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    
    def test_directory_scanning_completeness(self):
        """
        Property: Directory scanning should be complete and consistent.
        
        Directory scanning should:
        1. Find all Python and configuration files
        2. Skip excluded directories (venv, __pycache__)
        3. Report issues by file
        4. Provide comprehensive coverage
        """
        # Create directory structure with various file types
        test_structure = {
            "app/main.py": "api_key = 'hardcoded_key'",
            "app/config.py": "import os\napi_key = os.getenv('API_KEY')",
            "tests/test_main.py": "print('test output')",  # Should be allowed in tests
            "venv/lib/python.py": "secret = 'should_be_ignored'",  # Should be ignored
            ".env": "API_KEY=test_value",
            "requirements.txt": "fastapi==0.68.0"
        }
        
        for file_path, content in test_structure.items():
            full_path = Path(self.temp_dir) / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            with open(full_path, 'w') as f:
                f.write(content)
        
        # Scan directory
        issues_by_file = self.auditor.scan_directory_for_secrets(Path(self.temp_dir))
        
        # Verify venv files are excluded
        venv_files = [f for f in issues_by_file.keys() if 'venv' in f]
        assert len(venv_files) == 0
        
        # Verify main.py has issues (hardcoded key)
        main_py_files = [f for f in issues_by_file.keys() if 'main.py' in f and 'venv' not in f]
        assert len(main_py_files) > 0
        
        # Verify config.py has no issues (uses os.getenv)
        config_issues = []
        for file_path, issues in issues_by_file.items():
            if 'config.py' in file_path:
                config_issues.extend(issues)
        # Config.py should have no hardcoded secret issues
        hardcoded_issues = [issue for issue in config_issues if 'hardcoded' in issue.get('description', '').lower()]
        assert len(hardcoded_issues) == 0
    
    def test_print_statement_detection(self):
        """
        Property: Print statement detection should be accurate and context-aware.
        
        Print statement detection should:
        1. Find all print statements in non-test files
        2. Allow print statements in test files
        3. Ignore commented print statements
        4. Provide accurate line numbers
        """
        # Create test files with various print statement scenarios
        test_files = {
            "main.py": """
import logging
logger = logging.getLogger(__name__)

def process_data():
    print("Debug info")  # Should be flagged
    logger.info("Proper logging")
    # print("Commented out")  # Should not be flagged
""",
            "test_main.py": """
def test_function():
    print("Test output")  # Should NOT be flagged (test file)
    assert True
""",
            "utils.py": """
def helper():
    logger.debug("Helper function")
    print('Another debug')  # Should be flagged
"""
        }
        
        for file_name, content in test_files.items():
            file_path = Path(self.temp_dir) / file_name
            with open(file_path, 'w') as f:
                f.write(content)
        
        # Check print statements
        violations = self.auditor.check_print_statements(Path(self.temp_dir))
        
        # Verify test files are excluded from violations
        test_violations = [v for v in violations if 'test_' in v['file']]
        assert len(test_violations) == 0
        
        # Verify non-test files have violations
        main_violations = [v for v in violations if 'main.py' in v['file'] and 'test_' not in v['file']]
        utils_violations = [v for v in violations if 'utils.py' in v['file']]
        
        assert len(main_violations) > 0
        assert len(utils_violations) > 0
        
        # Verify violation structure
        for violation in violations:
            assert "file" in violation
            assert "line" in violation
            assert "content" in violation
            assert "severity" in violation
            assert "description" in violation
            assert "recommendation" in violation
            assert violation["line"] > 0

class TestValidationSecurityWrapper:
    """Property-based tests for validation security wrapper."""
    
    def setup_method(self):
        """Set up test environment."""
        self.wrapper = ValidationSecurityWrapper()
    
    @given(
        valid_data=st.dictionaries(
            keys=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll'))),
            values=st.one_of(st.text(), st.integers(), st.booleans()),
            min_size=1,
            max_size=5
        ),
        user_id=user_ids
    )
    def test_validation_security_logging(self, valid_data: Dict[str, Any], user_id: Optional[str]):
        """
        Property: Validation security wrapper should log all validation attempts.
        
        For any validation attempt, the wrapper should:
        1. Log successful validations
        2. Log validation failures
        3. Include user context
        4. Provide security metadata
        """
        # Create a simple Pydantic model for testing
        from pydantic import BaseModel
        
        class TestModel(BaseModel):
            name: str
            value: int = 0
        
        # Test with valid data that matches model
        test_data = {"name": "test", "value": 42}
        
        with patch.object(self.wrapper.security_logger, 'log_security_event') as mock_log:
            result = self.wrapper.validate_with_security(
                TestModel,
                test_data,
                user_id=user_id,
                operation="test_validation"
            )
            
            # Verify successful validation
            assert result.is_valid
            assert result.security_score == 1.0
            
            # Verify logging was called
            mock_log.assert_called()
            call_args = mock_log.call_args[1]
            assert call_args['event_type'] == SecurityEventType.DATA_ACCESS
            assert call_args['user_id'] == user_id
            assert call_args['result'] == 'SUCCESS'
        
        # Test with invalid data
        invalid_data = {"invalid_field": "value"}
        
        with patch.object(self.wrapper.security_logger, 'log_security_event') as mock_log:
            result = self.wrapper.validate_with_security(
                TestModel,
                invalid_data,
                user_id=user_id,
                operation="test_validation"
            )
            
            # Verify failed validation
            assert not result.is_valid
            assert result.security_score == 0.0
            
            # Verify logging was called for failure
            mock_log.assert_called()
            call_args = mock_log.call_args[1]
            assert call_args['result'] == 'FAILURE'

# Stateful Testing for Security Event History

class SecurityEventStateMachine(RuleBasedStateMachine):
    """
    Stateful property testing for security event logging.
    
    This tests that security event logging maintains consistency
    regardless of the sequence of operations.
    """
    
    def __init__(self):
        super().__init__()
        self.security_logger = SecurityEventLogger()
        self.logged_events = []
    
    @rule(
        event_type=event_types,
        user_id=user_ids,
        resource=resource_names,
        action=action_names
    )
    def log_security_event(
        self,
        event_type: SecurityEventType,
        user_id: Optional[str],
        resource: str,
        action: str
    ):
        """Log a security event."""
        assume(len(resource.strip()) > 0)
        assume(len(action.strip()) > 0)
        
        initial_count = len(self.security_logger.event_history)
        
        self.security_logger.log_security_event(
            event_type=event_type,
            user_id=user_id,
            resource=resource,
            action=action
        )
        
        # Track what we logged
        self.logged_events.append({
            'event_type': event_type,
            'user_id': user_id,
            'resource': resource,
            'action': action
        })
        
        # Verify event was added
        assert len(self.security_logger.event_history) == initial_count + 1
    
    @rule(user_id=user_ids)
    def query_events_by_user(self, user_id: Optional[str]):
        """Query events by user ID."""
        if user_id is None:
            return
        
        events = self.security_logger.get_security_events(user_id=user_id)
        
        # All returned events should match the user ID
        for event in events:
            assert event.get('user_id') == user_id
    
    @rule(event_type=event_types)
    def query_events_by_type(self, event_type: SecurityEventType):
        """Query events by type."""
        events = self.security_logger.get_security_events(event_type=event_type)
        
        # All returned events should match the event type
        for event in events:
            assert event.get('event_type') == event_type.value
    
    @invariant()
    def event_history_is_consistent(self):
        """Event history should always be consistent."""
        # Event history should not exceed maximum size
        assert len(self.security_logger.event_history) <= 1000
        
        # All events should have required fields
        for event in self.security_logger.event_history:
            assert 'event_id' in event
            assert 'timestamp' in event
            assert 'event_type' in event
            
        # Event IDs should be unique
        event_ids = [event['event_id'] for event in self.security_logger.event_history]
        assert len(set(event_ids)) == len(event_ids)

# Configure Hypothesis settings for property tests
TestSecurityEventStateMachine = SecurityEventStateMachine.TestCase
TestSecurityEventStateMachine.settings = hypothesis_settings(
    max_examples=50,
    stateful_step_count=20,
    deadline=None
)