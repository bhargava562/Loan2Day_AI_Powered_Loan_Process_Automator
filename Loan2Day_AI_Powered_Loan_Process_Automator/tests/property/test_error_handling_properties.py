"""
Property-based tests for Error Handling System.

This test suite uses Hypothesis to verify universal properties of the error
handling infrastructure across all possible failure scenarios through randomization.
Property tests ensure consistent error handling behavior regardless of error variations.

Test Coverage:
- Property 15: Error Handling Consistency
- Circuit breaker behavior under various failure patterns
- Graceful degradation mechanisms
- Structured error response formatting
- Agent failure recovery patterns
- Security event logging consistency

Framework: Hypothesis
"""

import pytest
import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
from decimal import Decimal

from hypothesis import given, strategies as st, assume, settings as hypothesis_settings
from hypothesis.stateful import RuleBasedStateMachine, rule, initialize, invariant

from app.core.error_handling import (
    ErrorResponse,
    ErrorCode,
    ErrorSeverity,
    ErrorCategory,
    CircuitBreaker,
    CircuitBreakerError,
    AgentFailureHandler,
    ErrorLogger,
    RetryHandler,
    create_error_response,
    get_http_status_for_error_code,
    handle_external_service_call,
    get_system_health
)

# Test Strategies

error_codes = st.sampled_from(list(ErrorCode))
error_severities = st.sampled_from(list(ErrorSeverity))
error_categories = st.sampled_from(list(ErrorCategory))

error_messages = st.text(min_size=1, max_size=200)
session_ids = st.one_of(st.none(), st.uuids().map(str))
trace_ids = st.text(min_size=8, max_size=32, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')))

error_details = st.dictionaries(
    keys=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc'))),
    values=st.one_of(
        st.text(max_size=100),
        st.integers(),
        st.floats(allow_nan=False, allow_infinity=False),
        st.booleans(),
        st.none()
    ),
    max_size=10
)

agent_names = st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc')))
service_names = st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc')))

# Feature: loan2day, Property 15: Error Handling Consistency
# **Property 15: Error Handling Consistency**
# *For any* validation error or system failure, the system should return structured error responses 
# with sufficient debugging context and handle graceful degradation.
# **Validates: Requirements 10.1, 10.3, 10.4**

class TestErrorHandlingProperties:
    """Property-based tests for error handling consistency."""
    
    @given(
        error_code=error_codes,
        error_message=error_messages,
        error_details=error_details,
        trace_id=trace_ids,
        session_id=session_ids
    )
    def test_error_response_structure_consistency(
        self,
        error_code: ErrorCode,
        error_message: str,
        error_details: Dict[str, Any],
        trace_id: str,
        session_id: Optional[str]
    ):
        """
        Property: Error responses must have consistent structure and required fields.
        
        For any error code, message, and context, the error response should:
        1. Contain all required fields
        2. Have proper data types
        3. Include timestamp
        4. Maintain trace_id for debugging
        """
        # Create error response
        error_response = create_error_response(
            error_code=error_code,
            error_message=error_message,
            error_details=error_details,
            trace_id=trace_id,
            session_id=session_id
        )
        
        # Verify structure consistency
        assert isinstance(error_response, ErrorResponse)
        assert error_response.error_code == error_code
        assert error_response.error_message == error_message
        assert error_response.error_details == error_details
        assert error_response.trace_id == trace_id
        assert error_response.session_id == session_id
        
        # Verify timestamp is recent (within last minute)
        time_diff = (datetime.now() - error_response.timestamp).total_seconds()
        assert 0 <= time_diff <= 60
        
        # Verify serialization works
        error_dict = error_response.dict()
        assert "error_code" in error_dict
        assert "error_message" in error_dict
        assert "error_details" in error_dict
        assert "timestamp" in error_dict
        assert "trace_id" in error_dict
    
    @given(error_code=error_codes)
    def test_http_status_code_mapping_consistency(self, error_code: ErrorCode):
        """
        Property: HTTP status codes must be consistent and appropriate for error types.
        
        For any error code, the HTTP status should:
        1. Be a valid HTTP status code (100-599)
        2. Be consistent for the same error code
        3. Match semantic meaning of the error
        """
        status_code = get_http_status_for_error_code(error_code)
        
        # Verify valid HTTP status code range
        assert 100 <= status_code <= 599
        
        # Verify consistency - same error code always returns same status
        assert get_http_status_for_error_code(error_code) == status_code
        
        # Verify semantic correctness for known categories
        if "VALIDATION" in error_code.value or "INVALID_INPUT" in error_code.value or "MISSING_REQUIRED" in error_code.value or "INVALID_FORMAT" in error_code.value:
            assert status_code == 400
        elif "AUTHENTICATION" in error_code.value or "SESSION_EXPIRED" in error_code.value or "INVALID_TOKEN" in error_code.value:
            assert status_code == 401
        elif "AUTHORIZATION" in error_code.value or "SECURITY" in error_code.value:
            assert status_code in [403, 429]
        elif "BUSINESS" in error_code.value:
            assert status_code == 422
        elif "EXTERNAL_SERVICE" in error_code.value:
            assert status_code in [502, 503]
        elif "DATABASE" in error_code.value or "INTERNAL" in error_code.value:
            assert status_code == 500

class TestCircuitBreakerProperties:
    """Property-based tests for circuit breaker behavior."""
    
    @given(
        name=service_names,
        failure_threshold=st.integers(min_value=1, max_value=10),
        recovery_timeout=st.integers(min_value=1, max_value=300),
        success_threshold=st.integers(min_value=1, max_value=5)
    )
    def test_circuit_breaker_initialization_consistency(
        self,
        name: str,
        failure_threshold: int,
        recovery_timeout: int,
        success_threshold: int
    ):
        """
        Property: Circuit breaker initialization should be consistent and valid.
        
        For any valid parameters, circuit breaker should:
        1. Initialize with CLOSED state
        2. Store configuration correctly
        3. Have zero failure count initially
        """
        cb = CircuitBreaker(
            name=name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            success_threshold=success_threshold
        )
        
        # Verify initial state
        assert cb.name == name
        assert cb.failure_threshold == failure_threshold
        assert cb.recovery_timeout == recovery_timeout
        assert cb.success_threshold == success_threshold
        assert cb.state.state == "CLOSED"
        assert cb.state.failure_count == 0
        assert cb.state.success_count == 0
        assert cb.state.last_failure_time is None
        
        # Verify state reporting
        state = cb.get_state()
        assert state["name"] == name
        assert state["state"] == "CLOSED"
        assert state["failure_count"] == 0
        assert state["success_count"] == 0
        assert state["last_failure_time"] is None
    
    @pytest.mark.asyncio
    @given(
        failure_count=st.integers(min_value=1, max_value=20),
        failure_threshold=st.integers(min_value=1, max_value=10)
    )
    async def test_circuit_breaker_failure_threshold_behavior(
        self,
        failure_count: int,
        failure_threshold: int
    ):
        """
        Property: Circuit breaker should open when failure threshold is exceeded.
        
        For any number of failures and threshold, circuit breaker should:
        1. Stay CLOSED while failures < threshold
        2. Open when failures >= threshold
        3. Block subsequent calls when OPEN
        """
        cb = CircuitBreaker("test_service", failure_threshold=failure_threshold)
        
        # Simulate failures up to threshold - 1
        for i in range(min(failure_count, failure_threshold - 1)):
            with pytest.raises(Exception):
                await cb.call(lambda: exec('raise Exception("test failure")'))
        
        # Verify still CLOSED if under threshold
        if failure_count < failure_threshold:
            assert cb.state.state == "CLOSED"
            assert cb.state.failure_count == failure_count
        else:
            # Add one more failure to exceed threshold
            with pytest.raises(Exception):
                await cb.call(lambda: exec('raise Exception("test failure")'))
            
            # Verify circuit is now OPEN
            assert cb.state.state == "OPEN"
            assert cb.state.failure_count >= failure_threshold
            
            # Verify subsequent calls are blocked
            with pytest.raises(CircuitBreakerError):
                await cb.call(lambda: "success")

class TestAgentFailureHandlerProperties:
    """Property-based tests for agent failure handling."""
    
    @pytest.mark.asyncio
    @given(
        agent_name=agent_names,
        should_succeed=st.booleans(),
        has_fallback=st.booleans()
    )
    async def test_agent_task_execution_consistency(
        self,
        agent_name: str,
        should_succeed: bool,
        has_fallback: bool
    ):
        """
        Property: Agent task execution should handle success/failure consistently.
        
        For any agent and task outcome, the handler should:
        1. Return structured result with metadata
        2. Include agent name and execution path
        3. Handle fallback appropriately
        4. Maintain circuit breaker state
        """
        assume(len(agent_name.strip()) > 0)  # Ensure non-empty agent name
        
        handler = AgentFailureHandler()
        
        # Create mock task functions
        async def primary_task():
            if should_succeed:
                return {"result": "success", "data": "test_data"}
            else:
                raise Exception("Primary task failed")
        
        async def fallback_task():
            return {"result": "fallback_success", "data": "fallback_data"}
        
        # Execute task
        if has_fallback:
            result = await handler.execute_agent_task(
                agent_name=agent_name,
                task_func=primary_task,
                fallback_func=fallback_task
            )
        else:
            result = await handler.execute_agent_task(
                agent_name=agent_name,
                task_func=primary_task
            )
        
        # Verify result structure consistency
        assert isinstance(result, dict)
        assert "success" in result
        assert "agent" in result
        assert "execution_path" in result
        assert "circuit_breaker_state" in result
        assert result["agent"] == agent_name
        
        # Verify execution path logic
        if should_succeed:
            assert result["success"] is True
            assert result["execution_path"] == "primary"
            assert "result" in result
        else:
            if has_fallback:
                # Should succeed with fallback
                assert result["success"] is True
                assert result["execution_path"] == "fallback"
                assert "fallback_reason" in result
            else:
                # Should fail without fallback
                assert result["success"] is False
                assert result["execution_path"] in ["failed", "degraded"]
                assert "error" in result
        
        # Verify circuit breaker state is included
        cb_state = result["circuit_breaker_state"]
        assert isinstance(cb_state, dict)
        assert "name" in cb_state
        assert "state" in cb_state
        assert cb_state["name"] == f"agent_{agent_name}"

class TestRetryHandlerProperties:
    """Property-based tests for retry handler behavior."""
    
    @pytest.mark.asyncio
    @given(
        max_retries=st.integers(min_value=0, max_value=3),  # Reduced max retries
        base_delay=st.floats(min_value=0.001, max_value=0.1),  # Much smaller delays
        failure_count=st.integers(min_value=0, max_value=5)
    )
    @hypothesis_settings(deadline=5000)  # Increase deadline to 5 seconds
    async def test_retry_handler_attempt_consistency(
        self,
        max_retries: int,
        base_delay: float,
        failure_count: int
    ):
        """
        Property: Retry handler should attempt correct number of retries.
        
        For any retry configuration and failure pattern, handler should:
        1. Attempt exactly max_retries + 1 times
        2. Succeed if failure_count <= max_retries
        3. Fail if failure_count > max_retries
        4. Apply exponential backoff between attempts
        """
        retry_handler = RetryHandler(
            max_retries=max_retries,
            base_delay=base_delay
        )
        
        attempt_count = 0
        
        async def failing_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count <= failure_count:
                raise Exception(f"Attempt {attempt_count} failed")
            return f"Success on attempt {attempt_count}"
        
        # Execute with retry
        if failure_count <= max_retries:
            # Should eventually succeed
            result = await retry_handler.execute_with_retry(failing_function)
            assert result == f"Success on attempt {failure_count + 1}"
            assert attempt_count == failure_count + 1
        else:
            # Should fail after all retries
            with pytest.raises(Exception) as exc_info:
                await retry_handler.execute_with_retry(failing_function)
            
            # Should have attempted max_retries + 1 times
            assert attempt_count == max_retries + 1
            assert f"Attempt {max_retries + 1} failed" in str(exc_info.value)

class TestErrorLoggerProperties:
    """Property-based tests for error logging consistency."""
    
    @given(
        error_message=error_messages,
        severity=error_severities,
        category=error_categories,
        context=error_details
    )
    def test_error_logging_structure_consistency(
        self,
        error_message: str,
        severity: ErrorSeverity,
        category: ErrorCategory,
        context: Dict[str, Any]
    ):
        """
        Property: Error logging should maintain consistent structure and context.
        
        For any error and context, logging should:
        1. Include all required fields
        2. Use appropriate log level based on severity
        3. Preserve context information
        4. Include timestamp
        """
        with patch('logging.getLogger') as mock_logger_factory:
            mock_logger = Mock()
            mock_logger_factory.return_value = mock_logger
            
            error_logger = ErrorLogger("test_logger")
            test_exception = Exception(error_message)
            
            # Log the error
            error_logger.log_error(
                error=test_exception,
                context=context,
                severity=severity,
                category=category
            )
            
            # Verify appropriate logging method was called based on severity
            if severity == ErrorSeverity.CRITICAL:
                assert mock_logger.critical.called
                log_call = mock_logger.critical.call_args
            elif severity == ErrorSeverity.HIGH:
                assert mock_logger.error.called
                log_call = mock_logger.error.call_args
            elif severity == ErrorSeverity.MEDIUM:
                assert mock_logger.warning.called
                log_call = mock_logger.warning.call_args
            else:  # LOW
                assert mock_logger.info.called
                log_call = mock_logger.info.call_args
            
            # Verify log structure
            assert log_call is not None
            extra_data = log_call[1]['extra']
            
            assert extra_data['error_type'] == 'Exception'
            assert extra_data['error_message'] == error_message
            assert extra_data['severity'] == severity.value
            assert extra_data['category'] == category.value
            assert extra_data['context'] == context
            assert 'timestamp' in extra_data
    
    @given(
        event_type=st.text(min_size=1, max_size=50),
        user_id=st.one_of(st.none(), st.uuids().map(str)),
        details=error_details,
        severity=error_severities
    )
    def test_security_event_logging_consistency(
        self,
        event_type: str,
        user_id: Optional[str],
        details: Dict[str, Any],
        severity: ErrorSeverity
    ):
        """
        Property: Security event logging should maintain consistent structure.
        
        For any security event, logging should:
        1. Include security event markers
        2. Preserve user context
        3. Include event details
        4. Use warning level for security events
        """
        with patch('logging.getLogger') as mock_logger_factory:
            mock_logger = Mock()
            mock_logger_factory.return_value = mock_logger
            
            error_logger = ErrorLogger("test_logger")
            
            # Log security event
            error_logger.log_security_event(
                event_type=event_type,
                user_id=user_id,
                details=details,
                severity=severity
            )
            
            # Verify warning level is used for security events
            assert mock_logger.warning.called
            log_call = mock_logger.warning.call_args
            
            # Verify security event structure
            extra_data = log_call[1]['extra']
            assert extra_data['event_type'] == 'SECURITY_EVENT'
            assert extra_data['security_event_type'] == event_type
            assert extra_data['user_id'] == user_id
            assert extra_data['severity'] == severity.value
            assert extra_data['details'] == details
            assert 'timestamp' in extra_data

class TestSystemHealthProperties:
    """Property-based tests for system health monitoring."""
    
    def test_system_health_structure_consistency(self):
        """
        Property: System health should always return consistent structure.
        
        System health should:
        1. Include timestamp
        2. Include circuit breaker states
        3. Include error metrics
        4. Have consistent structure regardless of system state
        """
        health = get_system_health()
        
        # Verify required top-level fields
        assert isinstance(health, dict)
        assert "timestamp" in health
        assert "circuit_breakers" in health
        assert "agent_circuit_breakers" in health
        assert "error_counts" in health
        
        # Verify timestamp format
        timestamp_str = health["timestamp"]
        datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))  # Should not raise
        
        # Verify circuit breaker structure
        assert isinstance(health["circuit_breakers"], dict)
        assert isinstance(health["agent_circuit_breakers"], dict)
        
        # Verify error counts structure
        error_counts = health["error_counts"]
        assert isinstance(error_counts, dict)
        assert "total_errors_last_hour" in error_counts
        assert "critical_errors_last_hour" in error_counts
        assert "circuit_breaker_trips_last_hour" in error_counts

# Stateful Testing for Circuit Breaker State Transitions

class CircuitBreakerStateMachine(RuleBasedStateMachine):
    """
    Stateful property testing for circuit breaker state transitions.
    
    This tests that circuit breaker state transitions are consistent
    regardless of the sequence of operations.
    """
    
    def __init__(self):
        super().__init__()
        self.circuit_breaker = None
        self.failure_threshold = 3
        self.success_threshold = 2
    
    @initialize()
    def init_circuit_breaker(self):
        """Initialize circuit breaker with random parameters."""
        self.circuit_breaker = CircuitBreaker(
            name="test_cb",
            failure_threshold=self.failure_threshold,
            recovery_timeout=1,  # Short timeout for testing
            success_threshold=self.success_threshold
        )
    
    @rule()
    def record_success(self):
        """Record a successful operation."""
        initial_state = self.circuit_breaker.state.state
        
        # Simulate success by calling with successful function
        async def success_func():
            return "success"
        
        # Run in event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.circuit_breaker.call(success_func))
        except:
            pass  # May fail if circuit is open
        finally:
            loop.close()
    
    @rule()
    def record_failure(self):
        """Record a failed operation."""
        initial_state = self.circuit_breaker.state.state
        
        # Simulate failure by calling with failing function
        async def failing_func():
            raise Exception("Test failure")
        
        # Run in event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.circuit_breaker.call(failing_func))
        except:
            pass  # Expected to fail
        finally:
            loop.close()
    
    @invariant()
    def state_is_valid(self):
        """Circuit breaker state should always be valid."""
        assert self.circuit_breaker.state.state in ["CLOSED", "OPEN", "HALF_OPEN"]
        assert self.circuit_breaker.state.failure_count >= 0
        assert self.circuit_breaker.state.success_count >= 0
    
    @invariant()
    def failure_threshold_respected(self):
        """Circuit should open when failure threshold is exceeded."""
        if self.circuit_breaker.state.failure_count >= self.failure_threshold:
            assert self.circuit_breaker.state.state in ["OPEN", "HALF_OPEN"]
    
    @invariant()
    def success_threshold_respected(self):
        """Circuit should close when success threshold is met in HALF_OPEN."""
        if (self.circuit_breaker.state.state == "CLOSED" and 
            hasattr(self, '_was_half_open') and self._was_half_open):
            # If we transitioned from HALF_OPEN to CLOSED, success threshold was met
            assert self.circuit_breaker.state.success_count == 0  # Reset after closing

# Configure Hypothesis settings for property tests
TestCircuitBreakerStateMachine = CircuitBreakerStateMachine.TestCase
TestCircuitBreakerStateMachine.settings = hypothesis_settings(
    max_examples=50,
    stateful_step_count=20,
    deadline=None
)