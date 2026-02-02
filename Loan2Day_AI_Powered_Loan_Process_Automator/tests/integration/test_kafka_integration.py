"""
Integration tests for Kafka messaging system.

Tests the complete Kafka integration including producer, consumer,
and dead letter queue functionality following the LQM Standard.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from decimal import Decimal

from app.services.kafka_integration import AgentKafkaIntegration, MessagePriority
from app.services.kafka_service import KafkaService, KafkaMessage, MessageType
from app.models.pydantic_models import AgentState, SentimentScore


class TestKafkaIntegration:
    """Test suite for Kafka integration functionality."""
    
    @pytest.fixture
    def mock_kafka_service(self):
        """Create a mock Kafka service for testing."""
        mock_service = AsyncMock(spec=KafkaService)
        mock_service.start = AsyncMock()
        mock_service.stop = AsyncMock()
        mock_service.add_consumer = AsyncMock()
        
        # Mock producer context manager
        mock_producer = AsyncMock()
        mock_producer.send_agent_communication = AsyncMock(return_value=True)
        mock_producer.send_audit_log = AsyncMock(return_value=True)
        mock_producer.send_security_event = AsyncMock(return_value=True)
        
        # Create a proper async context manager mock
        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__ = AsyncMock(return_value=mock_producer)
        mock_context_manager.__aexit__ = AsyncMock(return_value=None)
        
        mock_service.get_producer = MagicMock(return_value=mock_context_manager)
        
        return mock_service
    
    @pytest.fixture
    def kafka_integration(self, mock_kafka_service):
        """Create Kafka integration instance with mock service."""
        integration = AgentKafkaIntegration(mock_kafka_service)
        return integration
    
    @pytest.mark.asyncio
    async def test_initialization(self, kafka_integration, mock_kafka_service):
        """Test Kafka integration initialization."""
        # Test initialization
        await kafka_integration.initialize()
        
        # Verify service was started
        mock_kafka_service.start.assert_called_once()
        
        # Verify consumers were added
        assert mock_kafka_service.add_consumer.call_count == 4
        
        # Verify initialization flag
        assert kafka_integration._is_initialized is True
    
    @pytest.mark.asyncio
    async def test_shutdown(self, kafka_integration, mock_kafka_service):
        """Test Kafka integration shutdown."""
        # Initialize first
        await kafka_integration.initialize()
        
        # Test shutdown
        await kafka_integration.shutdown()
        
        # Verify service was stopped
        mock_kafka_service.stop.assert_called_once()
        
        # Verify shutdown flag
        assert kafka_integration._is_initialized is False
    
    @pytest.mark.asyncio
    async def test_send_agent_message(self, kafka_integration, mock_kafka_service):
        """Test sending inter-agent messages."""
        # Initialize integration
        await kafka_integration.initialize()
        
        # Test sending message
        result = await kafka_integration.send_agent_message(
            from_agent="master",
            to_agent="sales",
            message_type="task_request",
            payload={"action": "analyze_sentiment", "text": "I need a loan"},
            session_id="test-session-123",
            user_id="user-456",
            priority=MessagePriority.HIGH
        )
        
        # Verify message was sent
        assert result is True
        
        # Get the mock producer and verify it was called
        mock_producer_context = mock_kafka_service.get_producer.return_value
        mock_producer = await mock_producer_context.__aenter__()
        
        mock_producer.send_agent_communication.assert_called_once()
        call_args = mock_producer.send_agent_communication.call_args
        
        assert call_args[1]["source_agent"] == "master"
        assert call_args[1]["target_agent"] == "sales"
        assert call_args[1]["session_id"] == "test-session-123"
        assert call_args[1]["user_id"] == "user-456"
        assert call_args[1]["priority"] == MessagePriority.HIGH
    
    @pytest.mark.asyncio
    async def test_log_agent_action(self, kafka_integration, mock_kafka_service):
        """Test logging agent actions for audit compliance."""
        # Initialize integration
        await kafka_integration.initialize()
        
        # Test logging action
        result = await kafka_integration.log_agent_action(
            agent_name="verification",
            action="kyc_document_processed",
            details={
                "document_type": "pan_card",
                "verification_status": "verified",
                "fraud_score": Decimal("0.15")
            },
            session_id="test-session-123",
            user_id="user-456"
        )
        
        # Verify log was sent
        assert result is True
        
        # Get the mock producer and verify it was called
        mock_producer_context = mock_kafka_service.get_producer.return_value
        mock_producer = await mock_producer_context.__aenter__()
        
        mock_producer.send_audit_log.assert_called_once()
        call_args = mock_producer.send_audit_log.call_args
        
        assert call_args[1]["source_agent"] == "verification"
        assert call_args[1]["action"] == "kyc_document_processed"
        assert call_args[1]["session_id"] == "test-session-123"
        assert call_args[1]["user_id"] == "user-456"
    
    @pytest.mark.asyncio
    async def test_log_security_event(self, kafka_integration, mock_kafka_service):
        """Test logging security events for SGS monitoring."""
        # Initialize integration
        await kafka_integration.initialize()
        
        # Test logging security event
        result = await kafka_integration.log_security_event(
            agent_name="verification",
            event_type="deepfake_detected",
            severity="HIGH",
            details={
                "file_path": "/uploads/suspicious_image.jpg",
                "sgs_score": Decimal("0.85"),
                "detection_confidence": Decimal("0.92")
            },
            session_id="test-session-123",
            user_id="user-456"
        )
        
        # Verify event was logged
        assert result is True
        
        # Get the mock producer and verify it was called
        mock_producer_context = mock_kafka_service.get_producer.return_value
        mock_producer = await mock_producer_context.__aenter__()
        
        mock_producer.send_security_event.assert_called_once()
        call_args = mock_producer.send_security_event.call_args
        
        assert call_args[1]["source_agent"] == "verification"
        assert call_args[1]["event_type"] == "deepfake_detected"
        assert call_args[1]["severity"] == "HIGH"
        assert call_args[1]["session_id"] == "test-session-123"
        assert call_args[1]["user_id"] == "user-456"
    
    @pytest.mark.asyncio
    async def test_broadcast_state_change(self, kafka_integration, mock_kafka_service):
        """Test broadcasting agent state changes."""
        # Initialize integration
        await kafka_integration.initialize()
        
        # Create test agent states
        old_state = AgentState(
            session_id="test-session-123",
            user_id="user-456",
            current_step="KYC",
            loan_details={"amount_in_cents": Decimal("500000")},
            kyc_status="PENDING",
            fraud_score=0.0,
            sentiment_history=[
                SentimentScore(
                    polarity=0.0,
                    subjectivity=0.5,
                    emotion="neutral",
                    confidence=0.8
                )
            ],
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        new_state = AgentState(
            session_id="test-session-123",
            user_id="user-456",
            current_step="NEGOTIATION",
            loan_details={"amount_in_cents": Decimal("500000")},
            kyc_status="VERIFIED",
            fraud_score=0.15,
            sentiment_history=[
                SentimentScore(
                    polarity=0.0,
                    subjectivity=0.5,
                    emotion="neutral",
                    confidence=0.8
                ),
                SentimentScore(
                    polarity=0.6,
                    subjectivity=0.7,
                    emotion="positive",
                    confidence=0.9
                )
            ],
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Test broadcasting state change
        result = await kafka_integration.broadcast_state_change(
            agent_name="verification",
            old_state=old_state,
            new_state=new_state,
            change_reason="kyc_verification_completed"
        )
        
        # Verify broadcast was sent
        assert result is True
        
        # Get the mock producer and verify it was called
        mock_producer_context = mock_kafka_service.get_producer.return_value
        mock_producer = await mock_producer_context.__aenter__()
        
        mock_producer.send_agent_communication.assert_called_once()
        call_args = mock_producer.send_agent_communication.call_args
        
        assert call_args[1]["source_agent"] == "verification"
        assert call_args[1]["target_agent"] == "ALL_AGENTS"
        assert call_args[1]["priority"] == MessagePriority.HIGH
    
    @pytest.mark.asyncio
    async def test_message_handler_registration(self, kafka_integration):
        """Test registering message handlers for agents."""
        # Create mock handler
        mock_handler = AsyncMock(return_value=True)
        
        # Register handler
        kafka_integration.register_message_handler("sales", mock_handler)
        
        # Verify handler was registered
        assert "sales" in kafka_integration._message_handlers
        assert kafka_integration._message_handlers["sales"] == mock_handler
    
    @pytest.mark.asyncio
    async def test_handle_agent_communication(self, kafka_integration):
        """Test handling incoming agent communication messages."""
        # Register mock handler
        mock_handler = AsyncMock(return_value=True)
        kafka_integration.register_message_handler("sales", mock_handler)
        
        # Create test message
        test_message = KafkaMessage(
            message_id="test-msg-123",
            message_type=MessageType.AGENT_COMMUNICATION,
            priority=MessagePriority.NORMAL,
            source_agent="master",
            target_agent="sales",
            session_id="test-session-123",
            user_id="user-456",
            payload={"action": "analyze_sentiment", "text": "I need help"},
            timestamp=datetime.now()
        )
        
        # Test handling message
        result = await kafka_integration._handle_agent_communication(test_message)
        
        # Verify handler was called and message processed
        assert result is True
        mock_handler.assert_called_once_with(test_message)
    
    @pytest.mark.asyncio
    async def test_handle_broadcast_message(self, kafka_integration):
        """Test handling broadcast messages to all agents."""
        # Register multiple mock handlers
        mock_handler1 = AsyncMock(return_value=True)
        mock_handler2 = AsyncMock(return_value=True)
        kafka_integration.register_message_handler("sales", mock_handler1)
        kafka_integration.register_message_handler("verification", mock_handler2)
        
        # Create broadcast message
        broadcast_message = KafkaMessage(
            message_id="broadcast-msg-123",
            message_type=MessageType.AGENT_COMMUNICATION,
            priority=MessagePriority.HIGH,
            source_agent="master",
            target_agent="ALL_AGENTS",
            session_id="test-session-123",
            user_id="user-456",
            payload={"action": "system_shutdown", "reason": "maintenance"},
            timestamp=datetime.now()
        )
        
        # Test handling broadcast
        result = await kafka_integration._handle_agent_communication(broadcast_message)
        
        # Verify all handlers were called
        assert result is True
        mock_handler1.assert_called_once_with(broadcast_message)
        mock_handler2.assert_called_once_with(broadcast_message)


class TestKafkaMessage:
    """Test suite for KafkaMessage serialization and deserialization."""
    
    def test_message_serialization_with_decimals(self):
        """Test message serialization with Decimal values (LQM Standard)."""
        # Create message with Decimal values
        message = KafkaMessage(
            message_id="test-msg-123",
            message_type=MessageType.AGENT_COMMUNICATION,
            priority=MessagePriority.NORMAL,
            source_agent="underwriting",
            target_agent="master",
            session_id="test-session-123",
            user_id="user-456",
            payload={
                "loan_amount": Decimal("500000.00"),
                "emi_amount": Decimal("12500.50"),
                "interest_rate": Decimal("12.5"),
                "timestamp": datetime(2024, 1, 15, 10, 30, 0)
            },
            timestamp=datetime(2024, 1, 15, 10, 30, 0)
        )
        
        # Serialize to dictionary
        serialized = message.to_dict()
        
        # Verify Decimal serialization
        assert serialized["payload"]["loan_amount"]["__decimal__"] == "500000.00"
        assert serialized["payload"]["emi_amount"]["__decimal__"] == "12500.50"
        assert serialized["payload"]["interest_rate"]["__decimal__"] == "12.5"
        assert serialized["payload"]["timestamp"]["__datetime__"] == "2024-01-15T10:30:00"
    
    def test_message_deserialization_with_decimals(self):
        """Test message deserialization with Decimal values (LQM Standard)."""
        # Create serialized message data
        serialized_data = {
            "message_id": "test-msg-123",
            "message_type": "agent_communication",
            "priority": 2,
            "source_agent": "underwriting",
            "target_agent": "master",
            "session_id": "test-session-123",
            "user_id": "user-456",
            "payload": {
                "loan_amount": {"__decimal__": "500000.00"},
                "emi_amount": {"__decimal__": "12500.50"},
                "interest_rate": {"__decimal__": "12.5"},
                "timestamp": {"__datetime__": "2024-01-15T10:30:00"}
            },
            "timestamp": "2024-01-15T10:30:00",
            "retry_count": 0,
            "max_retries": 3
        }
        
        # Deserialize message
        message = KafkaMessage.from_dict(serialized_data)
        
        # Verify Decimal deserialization
        assert isinstance(message.payload["loan_amount"], Decimal)
        assert message.payload["loan_amount"] == Decimal("500000.00")
        assert isinstance(message.payload["emi_amount"], Decimal)
        assert message.payload["emi_amount"] == Decimal("12500.50")
        assert isinstance(message.payload["interest_rate"], Decimal)
        assert message.payload["interest_rate"] == Decimal("12.5")
        assert isinstance(message.payload["timestamp"], datetime)
        assert message.payload["timestamp"] == datetime(2024, 1, 15, 10, 30, 0)
    
    def test_message_roundtrip_serialization(self):
        """Test complete serialization/deserialization roundtrip."""
        # Create original message
        original_message = KafkaMessage(
            message_id="roundtrip-test-123",
            message_type=MessageType.AUDIT_LOG,
            priority=MessagePriority.HIGH,
            source_agent="verification",
            target_agent=None,
            session_id="test-session-456",
            user_id="user-789",
            payload={
                "action": "fraud_detection",
                "fraud_score": Decimal("0.85"),
                "confidence": Decimal("0.92"),
                "detected_at": datetime(2024, 1, 15, 14, 45, 30)
            },
            timestamp=datetime(2024, 1, 15, 14, 45, 30),
            retry_count=1,
            max_retries=5
        )
        
        # Serialize and deserialize
        serialized = original_message.to_dict()
        deserialized_message = KafkaMessage.from_dict(serialized)
        
        # Verify all fields match
        assert deserialized_message.message_id == original_message.message_id
        assert deserialized_message.message_type == original_message.message_type
        assert deserialized_message.priority == original_message.priority
        assert deserialized_message.source_agent == original_message.source_agent
        assert deserialized_message.target_agent == original_message.target_agent
        assert deserialized_message.session_id == original_message.session_id
        assert deserialized_message.user_id == original_message.user_id
        assert deserialized_message.retry_count == original_message.retry_count
        assert deserialized_message.max_retries == original_message.max_retries
        
        # Verify payload with Decimal precision
        assert deserialized_message.payload["action"] == "fraud_detection"
        assert deserialized_message.payload["fraud_score"] == Decimal("0.85")
        assert deserialized_message.payload["confidence"] == Decimal("0.92")
        assert deserialized_message.payload["detected_at"] == datetime(2024, 1, 15, 14, 45, 30)