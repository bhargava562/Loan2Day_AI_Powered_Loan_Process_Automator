"""
Unit tests for Kafka Service functionality.

Tests message production, consumption, and dead letter queue functionality
following the LQM Standard with strict typing and Decimal precision.

Architecture: Tests individual components in isolation with proper mocking
Security: Validates message serialization and error handling
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
from datetime import datetime
from decimal import Decimal
import json

from app.services.kafka_service import (
    KafkaProducerService,
    KafkaConsumerService,
    KafkaService,
    KafkaMessage,
    MessageType,
    MessagePriority
)


class TestKafkaMessage:
    """Test suite for KafkaMessage serialization and validation."""
    
    def test_message_creation_with_decimals(self):
        """Test creating KafkaMessage with Decimal values (LQM Standard)."""
        # Create message with Decimal monetary values
        message = KafkaMessage(
            message_id="test-msg-001",
            message_type=MessageType.AGENT_COMMUNICATION,
            priority=MessagePriority.HIGH,
            source_agent="underwriting",
            target_agent="master",
            session_id="session-123",
            user_id="user-456",
            payload={
                "loan_amount_in_cents": Decimal("50000000"),  # $500,000.00
                "emi_in_cents": Decimal("1250050"),          # $12,500.50
                "interest_rate": Decimal("12.50"),
                "processing_fee": Decimal("2500.00")
            },
            timestamp=datetime(2024, 1, 15, 10, 30, 0)
        )
        
        # Verify message properties
        assert message.message_id == "test-msg-001"
        assert message.message_type == MessageType.AGENT_COMMUNICATION
        assert message.priority == MessagePriority.HIGH
        assert message.source_agent == "underwriting"
        assert message.target_agent == "master"
        assert message.session_id == "session-123"
        assert message.user_id == "user-456"
        
        # Verify Decimal values preserved
        assert isinstance(message.payload["loan_amount_in_cents"], Decimal)
        assert message.payload["loan_amount_in_cents"] == Decimal("50000000")
        assert isinstance(message.payload["emi_in_cents"], Decimal)
        assert message.payload["emi_in_cents"] == Decimal("1250050")
    
    def test_message_serialization_preserves_decimals(self):
        """Test that message serialization preserves Decimal precision."""
        # Create message with precise Decimal values
        message = KafkaMessage(
            message_id="decimal-test-001",
            message_type=MessageType.AUDIT_LOG,
            priority=MessagePriority.NORMAL,
            source_agent="verification",
            target_agent=None,
            session_id="session-789",
            user_id="user-123",
            payload={
                "fraud_score": Decimal("0.8500"),
                "trust_score": Decimal("0.9250"),
                "loan_amount": Decimal("250000.00"),
                "processing_timestamp": datetime(2024, 1, 15, 14, 30, 45)
            },
            timestamp=datetime(2024, 1, 15, 14, 30, 45)
        )
        
        # Serialize message
        serialized = message.to_dict()
        
        # Verify Decimal serialization format
        assert serialized["payload"]["fraud_score"]["__decimal__"] == "0.8500"
        assert serialized["payload"]["trust_score"]["__decimal__"] == "0.9250"
        assert serialized["payload"]["loan_amount"]["__decimal__"] == "250000.00"
        
        # Verify datetime serialization
        assert serialized["payload"]["processing_timestamp"]["__datetime__"] == "2024-01-15T14:30:45"
        assert serialized["timestamp"] == "2024-01-15T14:30:45"
    
    def test_message_deserialization_restores_decimals(self):
        """Test that message deserialization restores Decimal precision."""
        # Create serialized message data
        serialized_data = {
            "message_id": "deserialize-test-001",
            "message_type": "agent_communication",
            "priority": 3,
            "source_agent": "underwriting",
            "target_agent": "sales",
            "session_id": "session-456",
            "user_id": "user-789",
            "payload": {
                "emi_calculation": {"__decimal__": "15750.25"},
                "interest_rate": {"__decimal__": "11.75"},
                "loan_tenure_months": 60,
                "calculated_at": {"__datetime__": "2024-01-15T16:45:30"}
            },
            "timestamp": "2024-01-15T16:45:30",
            "retry_count": 0,
            "max_retries": 3
        }
        
        # Deserialize message
        message = KafkaMessage.from_dict(serialized_data)
        
        # Verify Decimal restoration
        assert isinstance(message.payload["emi_calculation"], Decimal)
        assert message.payload["emi_calculation"] == Decimal("15750.25")
        assert isinstance(message.payload["interest_rate"], Decimal)
        assert message.payload["interest_rate"] == Decimal("11.75")
        
        # Verify datetime restoration
        assert isinstance(message.payload["calculated_at"], datetime)
        assert message.payload["calculated_at"] == datetime(2024, 1, 15, 16, 45, 30)
        assert isinstance(message.timestamp, datetime)
        assert message.timestamp == datetime(2024, 1, 15, 16, 45, 30)
    
    def test_message_roundtrip_serialization(self):
        """Test complete serialization/deserialization roundtrip maintains precision."""
        # Create original message with complex nested Decimal values
        original_message = KafkaMessage(
            message_id="roundtrip-test-001",
            message_type=MessageType.SECURITY_EVENT,
            priority=MessagePriority.CRITICAL,
            source_agent="verification",
            target_agent="SGS_Monitor",
            session_id="security-session-123",
            user_id="user-456",
            payload={
                "security_analysis": {
                    "deepfake_probability": Decimal("0.8750"),
                    "document_authenticity": Decimal("0.1250"),
                    "risk_assessment": {
                        "financial_risk": Decimal("0.9500"),
                        "identity_risk": Decimal("0.8000")
                    }
                },
                "detection_timestamp": datetime(2024, 1, 15, 18, 20, 15),
                "requires_manual_review": True
            },
            timestamp=datetime(2024, 1, 15, 18, 20, 15),
            retry_count=2,
            max_retries=5
        )
        
        # Perform roundtrip serialization
        serialized = original_message.to_dict()
        deserialized_message = KafkaMessage.from_dict(serialized)
        
        # Verify all properties match exactly
        assert deserialized_message.message_id == original_message.message_id
        assert deserialized_message.message_type == original_message.message_type
        assert deserialized_message.priority == original_message.priority
        assert deserialized_message.source_agent == original_message.source_agent
        assert deserialized_message.target_agent == original_message.target_agent
        assert deserialized_message.session_id == original_message.session_id
        assert deserialized_message.user_id == original_message.user_id
        assert deserialized_message.retry_count == original_message.retry_count
        assert deserialized_message.max_retries == original_message.max_retries
        assert deserialized_message.timestamp == original_message.timestamp
        
        # Verify nested Decimal precision preserved
        security_analysis = deserialized_message.payload["security_analysis"]
        assert security_analysis["deepfake_probability"] == Decimal("0.8750")
        assert security_analysis["document_authenticity"] == Decimal("0.1250")
        
        risk_assessment = security_analysis["risk_assessment"]
        assert risk_assessment["financial_risk"] == Decimal("0.9500")
        assert risk_assessment["identity_risk"] == Decimal("0.8000")
        
        # Verify datetime precision preserved
        assert deserialized_message.payload["detection_timestamp"] == datetime(2024, 1, 15, 18, 20, 15)


class TestKafkaProducerService:
    """Test suite for Kafka producer functionality."""
    
    @pytest.fixture
    def mock_kafka_config(self):
        """Mock Kafka configuration."""
        return {
            'bootstrap_servers': ['localhost:9092'],
            'retry_backoff_ms': 100,
            'request_timeout_ms': 30000,
            'dead_letter_topic': 'loan2day-dead-letter',
            'group_id': 'loan2day-test-group',
            'auto_offset_reset': 'earliest',
            'enable_auto_commit': True,
            'max_poll_records': 100,
            'session_timeout_ms': 30000,
            'heartbeat_interval_ms': 3000
        }
    
    @pytest.fixture
    def producer_service(self, mock_kafka_config):
        """Create KafkaProducerService with mocked configuration."""
        with patch('app.services.kafka_service.get_kafka_config', return_value=mock_kafka_config):
            return KafkaProducerService()
    
    @pytest.mark.asyncio
    async def test_producer_start_success(self, producer_service):
        """Test successful producer startup."""
        with patch('app.services.kafka_service.AIOKafkaProducer') as mock_producer_class:
            mock_producer = AsyncMock()
            mock_producer.start = AsyncMock()
            mock_producer_class.return_value = mock_producer
            
            # Test producer start
            await producer_service.start()
            
            # Verify producer was created with correct configuration
            mock_producer_class.assert_called_once()
            call_kwargs = mock_producer_class.call_args[1]
            assert call_kwargs['bootstrap_servers'] == ['localhost:9092']
            assert call_kwargs['retry_backoff_ms'] == 100
            assert call_kwargs['request_timeout_ms'] == 30000
            assert call_kwargs['acks'] == 'all'
            
            # Verify producer was started
            mock_producer.start.assert_called_once()
            assert producer_service._is_running is True
            assert producer_service.producer == mock_producer
    
    @pytest.mark.asyncio
    async def test_producer_start_failure(self, producer_service):
        """Test producer startup failure handling."""
        with patch('app.services.kafka_service.AIOKafkaProducer') as mock_producer_class:
            mock_producer = AsyncMock()
            mock_producer.start = AsyncMock(side_effect=Exception("Connection failed"))
            mock_producer_class.return_value = mock_producer
            
            # Test producer start failure
            with pytest.raises(Exception, match="Connection failed"):
                await producer_service.start()
            
            # Verify producer is not marked as running
            assert producer_service._is_running is False
    
    @pytest.mark.asyncio
    async def test_producer_stop(self, producer_service):
        """Test producer shutdown."""
        with patch('app.services.kafka_service.AIOKafkaProducer') as mock_producer_class:
            mock_producer = AsyncMock()
            mock_producer.start = AsyncMock()
            mock_producer.stop = AsyncMock()
            mock_producer_class.return_value = mock_producer
            
            # Start and then stop producer
            await producer_service.start()
            await producer_service.stop()
            
            # Verify producer was stopped
            mock_producer.stop.assert_called_once()
            assert producer_service._is_running is False
    
    @pytest.mark.asyncio
    async def test_send_message_success(self, producer_service):
        """Test successful message sending."""
        with patch('app.services.kafka_service.AIOKafkaProducer') as mock_producer_class:
            mock_producer = AsyncMock()
            mock_producer.start = AsyncMock()
            mock_producer.send_and_wait = AsyncMock()
            
            # Mock record metadata
            mock_metadata = MagicMock()
            mock_metadata.partition = 0
            mock_metadata.offset = 12345
            mock_producer.send_and_wait.return_value = mock_metadata
            
            mock_producer_class.return_value = mock_producer
            
            # Start producer
            await producer_service.start()
            
            # Create test message
            test_message = KafkaMessage(
                message_id="send-test-001",
                message_type=MessageType.AGENT_COMMUNICATION,
                priority=MessagePriority.NORMAL,
                source_agent="master",
                target_agent="sales",
                session_id="session-123",
                user_id="user-456",
                payload={"action": "analyze_sentiment", "text": "I need a loan"},
                timestamp=datetime.now()
            )
            
            # Test message sending
            result = await producer_service.send_message("test-topic", test_message)
            
            # Verify message was sent successfully
            assert result is True
            mock_producer.send_and_wait.assert_called_once()
            
            # Verify call arguments
            call_args = mock_producer.send_and_wait.call_args
            assert call_args[1]["topic"] == "test-topic"
            assert call_args[1]["key"] == "session-123"  # Should use session_id as key
    
    @pytest.mark.asyncio
    async def test_send_message_not_running(self, producer_service):
        """Test sending message when producer is not running."""
        # Create test message
        test_message = KafkaMessage(
            message_id="not-running-test-001",
            message_type=MessageType.AGENT_COMMUNICATION,
            priority=MessagePriority.NORMAL,
            source_agent="master",
            target_agent="sales",
            session_id="session-123",
            user_id="user-456",
            payload={"action": "test"},
            timestamp=datetime.now()
        )
        
        # Test sending when not running
        result = await producer_service.send_message("test-topic", test_message)
        
        # Verify failure
        assert result is False
    
    @pytest.mark.asyncio
    async def test_send_agent_communication(self, producer_service):
        """Test sending inter-agent communication messages."""
        with patch('app.services.kafka_service.AIOKafkaProducer') as mock_producer_class:
            mock_producer = AsyncMock()
            mock_producer.start = AsyncMock()
            mock_producer.send_and_wait = AsyncMock()
            
            # Mock successful send
            mock_metadata = MagicMock()
            mock_metadata.partition = 0
            mock_metadata.offset = 67890
            mock_producer.send_and_wait.return_value = mock_metadata
            
            mock_producer_class.return_value = mock_producer
            
            # Start producer
            await producer_service.start()
            
            # Test agent communication
            result = await producer_service.send_agent_communication(
                source_agent="master",
                target_agent="underwriting",
                payload={
                    "action": "calculate_emi",
                    "loan_amount": Decimal("500000.00"),
                    "tenure_months": 60,
                    "interest_rate": Decimal("12.5")
                },
                session_id="emi-calc-session-123",
                user_id="user-789",
                priority=MessagePriority.HIGH
            )
            
            # Verify message was sent
            assert result is True
            mock_producer.send_and_wait.assert_called_once()
            
            # Verify message content
            call_args = mock_producer.send_and_wait.call_args
            message_data = call_args[1]["value"]
            
            assert message_data["source_agent"] == "master"
            assert message_data["target_agent"] == "underwriting"
            assert message_data["session_id"] == "emi-calc-session-123"
            assert message_data["user_id"] == "user-789"
            assert message_data["priority"] == MessagePriority.HIGH.value
            
            # Verify Decimal serialization in payload
            payload = message_data["payload"]
            assert payload["loan_amount"]["__decimal__"] == "500000.00"
            assert payload["interest_rate"]["__decimal__"] == "12.5"
    
    @pytest.mark.asyncio
    async def test_send_audit_log(self, producer_service):
        """Test sending audit log messages for compliance."""
        with patch('app.services.kafka_service.AIOKafkaProducer') as mock_producer_class:
            mock_producer = AsyncMock()
            mock_producer.start = AsyncMock()
            mock_producer.send_and_wait = AsyncMock()
            
            # Mock successful send
            mock_metadata = MagicMock()
            mock_metadata.partition = 1
            mock_metadata.offset = 11111
            mock_producer.send_and_wait.return_value = mock_metadata
            
            mock_producer_class.return_value = mock_producer
            
            # Start producer
            await producer_service.start()
            
            # Test audit log
            result = await producer_service.send_audit_log(
                source_agent="verification",
                action="kyc_document_processed",
                details={
                    "document_type": "pan_card",
                    "verification_result": "verified",
                    "fraud_score": Decimal("0.15"),
                    "processing_time_ms": 2500
                },
                session_id="kyc-session-456",
                user_id="user-123"
            )
            
            # Verify audit log was sent
            assert result is True
            mock_producer.send_and_wait.assert_called_once()
            
            # Verify audit log content
            call_args = mock_producer.send_and_wait.call_args
            message_data = call_args[1]["value"]
            
            assert message_data["message_type"] == MessageType.AUDIT_LOG.value
            assert message_data["priority"] == MessagePriority.HIGH.value
            assert message_data["source_agent"] == "verification"
            assert message_data["target_agent"] is None
            
            # Verify compliance flag
            payload = message_data["payload"]
            assert payload["compliance_required"] is True
            assert payload["action"] == "kyc_document_processed"
            assert payload["details"]["fraud_score"]["__decimal__"] == "0.15"
    
    @pytest.mark.asyncio
    async def test_send_security_event(self, producer_service):
        """Test sending security events for SGS monitoring."""
        with patch('app.services.kafka_service.AIOKafkaProducer') as mock_producer_class:
            mock_producer = AsyncMock()
            mock_producer.start = AsyncMock()
            mock_producer.send_and_wait = AsyncMock()
            
            # Mock successful send
            mock_metadata = MagicMock()
            mock_metadata.partition = 2
            mock_metadata.offset = 22222
            mock_producer.send_and_wait.return_value = mock_metadata
            
            mock_producer_class.return_value = mock_producer
            
            # Start producer
            await producer_service.start()
            
            # Test security event
            result = await producer_service.send_security_event(
                source_agent="verification",
                event_type="deepfake_detected",
                severity="CRITICAL",
                details={
                    "file_path": "/uploads/suspicious_document.jpg",
                    "sgs_score": Decimal("0.95"),
                    "detection_confidence": Decimal("0.88"),
                    "threat_level": "HIGH"
                },
                session_id="security-session-789",
                user_id="user-456"
            )
            
            # Verify security event was sent
            assert result is True
            mock_producer.send_and_wait.assert_called_once()
            
            # Verify security event content
            call_args = mock_producer.send_and_wait.call_args
            message_data = call_args[1]["value"]
            
            assert message_data["message_type"] == MessageType.SECURITY_EVENT.value
            assert message_data["priority"] == MessagePriority.CRITICAL.value
            assert message_data["source_agent"] == "verification"
            assert message_data["target_agent"] == "SGS_Monitor"
            
            # Verify security event payload
            payload = message_data["payload"]
            assert payload["event_type"] == "deepfake_detected"
            assert payload["severity"] == "CRITICAL"
            assert payload["requires_immediate_attention"] is True
            assert payload["details"]["sgs_score"]["__decimal__"] == "0.95"
            assert payload["details"]["detection_confidence"]["__decimal__"] == "0.88"


class TestKafkaConsumerService:
    """Test suite for Kafka consumer functionality."""
    
    @pytest.fixture
    def mock_kafka_config(self):
        """Mock Kafka configuration."""
        return {
            'bootstrap_servers': ['localhost:9092'],
            'group_id': 'loan2day-consumer-group',
            'auto_offset_reset': 'earliest',
            'enable_auto_commit': True,
            'max_poll_records': 100,
            'session_timeout_ms': 30000,
            'heartbeat_interval_ms': 3000,
            'dead_letter_topic': 'loan2day-dead-letter',
            'retry_backoff_ms': 100,
            'request_timeout_ms': 30000
        }
    
    @pytest.fixture
    def mock_message_handler(self):
        """Mock message handler function."""
        return AsyncMock(return_value=True)
    
    @pytest.fixture
    def consumer_service(self, mock_kafka_config, mock_message_handler):
        """Create KafkaConsumerService with mocked configuration."""
        with patch('app.services.kafka_service.get_kafka_config', return_value=mock_kafka_config):
            return KafkaConsumerService(
                topics=["test-topic"],
                message_handler=mock_message_handler
            )
    
    @pytest.mark.asyncio
    async def test_consumer_start_success(self, consumer_service):
        """Test successful consumer startup."""
        with patch('app.services.kafka_service.AIOKafkaConsumer') as mock_consumer_class, \
             patch('app.services.kafka_service.KafkaProducerService') as mock_producer_class:
            
            # Mock consumer
            mock_consumer = AsyncMock()
            mock_consumer.start = AsyncMock()
            mock_consumer_class.return_value = mock_consumer
            
            # Mock producer for dead letter queue
            mock_producer = AsyncMock()
            mock_producer.start = AsyncMock()
            mock_producer_class.return_value = mock_producer
            
            # Test consumer start
            await consumer_service.start()
            
            # Verify consumer and producer were started
            mock_consumer.start.assert_called_once()
            mock_producer.start.assert_called_once()
            assert consumer_service._is_running is True
            assert consumer_service.consumer == mock_consumer
            assert consumer_service.producer_service == mock_producer
    
    @pytest.mark.asyncio
    async def test_consumer_stop(self, consumer_service):
        """Test consumer shutdown."""
        with patch('app.services.kafka_service.AIOKafkaConsumer') as mock_consumer_class, \
             patch('app.services.kafka_service.KafkaProducerService') as mock_producer_class:
            
            # Mock consumer and producer
            mock_consumer = AsyncMock()
            mock_consumer.start = AsyncMock()
            mock_consumer.stop = AsyncMock()
            mock_consumer_class.return_value = mock_consumer
            
            mock_producer = AsyncMock()
            mock_producer.start = AsyncMock()
            mock_producer.stop = AsyncMock()
            mock_producer_class.return_value = mock_producer
            
            # Start and then stop consumer
            await consumer_service.start()
            await consumer_service.stop()
            
            # Verify consumer and producer were stopped
            mock_consumer.stop.assert_called_once()
            mock_producer.stop.assert_called_once()
            assert consumer_service._is_running is False
    
    @pytest.mark.asyncio
    async def test_process_message_success(self, consumer_service, mock_message_handler):
        """Test successful message processing."""
        # Create mock raw message
        mock_raw_message = MagicMock()
        mock_raw_message.value = {
            "message_id": "process-test-001",
            "message_type": "agent_communication",
            "priority": 2,
            "source_agent": "master",
            "target_agent": "sales",
            "session_id": "session-123",
            "user_id": "user-456",
            "payload": {"action": "test"},
            "timestamp": "2024-01-15T10:30:00",
            "retry_count": 0,
            "max_retries": 3
        }
        
        # Test message processing
        await consumer_service._process_message(mock_raw_message)
        
        # Verify handler was called
        mock_message_handler.assert_called_once()
        
        # Verify message was properly deserialized
        call_args = mock_message_handler.call_args[0][0]
        assert call_args.message_id == "process-test-001"
        assert call_args.source_agent == "master"
        assert call_args.target_agent == "sales"
    
    @pytest.mark.asyncio
    async def test_process_message_handler_failure(self, consumer_service, mock_message_handler):
        """Test message processing when handler fails."""
        # Configure handler to return False (failure)
        mock_message_handler.return_value = False
        
        # Mock producer for dead letter queue
        mock_producer = AsyncMock()
        mock_producer.send_message = AsyncMock(return_value=True)
        consumer_service.producer_service = mock_producer
        
        # Create mock raw message
        mock_raw_message = MagicMock()
        mock_raw_message.value = {
            "message_id": "failure-test-001",
            "message_type": "agent_communication",
            "priority": 2,
            "source_agent": "master",
            "target_agent": "sales",
            "session_id": "session-123",
            "user_id": "user-456",
            "payload": {"action": "test"},
            "timestamp": "2024-01-15T10:30:00",
            "retry_count": 0,
            "max_retries": 1  # Low retry count for faster test
        }
        
        # Test message processing with failure
        await consumer_service._process_message(mock_raw_message)
        
        # Verify handler was called multiple times (original + retries)
        assert mock_message_handler.call_count >= 2
        
        # Verify message was sent to dead letter queue
        mock_producer.send_message.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_handle_failed_message_retry_success(self, consumer_service, mock_message_handler):
        """Test failed message retry logic with eventual success."""
        # Configure handler to succeed on retry
        mock_message_handler.return_value = True
        
        # Mock producer for dead letter queue (not needed for successful retry)
        mock_producer = AsyncMock()
        mock_producer.send_message = AsyncMock(return_value=True)
        consumer_service.producer_service = mock_producer
        
        # Create test message
        test_message = KafkaMessage(
            message_id="retry-test-001",
            message_type=MessageType.AGENT_COMMUNICATION,
            priority=MessagePriority.NORMAL,
            source_agent="master",
            target_agent="sales",
            session_id="session-123",
            user_id="user-456",
            payload={"action": "retry_test"},
            timestamp=datetime.now(),
            retry_count=0,
            max_retries=3
        )
        
        # Test retry handling - this should retry and succeed
        await consumer_service._handle_failed_message(test_message, "Handler returned False")
        
        # Verify handler was called once during retry
        assert mock_message_handler.call_count == 1
        
        # Verify retry count was incremented
        assert test_message.retry_count == 1
        
        # Verify message was NOT sent to dead letter queue (since retry succeeded)
        mock_producer.send_message.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_send_to_dead_letter_queue(self, consumer_service):
        """Test sending messages to dead letter queue."""
        # Mock producer for dead letter queue
        mock_producer = AsyncMock()
        mock_producer.send_message = AsyncMock(return_value=True)
        consumer_service.producer_service = mock_producer
        
        # Create test message that exceeded retries
        failed_message = KafkaMessage(
            message_id="dlq-test-001",
            message_type=MessageType.AGENT_COMMUNICATION,
            priority=MessagePriority.NORMAL,
            source_agent="master",
            target_agent="sales",
            session_id="session-456",
            user_id="user-789",
            payload={"action": "failed_action"},
            timestamp=datetime.now(),
            retry_count=3,
            max_retries=3
        )
        
        # Test sending to dead letter queue
        await consumer_service._send_to_dead_letter_queue(failed_message, "Max retries exceeded")
        
        # Verify dead letter message was sent
        mock_producer.send_message.assert_called_once()
        
        # Verify dead letter message content
        call_args = mock_producer.send_message.call_args
        topic = call_args[0][0]
        dlq_message = call_args[0][1]
        
        assert topic == "loan2day-dead-letter"
        assert dlq_message.message_type == MessageType.DEAD_LETTER
        assert dlq_message.priority == MessagePriority.HIGH
        assert dlq_message.source_agent == "KafkaConsumerService"
        assert dlq_message.target_agent == "DeadLetterProcessor"
        
        # Verify original message is preserved in payload
        original_message = dlq_message.payload["original_message"]
        assert original_message["message_id"] == "dlq-test-001"
        assert dlq_message.payload["error_reason"] == "Max retries exceeded"
        assert dlq_message.payload["retry_count"] == 3


class TestKafkaService:
    """Test suite for main Kafka service orchestration."""
    
    @pytest.fixture
    def mock_kafka_config(self):
        """Mock Kafka configuration."""
        return {
            'bootstrap_servers': ['localhost:9092'],
            'group_id': 'loan2day-service-group',
            'dead_letter_topic': 'loan2day-dead-letter',
            'retry_backoff_ms': 100,
            'request_timeout_ms': 30000,
            'auto_offset_reset': 'earliest',
            'enable_auto_commit': True,
            'max_poll_records': 100,
            'session_timeout_ms': 30000,
            'heartbeat_interval_ms': 3000
        }
    
    @pytest.fixture
    def kafka_service_instance(self, mock_kafka_config):
        """Create KafkaService with mocked configuration."""
        with patch('app.services.kafka_service.get_kafka_config', return_value=mock_kafka_config):
            return KafkaService()
    
    @pytest.mark.asyncio
    async def test_service_start_success(self, kafka_service_instance):
        """Test successful Kafka service startup."""
        with patch.object(kafka_service_instance, 'producer') as mock_producer:
            mock_producer.start = AsyncMock()
            
            # Test service start
            await kafka_service_instance.start()
            
            # Verify producer was started
            mock_producer.start.assert_called_once()
            assert kafka_service_instance._is_running is True
    
    @pytest.mark.asyncio
    async def test_service_stop(self, kafka_service_instance):
        """Test Kafka service shutdown."""
        with patch.object(kafka_service_instance, 'producer') as mock_producer:
            mock_producer.start = AsyncMock()
            mock_producer.stop = AsyncMock()
            
            # Start and then stop service
            await kafka_service_instance.start()
            await kafka_service_instance.stop()
            
            # Verify producer was stopped
            mock_producer.stop.assert_called_once()
            assert kafka_service_instance._is_running is False
    
    @pytest.mark.asyncio
    async def test_add_consumer_success(self, kafka_service_instance):
        """Test adding a consumer to the service."""
        with patch.object(kafka_service_instance, 'producer') as mock_producer, \
             patch('app.services.kafka_service.KafkaConsumerService') as mock_consumer_class:
            
            # Mock producer
            mock_producer.start = AsyncMock()
            
            # Mock consumer
            mock_consumer = AsyncMock()
            mock_consumer.start = AsyncMock()
            mock_consumer_class.return_value = mock_consumer
            
            # Start service
            await kafka_service_instance.start()
            
            # Mock message handler
            mock_handler = AsyncMock(return_value=True)
            
            # Test adding consumer
            await kafka_service_instance.add_consumer(
                name="test_consumer",
                topics=["test-topic"],
                message_handler=mock_handler
            )
            
            # Verify consumer was created and started
            mock_consumer_class.assert_called_once_with(["test-topic"], mock_handler)
            mock_consumer.start.assert_called_once()
            assert "test_consumer" in kafka_service_instance.consumers
            assert kafka_service_instance.consumers["test_consumer"] == mock_consumer
    
    @pytest.mark.asyncio
    async def test_add_duplicate_consumer_failure(self, kafka_service_instance):
        """Test adding a consumer with duplicate name fails."""
        with patch.object(kafka_service_instance, 'producer') as mock_producer, \
             patch('app.services.kafka_service.KafkaConsumerService') as mock_consumer_class:
            
            # Mock producer and consumer
            mock_producer.start = AsyncMock()
            
            mock_consumer = AsyncMock()
            mock_consumer.start = AsyncMock()
            mock_consumer_class.return_value = mock_consumer
            
            # Start service and add first consumer
            await kafka_service_instance.start()
            
            mock_handler = AsyncMock(return_value=True)
            await kafka_service_instance.add_consumer(
                name="duplicate_consumer",
                topics=["topic1"],
                message_handler=mock_handler
            )
            
            # Test adding duplicate consumer
            with pytest.raises(ValueError, match="Consumer duplicate_consumer already exists"):
                await kafka_service_instance.add_consumer(
                    name="duplicate_consumer",
                    topics=["topic2"],
                    message_handler=mock_handler
                )
    
    @pytest.mark.asyncio
    async def test_remove_consumer_success(self, kafka_service_instance):
        """Test removing a consumer from the service."""
        with patch.object(kafka_service_instance, 'producer') as mock_producer, \
             patch('app.services.kafka_service.KafkaConsumerService') as mock_consumer_class:
            
            # Mock producer and consumer
            mock_producer.start = AsyncMock()
            
            mock_consumer = AsyncMock()
            mock_consumer.start = AsyncMock()
            mock_consumer.stop = AsyncMock()
            mock_consumer_class.return_value = mock_consumer
            
            # Start service and add consumer
            await kafka_service_instance.start()
            
            mock_handler = AsyncMock(return_value=True)
            await kafka_service_instance.add_consumer(
                name="removable_consumer",
                topics=["test-topic"],
                message_handler=mock_handler
            )
            
            # Test removing consumer
            await kafka_service_instance.remove_consumer("removable_consumer")
            
            # Verify consumer was stopped and removed
            mock_consumer.stop.assert_called_once()
            assert "removable_consumer" not in kafka_service_instance.consumers
    
    @pytest.mark.asyncio
    async def test_remove_nonexistent_consumer_failure(self, kafka_service_instance):
        """Test removing a non-existent consumer fails."""
        with patch.object(kafka_service_instance, 'producer') as mock_producer:
            mock_producer.start = AsyncMock()
            
            # Start service
            await kafka_service_instance.start()
            
            # Test removing non-existent consumer
            with pytest.raises(ValueError, match="Consumer nonexistent_consumer not found"):
                await kafka_service_instance.remove_consumer("nonexistent_consumer")
    
    @pytest.mark.asyncio
    async def test_get_producer_context_manager(self, kafka_service_instance):
        """Test producer context manager functionality."""
        with patch.object(kafka_service_instance, 'producer') as mock_producer:
            mock_producer.start = AsyncMock()
            
            # Start service
            await kafka_service_instance.start()
            
            # Test context manager
            async with kafka_service_instance.get_producer() as producer:
                assert producer == mock_producer
    
    @pytest.mark.asyncio
    async def test_get_producer_not_running_failure(self, kafka_service_instance):
        """Test producer context manager when service is not running."""
        # Test context manager when service not running
        with pytest.raises(RuntimeError, match="Kafka service not running"):
            async with kafka_service_instance.get_producer():
                pass