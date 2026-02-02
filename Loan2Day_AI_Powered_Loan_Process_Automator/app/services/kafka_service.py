"""
Kafka Service for Loan2Day Platform.

This module provides async Kafka producer and consumer functionality for
inter-agent communication and high-volume logging following the LQM Standard.

Architecture: Master-Worker Agent pattern with async message streaming
Security: All configuration through SecretManager, no hardcoded values
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, List, Callable, Union, Awaitable
from datetime import datetime
from decimal import Decimal
from dataclasses import dataclass, asdict
from enum import Enum
from contextlib import asynccontextmanager

import aiokafka
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
from aiokafka.errors import KafkaError, KafkaTimeoutError

from app.core.config import get_kafka_config

# Configure logger
logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Message types for Kafka topics following the Master-Worker pattern."""
    
    AGENT_COMMUNICATION = "agent_communication"
    AUDIT_LOG = "audit_log"
    ERROR_LOG = "error_log"
    PERFORMANCE_METRIC = "performance_metric"
    SECURITY_EVENT = "security_event"
    STATE_CHANGE = "state_change"
    DEAD_LETTER = "dead_letter"


class MessagePriority(Enum):
    """Message priority levels for processing order."""
    
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class KafkaMessage:
    """
    Standardized Kafka message structure following LQM Standard.
    
    All monetary values use Decimal type for zero-hallucination mathematics.
    """
    
    message_id: str
    message_type: MessageType
    priority: MessagePriority
    source_agent: str
    target_agent: Optional[str]
    session_id: Optional[str]
    user_id: Optional[str]
    payload: Dict[str, Any]
    timestamp: datetime
    retry_count: int = 0
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for JSON serialization."""
        return {
            'message_id': self.message_id,
            'message_type': self.message_type.value,
            'priority': self.priority.value,
            'source_agent': self.source_agent,
            'target_agent': self.target_agent,
            'session_id': self.session_id,
            'user_id': self.user_id,
            'payload': self._serialize_payload(self.payload),
            'timestamp': self.timestamp.isoformat(),
            'retry_count': self.retry_count,
            'max_retries': self.max_retries
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KafkaMessage':
        """Create message from dictionary."""
        return cls(
            message_id=data['message_id'],
            message_type=MessageType(data['message_type']),
            priority=MessagePriority(data['priority']),
            source_agent=data['source_agent'],
            target_agent=data.get('target_agent'),
            session_id=data.get('session_id'),
            user_id=data.get('user_id'),
            payload=cls._deserialize_payload(data['payload']),
            timestamp=datetime.fromisoformat(data['timestamp']),
            retry_count=data.get('retry_count', 0),
            max_retries=data.get('max_retries', 3)
        )
    
    def _serialize_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize payload with Decimal handling for LQM Standard."""
        serialized = {}
        for key, value in payload.items():
            if isinstance(value, Decimal):
                serialized[key] = {'__decimal__': str(value)}
            elif isinstance(value, datetime):
                serialized[key] = {'__datetime__': value.isoformat()}
            elif isinstance(value, dict):
                serialized[key] = self._serialize_payload(value)
            else:
                serialized[key] = value
        return serialized
    
    @classmethod
    def _deserialize_payload(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize payload with Decimal handling for LQM Standard."""
        deserialized = {}
        for key, value in payload.items():
            if isinstance(value, dict):
                if '__decimal__' in value:
                    deserialized[key] = Decimal(value['__decimal__'])
                elif '__datetime__' in value:
                    deserialized[key] = datetime.fromisoformat(value['__datetime__'])
                else:
                    deserialized[key] = cls._deserialize_payload(value)
            else:
                deserialized[key] = value
        return deserialized


class KafkaProducerService:
    """
    Async Kafka producer for inter-agent communication and logging.
    
    Follows the LQM Standard with strict typing and error handling.
    """
    
    def __init__(self):
        """Initialize Kafka producer with configuration."""
        self.config = get_kafka_config()
        self.producer: Optional[AIOKafkaProducer] = None
        self._is_running = False
        
    async def start(self) -> None:
        """Start the Kafka producer."""
        try:
            self.producer = AIOKafkaProducer(
                bootstrap_servers=self.config['bootstrap_servers'],
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                retry_backoff_ms=self.config['retry_backoff_ms'],
                request_timeout_ms=self.config['request_timeout_ms'],
                acks='all'  # Wait for all replicas to acknowledge
            )
            
            await self.producer.start()
            self._is_running = True
            logger.info("Kafka producer started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start Kafka producer: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the Kafka producer."""
        if self.producer and self._is_running:
            try:
                await self.producer.stop()
                self._is_running = False
                logger.info("Kafka producer stopped successfully")
            except Exception as e:
                logger.error(f"Error stopping Kafka producer: {e}")
    
    async def send_message(
        self,
        topic: str,
        message: KafkaMessage,
        partition_key: Optional[str] = None
    ) -> bool:
        """
        Send a message to Kafka topic.
        
        Args:
            topic: Kafka topic name
            message: KafkaMessage to send
            partition_key: Optional partition key for message ordering
            
        Returns:
            bool: True if message sent successfully
        """
        if not self._is_running or not self.producer:
            logger.error("Kafka producer not running")
            return False
        
        try:
            # Use session_id as partition key for message ordering per session
            key = partition_key or message.session_id or message.message_id
            
            # Send message with metadata
            record_metadata = await self.producer.send_and_wait(
                topic=topic,
                value=message.to_dict(),
                key=key
            )
            
            logger.debug(
                f"Message sent to topic {topic}, partition {record_metadata.partition}, "
                f"offset {record_metadata.offset}"
            )
            return True
            
        except KafkaTimeoutError:
            logger.error(f"Timeout sending message to topic {topic}")
            return False
        except KafkaError as e:
            logger.error(f"Kafka error sending message to topic {topic}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending message to topic {topic}: {e}")
            return False
    
    async def send_agent_communication(
        self,
        source_agent: str,
        target_agent: str,
        payload: Dict[str, Any],
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        priority: MessagePriority = MessagePriority.NORMAL
    ) -> bool:
        """Send inter-agent communication message."""
        message = KafkaMessage(
            message_id=f"{source_agent}_{target_agent}_{datetime.now().timestamp()}",
            message_type=MessageType.AGENT_COMMUNICATION,
            priority=priority,
            source_agent=source_agent,
            target_agent=target_agent,
            session_id=session_id,
            user_id=user_id,
            payload=payload,
            timestamp=datetime.now()
        )
        
        return await self.send_message("agent-communication", message)
    
    async def send_audit_log(
        self,
        source_agent: str,
        action: str,
        details: Dict[str, Any],
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> bool:
        """Send audit log message for compliance tracking."""
        message = KafkaMessage(
            message_id=f"audit_{source_agent}_{datetime.now().timestamp()}",
            message_type=MessageType.AUDIT_LOG,
            priority=MessagePriority.HIGH,
            source_agent=source_agent,
            target_agent=None,
            session_id=session_id,
            user_id=user_id,
            payload={
                'action': action,
                'details': details,
                'compliance_required': True
            },
            timestamp=datetime.now()
        )
        
        return await self.send_message("audit-logs", message)
    
    async def send_security_event(
        self,
        source_agent: str,
        event_type: str,
        severity: str,
        details: Dict[str, Any],
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> bool:
        """Send security event for SGS monitoring."""
        message = KafkaMessage(
            message_id=f"security_{source_agent}_{datetime.now().timestamp()}",
            message_type=MessageType.SECURITY_EVENT,
            priority=MessagePriority.CRITICAL,
            source_agent=source_agent,
            target_agent="SGS_Monitor",
            session_id=session_id,
            user_id=user_id,
            payload={
                'event_type': event_type,
                'severity': severity,
                'details': details,
                'requires_immediate_attention': severity in ['HIGH', 'CRITICAL']
            },
            timestamp=datetime.now()
        )
        
        return await self.send_message("security-events", message)


class KafkaConsumerService:
    """
    Async Kafka consumer for processing messages with dead letter queue support.
    
    Follows the Master-Worker pattern for message processing.
    """
    
    def __init__(self, topics: List[str], message_handler: Callable[[KafkaMessage], Awaitable[bool]]):
        """
        Initialize Kafka consumer.
        
        Args:
            topics: List of topics to subscribe to
            message_handler: Async function to handle received messages
        """
        self.config = get_kafka_config()
        self.topics = topics
        self.message_handler = message_handler
        self.consumer: Optional[AIOKafkaConsumer] = None
        self.producer_service: Optional[KafkaProducerService] = None
        self._is_running = False
        self._consumer_task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start the Kafka consumer."""
        try:
            # Initialize producer for dead letter queue
            self.producer_service = KafkaProducerService()
            await self.producer_service.start()
            
            # Initialize consumer
            self.consumer = AIOKafkaConsumer(
                *self.topics,
                bootstrap_servers=self.config['bootstrap_servers'],
                group_id=self.config['group_id'],
                auto_offset_reset=self.config['auto_offset_reset'],
                enable_auto_commit=self.config['enable_auto_commit'],
                max_poll_records=self.config['max_poll_records'],
                session_timeout_ms=self.config['session_timeout_ms'],
                heartbeat_interval_ms=self.config['heartbeat_interval_ms'],
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                key_deserializer=lambda k: k.decode('utf-8') if k else None
            )
            
            await self.consumer.start()
            self._is_running = True
            
            # Start consumer loop
            self._consumer_task = asyncio.create_task(self._consume_messages())
            
            logger.info(f"Kafka consumer started for topics: {self.topics}")
            
        except Exception as e:
            logger.error(f"Failed to start Kafka consumer: {e}")
            await self.stop()
            raise
    
    async def stop(self) -> None:
        """Stop the Kafka consumer."""
        self._is_running = False
        
        if self._consumer_task:
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass
        
        if self.consumer:
            try:
                await self.consumer.stop()
                logger.info("Kafka consumer stopped successfully")
            except Exception as e:
                logger.error(f"Error stopping Kafka consumer: {e}")
        
        if self.producer_service:
            await self.producer_service.stop()
    
    async def _consume_messages(self) -> None:
        """Main consumer loop for processing messages."""
        while self._is_running:
            try:
                # Poll for messages
                msg_pack = await self.consumer.getmany(timeout_ms=1000)
                
                for topic_partition, messages in msg_pack.items():
                    for message in messages:
                        await self._process_message(message)
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in consumer loop: {e}")
                await asyncio.sleep(1)  # Brief pause before retrying
    
    async def _process_message(self, raw_message) -> None:
        """
        Process a single Kafka message with retry and dead letter queue support.
        
        Args:
            raw_message: Raw Kafka message from aiokafka
        """
        try:
            # Deserialize message
            kafka_message = KafkaMessage.from_dict(raw_message.value)
            
            logger.debug(f"Processing message {kafka_message.message_id} from {kafka_message.source_agent}")
            
            # Attempt to process message
            success = await self.message_handler(kafka_message)
            
            if not success:
                await self._handle_failed_message(kafka_message, "Handler returned False")
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            
            # Try to extract message for dead letter queue
            try:
                kafka_message = KafkaMessage.from_dict(raw_message.value)
                await self._handle_failed_message(kafka_message, str(e))
            except Exception as dlq_error:
                logger.error(f"Failed to send message to dead letter queue: {dlq_error}")
    
    async def _handle_failed_message(self, message: KafkaMessage, error_reason: str) -> None:
        """
        Handle failed message processing with retry logic and dead letter queue.
        
        Args:
            message: Failed KafkaMessage
            error_reason: Reason for failure
        """
        message.retry_count += 1
        
        if message.retry_count <= message.max_retries:
            # Retry processing
            logger.warning(
                f"Retrying message {message.message_id} "
                f"(attempt {message.retry_count}/{message.max_retries})"
            )
            
            # Add exponential backoff
            await asyncio.sleep(2 ** message.retry_count)
            
            # Retry processing
            try:
                success = await self.message_handler(message)
                if success:
                    logger.info(f"Message {message.message_id} processed successfully on retry")
                    return
            except Exception as retry_error:
                logger.error(f"Retry failed for message {message.message_id}: {retry_error}")
        
        # Send to dead letter queue
        await self._send_to_dead_letter_queue(message, error_reason)
    
    async def _send_to_dead_letter_queue(self, message: KafkaMessage, error_reason: str) -> None:
        """
        Send failed message to dead letter queue for manual processing.
        
        Args:
            message: Failed KafkaMessage
            error_reason: Reason for failure
        """
        if not self.producer_service:
            logger.error("Producer service not available for dead letter queue")
            return
        
        # Create dead letter message
        dlq_message = KafkaMessage(
            message_id=f"dlq_{message.message_id}",
            message_type=MessageType.DEAD_LETTER,
            priority=MessagePriority.HIGH,
            source_agent="KafkaConsumerService",
            target_agent="DeadLetterProcessor",
            session_id=message.session_id,
            user_id=message.user_id,
            payload={
                'original_message': message.to_dict(),
                'error_reason': error_reason,
                'failed_at': datetime.now().isoformat(),
                'retry_count': message.retry_count
            },
            timestamp=datetime.now()
        )
        
        success = await self.producer_service.send_message(
            self.config['dead_letter_topic'],
            dlq_message
        )
        
        if success:
            logger.warning(f"Message {message.message_id} sent to dead letter queue: {error_reason}")
        else:
            logger.error(f"Failed to send message {message.message_id} to dead letter queue")


class KafkaService:
    """
    Main Kafka service that manages both producer and consumer functionality.
    
    Provides high-level interface for inter-agent communication and logging.
    """
    
    def __init__(self):
        """Initialize Kafka service."""
        self.producer = KafkaProducerService()
        self.consumers: Dict[str, KafkaConsumerService] = {}
        self._is_running = False
    
    async def start(self) -> None:
        """Start Kafka service with producer and consumers."""
        try:
            await self.producer.start()
            self._is_running = True
            logger.info("Kafka service started successfully")
        except Exception as e:
            logger.error(f"Failed to start Kafka service: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop Kafka service and all consumers."""
        self._is_running = False
        
        # Stop all consumers
        for consumer_name, consumer in self.consumers.items():
            try:
                await consumer.stop()
                logger.info(f"Stopped consumer: {consumer_name}")
            except Exception as e:
                logger.error(f"Error stopping consumer {consumer_name}: {e}")
        
        # Stop producer
        await self.producer.stop()
        logger.info("Kafka service stopped successfully")
    
    async def add_consumer(
        self,
        name: str,
        topics: List[str],
        message_handler: Callable[[KafkaMessage], Awaitable[bool]]
    ) -> None:
        """
        Add a new consumer to the service.
        
        Args:
            name: Consumer name for identification
            topics: List of topics to subscribe to
            message_handler: Async function to handle messages
        """
        if name in self.consumers:
            raise ValueError(f"Consumer {name} already exists")
        
        consumer = KafkaConsumerService(topics, message_handler)
        await consumer.start()
        self.consumers[name] = consumer
        
        logger.info(f"Added consumer {name} for topics: {topics}")
    
    async def remove_consumer(self, name: str) -> None:
        """Remove and stop a consumer."""
        if name not in self.consumers:
            raise ValueError(f"Consumer {name} not found")
        
        await self.consumers[name].stop()
        del self.consumers[name]
        
        logger.info(f"Removed consumer: {name}")
    
    @asynccontextmanager
    async def get_producer(self):
        """Context manager for accessing the producer."""
        if not self._is_running:
            raise RuntimeError("Kafka service not running")
        yield self.producer


# Global Kafka service instance
kafka_service = KafkaService()