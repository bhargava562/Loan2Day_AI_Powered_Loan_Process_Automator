"""
Kafka Integration for Loan2Day Agent Communication.

This module provides high-level integration between Kafka messaging and
the Master-Worker Agent architecture following the LQM Standard.

Architecture: Seamless integration with existing agent workflows
Security: All messages validated and logged for audit compliance
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from decimal import Decimal

from app.services.kafka_service import (
    KafkaService, 
    KafkaMessage, 
    MessageType, 
    MessagePriority,
    kafka_service
)
from app.models.pydantic_models import AgentState

# Configure logger
logger = logging.getLogger(__name__)


class AgentKafkaIntegration:
    """
    Integration layer between Kafka messaging and Agent architecture.
    
    Provides high-level methods for agent communication and state synchronization.
    """
    
    def __init__(self, kafka_service: KafkaService):
        """Initialize with Kafka service instance."""
        self.kafka = kafka_service
        self._message_handlers: Dict[str, Callable] = {}
        self._is_initialized = False
    
    async def initialize(self) -> None:
        """Initialize Kafka integration with default consumers."""
        if self._is_initialized:
            return
        
        try:
            # Start Kafka service
            await self.kafka.start()
            
            # Add consumer for agent communication
            await self.kafka.add_consumer(
                name="agent_communication",
                topics=["agent-communication"],
                message_handler=self._handle_agent_communication
            )
            
            # Add consumer for audit logs
            await self.kafka.add_consumer(
                name="audit_logs",
                topics=["audit-logs"],
                message_handler=self._handle_audit_log
            )
            
            # Add consumer for security events
            await self.kafka.add_consumer(
                name="security_events",
                topics=["security-events"],
                message_handler=self._handle_security_event
            )
            
            # Add consumer for dead letter queue
            await self.kafka.add_consumer(
                name="dead_letter_queue",
                topics=["loan2day-dead-letter"],
                message_handler=self._handle_dead_letter_message
            )
            
            self._is_initialized = True
            logger.info("Agent Kafka integration initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Kafka integration: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown Kafka integration."""
        if self._is_initialized:
            await self.kafka.stop()
            self._is_initialized = False
            logger.info("Agent Kafka integration shutdown completed")
    
    def register_message_handler(self, agent_name: str, handler: Callable) -> None:
        """
        Register a message handler for a specific agent.
        
        Args:
            agent_name: Name of the agent (e.g., 'master', 'sales', 'verification')
            handler: Async function to handle messages for this agent
        """
        self._message_handlers[agent_name] = handler
        logger.info(f"Registered message handler for agent: {agent_name}")
    
    async def send_agent_message(
        self,
        from_agent: str,
        to_agent: str,
        message_type: str,
        payload: Dict[str, Any],
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        priority: MessagePriority = MessagePriority.NORMAL
    ) -> bool:
        """
        Send a message between agents.
        
        Args:
            from_agent: Source agent name
            to_agent: Target agent name
            message_type: Type of message (e.g., 'state_update', 'task_request')
            payload: Message payload data
            session_id: Optional session ID for context
            user_id: Optional user ID for context
            priority: Message priority level
            
        Returns:
            bool: True if message sent successfully
        """
        async with self.kafka.get_producer() as producer:
            return await producer.send_agent_communication(
                source_agent=from_agent,
                target_agent=to_agent,
                payload={
                    'message_type': message_type,
                    'data': payload
                },
                session_id=session_id,
                user_id=user_id,
                priority=priority
            )
    
    async def log_agent_action(
        self,
        agent_name: str,
        action: str,
        details: Dict[str, Any],
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> bool:
        """
        Log an agent action for audit compliance.
        
        Args:
            agent_name: Name of the agent performing the action
            action: Action being performed
            details: Action details and context
            session_id: Optional session ID
            user_id: Optional user ID
            
        Returns:
            bool: True if log sent successfully
        """
        async with self.kafka.get_producer() as producer:
            return await producer.send_audit_log(
                source_agent=agent_name,
                action=action,
                details=details,
                session_id=session_id,
                user_id=user_id
            )
    
    async def log_security_event(
        self,
        agent_name: str,
        event_type: str,
        severity: str,
        details: Dict[str, Any],
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> bool:
        """
        Log a security event for SGS monitoring.
        
        Args:
            agent_name: Name of the agent detecting the event
            event_type: Type of security event
            severity: Event severity (LOW, MEDIUM, HIGH, CRITICAL)
            details: Event details and context
            session_id: Optional session ID
            user_id: Optional user ID
            
        Returns:
            bool: True if event logged successfully
        """
        async with self.kafka.get_producer() as producer:
            return await producer.send_security_event(
                source_agent=agent_name,
                event_type=event_type,
                severity=severity,
                details=details,
                session_id=session_id,
                user_id=user_id
            )
    
    async def broadcast_state_change(
        self,
        agent_name: str,
        old_state: AgentState,
        new_state: AgentState,
        change_reason: str
    ) -> bool:
        """
        Broadcast agent state changes to all interested agents.
        
        Args:
            agent_name: Name of the agent whose state changed
            old_state: Previous agent state
            new_state: New agent state
            change_reason: Reason for the state change
            
        Returns:
            bool: True if broadcast sent successfully
        """
        payload = {
            'agent_name': agent_name,
            'old_state': old_state.model_dump() if old_state else None,
            'new_state': new_state.model_dump(),
            'change_reason': change_reason,
            'timestamp': datetime.now().isoformat()
        }
        
        return await self.send_agent_message(
            from_agent=agent_name,
            to_agent="ALL_AGENTS",
            message_type="state_change",
            payload=payload,
            session_id=new_state.session_id,
            user_id=new_state.user_id,
            priority=MessagePriority.HIGH
        )
    
    async def _handle_agent_communication(self, message: KafkaMessage) -> bool:
        """
        Handle inter-agent communication messages.
        
        Args:
            message: Received Kafka message
            
        Returns:
            bool: True if message processed successfully
        """
        try:
            target_agent = message.target_agent
            
            # Check if we have a handler for the target agent
            if target_agent in self._message_handlers:
                handler = self._message_handlers[target_agent]
                return await handler(message)
            elif target_agent == "ALL_AGENTS":
                # Broadcast to all registered handlers
                success_count = 0
                for agent_name, handler in self._message_handlers.items():
                    try:
                        if await handler(message):
                            success_count += 1
                    except Exception as e:
                        logger.error(f"Error broadcasting to {agent_name}: {e}")
                
                # Consider successful if at least one handler processed it
                return success_count > 0
            else:
                logger.warning(f"No handler registered for agent: {target_agent}")
                return False
                
        except Exception as e:
            logger.error(f"Error handling agent communication: {e}")
            return False
    
    async def _handle_audit_log(self, message: KafkaMessage) -> bool:
        """
        Handle audit log messages for compliance tracking.
        
        Args:
            message: Received audit log message
            
        Returns:
            bool: True if message processed successfully
        """
        try:
            # Log audit message for compliance
            logger.info(
                f"AUDIT: Agent={message.source_agent}, "
                f"Action={message.payload.get('action')}, "
                f"Session={message.session_id}, "
                f"User={message.user_id}, "
                f"Details={message.payload.get('details')}"
            )
            
            # Here you could also store to database for compliance reporting
            # await self._store_audit_log(message)
            
            return True
            
        except Exception as e:
            logger.error(f"Error handling audit log: {e}")
            return False
    
    async def _handle_security_event(self, message: KafkaMessage) -> bool:
        """
        Handle security events for SGS monitoring.
        
        Args:
            message: Received security event message
            
        Returns:
            bool: True if message processed successfully
        """
        try:
            event_type = message.payload.get('event_type')
            severity = message.payload.get('severity')
            details = message.payload.get('details')
            
            # Log security event
            logger.warning(
                f"SECURITY EVENT: Type={event_type}, "
                f"Severity={severity}, "
                f"Agent={message.source_agent}, "
                f"Session={message.session_id}, "
                f"Details={details}"
            )
            
            # For critical events, trigger immediate alerts
            if severity == 'CRITICAL':
                await self._trigger_security_alert(message)
            
            return True
            
        except Exception as e:
            logger.error(f"Error handling security event: {e}")
            return False
    
    async def _handle_dead_letter_message(self, message: KafkaMessage) -> bool:
        """
        Handle messages from the dead letter queue for manual processing.
        
        Args:
            message: Dead letter queue message
            
        Returns:
            bool: True if message processed successfully
        """
        try:
            original_message = message.payload.get('original_message')
            error_reason = message.payload.get('error_reason')
            retry_count = message.payload.get('retry_count')
            
            # Log dead letter message for manual investigation
            logger.error(
                f"DEAD LETTER: OriginalID={original_message.get('message_id')}, "
                f"Reason={error_reason}, "
                f"Retries={retry_count}, "
                f"RequiresManualProcessing=True"
            )
            
            # Here you could implement manual processing logic
            # or store for administrator review
            
            return True
            
        except Exception as e:
            logger.error(f"Error handling dead letter message: {e}")
            return False
    
    async def _trigger_security_alert(self, message: KafkaMessage) -> None:
        """
        Trigger immediate security alert for critical events.
        
        Args:
            message: Critical security event message
        """
        try:
            # This could integrate with external alerting systems
            # For now, we'll log with high priority
            logger.critical(
                f"CRITICAL SECURITY ALERT: {message.payload.get('event_type')} "
                f"detected by {message.source_agent}. "
                f"Immediate attention required. Details: {message.payload.get('details')}"
            )
            
            # Could also send notifications via email, SMS, etc.
            
        except Exception as e:
            logger.error(f"Error triggering security alert: {e}")


# Global integration instance
agent_kafka_integration = AgentKafkaIntegration(kafka_service)