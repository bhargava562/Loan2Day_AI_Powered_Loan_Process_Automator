"""
Agent Messaging Utilities for Loan2Day Platform.

This module provides high-level utilities for agent communication through Kafka,
following the Master-Worker pattern and LQM Standard.

Architecture: Simplified interface for agent-to-agent communication
Security: All messages validated and logged for audit compliance
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from decimal import Decimal
from contextlib import asynccontextmanager

from app.services.kafka_integration import agent_kafka_integration, MessagePriority
from app.models.pydantic_models import AgentState

# Configure logger
logger = logging.getLogger(__name__)


class AgentMessenger:
    """
    High-level messaging interface for agents.
    
    Provides simplified methods for common agent communication patterns.
    """
    
    def __init__(self, agent_name: str):
        """
        Initialize messenger for a specific agent.
        
        Args:
            agent_name: Name of the agent using this messenger
        """
        self.agent_name = agent_name
        self._is_initialized = False
    
    async def initialize(self) -> None:
        """Initialize the messenger and register with Kafka integration."""
        if not self._is_initialized:
            await agent_kafka_integration.initialize()
            agent_kafka_integration.register_message_handler(
                self.agent_name, 
                self._handle_incoming_message
            )
            self._is_initialized = True
            logger.info(f"Agent messenger initialized for: {self.agent_name}")
    
    async def shutdown(self) -> None:
        """Shutdown the messenger."""
        if self._is_initialized:
            self._is_initialized = False
            logger.info(f"Agent messenger shutdown for: {self.agent_name}")
    
    async def send_to_master(
        self,
        message_type: str,
        payload: Dict[str, Any],
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        priority: MessagePriority = MessagePriority.NORMAL
    ) -> bool:
        """
        Send a message to the Master Agent.
        
        Args:
            message_type: Type of message (e.g., 'task_complete', 'error_report')
            payload: Message payload data
            session_id: Optional session ID for context
            user_id: Optional user ID for context
            priority: Message priority level
            
        Returns:
            bool: True if message sent successfully
        """
        return await agent_kafka_integration.send_agent_message(
            from_agent=self.agent_name,
            to_agent="master",
            message_type=message_type,
            payload=payload,
            session_id=session_id,
            user_id=user_id,
            priority=priority
        )
    
    async def send_to_worker(
        self,
        target_agent: str,
        message_type: str,
        payload: Dict[str, Any],
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        priority: MessagePriority = MessagePriority.NORMAL
    ) -> bool:
        """
        Send a message to a Worker Agent.
        
        Args:
            target_agent: Name of the target worker agent
            message_type: Type of message
            payload: Message payload data
            session_id: Optional session ID for context
            user_id: Optional user ID for context
            priority: Message priority level
            
        Returns:
            bool: True if message sent successfully
        """
        return await agent_kafka_integration.send_agent_message(
            from_agent=self.agent_name,
            to_agent=target_agent,
            message_type=message_type,
            payload=payload,
            session_id=session_id,
            user_id=user_id,
            priority=priority
        )
    
    async def broadcast_to_all(
        self,
        message_type: str,
        payload: Dict[str, Any],
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        priority: MessagePriority = MessagePriority.HIGH
    ) -> bool:
        """
        Broadcast a message to all agents.
        
        Args:
            message_type: Type of message
            payload: Message payload data
            session_id: Optional session ID for context
            user_id: Optional user ID for context
            priority: Message priority level
            
        Returns:
            bool: True if message sent successfully
        """
        return await agent_kafka_integration.send_agent_message(
            from_agent=self.agent_name,
            to_agent="ALL_AGENTS",
            message_type=message_type,
            payload=payload,
            session_id=session_id,
            user_id=user_id,
            priority=priority
        )
    
    async def log_action(
        self,
        action: str,
        details: Dict[str, Any],
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> bool:
        """
        Log an agent action for audit compliance.
        
        Args:
            action: Action being performed
            details: Action details and context
            session_id: Optional session ID
            user_id: Optional user ID
            
        Returns:
            bool: True if log sent successfully
        """
        return await agent_kafka_integration.log_agent_action(
            agent_name=self.agent_name,
            action=action,
            details=details,
            session_id=session_id,
            user_id=user_id
        )
    
    async def log_security_event(
        self,
        event_type: str,
        severity: str,
        details: Dict[str, Any],
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> bool:
        """
        Log a security event for SGS monitoring.
        
        Args:
            event_type: Type of security event
            severity: Event severity (LOW, MEDIUM, HIGH, CRITICAL)
            details: Event details and context
            session_id: Optional session ID
            user_id: Optional user ID
            
        Returns:
            bool: True if event logged successfully
        """
        return await agent_kafka_integration.log_security_event(
            agent_name=self.agent_name,
            event_type=event_type,
            severity=severity,
            details=details,
            session_id=session_id,
            user_id=user_id
        )
    
    async def notify_state_change(
        self,
        old_state: Optional[AgentState],
        new_state: AgentState,
        change_reason: str
    ) -> bool:
        """
        Notify other agents of state changes.
        
        Args:
            old_state: Previous agent state (None for initial state)
            new_state: New agent state
            change_reason: Reason for the state change
            
        Returns:
            bool: True if notification sent successfully
        """
        return await agent_kafka_integration.broadcast_state_change(
            agent_name=self.agent_name,
            old_state=old_state,
            new_state=new_state,
            change_reason=change_reason
        )
    
    async def _handle_incoming_message(self, message) -> bool:
        """
        Handle incoming messages for this agent.
        
        This is a default handler that can be overridden by specific agents.
        
        Args:
            message: Incoming Kafka message
        