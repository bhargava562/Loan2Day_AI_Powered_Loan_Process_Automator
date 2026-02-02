#!/usr/bin/env python3
"""
Kafka Integration Demo for Loan2Day Platform.

This script demonstrates the Kafka integration functionality including:
- Inter-agent communication
- Audit logging
- Security event logging
- State change broadcasting
- Dead letter queue handling

Note: This demo uses mocks since no real Kafka server is running.
"""

import asyncio
import logging
from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

from app.services.kafka_integration import AgentKafkaIntegration, MessagePriority
from app.services.kafka_service import KafkaService, KafkaMessage, MessageType
from app.models.pydantic_models import AgentState, SentimentScore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def create_mock_kafka_service():
    """Create a mock Kafka service for demonstration."""
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


async def demo_agent_communication(integration: AgentKafkaIntegration):
    """Demonstrate inter-agent communication."""
    print("\n🔄 Demo: Inter-Agent Communication")
    print("=" * 50)
    
    # Master Agent sends task to Sales Agent
    result = await integration.send_agent_message(
        from_agent="master",
        to_agent="sales",
        message_type="task_request",
        payload={
            "action": "analyze_sentiment",
            "user_input": "I really need this loan for my business",
            "context": "loan_application"
        },
        session_id="demo-session-001",
        user_id="user-12345",
        priority=MessagePriority.HIGH
    )
    
    print(f"✅ Master → Sales communication: {'Success' if result else 'Failed'}")
    
    # Sales Agent responds to Master Agent
    result = await integration.send_agent_message(
        from_agent="sales",
        to_agent="master",
        message_type="task_response",
        payload={
            "sentiment_analysis": {
                "polarity": 0.7,
                "subjectivity": 0.8,
                "emotion": "hopeful",
                "confidence": 0.9
            },
            "recommended_action": "proceed_with_empathy"
        },
        session_id="demo-session-001",
        user_id="user-12345",
        priority=MessagePriority.NORMAL
    )
    
    print(f"✅ Sales → Master response: {'Success' if result else 'Failed'}")


async def demo_audit_logging(integration: AgentKafkaIntegration):
    """Demonstrate audit logging for compliance."""
    print("\n📋 Demo: Audit Logging")
    print("=" * 50)
    
    # Log KYC document processing
    result = await integration.log_agent_action(
        agent_name="verification",
        action="kyc_document_processed",
        details={
            "document_type": "pan_card",
            "file_size_bytes": 2048576,
            "ocr_confidence": Decimal("0.95"),
            "verification_status": "verified",
            "processing_time_ms": 1250
        },
        session_id="demo-session-001",
        user_id="user-12345"
    )
    
    print(f"✅ KYC processing audit: {'Logged' if result else 'Failed'}")
    
    # Log EMI calculation
    result = await integration.log_agent_action(
        agent_name="underwriting",
        action="emi_calculated",
        details={
            "loan_amount": Decimal("500000.00"),
            "interest_rate": Decimal("12.5"),
            "tenure_months": 36,
            "emi_amount": Decimal("16680.28"),
            "calculation_method": "reducing_balance"
        },
        session_id="demo-session-001",
        user_id="user-12345"
    )
    
    print(f"✅ EMI calculation audit: {'Logged' if result else 'Failed'}")


async def demo_security_events(integration: AgentKafkaIntegration):
    """Demonstrate security event logging."""
    print("\n🔒 Demo: Security Event Logging")
    print("=" * 50)
    
    # Log deepfake detection
    result = await integration.log_security_event(
        agent_name="verification",
        event_type="deepfake_detected",
        severity="HIGH",
        details={
            "file_path": "/uploads/suspicious_selfie.jpg",
            "sgs_score": Decimal("0.87"),
            "detection_confidence": Decimal("0.94"),
            "threat_indicators": ["facial_inconsistency", "metadata_anomaly"]
        },
        session_id="demo-session-001",
        user_id="user-12345"
    )
    
    print(f"🚨 Deepfake detection event: {'Logged' if result else 'Failed'}")
    
    # Log suspicious activity
    result = await integration.log_security_event(
        agent_name="master",
        event_type="suspicious_activity",
        severity="MEDIUM",
        details={
            "activity_type": "rapid_application_attempts",
            "attempt_count": 5,
            "time_window_minutes": 2,
            "ip_address": "192.168.1.100",
            "user_agent": "suspicious_bot_pattern"
        },
        session_id="demo-session-001",
        user_id="user-12345"
    )
    
    print(f"⚠️  Suspicious activity event: {'Logged' if result else 'Failed'}")


async def demo_state_broadcasting(integration: AgentKafkaIntegration):
    """Demonstrate agent state change broadcasting."""
    print("\n📡 Demo: State Change Broadcasting")
    print("=" * 50)
    
    # Create old and new states
    old_state = AgentState(
        session_id="demo-session-001",
        user_id="user-12345",
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
        session_id="demo-session-001",
        user_id="user-12345",
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
                polarity=0.7,
                subjectivity=0.8,
                emotion="hopeful",
                confidence=0.9
            )
        ],
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    
    # Broadcast state change
    result = await integration.broadcast_state_change(
        agent_name="verification",
        old_state=old_state,
        new_state=new_state,
        change_reason="kyc_verification_completed"
    )
    
    print(f"📢 State change broadcast: {'Success' if result else 'Failed'}")
    print(f"   Old Step: {old_state.current_step} → New Step: {new_state.current_step}")
    print(f"   KYC Status: {old_state.kyc_status} → {new_state.kyc_status}")


async def demo_message_handling(integration: AgentKafkaIntegration):
    """Demonstrate message handler registration and processing."""
    print("\n🎯 Demo: Message Handler Registration")
    print("=" * 50)
    
    # Create mock message handlers
    async def sales_handler(message: KafkaMessage) -> bool:
        print(f"📨 Sales Agent received: {message.payload.get('action', 'unknown')}")
        return True
    
    async def verification_handler(message: KafkaMessage) -> bool:
        print(f"📨 Verification Agent received: {message.payload.get('action', 'unknown')}")
        return True
    
    # Register handlers
    integration.register_message_handler("sales", sales_handler)
    integration.register_message_handler("verification", verification_handler)
    
    print("✅ Registered message handlers for Sales and Verification agents")
    
    # Simulate incoming message
    test_message = KafkaMessage(
        message_id="demo-msg-001",
        message_type=MessageType.AGENT_COMMUNICATION,
        priority=MessagePriority.NORMAL,
        source_agent="master",
        target_agent="sales",
        session_id="demo-session-001",
        user_id="user-12345",
        payload={"action": "process_loan_application", "urgency": "high"},
        timestamp=datetime.now()
    )
    
    # Process message
    result = await integration._handle_agent_communication(test_message)
    print(f"✅ Message processing: {'Success' if result else 'Failed'}")


async def main():
    """Run the Kafka integration demonstration."""
    print("🚀 Loan2Day Kafka Integration Demo")
    print("=" * 60)
    print("This demo shows Kafka integration capabilities using mocks")
    print("(no real Kafka server required)")
    
    # Create mock Kafka service and integration
    mock_kafka_service = await create_mock_kafka_service()
    integration = AgentKafkaIntegration(mock_kafka_service)
    
    try:
        # Initialize integration
        print("\n🔧 Initializing Kafka integration...")
        await integration.initialize()
        print("✅ Kafka integration initialized successfully")
        
        # Run demonstrations
        await demo_agent_communication(integration)
        await demo_audit_logging(integration)
        await demo_security_events(integration)
        await demo_state_broadcasting(integration)
        await demo_message_handling(integration)
        
        print("\n🎉 Demo completed successfully!")
        print("\nKey Features Demonstrated:")
        print("• Inter-agent communication with priority levels")
        print("• Audit logging for compliance tracking")
        print("• Security event logging with severity levels")
        print("• Agent state change broadcasting")
        print("• Message handler registration and processing")
        print("• Decimal precision for financial data (LQM Standard)")
        print("• Dead letter queue support (built-in)")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print("\n🧹 Shutting down integration...")
        await integration.shutdown()
        print("✅ Shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())