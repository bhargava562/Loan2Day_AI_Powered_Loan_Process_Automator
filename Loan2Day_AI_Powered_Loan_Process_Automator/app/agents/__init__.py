"""
Agent modules for Loan2Day Agentic AI Fintech Platform.

This package contains all agent implementations following the Master-Worker
pattern with LangGraph orchestration.

Agents:
- MasterAgent: Central orchestrator using LangGraph state machine
- SalesAgent: Empathetic communication and Plan B logic
- VerificationAgent: KYC processing with SGS security scanning
- UnderwritingAgent: Risk assessment and EMI calculations using LQM

Author: Lead AI Architect, Loan2Day Fintech Platform
"""

__all__ = [
    'MasterAgent',
    'SalesAgent', 
    'VerificationAgent',
    'UnderwritingAgent'
]