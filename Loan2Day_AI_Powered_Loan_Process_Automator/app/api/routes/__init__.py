"""
API routes for Loan2Day Agentic AI Fintech Platform.

This package contains all API route implementations following the
Routes -> Services -> Repositories pattern for proper separation
of concerns and maintainable architecture.

Routes:
- chat: Main conversation endpoint for Master Agent orchestration
- upload: KYC document upload with SGS security scanning
- voice: Twilio webhook for voice interface processing

Author: Lead AI Architect, Loan2Day Fintech Platform
"""

__all__ = ['chat', 'upload', 'voice']