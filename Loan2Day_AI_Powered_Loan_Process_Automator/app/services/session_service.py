"""
Session Service - AgentState Management with Redis Persistence

This service handles session lifecycle management for the Master-Worker Agent
pattern, providing sub-millisecond AgentState retrieval through Redis caching
and proper session recovery capabilities.

Key Features:
- Redis-based session persistence with TTL management
- AgentState serialization/deserialization with Decimal support
- Session recovery and restoration capabilities
- Performance optimization for sub-millisecond retrieval
- Proper error handling and graceful degradation

Architecture: Service layer in Routes -> Services -> Repositories pattern
Performance: Sub-millisecond AgentState retrieval from Redis
Persistence: Session data with configurable TTL

Author: Lead AI Architect, Loan2Day Fintech Platform
"""

from typing import Dict, List, Optional, Any, Union
from decimal import Decimal
from datetime import datetime, timedelta
import logging
import json
import uuid

# Redis and async support
import redis.asyncio as redis
from redis.asyncio import Redis

# Import core modules
from app.models.pydantic_models import (
    AgentState, AgentStep, KYCStatus, UserProfile, 
    LoanRequest, EMICalculation, KYCDocument, SentimentScore
)
from app.core.config import settings

# Configure logger
logger = logging.getLogger(__name__)

class SessionServiceError(Exception):
    """Base exception for session service errors."""
    pass

class SessionNotFoundError(SessionServiceError):
    """Raised when session is not found."""
    pass

class SessionSerializationError(SessionServiceError):
    """Raised when session serialization/deserialization fails."""
    pass

class RedisConnectionError(SessionServiceError):
    """Raised when Redis connection fails."""
    pass

class SessionService:
    """
    Session Service for AgentState management with Redis persistence.
    
    This service provides comprehensive session lifecycle management
    with high-performance Redis caching and proper error handling.
    All monetary values maintain decimal.Decimal precision following
    the LQM Standard.
    """
    
    def __init__(self, redis_url: Optional[str] = None):
        """
        Initialize Session Service with Redis connection.
        
        Args:
            redis_url: Redis connection URL (optional, uses config default)
        """
        self.redis_url = redis_url or settings.redis_url
        self.redis_client: Optional[Redis] = None
        self.session_ttl = settings.redis_session_ttl_seconds
        
        logger.info("SessionService initialized")
    
    async def _get_redis_client(self) -> Redis:
        """
        Get Redis client with connection management.
        
        Returns:
            Redis: Connected Redis client
            
        Raises:
            RedisConnectionError: If Redis connection fails
        """
        if not self.redis_client:
            try:
                self.redis_client = redis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True
                )
                
                # Test connection
                await self.redis_client.ping()
                logger.info("Redis connection established successfully")
                
            except Exception as e:
                logger.error(f"Redis connection failed: {str(e)}")
                raise RedisConnectionError(f"Failed to connect to Redis: {str(e)}")
        
        return self.redis_client
    
    def _generate_session_key(self, session_id: str) -> str:
        """Generate Redis key for session storage."""
        return f"loan2day:session:{session_id}"
    
    def _serialize_agent_state(self, agent_state: AgentState) -> str:
        """
        Serialize AgentState to JSON string with Decimal support.
        
        Args:
            agent_state: AgentState object to serialize
            
        Returns:
            str: JSON string representation
            
        Raises:
            SessionSerializationError: If serialization fails
        """
        try:
            # Convert AgentState to dictionary with proper Decimal handling
            state_dict = {
                "session_id": agent_state.session_id,
                "user_id": agent_state.user_id,
                "current_step": agent_state.current_step.value,
                "kyc_status": agent_state.kyc_status.value,
                "fraud_score": agent_state.fraud_score,
                "trust_score": agent_state.trust_score,
                "created_at": agent_state.created_at.isoformat(),
                "updated_at": agent_state.updated_at.isoformat(),
                "conversation_context": agent_state.conversation_context,
                "plan_b_offers": agent_state.plan_b_offers
            }
            
            # Serialize loan_details with Decimal conversion
            state_dict["loan_details"] = {
                k: str(v) for k, v in agent_state.loan_details.items()
            }
            
            # Serialize complex objects
            if agent_state.user_profile:
                profile_dict = agent_state.user_profile.dict()
                # Convert Decimal fields to strings
                profile_dict["income_in_cents"] = str(profile_dict["income_in_cents"])
                state_dict["user_profile"] = profile_dict
            
            if agent_state.loan_request:
                request_dict = agent_state.loan_request.dict()
                # Convert Decimal fields to strings
                request_dict["amount_in_cents"] = str(request_dict["amount_in_cents"])
                if request_dict.get("requested_rate"):
                    request_dict["requested_rate"] = str(request_dict["requested_rate"])
                if request_dict.get("monthly_income_in_cents"):
                    request_dict["monthly_income_in_cents"] = str(request_dict["monthly_income_in_cents"])
                if request_dict.get("existing_emi_in_cents"):
                    request_dict["existing_emi_in_cents"] = str(request_dict["existing_emi_in_cents"])
                state_dict["loan_request"] = request_dict
            
            if agent_state.emi_calculation:
                emi_dict = agent_state.emi_calculation.dict()
                # Convert all Decimal fields to strings
                for field in ["principal_in_cents", "rate_per_annum", "emi_in_cents", 
                             "total_interest_in_cents", "total_amount_in_cents"]:
                    if field in emi_dict:
                        emi_dict[field] = str(emi_dict[field])
                state_dict["emi_calculation"] = emi_dict
            
            # Serialize KYC documents
            if agent_state.kyc_documents:
                state_dict["kyc_documents"] = [doc.dict() for doc in agent_state.kyc_documents]
            
            # Serialize sentiment history
            if agent_state.sentiment_history:
                sentiment_list = []
                for sentiment in agent_state.sentiment_history:
                    sentiment_dict = sentiment.dict()
                    sentiment_dict["timestamp"] = sentiment_dict["timestamp"].isoformat()
                    sentiment_list.append(sentiment_dict)
                state_dict["sentiment_history"] = sentiment_list
            
            # Convert to JSON string
            json_string = json.dumps(state_dict, ensure_ascii=False, separators=(',', ':'))
            
            logger.debug(f"AgentState serialized - Session: {agent_state.session_id}")
            return json_string
            
        except Exception as e:
            logger.error(f"AgentState serialization failed: {str(e)}")
            raise SessionSerializationError(f"Failed to serialize AgentState: {str(e)}")
    
    async def deserialize_agent_state(self, json_string: Union[str, Dict[str, Any]]) -> AgentState:
        """
        Deserialize AgentState from JSON string with Decimal restoration.
        
        Args:
            json_string: JSON string or dict representation
            
        Returns:
            AgentState: Reconstructed AgentState object
            
        Raises:
            SessionSerializationError: If deserialization fails
        """
        try:
            # Handle both string and dict inputs
            if isinstance(json_string, str):
                state_dict = json.loads(json_string)
            else:
                state_dict = json_string
            
            # Convert loan_details back to Decimal
            loan_details = {}
            for k, v in state_dict.get("loan_details", {}).items():
                loan_details[k] = Decimal(str(v))
            
            # Create base AgentState
            agent_state = AgentState(
                session_id=state_dict["session_id"],
                user_id=state_dict["user_id"],
                current_step=AgentStep(state_dict["current_step"]),
                loan_details=loan_details,
                kyc_status=KYCStatus(state_dict["kyc_status"]),
                fraud_score=state_dict["fraud_score"],
                trust_score=state_dict.get("trust_score"),
                created_at=datetime.fromisoformat(state_dict["created_at"]),
                updated_at=datetime.fromisoformat(state_dict["updated_at"]),
                conversation_context=state_dict.get("conversation_context", {}),
                plan_b_offers=state_dict.get("plan_b_offers", [])
            )
            
            # Deserialize user profile
            if state_dict.get("user_profile"):
                profile_dict = state_dict["user_profile"]
                # Convert Decimal fields back
                profile_dict["income_in_cents"] = Decimal(str(profile_dict["income_in_cents"]))
                agent_state.user_profile = UserProfile(**profile_dict)
            
            # Deserialize loan request
            if state_dict.get("loan_request"):
                request_dict = state_dict["loan_request"]
                # Convert Decimal fields back
                request_dict["amount_in_cents"] = Decimal(str(request_dict["amount_in_cents"]))
                if request_dict.get("requested_rate"):
                    request_dict["requested_rate"] = Decimal(str(request_dict["requested_rate"]))
                if request_dict.get("monthly_income_in_cents"):
                    request_dict["monthly_income_in_cents"] = Decimal(str(request_dict["monthly_income_in_cents"]))
                if request_dict.get("existing_emi_in_cents"):
                    request_dict["existing_emi_in_cents"] = Decimal(str(request_dict["existing_emi_in_cents"]))
                agent_state.loan_request = LoanRequest(**request_dict)
            
            # Deserialize EMI calculation
            if state_dict.get("emi_calculation"):
                emi_dict = state_dict["emi_calculation"]
                # Convert Decimal fields back
                for field in ["principal_in_cents", "rate_per_annum", "emi_in_cents", 
                             "total_interest_in_cents", "total_amount_in_cents"]:
                    if field in emi_dict:
                        emi_dict[field] = Decimal(str(emi_dict[field]))
                agent_state.emi_calculation = EMICalculation(**emi_dict)
            
            # Deserialize KYC documents
            if state_dict.get("kyc_documents"):
                kyc_docs = []
                for doc_dict in state_dict["kyc_documents"]:
                    kyc_docs.append(KYCDocument(**doc_dict))
                agent_state.kyc_documents = kyc_docs
            
            # Deserialize sentiment history
            if state_dict.get("sentiment_history"):
                sentiment_list = []
                for sentiment_dict in state_dict["sentiment_history"]:
                    sentiment_dict["timestamp"] = datetime.fromisoformat(sentiment_dict["timestamp"])
                    sentiment_list.append(SentimentScore(**sentiment_dict))
                agent_state.sentiment_history = sentiment_list
            
            logger.debug(f"AgentState deserialized - Session: {agent_state.session_id}")
            return agent_state
            
        except Exception as e:
            logger.error(f"AgentState deserialization failed: {str(e)}")
            raise SessionSerializationError(f"Failed to deserialize AgentState: {str(e)}")
    
    async def create_session(self, session_id: str, user_id: str) -> AgentState:
        """
        Create new session with initial AgentState.
        
        Args:
            session_id: Unique session identifier
            user_id: User identifier
            
        Returns:
            AgentState: New AgentState object
            
        Raises:
            SessionServiceError: If session creation fails
        """
        logger.info(f"Creating new session - Session: {session_id}, User: {user_id}")
        
        try:
            # Create initial AgentState
            agent_state = AgentState(
                session_id=session_id,
                user_id=user_id,
                current_step=AgentStep.GREETING,
                loan_details={},
                kyc_status=KYCStatus.PENDING,
                fraud_score=0.0,
                trust_score=None,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                conversation_context={},
                plan_b_offers=[]
            )
            
            # Store in Redis
            await self.store_session(agent_state)
            
            logger.info(f"Session created successfully - Session: {session_id}")
            return agent_state
            
        except Exception as e:
            logger.error(f"Session creation failed: {str(e)}")
            raise SessionServiceError(f"Failed to create session: {str(e)}")
    
    async def get_session(self, session_id: str, user_id: str) -> Optional[AgentState]:
        """
        Retrieve session from Redis with sub-millisecond performance.
        
        Args:
            session_id: Session identifier
            user_id: User identifier for authorization
            
        Returns:
            Optional[AgentState]: AgentState if found, None otherwise
            
        Raises:
            SessionServiceError: If retrieval fails
        """
        logger.debug(f"Retrieving session - Session: {session_id}, User: {user_id}")
        
        try:
            redis_client = await self._get_redis_client()
            session_key = self._generate_session_key(session_id)
            
            # Retrieve from Redis
            session_data = await redis_client.get(session_key)
            
            if not session_data:
                logger.debug(f"Session not found in Redis - Session: {session_id}")
                return None
            
            # Deserialize AgentState
            agent_state = await self.deserialize_agent_state(session_data)
            
            # Verify user authorization
            if agent_state.user_id != user_id:
                logger.warning(
                    f"Session access denied - Session: {session_id}, "
                    f"Expected User: {user_id}, Actual User: {agent_state.user_id}"
                )
                return None
            
            logger.debug(f"Session retrieved successfully - Session: {session_id}")
            return agent_state
            
        except (SessionSerializationError, RedisConnectionError):
            raise
        except Exception as e:
            logger.error(f"Session retrieval failed: {str(e)}")
            raise SessionServiceError(f"Failed to retrieve session: {str(e)}")
    
    async def store_session(self, agent_state: AgentState) -> bool:
        """
        Store session in Redis with TTL.
        
        Args:
            agent_state: AgentState to store
            
        Returns:
            bool: True if stored successfully
            
        Raises:
            SessionServiceError: If storage fails
        """
        logger.debug(f"Storing session - Session: {agent_state.session_id}")
        
        try:
            redis_client = await self._get_redis_client()
            session_key = self._generate_session_key(agent_state.session_id)
            
            # Update timestamp
            agent_state.updated_at = datetime.now()
            
            # Serialize AgentState
            session_data = self._serialize_agent_state(agent_state)
            
            # Store in Redis with TTL
            await redis_client.setex(
                session_key,
                self.session_ttl,
                session_data
            )
            
            logger.debug(f"Session stored successfully - Session: {agent_state.session_id}")
            return True
            
        except (SessionSerializationError, RedisConnectionError):
            raise
        except Exception as e:
            logger.error(f"Session storage failed: {str(e)}")
            raise SessionServiceError(f"Failed to store session: {str(e)}")
    
    async def update_session(self, agent_state: AgentState) -> bool:
        """
        Update existing session in Redis.
        
        Args:
            agent_state: Updated AgentState
            
        Returns:
            bool: True if updated successfully
            
        Raises:
            SessionServiceError: If update fails
        """
        logger.debug(f"Updating session - Session: {agent_state.session_id}")
        
        try:
            # Store updated session (same as store_session)
            return await self.store_session(agent_state)
            
        except Exception as e:
            logger.error(f"Session update failed: {str(e)}")
            raise SessionServiceError(f"Failed to update session: {str(e)}")
    
    async def delete_session(self, session_id: str, user_id: str) -> bool:
        """
        Delete session from Redis.
        
        Args:
            session_id: Session identifier
            user_id: User identifier for authorization
            
        Returns:
            bool: True if deleted successfully
            
        Raises:
            SessionServiceError: If deletion fails
        """
        logger.info(f"Deleting session - Session: {session_id}, User: {user_id}")
        
        try:
            # Verify session exists and user authorization
            agent_state = await self.get_session(session_id, user_id)
            if not agent_state:
                logger.warning(f"Session not found for deletion - Session: {session_id}")
                return False
            
            redis_client = await self._get_redis_client()
            session_key = self._generate_session_key(session_id)
            
            # Delete from Redis
            deleted_count = await redis_client.delete(session_key)
            
            success = deleted_count > 0
            if success:
                logger.info(f"Session deleted successfully - Session: {session_id}")
            else:
                logger.warning(f"Session deletion failed - Session: {session_id}")
            
            return success
            
        except (SessionServiceError, RedisConnectionError):
            raise
        except Exception as e:
            logger.error(f"Session deletion failed: {str(e)}")
            raise SessionServiceError(f"Failed to delete session: {str(e)}")
    
    async def extend_session_ttl(self, session_id: str, user_id: str) -> bool:
        """
        Extend session TTL to keep active sessions alive.
        
        Args:
            session_id: Session identifier
            user_id: User identifier for authorization
            
        Returns:
            bool: True if TTL extended successfully
            
        Raises:
            SessionServiceError: If TTL extension fails
        """
        logger.debug(f"Extending session TTL - Session: {session_id}")
        
        try:
            # Verify session exists
            agent_state = await self.get_session(session_id, user_id)
            if not agent_state:
                return False
            
            redis_client = await self._get_redis_client()
            session_key = self._generate_session_key(session_id)
            
            # Extend TTL
            await redis_client.expire(session_key, self.session_ttl)
            
            logger.debug(f"Session TTL extended - Session: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Session TTL extension failed: {str(e)}")
            raise SessionServiceError(f"Failed to extend session TTL: {str(e)}")
    
    async def get_user_sessions(self, user_id: str) -> List[str]:
        """
        Get all active session IDs for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List[str]: List of active session IDs
            
        Raises:
            SessionServiceError: If retrieval fails
        """
        logger.debug(f"Getting user sessions - User: {user_id}")
        
        try:
            redis_client = await self._get_redis_client()
            
            # Scan for user sessions (this is not optimal for large scale)
            # In production, consider maintaining a separate user->sessions mapping
            pattern = f"loan2day:session:*"
            session_keys = []
            
            async for key in redis_client.scan_iter(match=pattern):
                session_keys.append(key)
            
            # Filter sessions by user_id
            user_sessions = []
            for key in session_keys:
                try:
                    session_data = await redis_client.get(key)
                    if session_data:
                        state_dict = json.loads(session_data)
                        if state_dict.get("user_id") == user_id:
                            session_id = state_dict.get("session_id")
                            if session_id:
                                user_sessions.append(session_id)
                except Exception:
                    continue  # Skip invalid sessions
            
            logger.debug(f"Found {len(user_sessions)} sessions for user: {user_id}")
            return user_sessions
            
        except Exception as e:
            logger.error(f"Failed to get user sessions: {str(e)}")
            raise SessionServiceError(f"Failed to get user sessions: {str(e)}")
    
    async def cleanup_expired_sessions(self) -> int:
        """
        Cleanup expired sessions (Redis handles this automatically with TTL).
        
        Returns:
            int: Number of sessions cleaned up
        """
        logger.info("Session cleanup requested (Redis TTL handles automatic cleanup)")
        
        # Redis automatically handles TTL expiration
        # This method is provided for compatibility and manual cleanup if needed
        
        try:
            redis_client = await self._get_redis_client()
            
            # Get Redis info about expired keys (if available)
            info = await redis_client.info("keyspace")
            
            logger.info("Session cleanup completed (automatic via Redis TTL)")
            return 0  # Redis handles cleanup automatically
            
        except Exception as e:
            logger.error(f"Session cleanup failed: {str(e)}")
            return 0
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on session service.
        
        Returns:
            Dict[str, Any]: Health check results
        """
        try:
            redis_client = await self._get_redis_client()
            
            # Test Redis connection
            start_time = datetime.now()
            await redis_client.ping()
            end_time = datetime.now()
            
            ping_time_ms = (end_time - start_time).total_seconds() * 1000
            
            # Get Redis info
            info = await redis_client.info()
            
            return {
                "status": "healthy",
                "redis_connected": True,
                "ping_time_ms": ping_time_ms,
                "redis_version": info.get("redis_version"),
                "used_memory": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "session_ttl_seconds": self.session_ttl,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Session service health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "redis_connected": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

# Export main classes
__all__ = [
    'SessionService',
    'SessionServiceError',
    'SessionNotFoundError',
    'SessionSerializationError',
    'RedisConnectionError'
]