"""
Dependency Injection Container for Loan2Day Platform

This module provides centralized dependency injection for all services,
agents, and components following the LQM Standard for proper architecture.
It ensures proper initialization order, singleton patterns where appropriate,
and clean separation of concerns.

Key Features:
- Centralized service initialization
- Proper dependency management
- Singleton patterns for stateful services
- Configuration-based initialization
- Health check integration

Architecture: Dependency Injection Container pattern
Performance: Lazy initialization with caching
Security: Proper secret management integration

Author: Lead AI Architect, Loan2Day Fintech Platform
"""

from typing import Dict, Any, Optional
import logging
from functools import lru_cache
from contextlib import asynccontextmanager

# Core services
from app.services.session_service import SessionService
from app.services.pdf_service import PDFService
from app.services.kafka_integration import agent_kafka_integration

# Agents
from app.agents.master import MasterAgent
from app.agents.sales import SalesAgent
from app.agents.verification import VerificationAgent
from app.agents.underwriting import UnderwritingAgent

# Core modules
from app.core.config import settings
from app.core.error_handling import error_logger, agent_failure_handler

# Configure logger
logger = logging.getLogger(__name__)

class DependencyContainer:
    """
    Centralized dependency injection container.
    
    This container manages the lifecycle of all services and agents,
    ensuring proper initialization order and dependency resolution.
    """
    
    def __init__(self):
        """Initialize dependency container."""
        self._services: Dict[str, Any] = {}
        self._agents: Dict[str, Any] = {}
        self._initialized = False
        
        logger.info("DependencyContainer initialized")
    
    async def initialize(self) -> None:
        """
        Initialize all services and agents in proper order.
        
        This method ensures that dependencies are initialized in the
        correct order and all connections are established.
        """
        if self._initialized:
            logger.info("DependencyContainer already initialized")
            return
        
        logger.info("🔄 Initializing DependencyContainer...")
        
        try:
            # Step 1: Initialize core services
            await self._initialize_core_services()
            
            # Step 2: Initialize agents
            await self._initialize_agents()
            
            # Step 3: Initialize integrations
            await self._initialize_integrations()
            
            # Step 4: Validate all dependencies
            await self._validate_dependencies()
            
            self._initialized = True
            logger.info("✅ DependencyContainer initialization complete")
            
        except Exception as e:
            logger.error(f"❌ DependencyContainer initialization failed: {str(e)}")
            raise
    
    async def _initialize_core_services(self) -> None:
        """Initialize core services."""
        logger.info("🔄 Initializing core services...")
        
        # Initialize Session Service
        session_service = SessionService()
        self._services["session_service"] = session_service
        logger.info("✅ SessionService initialized")
        
        # Initialize PDF Service
        pdf_service = PDFService()
        self._services["pdf_service"] = pdf_service
        logger.info("✅ PDFService initialized")
        
        # Test Redis connection for session service
        try:
            health_check = await session_service.health_check()
            if health_check["status"] != "healthy":
                logger.warning(f"SessionService health check warning: {health_check}")
        except Exception as e:
            logger.warning(f"SessionService health check failed: {str(e)}")
        
        logger.info("✅ Core services initialized")
    
    async def _initialize_agents(self) -> None:
        """Initialize all agents."""
        logger.info("🔄 Initializing agents...")
        
        # Initialize Worker Agents first
        sales_agent = SalesAgent()
        self._agents["sales_agent"] = sales_agent
        logger.info("✅ SalesAgent initialized")
        
        verification_agent = VerificationAgent()
        self._agents["verification_agent"] = verification_agent
        logger.info("✅ VerificationAgent initialized")
        
        underwriting_agent = UnderwritingAgent()
        self._agents["underwriting_agent"] = underwriting_agent
        logger.info("✅ UnderwritingAgent initialized")
        
        # Initialize Master Agent last (depends on Worker Agents)
        master_agent = MasterAgent()
        self._agents["master_agent"] = master_agent
        logger.info("✅ MasterAgent initialized")
        
        logger.info("✅ All agents initialized")
    
    async def _initialize_integrations(self) -> None:
        """Initialize external integrations."""
        logger.info("🔄 Initializing integrations...")
        
        # Initialize Kafka integration
        try:
            await agent_kafka_integration.initialize()
            self._services["kafka_integration"] = agent_kafka_integration
            logger.info("✅ Kafka integration initialized")
        except Exception as e:
            logger.warning(f"Kafka integration failed: {str(e)}")
            # Don't fail initialization if Kafka is not available
        
        logger.info("✅ Integrations initialized")
    
    async def _validate_dependencies(self) -> None:
        """Validate all dependencies are properly initialized."""
        logger.info("🔍 Validating dependencies...")
        
        # Validate services
        required_services = ["session_service", "pdf_service"]
        for service_name in required_services:
            if service_name not in self._services:
                raise RuntimeError(f"Required service not initialized: {service_name}")
        
        # Validate agents
        required_agents = ["sales_agent", "verification_agent", "underwriting_agent", "master_agent"]
        for agent_name in required_agents:
            if agent_name not in self._agents:
                raise RuntimeError(f"Required agent not initialized: {agent_name}")
        
        logger.info("✅ All dependencies validated")
    
    async def shutdown(self) -> None:
        """
        Shutdown all services and agents gracefully.
        """
        logger.info("🛑 Shutting down DependencyContainer...")
        
        try:
            # Shutdown Kafka integration
            if "kafka_integration" in self._services:
                await self._services["kafka_integration"].shutdown()
                logger.info("✅ Kafka integration shutdown")
            
            # Close Redis connections
            if "session_service" in self._services:
                session_service = self._services["session_service"]
                if hasattr(session_service, 'redis_client') and session_service.redis_client:
                    await session_service.redis_client.close()
                    logger.info("✅ Redis connection closed")
            
            # Clear containers
            self._services.clear()
            self._agents.clear()
            self._initialized = False
            
            logger.info("✅ DependencyContainer shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ DependencyContainer shutdown error: {str(e)}")
    
    # Service Getters
    
    def get_session_service(self) -> SessionService:
        """Get SessionService instance."""
        if not self._initialized:
            raise RuntimeError("DependencyContainer not initialized")
        return self._services["session_service"]
    
    def get_pdf_service(self) -> PDFService:
        """Get PDFService instance."""
        if not self._initialized:
            raise RuntimeError("DependencyContainer not initialized")
        return self._services["pdf_service"]
    
    def get_kafka_integration(self):
        """Get Kafka integration instance."""
        if not self._initialized:
            raise RuntimeError("DependencyContainer not initialized")
        return self._services.get("kafka_integration")
    
    # Agent Getters
    
    def get_master_agent(self) -> MasterAgent:
        """Get MasterAgent instance."""
        if not self._initialized:
            raise RuntimeError("DependencyContainer not initialized")
        return self._agents["master_agent"]
    
    def get_sales_agent(self) -> SalesAgent:
        """Get SalesAgent instance."""
        if not self._initialized:
            raise RuntimeError("DependencyContainer not initialized")
        return self._agents["sales_agent"]
    
    def get_verification_agent(self) -> VerificationAgent:
        """Get VerificationAgent instance."""
        if not self._initialized:
            raise RuntimeError("DependencyContainer not initialized")
        return self._agents["verification_agent"]
    
    def get_underwriting_agent(self) -> UnderwritingAgent:
        """Get UnderwritingAgent instance."""
        if not self._initialized:
            raise RuntimeError("DependencyContainer not initialized")
        return self._agents["underwriting_agent"]
    
    # Health Check
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check on all components.
        
        Returns:
            Dict[str, Any]: Health check results
        """
        if not self._initialized:
            return {
                "status": "unhealthy",
                "error": "DependencyContainer not initialized",
                "timestamp": "unknown"
            }
        
        health_results = {
            "status": "healthy",
            "services": {},
            "agents": {},
            "integrations": {}
        }
        
        # Check services
        try:
            session_health = await self._services["session_service"].health_check()
            health_results["services"]["session_service"] = session_health
        except Exception as e:
            health_results["services"]["session_service"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Check PDF service
        try:
            pdf_health = self._services["pdf_service"].health_check()
            health_results["services"]["pdf_service"] = pdf_health
        except Exception as e:
            health_results["services"]["pdf_service"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Check Kafka integration
        if "kafka_integration" in self._services:
            try:
                kafka_health = await self._services["kafka_integration"].health_check()
                health_results["integrations"]["kafka"] = kafka_health
            except Exception as e:
                health_results["integrations"]["kafka"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
        
        # Check agents (basic validation)
        for agent_name, agent in self._agents.items():
            try:
                # Basic agent validation
                health_results["agents"][agent_name] = {
                    "status": "healthy",
                    "initialized": True,
                    "type": type(agent).__name__
                }
            except Exception as e:
                health_results["agents"][agent_name] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
        
        # Determine overall status
        all_healthy = True
        for category in ["services", "agents", "integrations"]:
            for component, status in health_results[category].items():
                if status.get("status") != "healthy":
                    all_healthy = False
                    break
        
        health_results["status"] = "healthy" if all_healthy else "degraded"
        
        return health_results

# Global dependency container instance
dependency_container = DependencyContainer()

# FastAPI dependency functions

async def get_dependency_container() -> DependencyContainer:
    """FastAPI dependency to get the dependency container."""
    return dependency_container

async def get_session_service() -> SessionService:
    """FastAPI dependency to get SessionService."""
    return dependency_container.get_session_service()

async def get_pdf_service() -> PDFService:
    """FastAPI dependency to get PDFService."""
    return dependency_container.get_pdf_service()

async def get_master_agent() -> MasterAgent:
    """FastAPI dependency to get MasterAgent."""
    return dependency_container.get_master_agent()

async def get_sales_agent() -> SalesAgent:
    """FastAPI dependency to get SalesAgent."""
    return dependency_container.get_sales_agent()

async def get_verification_agent() -> VerificationAgent:
    """FastAPI dependency to get VerificationAgent."""
    return dependency_container.get_verification_agent()

async def get_underwriting_agent() -> UnderwritingAgent:
    """FastAPI dependency to get UnderwritingAgent."""
    return dependency_container.get_underwriting_agent()

# Cached dependency functions for performance

@lru_cache(maxsize=1)
def get_cached_settings():
    """Get cached settings instance."""
    return settings

# Lifespan context manager for FastAPI

@asynccontextmanager
async def lifespan_manager(app):
    """
    FastAPI lifespan context manager for dependency initialization.
    
    This replaces the lifespan function in main.py to use the
    dependency container for proper initialization.
    """
    logger.info("🚀 Starting Loan2Day platform with DependencyContainer...")
    
    try:
        # Initialize dependency container
        await dependency_container.initialize()
        
        logger.info("✅ Loan2Day platform ready to serve requests")
        yield
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize platform: {e}")
        raise
    finally:
        logger.info("🛑 Shutting down Loan2Day platform...")
        
        try:
            await dependency_container.shutdown()
            logger.info("✅ Loan2Day platform shutdown complete")
        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}")

# Export main components
__all__ = [
    'DependencyContainer',
    'dependency_container',
    'get_dependency_container',
    'get_session_service',
    'get_pdf_service',
    'get_master_agent',
    'get_sales_agent',
    'get_verification_agent',
    'get_underwriting_agent',
    'get_cached_settings',
    'lifespan_manager'
]