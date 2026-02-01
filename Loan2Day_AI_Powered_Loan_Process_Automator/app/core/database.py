"""
Database Configuration for Loan2Day Agentic AI Fintech Platform

This module implements async SQLAlchemy database configuration with
connection pooling, proper error handling, and migration support.
Follows the LQM Standard for zero-hallucination database operations.

Key Features:
- Async SQLAlchemy with asyncpg driver for PostgreSQL
- Connection pooling for optimal performance
- Proper transaction management and error handling
- Database migration support with Alembic
- Health check functionality for monitoring

Author: Lead AI Architect, Loan2Day Fintech Platform
"""

import os
from typing import AsyncGenerator, Optional
import logging
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import text, event
from sqlalchemy.pool import NullPool

from app.core.config import settings
from app.models.db_models import Base

# Configure logger
logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Database manager for async SQLAlchemy operations.
    
    Handles connection pooling, session management, and health checks
    for the Loan2Day platform database operations.
    """
    
    def __init__(self):
        """Initialize database manager."""
        self.engine = None
        self.async_session_maker = None
        self._initialized = False
        logger.info("DatabaseManager initialized")
    
    async def initialize(self) -> None:
        """
        Initialize database engine and session maker.
        
        Creates async engine with proper connection pooling and
        configures session maker for dependency injection.
        """
        if self._initialized:
            logger.warning("Database already initialized")
            return
        
        try:
            # Create async engine with connection pooling
            self.engine = create_async_engine(
                settings.database_url,
                echo=settings.debug,  # Log SQL queries in debug mode
                pool_size=settings.db_pool_size,
                max_overflow=settings.db_max_overflow,
                pool_timeout=settings.db_pool_timeout,
                pool_recycle=settings.db_pool_recycle,
                pool_pre_ping=True,  # Validate connections before use
                # Use NullPool for testing to avoid connection issues
                poolclass=NullPool if settings.environment == "test" else None
            )
            
            # Create async session maker
            self.async_session_maker = async_sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=True,
                autocommit=False
            )
            
            # Test database connection
            await self.health_check()
            
            self._initialized = True
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            raise
    
    async def create_tables(self) -> None:
        """
        Create all database tables.
        
        This method creates all tables defined in the SQLAlchemy models.
        Should only be used in development/testing environments.
        """
        if not self.engine:
            raise RuntimeError("Database not initialized")
        
        logger.info("Creating database tables")
        
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            logger.info("Database tables created successfully")
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to create database tables: {str(e)}")
            raise
    
    async def drop_tables(self) -> None:
        """
        Drop all database tables.
        
        WARNING: This will delete all data. Only use in testing environments.
        """
        if not self.engine:
            raise RuntimeError("Database not initialized")
        
        if settings.environment == "production":
            raise RuntimeError("Cannot drop tables in production environment")
        
        logger.warning("Dropping all database tables")
        
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
            
            logger.warning("All database tables dropped")
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to drop database tables: {str(e)}")
            raise
    
    async def health_check(self) -> bool:
        """
        Perform database health check.
        
        Returns:
            bool: True if database is healthy, False otherwise
        """
        if not self.engine:
            logger.error("Database not initialized for health check")
            return False
        
        try:
            async with self.engine.begin() as conn:
                result = await conn.execute(text("SELECT 1"))
                result.scalar()
            
            logger.debug("Database health check passed")
            return True
            
        except Exception as e:
            logger.error(f"Database health check failed: {str(e)}")
            return False
    
    async def get_session(self) -> AsyncSession:
        """
        Get database session.
        
        Returns:
            AsyncSession: Database session
            
        Raises:
            RuntimeError: If database not initialized
        """
        if not self.async_session_maker:
            raise RuntimeError("Database not initialized")
        
        return self.async_session_maker()
    
    @asynccontextmanager
    async def get_session_context(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get database session with automatic cleanup.
        
        Yields:
            AsyncSession: Database session with automatic cleanup
        """
        if not self.async_session_maker:
            raise RuntimeError("Database not initialized")
        
        async with self.async_session_maker() as session:
            try:
                yield session
            except Exception as e:
                await session.rollback()
                logger.error(f"Database session error: {str(e)}")
                raise
            finally:
                await session.close()
    
    async def close(self) -> None:
        """Close database connections and cleanup resources."""
        if self.engine:
            await self.engine.dispose()
            logger.info("Database connections closed")
        
        self._initialized = False

# Global database manager instance
db_manager = DatabaseManager()

async def get_database() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency injection function for FastAPI routes.
    
    Yields:
        AsyncSession: Database session for request handling
    """
    async with db_manager.get_session_context() as session:
        yield session

async def init_database() -> None:
    """
    Initialize database for application startup.
    
    This function should be called during application startup
    to initialize the database connection and create tables if needed.
    """
    logger.info("Initializing database for application startup")
    
    try:
        await db_manager.initialize()
        
        # Create tables in development/testing environments
        if settings.environment in ["development", "test"]:
            await db_manager.create_tables()
        
        logger.info("Database initialization completed")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        raise

async def close_database() -> None:
    """
    Close database connections for application shutdown.
    
    This function should be called during application shutdown
    to properly close database connections and cleanup resources.
    """
    logger.info("Closing database connections for application shutdown")
    
    try:
        await db_manager.close()
        logger.info("Database connections closed successfully")
        
    except Exception as e:
        logger.error(f"Error closing database connections: {str(e)}")
        raise

# Database event listeners for logging and monitoring
@event.listens_for(db_manager.engine, "connect", once=True)
def receive_connect(dbapi_connection, connection_record):
    """Log database connection events."""
    logger.info("Database connection established")

@event.listens_for(db_manager.engine, "checkout")
def receive_checkout(dbapi_connection, connection_record, connection_proxy):
    """Log connection checkout from pool."""
    logger.debug("Database connection checked out from pool")

@event.listens_for(db_manager.engine, "checkin")
def receive_checkin(dbapi_connection, connection_record):
    """Log connection checkin to pool."""
    logger.debug("Database connection checked in to pool")

# Export database functions and manager
__all__ = [
    'DatabaseManager',
    'db_manager',
    'get_database',
    'init_database',
    'close_database'
]