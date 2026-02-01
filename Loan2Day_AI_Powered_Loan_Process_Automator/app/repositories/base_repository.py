"""
Base Repository for Loan2Day Agentic AI Fintech Platform

This module provides the base repository class with common database operations
using async SQLAlchemy. All repositories inherit from this base class to ensure
consistent patterns and error handling across the data access layer.

Key Features:
- Async SQLAlchemy operations for optimal I/O performance
- Generic CRUD operations with type safety
- Proper error handling and logging
- Connection pooling and transaction management
- LQM Standard compliance for monetary operations

Author: Lead AI Architect, Loan2Day Fintech Platform
"""

from typing import TypeVar, Generic, List, Optional, Dict, Any, Type
from decimal import Decimal
from uuid import UUID
import logging

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy import select, update, delete, and_, or_, func
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from pydantic import BaseModel

from app.models.db_models import Base

# Configure logger
logger = logging.getLogger(__name__)

# Generic type for database models
ModelType = TypeVar("ModelType", bound=Base)
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)

class RepositoryError(Exception):
    """Base exception for repository operations."""
    pass

class NotFoundError(RepositoryError):
    """Raised when requested entity is not found."""
    pass

class DuplicateError(RepositoryError):
    """Raised when attempting to create duplicate entity."""
    pass

class ValidationError(RepositoryError):
    """Raised when data validation fails."""
    pass

class BaseRepository(Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    """
    Base repository class providing common database operations.
    
    This class implements the repository pattern with async SQLAlchemy,
    providing type-safe CRUD operations and proper error handling.
    All monetary operations follow the LQM Standard using Decimal types.
    """
    
    def __init__(self, model: Type[ModelType]):
        """
        Initialize repository with model class.
        
        Args:
            model: SQLAlchemy model class
        """
        self.model = model
        logger.info(f"Repository initialized for model: {model.__name__}")
    
    async def create(
        self, 
        db: AsyncSession, 
        *, 
        obj_in: CreateSchemaType
    ) -> ModelType:
        """
        Create a new entity in the database.
        
        Args:
            db: Database session
            obj_in: Pydantic model with creation data
            
        Returns:
            ModelType: Created database entity
            
        Raises:
            DuplicateError: If entity already exists
            ValidationError: If data validation fails
        """
        logger.info(f"Creating new {self.model.__name__}")
        
        try:
            # Convert Pydantic model to dict
            obj_data = obj_in.dict()
            
            # Validate monetary fields if present
            self._validate_monetary_fields(obj_data)
            
            # Create database object
            db_obj = self.model(**obj_data)
            
            # Add to session and commit
            db.add(db_obj)
            await db.commit()
            await db.refresh(db_obj)
            
            logger.info(f"Successfully created {self.model.__name__} with ID: {db_obj.id}")
            return db_obj
            
        except IntegrityError as e:
            await db.rollback()
            logger.error(f"Integrity error creating {self.model.__name__}: {str(e)}")
            raise DuplicateError(f"Entity already exists: {str(e)}")
        except SQLAlchemyError as e:
            await db.rollback()
            logger.error(f"Database error creating {self.model.__name__}: {str(e)}")
            raise RepositoryError(f"Failed to create entity: {str(e)}")
        except Exception as e:
            await db.rollback()
            logger.error(f"Unexpected error creating {self.model.__name__}: {str(e)}")
            raise RepositoryError(f"Unexpected error: {str(e)}")
    
    async def get(
        self, 
        db: AsyncSession, 
        id: UUID
    ) -> Optional[ModelType]:
        """
        Get entity by ID.
        
        Args:
            db: Database session
            id: Entity UUID
            
        Returns:
            Optional[ModelType]: Entity if found, None otherwise
        """
        logger.debug(f"Getting {self.model.__name__} by ID: {id}")
        
        try:
            result = await db.execute(
                select(self.model).where(self.model.id == id)
            )
            entity = result.scalar_one_or_none()
            
            if entity:
                logger.debug(f"Found {self.model.__name__} with ID: {id}")
            else:
                logger.debug(f"No {self.model.__name__} found with ID: {id}")
            
            return entity
            
        except SQLAlchemyError as e:
            logger.error(f"Database error getting {self.model.__name__}: {str(e)}")
            raise RepositoryError(f"Failed to get entity: {str(e)}")
    
    async def get_multi(
        self,
        db: AsyncSession,
        *,
        skip: int = 0,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None
    ) -> List[ModelType]:
        """
        Get multiple entities with filtering and pagination.
        
        Args:
            db: Database session
            skip: Number of records to skip
            limit: Maximum number of records to return
            filters: Dictionary of field filters
            order_by: Field name to order by
            
        Returns:
            List[ModelType]: List of entities
        """
        logger.debug(f"Getting multiple {self.model.__name__} entities")
        
        try:
            query = select(self.model)
            
            # Apply filters
            if filters:
                for field, value in filters.items():
                    if hasattr(self.model, field):
                        query = query.where(getattr(self.model, field) == value)
            
            # Apply ordering
            if order_by and hasattr(self.model, order_by):
                query = query.order_by(getattr(self.model, order_by))
            
            # Apply pagination
            query = query.offset(skip).limit(limit)
            
            result = await db.execute(query)
            entities = result.scalars().all()
            
            logger.debug(f"Retrieved {len(entities)} {self.model.__name__} entities")
            return entities
            
        except SQLAlchemyError as e:
            logger.error(f"Database error getting multiple {self.model.__name__}: {str(e)}")
            raise RepositoryError(f"Failed to get entities: {str(e)}")
    
    async def update(
        self,
        db: AsyncSession,
        *,
        db_obj: ModelType,
        obj_in: UpdateSchemaType
    ) -> ModelType:
        """
        Update existing entity.
        
        Args:
            db: Database session
            db_obj: Existing database entity
            obj_in: Pydantic model with update data
            
        Returns:
            ModelType: Updated database entity
            
        Raises:
            ValidationError: If data validation fails
        """
        logger.info(f"Updating {self.model.__name__} with ID: {db_obj.id}")
        
        try:
            # Convert Pydantic model to dict, excluding unset fields
            obj_data = obj_in.dict(exclude_unset=True)
            
            # Validate monetary fields if present
            self._validate_monetary_fields(obj_data)
            
            # Update entity fields
            for field, value in obj_data.items():
                if hasattr(db_obj, field):
                    setattr(db_obj, field, value)
            
            # Commit changes
            await db.commit()
            await db.refresh(db_obj)
            
            logger.info(f"Successfully updated {self.model.__name__} with ID: {db_obj.id}")
            return db_obj
            
        except SQLAlchemyError as e:
            await db.rollback()
            logger.error(f"Database error updating {self.model.__name__}: {str(e)}")
            raise RepositoryError(f"Failed to update entity: {str(e)}")
        except Exception as e:
            await db.rollback()
            logger.error(f"Unexpected error updating {self.model.__name__}: {str(e)}")
            raise RepositoryError(f"Unexpected error: {str(e)}")
    
    async def delete(
        self,
        db: AsyncSession,
        *,
        id: UUID
    ) -> bool:
        """
        Delete entity by ID.
        
        Args:
            db: Database session
            id: Entity UUID
            
        Returns:
            bool: True if deleted, False if not found
        """
        logger.info(f"Deleting {self.model.__name__} with ID: {id}")
        
        try:
            result = await db.execute(
                delete(self.model).where(self.model.id == id)
            )
            
            deleted = result.rowcount > 0
            
            if deleted:
                await db.commit()
                logger.info(f"Successfully deleted {self.model.__name__} with ID: {id}")
            else:
                logger.warning(f"No {self.model.__name__} found to delete with ID: {id}")
            
            return deleted
            
        except SQLAlchemyError as e:
            await db.rollback()
            logger.error(f"Database error deleting {self.model.__name__}: {str(e)}")
            raise RepositoryError(f"Failed to delete entity: {str(e)}")
    
    async def count(
        self,
        db: AsyncSession,
        *,
        filters: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Count entities with optional filtering.
        
        Args:
            db: Database session
            filters: Dictionary of field filters
            
        Returns:
            int: Number of entities
        """
        logger.debug(f"Counting {self.model.__name__} entities")
        
        try:
            query = select(func.count(self.model.id))
            
            # Apply filters
            if filters:
                for field, value in filters.items():
                    if hasattr(self.model, field):
                        query = query.where(getattr(self.model, field) == value)
            
            result = await db.execute(query)
            count = result.scalar()
            
            logger.debug(f"Counted {count} {self.model.__name__} entities")
            return count
            
        except SQLAlchemyError as e:
            logger.error(f"Database error counting {self.model.__name__}: {str(e)}")
            raise RepositoryError(f"Failed to count entities: {str(e)}")
    
    async def exists(
        self,
        db: AsyncSession,
        *,
        filters: Dict[str, Any]
    ) -> bool:
        """
        Check if entity exists with given filters.
        
        Args:
            db: Database session
            filters: Dictionary of field filters
            
        Returns:
            bool: True if entity exists, False otherwise
        """
        logger.debug(f"Checking if {self.model.__name__} exists")
        
        try:
            query = select(self.model.id)
            
            # Apply filters
            for field, value in filters.items():
                if hasattr(self.model, field):
                    query = query.where(getattr(self.model, field) == value)
            
            query = query.limit(1)
            
            result = await db.execute(query)
            exists = result.scalar() is not None
            
            logger.debug(f"{self.model.__name__} exists: {exists}")
            return exists
            
        except SQLAlchemyError as e:
            logger.error(f"Database error checking {self.model.__name__} existence: {str(e)}")
            raise RepositoryError(f"Failed to check entity existence: {str(e)}")
    
    def _validate_monetary_fields(self, obj_data: Dict[str, Any]) -> None:
        """
        Validate monetary fields follow LQM Standard (Decimal types).
        
        Args:
            obj_data: Dictionary of object data
            
        Raises:
            ValidationError: If monetary fields are not Decimal type
        """
        monetary_field_suffixes = ['_in_cents', '_rate', '_score']
        
        for field_name, value in obj_data.items():
            # Check if field is monetary based on naming convention
            is_monetary = any(field_name.endswith(suffix) for suffix in monetary_field_suffixes)
            
            if is_monetary and value is not None:
                if not isinstance(value, Decimal):
                    try:
                        # Attempt to convert to Decimal
                        obj_data[field_name] = Decimal(str(value))
                        logger.debug(f"Converted {field_name} to Decimal: {obj_data[field_name]}")
                    except (ValueError, TypeError) as e:
                        raise ValidationError(
                            f"Monetary field {field_name} must be Decimal type, got {type(value)}: {str(e)}"
                        )

# Export base repository and exceptions
__all__ = [
    'BaseRepository',
    'RepositoryError',
    'NotFoundError',
    'DuplicateError',
    'ValidationError'
]