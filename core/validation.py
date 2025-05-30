"""Data validation service."""

from typing import Any, Dict, List, Optional, Type, TypeVar
from pydantic import BaseModel, ValidationError
from core.logging_config import setup_logging
from core.exceptions import DataValidationError

logger = setup_logging(__name__)
T = TypeVar('T', bound=BaseModel)

class DataValidator:
    """Data validation service."""
    
    @staticmethod
    def validate_data(data: Dict[str, Any], model: Type[T]) -> T:
        """
        Validate data against a Pydantic model.
        
        Args:
            data: Dictionary containing data to validate
            model: Pydantic model class to validate against
            
        Returns:
            Validated model instance
            
        Raises:
            DataValidationError: If validation fails
        """
        try:
            return model(**data)
        except ValidationError as e:
            logger.error(f"Validation error: {str(e)}")
            raise DataValidationError(str(e))
    
    @staticmethod
    def validate_list(data_list: List[Dict[str, Any]], model: Type[T]) -> List[T]:
        """Validate a list of data items."""
        return [DataValidator.validate_data(item, model) for item in data_list]
    
    @staticmethod
    def validate_optional(
        data: Optional[Dict[str, Any]],
        model: Type[T]
    ) -> Optional[T]:
        """Validate optional data."""
        if data is None:
            return None
        return DataValidator.validate_data(data, model)

# Create global validator instance
validator = DataValidator() 