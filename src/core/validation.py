import logging
from pydantic import BaseModel, ValidationError, validator
from typing import List, Optional

from src.core.logging_config import setup_logging
from src.core.exceptions import DataValidationError

# Initialize logging
logger = setup_logging()

# ... existing code ... 