"""Utility functions for error handling and logging."""

import functools
import time
import logging
from typing import Callable, Any
from .exceptions import FinanceAppError

logger = logging.getLogger("finance_app")

def log_execution_time(func: Callable) -> Callable:
    """Decorator to log function execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(
                f"Function '{func.__name__}' executed in {execution_time:.2f} seconds"
            )
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                f"Function '{func.__name__}' failed after {execution_time:.2f} seconds. Error: {str(e)}"
            )
            raise
    return wrapper

def handle_api_errors(func: Callable) -> Callable:
    """Decorator to handle API-related errors."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FinanceAppError as e:
            logger.error(f"API Error: {str(e)}", extra={"error_code": e.error_code})
            raise
        except Exception as e:
            logger.error(f"Unexpected API Error: {str(e)}")
            raise FinanceAppError(f"Unexpected error in {func.__name__}: {str(e)}")
    return wrapper

def retry_on_failure(retries: int = 3, delay: float = 1.0) -> Callable:
    """Decorator to retry failed operations."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            for attempt in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == retries - 1:
                        logger.error(
                            f"Failed after {retries} attempts. Error: {str(e)}"
                        )
                        raise
                    logger.warning(
                        f"Attempt {attempt + 1} failed. Retrying in {delay} seconds..."
                    )
                    time.sleep(delay)
            return None
        return wrapper
    return decorator 