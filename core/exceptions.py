"""Custom exceptions for the application."""

class FinanceAppError(Exception):
    """Base exception class for the application."""
    def __init__(self, message: str, error_code: str = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)

class APIError(FinanceAppError):
    """Raised when external API calls fail."""
    pass

class DataValidationError(FinanceAppError):
    """Raised when data validation fails."""
    pass

class ScrapingError(FinanceAppError):
    """Raised when web scraping operations fail."""
    pass

class DatabaseError(FinanceAppError):
    """Raised when database operations fail."""
    pass

class ConfigurationError(FinanceAppError):
    """Raised when there are configuration issues."""
    pass 