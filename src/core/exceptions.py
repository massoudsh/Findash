"""
Comprehensive exception handling for Quantum Trading Matrix™
Custom exceptions with error codes, context, and recovery strategies
"""

import logging
import traceback
from typing import Any, Dict, Optional, List, Union
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, asdict
import uuid


class ErrorSeverity(str, Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(str, Enum):
    """Error categories for classification"""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    BUSINESS_LOGIC = "business_logic"
    EXTERNAL_SERVICE = "external_service"
    DATABASE = "database"
    NETWORK = "network"
    SYSTEM = "system"
    TRADING = "trading"
    RISK_MANAGEMENT = "risk_management"
    ML_MODEL = "ml_model"
    CACHE = "cache"
    WEBSOCKET = "websocket"


class ErrorCode(str, Enum):
    """Standardized error codes"""
    # Authentication & Authorization
    INVALID_CREDENTIALS = "AUTH_001"
    TOKEN_EXPIRED = "AUTH_002"
    TOKEN_INVALID = "AUTH_003"
    INSUFFICIENT_PERMISSIONS = "AUTH_004"
    ACCOUNT_DISABLED = "AUTH_005"
    MFA_REQUIRED = "AUTH_006"
    
    # Validation
    INVALID_INPUT = "VAL_001"
    MISSING_REQUIRED_FIELD = "VAL_002"
    INVALID_FORMAT = "VAL_003"
    VALUE_OUT_OF_RANGE = "VAL_004"
    DUPLICATE_ENTRY = "VAL_005"
    
    # Trading
    INSUFFICIENT_BALANCE = "TRADE_001"
    INVALID_SYMBOL = "TRADE_002"
    MARKET_CLOSED = "TRADE_003"
    ORDER_REJECTED = "TRADE_004"
    POSITION_LIMIT_EXCEEDED = "TRADE_005"
    PRICE_OUT_OF_RANGE = "TRADE_006"
    TRADING_SUSPENDED = "TRADE_007"
    
    # Risk Management
    RISK_LIMIT_EXCEEDED = "RISK_001"
    MARGIN_CALL = "RISK_002"
    POSITION_SIZE_EXCEEDED = "RISK_003"
    EXPOSURE_LIMIT_EXCEEDED = "RISK_004"
    
    # External Services
    MARKET_DATA_UNAVAILABLE = "EXT_001"
    BROKER_API_ERROR = "EXT_002"
    PAYMENT_PROCESSOR_ERROR = "EXT_003"
    THIRD_PARTY_SERVICE_DOWN = "EXT_004"
    
    # Database
    DB_CONNECTION_FAILED = "DB_001"
    QUERY_TIMEOUT = "DB_002"
    CONSTRAINT_VIOLATION = "DB_003"
    TRANSACTION_FAILED = "DB_004"
    
    # System
    RESOURCE_EXHAUSTED = "SYS_001"
    SERVICE_UNAVAILABLE = "SYS_002"
    CONFIGURATION_ERROR = "SYS_003"
    INTERNAL_ERROR = "SYS_004"
    
    # ML Models
    MODEL_NOT_FOUND = "ML_001"
    PREDICTION_FAILED = "ML_002"
    MODEL_TRAINING_FAILED = "ML_003"
    INVALID_MODEL_INPUT = "ML_004"
    
    # Cache
    CACHE_CONNECTION_FAILED = "CACHE_001"
    CACHE_OPERATION_FAILED = "CACHE_002"
    
    # WebSocket
    WS_CONNECTION_FAILED = "WS_001"
    MESSAGE_SEND_FAILED = "WS_002"
    SUBSCRIPTION_FAILED = "WS_003"


@dataclass
class ErrorContext:
    """Error context information"""
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    endpoint: Optional[str] = None
    method: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None
    additional_data: Optional[Dict[str, Any]] = None


@dataclass
class ErrorDetail:
    """Detailed error information"""
    error_id: str
    code: ErrorCode
    message: str
    category: ErrorCategory
    severity: ErrorSeverity
    timestamp: datetime
    context: Optional[ErrorContext] = None
    stack_trace: Optional[str] = None
    recovery_suggestions: Optional[List[str]] = None
    related_errors: Optional[List[str]] = None


class QuantumTradingException(Exception):
    """Base exception for Quantum Trading Matrix"""
    
    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.INTERNAL_ERROR,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[ErrorContext] = None,
        recovery_suggestions: Optional[List[str]] = None,
        original_exception: Optional[Exception] = None
    ):
        super().__init__(message)
        
        self.error_id = str(uuid.uuid4())
        self.code = code
        self.category = category
        self.severity = severity
        self.context = context or ErrorContext()
        self.recovery_suggestions = recovery_suggestions or []
        self.original_exception = original_exception
        self.timestamp = datetime.utcnow()
        
        # Capture stack trace
        self.stack_trace = traceback.format_exc() if original_exception else None
        
        # Log the error
        self._log_error()
    
    def _log_error(self):
        """Log the error with appropriate level"""
        logger = logging.getLogger(__name__)
        
        log_data = {
            "error_id": self.error_id,
            "code": self.code.value,
            "category": self.category.value,
            "severity": self.severity.value,
            "message": str(self),
            "context": asdict(self.context) if self.context else None
        }
        
        if self.severity == ErrorSeverity.CRITICAL:
            logger.critical("Critical error", extra=log_data)
        elif self.severity == ErrorSeverity.HIGH:
            logger.error("High severity error", extra=log_data)
        elif self.severity == ErrorSeverity.MEDIUM:
            logger.warning("Medium severity error", extra=log_data)
        else:
            logger.info("Low severity error", extra=log_data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary"""
        return {
            "error_id": self.error_id,
            "code": self.code.value,
            "message": str(self),
            "category": self.category.value,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
            "context": asdict(self.context) if self.context else None,
            "recovery_suggestions": self.recovery_suggestions,
            "stack_trace": self.stack_trace
        }
    
    def get_error_detail(self) -> ErrorDetail:
        """Get detailed error information"""
        return ErrorDetail(
            error_id=self.error_id,
            code=self.code,
            message=str(self),
            category=self.category,
            severity=self.severity,
            timestamp=self.timestamp,
            context=self.context,
            stack_trace=self.stack_trace,
            recovery_suggestions=self.recovery_suggestions
        )


# Authentication & Authorization Exceptions
class AuthenticationError(QuantumTradingException):
    """Authentication related errors"""
    
    def __init__(self, message: str, code: ErrorCode = ErrorCode.INVALID_CREDENTIALS, **kwargs):
        super().__init__(
            message,
            code=code,
            category=ErrorCategory.AUTHENTICATION,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )


class AuthorizationError(QuantumTradingException):
    """Authorization related errors"""
    
    def __init__(self, message: str, code: ErrorCode = ErrorCode.INSUFFICIENT_PERMISSIONS, **kwargs):
        super().__init__(
            message,
            code=code,
            category=ErrorCategory.AUTHORIZATION,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )


class TokenExpiredError(AuthenticationError):
    """Token expired error"""
    
    def __init__(self, message: str = "Authentication token has expired", **kwargs):
        super().__init__(
            message,
            code=ErrorCode.TOKEN_EXPIRED,
            recovery_suggestions=["Please login again to get a new token"],
            **kwargs
        )


# Validation Exceptions
class ValidationError(QuantumTradingException):
    """Validation related errors"""
    
    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        self.field = field
        super().__init__(
            message,
            code=ErrorCode.INVALID_INPUT,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            **kwargs
        )


class MissingRequiredFieldError(ValidationError):
    """Missing required field error"""
    
    def __init__(self, field: str, **kwargs):
        super().__init__(
            f"Required field '{field}' is missing",
            field=field,
            code=ErrorCode.MISSING_REQUIRED_FIELD,
            **kwargs
        )


# Trading Exceptions
class TradingError(QuantumTradingException):
    """Trading related errors"""
    
    def __init__(self, message: str, code: ErrorCode = ErrorCode.ORDER_REJECTED, **kwargs):
        super().__init__(
            message,
            code=code,
            category=ErrorCategory.TRADING,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


class InsufficientBalanceError(TradingError):
    """Insufficient balance error"""
    
    def __init__(self, required: float, available: float, **kwargs):
        self.required = required
        self.available = available
        super().__init__(
            f"Insufficient balance. Required: {required}, Available: {available}",
            code=ErrorCode.INSUFFICIENT_BALANCE,
            recovery_suggestions=["Add funds to your account", "Reduce order size"],
            **kwargs
        )


class MarketClosedError(TradingError):
    """Market closed error"""
    
    def __init__(self, symbol: str, **kwargs):
        super().__init__(
            f"Market is closed for symbol {symbol}",
            code=ErrorCode.MARKET_CLOSED,
            recovery_suggestions=["Wait for market to open", "Check market hours"],
            **kwargs
        )


# Risk Management Exceptions
class RiskManagementError(QuantumTradingException):
    """Risk management related errors"""
    
    def __init__(self, message: str, code: ErrorCode = ErrorCode.RISK_LIMIT_EXCEEDED, **kwargs):
        super().__init__(
            message,
            code=code,
            category=ErrorCategory.RISK_MANAGEMENT,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


class RiskLimitExceededError(RiskManagementError):
    """Risk limit exceeded error"""
    
    def __init__(self, limit_type: str, current: float, limit: float, **kwargs):
        super().__init__(
            f"{limit_type} limit exceeded. Current: {current}, Limit: {limit}",
            recovery_suggestions=[
                "Reduce position size",
                "Close existing positions",
                "Contact risk management"
            ],
            **kwargs
        )


# External Service Exceptions
class ExternalServiceError(QuantumTradingException):
    """External service related errors"""
    
    def __init__(self, service: str, message: str, **kwargs):
        self.service = service
        super().__init__(
            f"External service '{service}' error: {message}",
            code=ErrorCode.THIRD_PARTY_SERVICE_DOWN,
            category=ErrorCategory.EXTERNAL_SERVICE,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


class MarketDataUnavailableError(ExternalServiceError):
    """Market data unavailable error"""
    
    def __init__(self, symbol: str, **kwargs):
        super().__init__(
            "market_data_provider",
            f"Market data unavailable for symbol {symbol}",
            code=ErrorCode.MARKET_DATA_UNAVAILABLE,
            recovery_suggestions=["Try again later", "Use alternative data source"],
            **kwargs
        )


# Database Exceptions
class DatabaseError(QuantumTradingException):
    """Database related errors"""
    
    def __init__(self, message: str, operation: Optional[str] = None, **kwargs):
        self.operation = operation
        super().__init__(
            message,
            code=ErrorCode.DB_CONNECTION_FAILED,
            category=ErrorCategory.DATABASE,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


class QueryTimeoutError(DatabaseError):
    """Database query timeout error"""
    
    def __init__(self, query: str, timeout: int, **kwargs):
        super().__init__(
            f"Query timeout after {timeout}s: {query[:100]}...",
            code=ErrorCode.QUERY_TIMEOUT,
            recovery_suggestions=["Optimize query", "Increase timeout", "Check database performance"],
            **kwargs
        )


# ML Model Exceptions
class MLModelError(QuantumTradingException):
    """ML model related errors"""
    
    def __init__(self, model: str, message: str, **kwargs):
        self.model = model
        super().__init__(
            f"ML model '{model}' error: {message}",
            code=ErrorCode.PREDICTION_FAILED,
            category=ErrorCategory.ML_MODEL,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )


class ModelNotFoundError(MLModelError):
    """Model not found error"""
    
    def __init__(self, model: str, **kwargs):
        super().__init__(
            model,
            f"Model not found",
            code=ErrorCode.MODEL_NOT_FOUND,
            recovery_suggestions=["Check model name", "Train the model", "Load model from backup"],
            **kwargs
        )


# Cache Exceptions
class CacheError(QuantumTradingException):
    """Cache related errors"""
    
    def __init__(self, message: str, operation: Optional[str] = None, **kwargs):
        self.operation = operation
        super().__init__(
            message,
            code=ErrorCode.CACHE_OPERATION_FAILED,
            category=ErrorCategory.CACHE,
            severity=ErrorSeverity.LOW,
            **kwargs
        )


# WebSocket Exceptions
class WebSocketError(QuantumTradingException):
    """WebSocket-related exceptions"""
    
    def __init__(self, message: str, client_id: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.WEBSOCKET,
            code=ErrorCode.WS_CONNECTION_FAILED,
            **kwargs
        )
        self.client_id = client_id


class StrategyError(QuantumTradingException):
    """Strategy-related exceptions"""
    
    def __init__(self, message: str, strategy_name: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.BUSINESS_LOGIC,
            code=ErrorCode.INTERNAL_ERROR,
            **kwargs
        )
        self.strategy_name = strategy_name


# Error Handler
class ErrorHandler:
    """Centralized error handling"""
    
    def __init__(self):
        self.error_counts = {}
        self.error_patterns = []
        
    def handle_exception(
        self,
        exception: Exception,
        context: Optional[ErrorContext] = None,
        reraise: bool = True
    ) -> QuantumTradingException:
        """Handle any exception and convert to QuantumTradingException"""
        
        if isinstance(exception, QuantumTradingException):
            if context:
                exception.context = context
            return exception
        
        # Convert standard exceptions
        if isinstance(exception, ValueError):
            qtm_exception = ValidationError(
                str(exception),
                context=context,
                original_exception=exception
            )
        elif isinstance(exception, PermissionError):
            qtm_exception = AuthorizationError(
                str(exception),
                context=context,
                original_exception=exception
            )
        elif isinstance(exception, ConnectionError):
            qtm_exception = ExternalServiceError(
                "unknown_service",
                str(exception),
                context=context,
                original_exception=exception
            )
        else:
            qtm_exception = QuantumTradingException(
                str(exception),
                context=context,
                original_exception=exception
            )
        
        # Track error patterns
        self._track_error(qtm_exception)
        
        if reraise:
            raise qtm_exception
        
        return qtm_exception
    
    def _track_error(self, exception: QuantumTradingException):
        """Track error for pattern analysis"""
        error_key = f"{exception.category.value}:{exception.code.value}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary statistics"""
        return {
            "total_errors": sum(self.error_counts.values()),
            "error_counts": self.error_counts.copy(),
            "top_errors": sorted(
                self.error_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }


# Global error handler instance
error_handler = ErrorHandler()


# Utility functions
def create_error_context(
    request=None,
    user_id: Optional[str] = None,
    **kwargs
) -> ErrorContext:
    """Create error context from request"""
    context = ErrorContext(**kwargs)
    
    if request:
        context.request_id = getattr(request.state, 'request_id', None)
        context.endpoint = getattr(request, 'url', {}).path if hasattr(request, 'url') else None
        context.method = getattr(request, 'method', None)
        # Add more request-specific context as needed
    
    if user_id:
        context.user_id = user_id
    
    return context


def handle_and_raise(
    exception: Exception,
    context: Optional[ErrorContext] = None
):
    """Handle exception and re-raise as QuantumTradingException"""
    error_handler.handle_exception(exception, context, reraise=True)


# API Exceptions
class APIError(QuantumTradingException):
    """API related errors"""
    
    def __init__(self, message: str, status_code: int = 500, **kwargs):
        self.status_code = status_code
        super().__init__(
            message,
            code=ErrorCode.INTERNAL_ERROR,
            category=ErrorCategory.EXTERNAL_SERVICE,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )


class DataValidationError(ValidationError):
    """Data validation errors"""
    
    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            field=field,
            code=ErrorCode.INVALID_INPUT,
            **kwargs
        ) 