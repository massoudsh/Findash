import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from ..core.config import settings

class AuthenticationError(Exception):
    """Custom authentication error for token verification."""
    pass

def create_access_token(data: Dict[str, Any]) -> str:
    """
    Creates a JWT access token.
    
    Args:
        data: The payload to include in the token (e.g., user identifier).

    Returns:
        A signed JWT string.
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=settings.auth.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire, "type": "access"})
    encoded_jwt = jwt.encode(to_encode, settings.auth.SECRET_KEY, algorithm=settings.auth.ALGORITHM)
    return encoded_jwt

def create_refresh_token(data: Dict[str, Any]) -> str:
    """
    Creates a JWT refresh token with a longer expiration.
    
    Args:
        data: The payload to include in the token.

    Returns:
        A signed JWT string for refreshing access tokens.
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=settings.auth.REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, settings.auth.SECRET_KEY, algorithm=settings.auth.ALGORITHM)
    return encoded_jwt

def verify_token(token: str, token_type: str = "access") -> Dict[str, Any]:
    """
    Verifies and decodes a JWT token.

    Args:
        token: The JWT string to decode.
        token_type: The expected type of token ('access' or 'refresh').

    Raises:
        AuthenticationError: If the token is expired, invalid, or of the wrong type.

    Returns:
        The decoded token payload.
    """
    try:
        payload = jwt.decode(token, settings.auth.SECRET_KEY, algorithms=[settings.auth.ALGORITHM])
        if payload.get("type") != token_type:
            raise AuthenticationError("Invalid token type")
        return payload
    except jwt.ExpiredSignatureError:
        raise AuthenticationError("Token has expired")
    except jwt.JWTError:
        raise AuthenticationError("Could not validate token credentials") 