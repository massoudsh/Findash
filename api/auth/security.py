from passlib.context import CryptContext
from typing import Optional
from sqlalchemy.orm import Session

# In a real app, you would have a more robust way of importing this
# But for now, let's assume a structure that allows this import.
from database.models import User
from ..core.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)

def hash_password(password: str) -> str:
    """Hash a password for storing."""
    return pwd_context.hash(password)

def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
    """
    Authenticate a user by email and password.
    
    - Fetches the user by email.
    - Verifies the provided password against the stored hash.
    - Returns the user object if authentication is successful, otherwise None.
    """
    user = db.query(User).filter(User.email == email).first()
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user 