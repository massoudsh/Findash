from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator

from ..core.config import settings

# Create the SQLAlchemy engine using the URL from our settings
engine = create_engine(
    settings.db.SQLALCHEMY_DATABASE_URL,
    pool_pre_ping=True
)

# Create a session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that provides a database session.

    This function is a generator that yields a new SQLAlchemy session for
    each request and ensures it's closed afterward, even if an error occurs.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() 