from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
import logging
import time

from src.database.postgres_connection import get_db, DatabaseConfig
from src.database.repositories import UserRepository, PortfolioRepository

# Configure logging
logging.basicConfig(level=logging.INFO)

# ... existing code ... 