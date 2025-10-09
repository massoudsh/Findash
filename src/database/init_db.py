import asyncio
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.models import Base, User, Portfolio, RiskLevel, MarketData, NewsArticle, RedditSentiment
from src.core.security import get_password_hash

# This is a synchronous example, consider using async for production
DATABASE_URL = "postgresql://user:password@localhost/dbname"

def create_default_user(session):
    # Check if user exists
    user = session.query(User).filter_by(email="admin@example.com").first()
    if not user:
        hashed_password = get_password_hash("admin")
        user = User(
            email="admin@example.com",
            hashed_password=hashed_password,
            is_admin=True
        )
        session.add(user)
        session.commit()
        print("Admin user created.")

        # Create a default portfolio for the user
        portfolio = Portfolio(
            user_id=user.id,
            name="Default Portfolio",
            risk_level=RiskLevel.LOW
        )
        session.add(portfolio)
        session.commit()
        print("Default portfolio created.")

def init_database():
    engine = create_engine(DATABASE_URL)
    Base.metadata.create_all(bind=engine)

    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionLocal()

    create_default_user(session)

if __name__ == "__main__":
    init_database() 