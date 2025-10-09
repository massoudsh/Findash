import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.database.models import Base, MarketData
from src.data_processing.tasks import ingest_market_data
import os
from main import app

DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://user:password@localhost/dbname')
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="module")
def db():
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()
    yield session
    session.close()
    Base.metadata.drop_all(bind=engine)

def test_ingest_market_data(db):
    symbol = "AAPL"  # Use a real or mock symbol
    result = ingest_market_data.apply(args=(symbol,)).get()
    assert result["status"] == "success"
    # Query the database for the inserted record
    record = db.query(MarketData).filter(MarketData.symbol == symbol).order_by(MarketData.time.desc()).first()
    assert record is not None
    assert record.symbol == symbol