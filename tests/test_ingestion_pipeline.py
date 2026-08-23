from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.database.models import Base, MarketData
from src.data_processing.ingestion.market_data import MarketData as MarketDataPayload
from src.data_processing.tasks import ingest_market_data
import os

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

def test_ingest_market_data(db, monkeypatch):
    symbol = "AAPL"
    monkeypatch.setattr(
        "src.data_processing.tasks.fetch_real_time_data",
        lambda _: MarketDataPayload(
            symbol=symbol,
            price=187.25,
            volume=1_000,
            timestamp=datetime.now(timezone.utc),
            exchange="NASDAQ"
        )
    )

    result = ingest_market_data.apply(args=(symbol,)).get()
    assert result["status"] == "success"
    # Query the database for the inserted record
    record = db.query(MarketData).filter(MarketData.symbol == symbol).order_by(MarketData.time.desc()).first()
    assert record is not None
    assert record.symbol == symbol