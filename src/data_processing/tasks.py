import pandas as pd
from src.core.celery_app import celery_app
from src.database.postgres_connection import get_db
import logging
from src.data_processing.ingestion.market_data import fetch_real_time_data
from src.data_processing.ingestion.news_scraper import scrape_fintech_news
from src.data_processing.ingestion.social_sentiment import analyze_reddit_sentiment
from src.database.crud import create_market_data, create_news_article, create_reddit_sentiment
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.database.models import Base
import os
from datetime import datetime

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://user:password@localhost/dbname')
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def create_time_series_table():
    """
    Creates the financial_time_series table and converts it to a hypertable.
    """
    db = get_db()
    try:
        # Check if the table already exists
        if not db.table_exists('financial_time_series'):
            logger.info("Creating 'financial_time_series' table.")
            # Create the table
            db.execute_query("""
                CREATE TABLE financial_time_series (
                    time TIMESTAMPTZ NOT NULL,
                    symbol TEXT NOT NULL,
                    price DOUBLE PRECISION,
                    open DOUBLE PRECISION,
                    high DOUBLE PRECISION,
                    low DOUBLE PRECISION,
                    volume BIGINT,
                    PRIMARY KEY (time, symbol)
                );
            """)
            
            # Convert the table to a hypertable
            db.execute_query("SELECT create_hypertable('financial_time_series', 'time');")
            logger.info("'financial_time_series' table created and converted to hypertable.")
        else:
            logger.info("'financial_time_series' table already exists.")
    finally:
        db.close()

@celery_app.task(name='data_processing.ingest_time_series_from_csv')
def ingest_time_series_from_csv(symbol: str, file_path: str):
    """
    Celery task to ingest time-series data from a CSV file into the database.

    Args:
        symbol (str): The ticker symbol for the data (e.g., 'BTC-USD').
        file_path (str): The path to the CSV file to ingest.
    """
    logger.info(f"Starting ingestion for {symbol} from {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        
        # --- Data Cleaning and Preparation ---
        df.rename(columns={
            'Date': 'time',
            'Price': 'price',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Vol.': 'volume'
        }, inplace=True)

        df['time'] = pd.to_datetime(df['time'], format='%m/%d/%Y')
        df['symbol'] = symbol

        # Clean numeric columns
        for col in ['price', 'open', 'high', 'low']:
            df[col] = df[col].astype(str).str.replace(',', '').astype(float)
        
        # Clean volume column (e.g., '42.66K' -> 42660)
        if 'volume' in df.columns and df['volume'].dtype == 'object':
            df['volume'] = df['volume'].str.replace('K', 'e3').str.replace('M', 'e6').str.replace('B', 'e9').astype(float).astype(int)
        
        # Select and reorder columns to match the table
        df = df[['time', 'symbol', 'price', 'open', 'high', 'low', 'volume']]
        
        # --- Database Insertion ---
        db = get_db()
        try:
            # Use a more efficient method for bulk insertion if available
            # For simplicity, we'll iterate here
            for _, row in df.iterrows():
                db.execute_query(
                    """
                    INSERT INTO financial_time_series (time, symbol, price, open, high, low, volume)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (time, symbol) DO NOTHING;
                    """,
                    params=tuple(row),
                    commit=False 
                )
            db.commit()
            logger.info(f"Successfully ingested {len(df)} records for {symbol}.")
        finally:
            db.close()

        return {"status": "success", "records_ingested": len(df)}

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return {"status": "error", "message": "File not found"}
    except Exception as e:
        logger.error(f"An error occurred during ingestion for {symbol}: {e}")
        return {"status": "error", "message": str(e)}

def initialize_database():
    """
    Main function to be called at startup to ensure the table exists.
    """
    create_time_series_table()

@celery_app.task(name='data_processing.ingest_market_data')
def ingest_market_data(symbol: str):
    """Celery task to ingest real-time market data for a symbol."""
    logger.info(f"Ingesting market data for {symbol}")
    db = SessionLocal()
    try:
        data = fetch_real_time_data(symbol)
        db_obj = create_market_data(db, data.dict())
        logger.info(f"Inserted market data: {db_obj.id}")
        return {"status": "success", "id": db_obj.id}
    except Exception as e:
        logger.error(f"Market data ingestion failed: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        db.close()

@celery_app.task(name='data_processing.ingest_news')
def ingest_news():
    """Celery task to ingest latest fintech news articles."""
    logger.info("Ingesting fintech news articles")
    db = SessionLocal()
    inserted_ids = []
    try:
        articles = scrape_fintech_news()
        for article in articles:
            db_obj = create_news_article(db, article.dict())
            inserted_ids.append(db_obj.id)
        logger.info(f"Inserted {len(inserted_ids)} news articles")
        return {"status": "success", "inserted_ids": inserted_ids}
    except Exception as e:
        logger.error(f"News ingestion failed: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        db.close()

@celery_app.task(name='data_processing.ingest_reddit_sentiment')
def ingest_reddit_sentiment(subreddit: str):
    """Celery task to ingest Reddit sentiment for a subreddit."""
    logger.info(f"Ingesting Reddit sentiment for r/{subreddit}")
    db = SessionLocal()
    try:
        sentiment = analyze_reddit_sentiment(subreddit)
        sentiment_dict = sentiment.dict()
        sentiment_dict['analyzed_at'] = datetime.utcnow()
        db_obj = create_reddit_sentiment(db, sentiment_dict)
        logger.info(f"Inserted Reddit sentiment: {db_obj.id}")
        return {"status": "success", "id": db_obj.id}
    except Exception as e:
        logger.error(f"Reddit sentiment ingestion failed: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        db.close()

@celery_app.task(name='data_processing.process_financial_data')
def process_financial_data_task(symbol: str):
    """Celery task to process financial data for a symbol."""
    logger.info(f"Processing financial data for {symbol}")
    try:
        # Ingest market data, news, and sentiment
        market_result = ingest_market_data.delay(symbol)
        news_result = ingest_news.delay()
        reddit_result = ingest_reddit_sentiment.delay('investing')
        
        return {
            "status": "success",
            "symbol": symbol,
            "tasks": {
                "market_data": market_result.id,
                "news": news_result.id,
                "reddit_sentiment": reddit_result.id
            }
        }
    except Exception as e:
        logger.error(f"Financial data processing failed for {symbol}: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == '__main__':
    # This allows for manual testing of the script
    initialize_database()
    # Example of how to manually trigger the task:
    # ingest_time_series_from_csv.delay('BTC-USD', 'datasets/Bitcoin 1year.D.csv')
    pass 