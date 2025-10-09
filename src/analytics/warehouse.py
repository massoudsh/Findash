import duckdb
import pandas as pd
from typing import List, Dict, Optional
import os
from datetime import datetime
import logging

# Use the platform's logging setup if available, otherwise basic config
try:
    from src.core.logging_config import setup_logging
    logger = setup_logging()
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = os.path.join(os.getcwd(), 'data', 'financial_warehouse.db')

class FinancialDataWarehouse:
    """
    A data warehouse for financial analytics using DuckDB.
    Provides an interface to store and query transactions, market data,
    risk metrics, and model predictions.
    """
    def __init__(self, db_path: Optional[str] = None):
        """
        Initializes the DuckDB connection and creates tables if they don't exist.
        Args:
            db_path (str, optional): Path to the DuckDB database file.
                                     Defaults to DUCKDB_PATH env var or ./data/financial_warehouse.db.
        """
        path = db_path or os.getenv('DUCKDB_PATH', DEFAULT_DB_PATH)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.conn = duckdb.connect(database=path, read_only=False)
        self._initialize_tables()

    def _initialize_tables(self):
        """Create the core tables for the data warehouse if they don't already exist."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                transaction_id INTEGER PRIMARY KEY,
                timestamp TIMESTAMP,
                amount DECIMAL(18, 4),
                description VARCHAR,
                category VARCHAR,
                sentiment_score DOUBLE
            );
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY,
                symbol VARCHAR,
                price DOUBLE,
                volume BIGINT,
                timestamp TIMESTAMP
            );
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS risk_metrics (
                id INTEGER PRIMARY KEY,
                metric_name VARCHAR,
                metric_value DOUBLE,
                timestamp TIMESTAMP
            );
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS model_predictions (
                id INTEGER PRIMARY KEY,
                model_name VARCHAR,
                prediction_type VARCHAR,
                prediction_value DOUBLE,
                confidence_score DOUBLE,
                timestamp TIMESTAMP
            );
        """)

    def insert_market_data(self, data: List[Dict]):
        """
        Inserts a list of market data records into the warehouse.
        Args:
            data (List[Dict]): A list of dictionaries, each representing a market data point.
                               Keys should match 'symbol', 'price', 'volume', 'timestamp'.
        """
        df = pd.DataFrame(data)
        self.conn.execute("INSERT INTO market_data SELECT * FROM df")
        logger.info(f"Inserted {len(df)} market data records.")

    def get_market_data_by_symbol(self, symbol: str, limit: int = 1000) -> pd.DataFrame:
        """Retrieves market data for a specific symbol, returned as a DataFrame."""
        return self.conn.execute(
            "SELECT * FROM market_data WHERE symbol = ? ORDER BY timestamp DESC LIMIT ?",
            (symbol, limit)
        ).df()

    def get_risk_metrics_summary(self) -> pd.DataFrame:
        """Retrieves a summary of all risk metrics, grouped by metric name."""
        return self.conn.execute("""
            SELECT
                metric_name,
                AVG(metric_value) as avg_value,
                MIN(metric_value) as min_value,
                MAX(metric_value) as max_value,
                COUNT(1) as count
            FROM risk_metrics
            GROUP BY metric_name
        """).df()

    def get_model_performance(self, model_name: str) -> pd.DataFrame:
        """Retrieve performance metrics for a specific model."""
        return self.conn.execute("""
            SELECT
                prediction_type,
                AVG(confidence_score) as avg_confidence,
                COUNT(1) as prediction_count
            FROM model_predictions
            WHERE model_name = ?
            GROUP BY prediction_type
        """, (model_name,)).df()

    def close(self):
        """Closes the DuckDB connection."""
        self.conn.close()
        logger.info("DuckDB connection closed.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close() 