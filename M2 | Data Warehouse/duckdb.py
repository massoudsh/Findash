import duckdb
import pandas as pd
from typing import List, Dict
import os
from datetime import datetime
import re
import logging


class FinancialDataWarehouse:
    def __init__(self, db_path: str = "financial_warehouse.db"):
        """Initialize the DuckDB financial data warehouse.
        
        Args:
            db_path (str): Path to the DuckDB database file
        """
        self.conn = duckdb.connect(db_path)
        self._initialize_tables()
    
    def _initialize_tables(self):
        """Create the core tables if they don't exist."""
        # Transactions table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                transaction_id INTEGER PRIMARY KEY,
                timestamp TIMESTAMP,
                amount DECIMAL(15,2),
                description VARCHAR,
                category VARCHAR,
                merchant VARCHAR,
                sentiment_score FLOAT
            )
        """)
        
        # Market data table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                market_data_id INTEGER PRIMARY KEY,
                symbol VARCHAR,
                price DECIMAL(15,2),
                volume INTEGER,
                timestamp TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Risk metrics table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS risk_metrics (
                metric_id INTEGER PRIMARY KEY,
                metric_name VARCHAR,
                metric_value FLOAT,
                timestamp TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Model predictions table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS model_predictions (
                prediction_id INTEGER PRIMARY KEY,
                model_name VARCHAR,
                prediction_type VARCHAR,
                prediction_value FLOAT,
                confidence_score FLOAT,
                timestamp TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

    def insert_transaction(self, transaction: Dict):
        """Insert a new financial transaction."""
        try:
            self._validate_timestamp(transaction.get('timestamp', datetime.now()))
            self._validate_amount(transaction['amount'])

            self.conn.execute("""
                INSERT INTO transactions (
                    timestamp, amount, description, category, 
                    merchant, sentiment_score
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                transaction.get('timestamp', datetime.now()),
                transaction['amount'],
                transaction['description'],
                transaction.get('category'),
                transaction.get('merchant'),
                transaction.get('sentiment_score')
            ))
            logging.info("Transaction inserted successfully: %s", transaction)

        except Exception as e:
            logging.error("Error inserting transaction: %s", e)
            raise

    def insert_market_data(self, market_data: Dict):
        """Insert new market data."""
        self.conn.execute("""
            INSERT INTO market_data (
                symbol, price, volume, timestamp
            ) VALUES (?, ?, ?, ?)
        """, (
            market_data['symbol'],
            market_data['price'],
            market_data['volume'],
            market_data.get('timestamp', datetime.now())
        ))

    def insert_risk_metric(self, metric: Dict):
        """Insert a new risk metric."""
        self.conn.execute("""
            INSERT INTO risk_metrics (
                metric_name, metric_value, timestamp
            ) VALUES (?, ?, ?)
        """, (
            metric['name'],
            metric['value'],
            metric.get('timestamp', datetime.now())
        ))

    def insert_model_prediction(self, prediction: Dict):
        """Insert a new model prediction."""
        self.conn.execute("""
            INSERT INTO model_predictions (
                model_name, prediction_type, prediction_value,
                confidence_score, timestamp
            ) VALUES (?, ?, ?, ?, ?)
        """, (
            prediction['model_name'],
            prediction['prediction_type'],
            prediction['prediction_value'],
            prediction.get('confidence_score', 0.0),
            prediction.get('timestamp', datetime.now())
        ))

    def get_transactions_by_date_range(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Retrieve transactions within a date range."""
        try:
            df = self.conn.execute("""
                SELECT * FROM transactions 
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY timestamp DESC
            """, (start_date, end_date)).df()
            logging.info("Retrieved transactions between %s and %s", start_date, end_date)
            return df
        except Exception as e:
            logging.error("Error retrieving transactions: %s", e)
            raise

    def get_market_data_by_symbol(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """Retrieve market data for a specific symbol."""
        return self.conn.execute("""
            SELECT * FROM market_data 
            WHERE symbol = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (symbol, limit)).df()

    def get_risk_metrics_summary(self) -> pd.DataFrame:
        """Get a summary of recent risk metrics."""
        return self.conn.execute("""
            SELECT 
                metric_name,
                AVG(metric_value) as avg_value,
                MIN(metric_value) as min_value,
                MAX(metric_value) as max_value,
                COUNT(*) as count
            FROM risk_metrics
            GROUP BY metric_name
        """).df()

    def get_model_performance(self, model_name: str) -> pd.DataFrame:
        """Get performance metrics for a specific model."""
        return self.conn.execute("""
            SELECT 
                prediction_type,
                AVG(confidence_score) as avg_confidence,
                COUNT(*) as prediction_count
            FROM model_predictions
            WHERE model_name = ?
            GROUP BY prediction_type
        """, (model_name,)).df()

    def close(self):
        """Close the database connection."""
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

        
    