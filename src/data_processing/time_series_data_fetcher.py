"""
Time Series Data Fetcher for Quantum Trading Matrix™
"""

import logging
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from src.database.postgres_connection import get_db

logger = logging.getLogger(__name__)

class TimeSeriesDataFetcher:
    """Fetches time series data from various sources"""
    
    def __init__(self):
        self.logger = logger
    
    def fetch_yahoo_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """
        Fetch data from Yahoo Finance
        
        Args:
            symbol: Financial symbol (e.g., 'AAPL', 'BTC-USD')
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
        
        Returns:
            DataFrame with time series data
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            # Reset index to make Date a column
            data.reset_index(inplace=True)
            
            # Rename columns to match our schema
            data.rename(columns={
                'Date': 'time',
                'Close': 'price',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Volume': 'volume'
            }, inplace=True)
            
            # Add symbol column
            data['symbol'] = symbol
            
            # Select relevant columns
            columns = ['time', 'symbol', 'price', 'open', 'high', 'low', 'volume']
            data = data[columns]
            
            self.logger.info(f"Fetched {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
            raise
    
    def fetch_database_data(self, symbol: str, start_date: Optional[datetime] = None, 
                           end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Fetch data from the database
        
        Args:
            symbol: Financial symbol
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
        
        Returns:
            DataFrame with time series data
        """
        try:
            db = get_db()
            
            # Build query
            query = "SELECT * FROM financial_time_series WHERE symbol = %s"
            params = [symbol]
            
            if start_date:
                query += " AND time >= %s"
                params.append(start_date)
            
            if end_date:
                query += " AND time <= %s"
                params.append(end_date)
            
            query += " ORDER BY time"
            
            # Execute query and convert to DataFrame
            results = db.fetch_all(query, params)
            
            if not results:
                self.logger.warning(f"No database data found for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            columns = ['time', 'symbol', 'price', 'open', 'high', 'low', 'volume']
            data = pd.DataFrame(results, columns=columns)
            
            # Ensure proper datetime format
            data['time'] = pd.to_datetime(data['time'])
            
            self.logger.info(f"Fetched {len(data)} records from database for {symbol}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching database data for {symbol}: {e}")
            raise
        finally:
            if 'db' in locals():
                db.close()
    
    def prepare_prophet_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for Prophet model training
        
        Args:
            data: Raw time series data
        
        Returns:
            DataFrame formatted for Prophet (ds, y columns)
        """
        try:
            if data.empty:
                raise ValueError("No data provided for Prophet preparation")
            
            # Prophet expects 'ds' (date) and 'y' (value) columns
            prophet_data = pd.DataFrame()
            prophet_data['ds'] = data['time']
            prophet_data['y'] = data['price']
            
            # Remove any NaN values
            prophet_data = prophet_data.dropna()
            
            # Ensure datetime format
            prophet_data['ds'] = pd.to_datetime(prophet_data['ds'])
            
            self.logger.info(f"Prepared {len(prophet_data)} records for Prophet")
            return prophet_data
            
        except Exception as e:
            self.logger.error(f"Error preparing Prophet data: {e}")
            raise
    
    def get_data_for_training(self, symbol: str, source: str = "yahoo", **kwargs) -> pd.DataFrame:
        """
        Get data for model training from specified source
        
        Args:
            symbol: Financial symbol
            source: Data source ('yahoo' or 'database')
            **kwargs: Additional parameters for data fetching
        
        Returns:
            DataFrame ready for training
        """
        try:
            if source == "yahoo":
                period = kwargs.get('period', '1y')
                data = self.fetch_yahoo_data(symbol, period)
            elif source == "database":
                start_date = kwargs.get('start_date')
                end_date = kwargs.get('end_date')
                data = self.fetch_database_data(symbol, start_date, end_date)
            else:
                raise ValueError(f"Unknown data source: {source}")
            
            if data.empty:
                raise ValueError(f"No data available for {symbol} from {source}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error getting training data for {symbol}: {e}")
            raise 