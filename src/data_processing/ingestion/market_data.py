from typing import Optional
from datetime import datetime
import os
import logging
import requests
from pydantic import BaseModel, ValidationError

# Platform logging and error handling (assume these exist in src/core)
from src.core.logging_config import setup_logging
from src.core.exceptions import APIError, DataValidationError
from src.core.validation import validator

setup_logging(log_level=os.getenv('LOG_LEVEL', 'INFO'))
logger = logging.getLogger(__name__)

class MarketData(BaseModel):
    symbol: str
    price: float
    volume: int
    timestamp: datetime
    exchange: str

def fetch_real_time_data(symbol: str, api_key: Optional[str] = None) -> MarketData:
    """Fetch real-time financial data from Alpha Vantage."""
    api_key = api_key or os.getenv('ALPHA_VANTAGE_API_KEY')
    url = 'https://www.alphavantage.co/query'
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": "1min",
        "apikey": api_key
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if "Time Series (1min)" not in data:
            logger.error(f"No time series data for {symbol}: {data}")
            raise APIError("No time series data available")
        time_series = data['Time Series (1min)']
        latest_time = next(iter(time_series.keys()))
        latest_data = time_series[latest_time]
        market_data = MarketData(
            symbol=symbol,
            price=float(latest_data['4. close']),
            volume=int(latest_data['5. volume']),
            timestamp=datetime.fromisoformat(latest_time),
            exchange='NASDAQ'  # or configurable
        )
        return market_data
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        raise APIError(f"API request failed: {str(e)}")
    except (KeyError, StopIteration) as e:
        logger.error(f"Data parsing failed: {e}")
        raise APIError(f"Data parsing failed: {str(e)}")
    except ValidationError as e:
        logger.error(f"Data validation failed: {e}")
        raise DataValidationError(f"Data validation failed: {str(e)}") 