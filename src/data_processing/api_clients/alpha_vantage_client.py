import requests
import logging
from typing import Dict, Any
from requests.adapters import HTTPAdapter, Retry
from decouple import config

logger = logging.getLogger(__name__)

class AlphaVantageClient:
    """
    A client for fetching real-time data from Alpha Vantage.
    """
    def __init__(self, api_key: str = None):
        self.api_key = api_key or config('ALPHA_VANTAGE_API_KEY')
        self.base_url = "https://www.alphavantage.co/query"
        self._session = self._create_session()

    def _create_session(self) -> requests.Session:
        """Creates a requests session with retry logic."""
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        return session

    def fetch_intraday_data(self, symbol: str, interval: str = '1min') -> Dict[str, Any]:
        """
        Fetches intraday time series data for a given symbol.
        """
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": interval,
            "apikey": self.api_key,
            "outputsize": "compact" # 'compact' for last 100, 'full' for all
        }
        logger.info(f"Fetching intraday data for {symbol} with interval {interval}.")
        
        try:
            response = self._session.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if "Time Series" not in str(data):
                logger.warning(f"No time series data found for {symbol} in response: {data}")
                raise ValueError("Invalid API response format")
            
            return data

        except requests.RequestException as e:
            logger.error(f"API request to Alpha Vantage failed for {symbol}: {e}")
            raise
        except ValueError as e:
            logger.error(f"Data parsing failed for {symbol}: {e}")
            raise 