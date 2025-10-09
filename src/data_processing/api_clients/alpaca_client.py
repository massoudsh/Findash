import alpaca_trade_api as tradeapi
import logging
from typing import List
from decouple import config

logger = logging.getLogger(__name__)

class AlpacaClient:
    """
    A client for interacting with the Alpaca trading API.
    """
    def __init__(self):
        try:
            self.api = tradeapi.REST(
                config('ALPACA_API_KEY'),
                config('ALPACA_SECRET_KEY'),
                base_url=config('ALPACA_BASE_URL', default='https://paper-api.alpaca.markets')
            )
            self.api.get_account() # Verify credentials
            logger.info("Alpaca client initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Alpaca client: {e}")
            self.api = None

    def submit_market_order(self, symbol: str, qty: float, side: str) -> tradeapi.entity.Order:
        """
        Submits a market order.
        
        Args:
            symbol (str): The ticker symbol to trade.
            qty (float): The number of shares to trade.
            side (str): 'buy' or 'sell'.

        Returns:
            The created Order entity.
        """
        if not self.api:
            raise ConnectionError("Alpaca API not initialized.")
        
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type='market',
                time_in_force='gtc'
            )
            logger.info(f"Submitted market order for {qty} shares of {symbol} ({side}). Order ID: {order.id}")
            return order
        except Exception as e:
            logger.error(f"Failed to submit market order for {symbol}: {e}")
            raise

    def list_positions(self) -> List[tradeapi.entity.Position]:
        """Retrieves a list of all open positions."""
        if not self.api:
            raise ConnectionError("Alpaca API not initialized.")
            
        try:
            positions = self.api.list_positions()
            logger.info(f"Retrieved {len(positions)} open positions.")
            return positions
        except Exception as e:
            logger.error(f"Failed to list positions: {e}")
            raise 