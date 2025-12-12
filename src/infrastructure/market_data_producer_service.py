#!/usr/bin/env python3
"""
Standalone Market Data Producer Service
Produces market data events to Redis Streams for real-time streaming
"""

import os
import sys
import time
import signal
import logging
import random
from datetime import datetime
from src.infrastructure.market_data_stream import MarketDataStreamProducer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Sample symbols
SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'V', 'JNJ']

def generate_market_data(symbol: str) -> dict:
    """Generate realistic market data for a symbol"""
    base_prices = {
        'AAPL': 175.0, 'MSFT': 380.0, 'GOOGL': 140.0, 'AMZN': 150.0,
        'TSLA': 250.0, 'META': 500.0, 'NVDA': 800.0, 'JPM': 150.0,
        'V': 250.0, 'JNJ': 160.0
    }
    
    base_price = base_prices.get(symbol, 100.0)
    price_change = random.uniform(-0.02, 0.02)  # ±2% change
    price = base_price * (1 + price_change)
    volume = random.randint(1000, 50000)
    
    return {
        'symbol': symbol,
        'price': round(price, 2),
        'volume': volume,
        'exchange': 'NASDAQ',
        'timestamp': datetime.utcnow().isoformat(),
        'event_type': 'price_update'
    }

def main():
    """Main entry point for the producer service"""
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    update_interval = float(os.getenv('UPDATE_INTERVAL', '1.0'))  # seconds
    
    producer = MarketDataStreamProducer(redis_url=redis_url)
    producer.connect()
    
    logger.info("Starting market data producer service...")
    logger.info(f"Redis: {redis_url}")
    logger.info(f"Update interval: {update_interval}s")
    logger.info(f"Symbols: {', '.join(SYMBOLS)}")
    
    def signal_handler(sig, frame):
        logger.info("Shutting down producer service...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        while True:
            for symbol in SYMBOLS:
                market_data = generate_market_data(symbol)
                producer.produce_market_data(
                    symbol=symbol,
                    price=market_data['price'],
                    volume=market_data['volume'],
                    exchange=market_data['exchange']
                )
                time.sleep(update_interval / len(SYMBOLS))
    except KeyboardInterrupt:
        signal_handler(None, None)

if __name__ == '__main__':
    main()

