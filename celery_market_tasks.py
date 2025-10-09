#!/usr/bin/env python3
"""
Production-Ready Celery Tasks for Market Data Collection
Multi-Source Data Ingestion with PostgreSQL storage

Supported Data Sources:
- Yahoo Finance (free, real-time quotes)
- Alpha Vantage (premium features, fundamental data)
- IEX Cloud (real-time and historical data)
- Polygon.io (high-frequency data)

Usage:
1. Start Redis: docker run -d -p 6380:6379 redis:alpine
2. Start worker: celery -A celery_market_tasks worker --loglevel=info
3. Start scheduler: celery -A celery_market_tasks beat --loglevel=info
"""

import asyncio
import logging
import time
import requests
import json
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Any, Optional
import asyncpg
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
from celery import Celery
from celery.schedules import crontab

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Celery configuration
celery_app = Celery(
    'octopus_market_data',
    broker='redis://localhost:6380/0',
    backend='redis://localhost:6380/0'
)

celery_app.conf.update(
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,  # 5 minutes
    task_soft_time_limit=240,  # 4 minutes
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_routes={
        'market_data.*': {'queue': 'market_data'},
        'data_collection.*': {'queue': 'data_collection'},
    },
)

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5433,
    'database': 'trading_db',
    'user': 'postgres',
    'password': 'postgres'
}

# API Configuration - Add your API keys here
API_KEYS = {
    'alpha_vantage': 'your_alpha_vantage_api_key_here',  # Get from: https://www.alphavantage.co/support/#api-key
    'iex_cloud': 'your_iex_cloud_token_here',           # Get from: https://iexcloud.io/
    'polygon': 'your_polygon_api_key_here',             # Get from: https://polygon.io/
    'finnhub': 'your_finnhub_api_key_here'              # Get from: https://finnhub.io/
}

# Rate limiting configuration for different sources
RATE_LIMITS = {
    'yahoo': {'calls_per_minute': 100, 'delay': 0.6},
    'alpha_vantage': {'calls_per_minute': 5, 'delay': 12},  # Free tier: 5 calls/min
    'iex_cloud': {'calls_per_minute': 100, 'delay': 0.6},
    'polygon': {'calls_per_minute': 5, 'delay': 12},
    'finnhub': {'calls_per_minute': 60, 'delay': 1}
}

MAX_RETRIES = 3

class DatabaseManager:
    """Database operations manager"""
    
    @staticmethod
    async def get_connection():
        """Get database connection"""
        return await asyncpg.connect(**DB_CONFIG)
    
    @staticmethod
    async def initialize_schema():
        """Initialize enhanced market data schema"""
        conn = await DatabaseManager.get_connection()
        try:
            # Enhanced table with more fields for multi-source data
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS market_data_enhanced (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    symbol VARCHAR(32) NOT NULL,
                    price DECIMAL(15,6),
                    open DECIMAL(15,6),
                    high DECIMAL(15,6),
                    low DECIMAL(15,6),
                    close DECIMAL(15,6),
                    volume BIGINT,
                    change_percent DECIMAL(8,4),
                    market_cap BIGINT,
                    pe_ratio DECIMAL(10,4),
                    dividend_yield DECIMAL(8,4),
                    beta DECIMAL(8,4),
                    eps DECIMAL(10,4),
                    week_52_high DECIMAL(15,6),
                    week_52_low DECIMAL(15,6),
                    avg_volume BIGINT,
                    shares_outstanding BIGINT,
                    source VARCHAR(32) NOT NULL,
                    data_type VARCHAR(20) DEFAULT 'realtime',
                    data_quality DECIMAL(3,2) DEFAULT 1.0,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE(timestamp, symbol, source, data_type)
                );
            """)
            
            # Create performance indexes
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_market_data_enhanced_symbol_time 
                ON market_data_enhanced (symbol, timestamp DESC);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_market_data_enhanced_source 
                ON market_data_enhanced (source, timestamp DESC);
            """)
            
            # Create TimescaleDB hypertable if available
            try:
                await conn.execute("""
                    SELECT create_hypertable('market_data_enhanced', 'timestamp', 
                                           if_not_exists => TRUE,
                                           chunk_time_interval => INTERVAL '1 day');
                """)
                logger.info("✅ TimescaleDB hypertable created")
            except Exception:
                logger.info("ℹ️  Using regular PostgreSQL table (TimescaleDB not available)")
            
            # Create fundamental data table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS fundamental_data (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(32) NOT NULL,
                    report_date DATE NOT NULL,
                    revenue DECIMAL(15,2),
                    gross_profit DECIMAL(15,2),
                    operating_income DECIMAL(15,2),
                    net_income DECIMAL(15,2),
                    total_assets DECIMAL(15,2),
                    total_debt DECIMAL(15,2),
                    book_value DECIMAL(15,2),
                    cash_flow DECIMAL(15,2),
                    source VARCHAR(32) NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE(symbol, report_date, source)
                );
            """)
            
            logger.info("✅ Enhanced database schema initialized")
            
        finally:
            await conn.close()

class MultiSourceDataFetcher:
    """Multi-source market data fetcher with intelligent source selection"""
    
    def __init__(self):
        self.alpha_vantage_ts = None
        self.alpha_vantage_fd = None
        if API_KEYS['alpha_vantage'] != 'your_alpha_vantage_api_key_here':
            self.alpha_vantage_ts = TimeSeries(key=API_KEYS['alpha_vantage'], output_format='pandas')
            self.alpha_vantage_fd = FundamentalData(key=API_KEYS['alpha_vantage'], output_format='pandas')
    
    async def fetch_yahoo_data(self, symbol: str, period: str = '1d', interval: str = '1m'):
        """Fetch data from Yahoo Finance"""
        try:
            logger.info(f"📊 [Yahoo] Fetching data for {symbol}")
            
            ticker = yf.Ticker(symbol)
            history = ticker.history(period=period, interval=interval)
            
            if history.empty:
                raise ValueError(f"No Yahoo data for {symbol}")
            
            latest = history.iloc[-1]
            info = {}
            try:
                info = ticker.info
            except Exception as e:
                logger.warning(f"Could not get Yahoo info for {symbol}: {e}")
            
            # Calculate change percentage
            if len(history) > 1:
                day_open = history.iloc[0]['Open']
            else:
                day_open = latest['Open']
            
            change_percent = ((latest['Close'] - day_open) / day_open) * 100 if day_open != 0 else 0
            
            return {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'price': Decimal(str(latest['Close'])),
                'open': Decimal(str(latest['Open'])),
                'high': Decimal(str(latest['High'])),
                'low': Decimal(str(latest['Low'])),
                'close': Decimal(str(latest['Close'])),
                'volume': int(latest['Volume']),
                'change_percent': Decimal(str(round(change_percent, 4))),
                'market_cap': info.get('marketCap', 0) or 0,
                'pe_ratio': Decimal(str(info.get('trailingPE', 0) or 0)),
                'dividend_yield': Decimal(str(info.get('dividendYield', 0) or 0)),
                'beta': Decimal(str(info.get('beta', 0) or 0)),
                'week_52_high': Decimal(str(info.get('fiftyTwoWeekHigh', 0) or 0)),
                'week_52_low': Decimal(str(info.get('fiftyTwoWeekLow', 0) or 0)),
                'source': 'yahoo',
                'data_quality': Decimal('0.85')  # Good quality, free source
            }
            
        except Exception as e:
            logger.error(f"❌ [Yahoo] Error for {symbol}: {e}")
            raise
    
    async def fetch_alpha_vantage_data(self, symbol: str):
        """Fetch data from Alpha Vantage"""
        try:
            if not self.alpha_vantage_ts:
                raise ValueError("Alpha Vantage API key not configured")
            
            logger.info(f"📊 [Alpha Vantage] Fetching data for {symbol}")
            
            # Get intraday data (1min intervals)
            data, meta_data = self.alpha_vantage_ts.get_intraday(symbol, interval='1min', outputsize='compact')
            
            if data.empty:
                raise ValueError(f"No Alpha Vantage data for {symbol}")
            
            # Get the latest data point
            latest_time = data.index[-1]
            latest = data.iloc[-1]
            
            # Get daily data for additional metrics
            daily_data, _ = self.alpha_vantage_ts.get_daily(symbol, outputsize='compact')
            
            # Calculate change percentage
            if len(data) > 1:
                day_open = data.iloc[0]['1. open']
            else:
                day_open = latest['1. open']
            
            change_percent = ((float(latest['4. close']) - float(day_open)) / float(day_open)) * 100
            
            # Get additional fundamental data
            overview = await self.get_alpha_vantage_overview(symbol)
            
            return {
                'symbol': symbol,
                'timestamp': latest_time,
                'price': Decimal(str(latest['4. close'])),
                'open': Decimal(str(latest['1. open'])),
                'high': Decimal(str(latest['2. high'])),
                'low': Decimal(str(latest['3. low'])),
                'close': Decimal(str(latest['4. close'])),
                'volume': int(latest['5. volume']),
                'change_percent': Decimal(str(round(change_percent, 4))),
                'market_cap': overview.get('MarketCapitalization', 0),
                'pe_ratio': Decimal(str(overview.get('PERatio', 0) or 0)),
                'dividend_yield': Decimal(str(overview.get('DividendYield', 0) or 0)),
                'beta': Decimal(str(overview.get('Beta', 0) or 0)),
                'eps': Decimal(str(overview.get('EPS', 0) or 0)),
                'week_52_high': Decimal(str(overview.get('52WeekHigh', 0) or 0)),
                'week_52_low': Decimal(str(overview.get('52WeekLow', 0) or 0)),
                'source': 'alpha_vantage',
                'data_quality': Decimal('0.95')  # High quality, premium source
            }
            
        except Exception as e:
            logger.error(f"❌ [Alpha Vantage] Error for {symbol}: {e}")
            raise
    
    async def get_alpha_vantage_overview(self, symbol: str):
        """Get company overview from Alpha Vantage"""
        try:
            if not self.alpha_vantage_fd:
                return {}
            
            overview, _ = self.alpha_vantage_fd.get_company_overview(symbol)
            return overview.iloc[0].to_dict() if not overview.empty else {}
            
        except Exception as e:
            logger.warning(f"Could not get Alpha Vantage overview for {symbol}: {e}")
            return {}
    
    async def fetch_iex_cloud_data(self, symbol: str):
        """Fetch data from IEX Cloud"""
        try:
            if API_KEYS['iex_cloud'] == 'your_iex_cloud_token_here':
                raise ValueError("IEX Cloud token not configured")
            
            logger.info(f"📊 [IEX Cloud] Fetching data for {symbol}")
            
            base_url = "https://cloud.iexapis.com/stable"
            token = API_KEYS['iex_cloud']
            
            # Get quote data
            quote_url = f"{base_url}/stock/{symbol}/quote?token={token}"
            response = requests.get(quote_url)
            response.raise_for_status()
            quote_data = response.json()
            
            # Get additional stats
            stats_url = f"{base_url}/stock/{symbol}/stats?token={token}"
            stats_response = requests.get(stats_url)
            stats_data = stats_response.json() if stats_response.status_code == 200 else {}
            
            return {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'price': Decimal(str(quote_data.get('latestPrice', 0))),
                'open': Decimal(str(quote_data.get('open', 0))),
                'high': Decimal(str(quote_data.get('high', 0))),
                'low': Decimal(str(quote_data.get('low', 0))),
                'close': Decimal(str(quote_data.get('previousClose', 0))),
                'volume': int(quote_data.get('volume', 0)),
                'change_percent': Decimal(str(quote_data.get('changePercent', 0) * 100)),
                'market_cap': quote_data.get('marketCap', 0),
                'pe_ratio': Decimal(str(quote_data.get('peRatio', 0) or 0)),
                'beta': Decimal(str(stats_data.get('beta', 0) or 0)),
                'week_52_high': Decimal(str(quote_data.get('week52High', 0) or 0)),
                'week_52_low': Decimal(str(quote_data.get('week52Low', 0) or 0)),
                'avg_volume': int(stats_data.get('avg30Volume', 0) or 0),
                'shares_outstanding': int(stats_data.get('sharesOutstanding', 0) or 0),
                'source': 'iex_cloud',
                'data_quality': Decimal('0.90')  # High quality, real-time
            }
            
        except Exception as e:
            logger.error(f"❌ [IEX Cloud] Error for {symbol}: {e}")
            raise
    
    async def fetch_polygon_data(self, symbol: str):
        """Fetch data from Polygon.io"""
        try:
            if API_KEYS['polygon'] == 'your_polygon_api_key_here':
                raise ValueError("Polygon API key not configured")
            
            logger.info(f"📊 [Polygon] Fetching data for {symbol}")
            
            # Get latest trade
            base_url = "https://api.polygon.io"
            api_key = API_KEYS['polygon']
            
            # Get previous close
            prev_close_url = f"{base_url}/v2/aggs/ticker/{symbol}/prev?adjusted=true&apikey={api_key}"
            response = requests.get(prev_close_url)
            response.raise_for_status()
            data = response.json()
            
            if not data.get('results'):
                raise ValueError(f"No Polygon data for {symbol}")
            
            result = data['results'][0]
            
            return {
                'symbol': symbol,
                'timestamp': datetime.fromtimestamp(result['t'] / 1000),
                'price': Decimal(str(result['c'])),  # close price
                'open': Decimal(str(result['o'])),
                'high': Decimal(str(result['h'])),
                'low': Decimal(str(result['l'])),
                'close': Decimal(str(result['c'])),
                'volume': int(result['v']),
                'change_percent': Decimal(str(((result['c'] - result['o']) / result['o']) * 100)),
                'source': 'polygon',
                'data_quality': Decimal('0.92')  # High quality, professional
            }
            
        except Exception as e:
            logger.error(f"❌ [Polygon] Error for {symbol}: {e}")
            raise
    
    async def fetch_with_fallback(self, symbol: str, preferred_sources: List[str] = None):
        """Fetch data with intelligent source fallback"""
        if not preferred_sources:
            preferred_sources = ['alpha_vantage', 'iex_cloud', 'yahoo', 'polygon']
        
        last_error = None
        
        for source in preferred_sources:
            try:
                # Apply rate limiting
                rate_limit = RATE_LIMITS.get(source, {'delay': 1})
                await asyncio.sleep(rate_limit['delay'])
                
                if source == 'yahoo':
                    return await self.fetch_yahoo_data(symbol)
                elif source == 'alpha_vantage':
                    return await self.fetch_alpha_vantage_data(symbol)
                elif source == 'iex_cloud':
                    return await self.fetch_iex_cloud_data(symbol)
                elif source == 'polygon':
                    return await self.fetch_polygon_data(symbol)
                
            except Exception as e:
                logger.warning(f"Source {source} failed for {symbol}: {e}")
                last_error = e
                continue
        
        # If all sources failed, raise the last error
        raise Exception(f"All data sources failed for {symbol}. Last error: {last_error}")

# Enhanced Celery Tasks

@celery_app.task(bind=True, name='market_data.fetch_multi_source')
def fetch_multi_source_data(self, symbol: str, preferred_sources: List[str] = None):
    """
    Fetch market data from multiple sources with intelligent fallback
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'TSLA')
        preferred_sources: List of preferred data sources in order
    
    Returns:
        Dict with market data and source information
    """
    async def _fetch_and_store():
        try:
            logger.info(f"🔄 [Multi-Source] Fetching data for {symbol}")
            
            fetcher = MultiSourceDataFetcher()
            market_data = await fetcher.fetch_with_fallback(symbol, preferred_sources)
            
            # Store in database
            conn = await DatabaseManager.get_connection()
            try:
                await conn.execute("""
                    INSERT INTO market_data_enhanced 
                    (timestamp, symbol, price, open, high, low, close, volume, 
                     change_percent, market_cap, pe_ratio, dividend_yield, beta, eps,
                     week_52_high, week_52_low, avg_volume, shares_outstanding, 
                     source, data_type, data_quality)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21)
                    ON CONFLICT (timestamp, symbol, source, data_type) DO UPDATE SET
                        price = EXCLUDED.price,
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume,
                        change_percent = EXCLUDED.change_percent,
                        market_cap = EXCLUDED.market_cap,
                        pe_ratio = EXCLUDED.pe_ratio,
                        dividend_yield = EXCLUDED.dividend_yield,
                        beta = EXCLUDED.beta,
                        eps = EXCLUDED.eps,
                        week_52_high = EXCLUDED.week_52_high,
                        week_52_low = EXCLUDED.week_52_low,
                        avg_volume = EXCLUDED.avg_volume,
                        shares_outstanding = EXCLUDED.shares_outstanding,
                        data_quality = EXCLUDED.data_quality,
                        created_at = NOW()
                """, 
                market_data['timestamp'],
                market_data['symbol'],
                market_data['price'],
                market_data['open'],
                market_data['high'],
                market_data['low'],
                market_data['close'],
                market_data['volume'],
                market_data['change_percent'],
                market_data.get('market_cap', 0),
                market_data.get('pe_ratio', Decimal('0')),
                market_data.get('dividend_yield', Decimal('0')),
                market_data.get('beta', Decimal('0')),
                market_data.get('eps', Decimal('0')),
                market_data.get('week_52_high', Decimal('0')),
                market_data.get('week_52_low', Decimal('0')),
                market_data.get('avg_volume', 0),
                market_data.get('shares_outstanding', 0),
                market_data['source'],
                'realtime',
                market_data['data_quality']
                )
                
                logger.info(f"✅ [Multi-Source] Stored {symbol} data from {market_data['source']}")
                
            finally:
                await conn.close()
            
            return {
                'status': 'success',
                'symbol': symbol,
                'source': market_data['source'],
                'price': float(market_data['price']),
                'change_percent': float(market_data['change_percent']),
                'data_quality': float(market_data['data_quality'])
            }
            
        except Exception as e:
            logger.error(f"❌ [Multi-Source] Error for {symbol}: {e}")
            return {
                'status': 'error',
                'symbol': symbol,
                'error': str(e)
            }
    
    # Run the async function
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(_fetch_and_store())
        return result
    finally:
        loop.close()

@celery_app.task(bind=True, name='market_data.fetch_real_time')
def fetch_real_time_data(self, symbol: str):
    """
    Fetch real-time market data for a symbol
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'TSLA')
    
    Returns:
        Dict with market data or error information
    """
    async def _fetch_and_store():
        try:
            logger.info(f"📊 Fetching real-time data for {symbol}")
            
            # Fetch data with retry logic
            ticker, history = MarketDataFetcher.fetch_with_retry(symbol, period='1d', interval='1m')
            
            # Get latest data point
            latest = history.iloc[-1]
            
            # Get company info (with error handling)
            try:
                info = ticker.info
                market_cap = info.get('marketCap', 0) or 0
            except Exception:
                market_cap = 0
            
            # Calculate change percentage
            if len(history) > 1:
                day_open = history.iloc[0]['Open']
            else:
                day_open = latest['Open']
            
            change_percent = ((latest['Close'] - day_open) / day_open) * 100 if day_open != 0 else 0
            
            # Prepare market data
            market_data = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'price': Decimal(str(latest['Close'])),
                'open': Decimal(str(latest['Open'])),
                'high': Decimal(str(latest['High'])),
                'low': Decimal(str(latest['Low'])),
                'close': Decimal(str(latest['Close'])),
                'volume': int(latest['Volume']),
                'change_percent': Decimal(str(round(change_percent, 4))),
                'market_cap': market_cap,
                'source': 'yahoo',
                'data_type': 'realtime'
            }
            
            # Store in database
            conn = await DatabaseManager.get_connection()
            try:
                await conn.execute("""
                    INSERT INTO market_data_celery 
                    (timestamp, symbol, price, open, high, low, close, volume, 
                     change_percent, market_cap, source, data_type)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                    ON CONFLICT (timestamp, symbol, data_type) DO UPDATE SET
                        price = EXCLUDED.price,
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume,
                        change_percent = EXCLUDED.change_percent,
                        market_cap = EXCLUDED.market_cap,
                        created_at = NOW()
                """, 
                market_data['timestamp'],
                market_data['symbol'],
                market_data['price'],
                market_data['open'],
                market_data['high'],
                market_data['low'],
                market_data['close'],
                market_data['volume'],
                market_data['change_percent'],
                market_data['market_cap'],
                market_data['source'],
                market_data['data_type']
                )
                
                logger.info(f"✅ Stored real-time data for {symbol}: ${market_data['price']} ({market_data['change_percent']:+}%)")
                
            finally:
                await conn.close()
            
            return {
                'status': 'success',
                'symbol': symbol,
                'price': float(market_data['price']),
                'change_percent': float(market_data['change_percent']),
                'volume': market_data['volume'],
                'timestamp': market_data['timestamp'].isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error fetching real-time data for {symbol}: {e}")
            return {
                'status': 'error',
                'symbol': symbol,
                'error': str(e)
            }
    
    # Run async function
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_fetch_and_store())
    finally:
        loop.close()

@celery_app.task(bind=True, name='market_data.fetch_historical')
def fetch_historical_data(self, symbol: str, period: str = '1mo', interval: str = '1d'):
    """
    Fetch historical market data for a symbol
    
    Args:
        symbol: Stock symbol
        period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
        interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
    
    Returns:
        Dict with result information
    """
    async def _fetch_and_store():
        try:
            logger.info(f"📈 Fetching historical data for {symbol} (period: {period}, interval: {interval})")
            
            # Fetch data with retry logic
            ticker, history = MarketDataFetcher.fetch_with_retry(symbol, period=period, interval=interval)
            
            # Prepare batch data
            batch_data = []
            for timestamp, row in history.iterrows():
                change_percent = ((row['Close'] - row['Open']) / row['Open']) * 100 if row['Open'] != 0 else 0
                
                batch_data.append((
                    timestamp,
                    symbol,
                    Decimal(str(row['Close'])),
                    Decimal(str(row['Open'])),
                    Decimal(str(row['High'])),
                    Decimal(str(row['Low'])),
                    Decimal(str(row['Close'])),
                    int(row['Volume']),
                    Decimal(str(round(change_percent, 4))),
                    0,  # market_cap not available in historical data
                    'yahoo',
                    'historical'
                ))
            
            # Batch insert into database
            conn = await DatabaseManager.get_connection()
            try:
                await conn.executemany("""
                    INSERT INTO market_data_celery 
                    (timestamp, symbol, price, open, high, low, close, volume, 
                     change_percent, market_cap, source, data_type)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                    ON CONFLICT (timestamp, symbol, data_type) DO UPDATE SET
                        price = EXCLUDED.price,
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume,
                        change_percent = EXCLUDED.change_percent
                """, batch_data)
                
                logger.info(f"✅ Stored {len(batch_data)} historical records for {symbol}")
                
            finally:
                await conn.close()
            
            return {
                'status': 'success',
                'symbol': symbol,
                'records_inserted': len(batch_data),
                'period': period,
                'interval': interval
            }
            
        except Exception as e:
            logger.error(f"❌ Error fetching historical data for {symbol}: {e}")
            return {
                'status': 'error',
                'symbol': symbol,
                'error': str(e)
            }
    
    # Run async function
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_fetch_and_store())
    finally:
        loop.close()

@celery_app.task(bind=True, name='market_data.batch_collect')
def batch_collect_market_data(self, symbols: List[str], data_type: str = 'realtime'):
    """
    Collect market data for multiple symbols
    
    Args:
        symbols: List of stock symbols
        data_type: 'realtime' or 'historical'
    
    Returns:
        Dict with batch results
    """
    logger.info(f"🔄 Starting batch {data_type} collection for {len(symbols)} symbols")
    
    results = []
    successful = 0
    failed = 0
    
    for i, symbol in enumerate(symbols):
        try:
            if data_type == 'realtime':
                result = fetch_real_time_data.delay(symbol)
            else:
                result = fetch_historical_data.delay(symbol)
            
            results.append({
                'symbol': symbol,
                'task_id': result.id,
                'status': 'queued',
                'position': i + 1
            })
            successful += 1
            
            # Rate limiting between symbols
            if i < len(symbols) - 1:  # Don't delay after last symbol
                time.sleep(RATE_LIMIT_DELAY)
                
        except Exception as e:
            logger.error(f"❌ Failed to queue task for {symbol}: {e}")
            results.append({
                'symbol': symbol,
                'status': 'error',
                'error': str(e),
                'position': i + 1
            })
            failed += 1
    
    logger.info(f"✅ Batch collection queued: {successful} successful, {failed} failed")
    
    return {
        'status': 'success',
        'data_type': data_type,
        'total_symbols': len(symbols),
        'queued_successfully': successful,
        'failed_to_queue': failed,
        'results': results,
        'timestamp': datetime.now().isoformat()
    }

@celery_app.task(bind=True, name='market_data.get_latest_prices')
def get_latest_prices(self, limit: int = 20):
    """
    Get latest market prices from database
    
    Args:
        limit: Maximum number of records to return
    
    Returns:
        List of latest price data
    """
    async def _get_prices():
        try:
            conn = await DatabaseManager.get_connection()
            try:
                rows = await conn.fetch("""
                    SELECT DISTINCT ON (symbol) 
                        symbol, price, change_percent, volume, timestamp, market_cap, data_type
                    FROM market_data_celery 
                    WHERE data_type = 'realtime'
                    ORDER BY symbol, timestamp DESC
                    LIMIT $1
                """, limit)
                
                results = []
                for row in rows:
                    results.append({
                        'symbol': row['symbol'],
                        'price': float(row['price']),
                        'change_percent': float(row['change_percent']),
                        'volume': row['volume'],
                        'timestamp': row['timestamp'].isoformat(),
                        'market_cap': row['market_cap'],
                        'data_type': row['data_type']
                    })
                
                return {
                    'status': 'success',
                    'count': len(results),
                    'data': results,
                    'retrieved_at': datetime.now().isoformat()
                }
                
            finally:
                await conn.close()
                
        except Exception as e:
            logger.error(f"❌ Error getting latest prices: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_get_prices())
    finally:
        loop.close()

@celery_app.task(bind=True, name='market_data.initialize_database')
def initialize_database(self):
    """Initialize database schema"""
    async def _init():
        try:
            await DatabaseManager.initialize_schema()
            return {'status': 'success', 'message': 'Database schema initialized'}
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_init())
    finally:
        loop.close()

# Periodic task configuration
@celery_app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    """Setup periodic tasks for automated data collection"""
    
    # Popular tech stocks for continuous monitoring
    WATCHLIST = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA']
    
    # Collect real-time data every 5 minutes during market hours
    sender.add_periodic_task(
        300.0,  # 5 minutes
        batch_collect_market_data.s(WATCHLIST, 'realtime'),
        name='collect_realtime_data_5min'
    )
    
    # Collect historical data once daily at 6 PM ET (after market close)
    sender.add_periodic_task(
        crontab(hour=23, minute=0),  # 11 PM UTC = 6 PM ET
        batch_collect_market_data.s(WATCHLIST, 'historical'),
        name='collect_historical_data_daily'
    )

if __name__ == '__main__':
    print("🚀 Octopus Trading Platform - Celery Market Data Tasks")
    print("=" * 60)
    print("\n📋 Available Tasks:")
    print("• fetch_real_time_data(symbol) - Get real-time quote")
    print("• fetch_historical_data(symbol, period, interval) - Get historical data")
    print("• batch_collect_market_data(symbols, data_type) - Batch collection")
    print("• get_latest_prices(limit) - Get latest prices from DB")
    print("• initialize_database() - Setup database schema")
    
    print("\n🔧 Setup Commands:")
    print("1. Start Redis: docker run -d -p 6380:6379 redis:alpine")
    print("2. Start worker: celery -A celery_market_tasks worker --loglevel=info")
    print("3. Start scheduler: celery -A celery_market_tasks beat --loglevel=info")
    
    print("\n📊 Usage Examples:")
    print("# Queue real-time data collection")
    print("from celery_market_tasks import fetch_real_time_data")
    print("result = fetch_real_time_data.delay('AAPL')")
    print("print(result.get())")
    
    print("\n# Queue historical data collection")
    print("from celery_market_tasks import fetch_historical_data")
    print("result = fetch_historical_data.delay('AAPL', '1mo', '1d')")
    print("print(result.get())")
    
    print("\n# Batch collection")
    print("from celery_market_tasks import batch_collect_market_data")
    print("result = batch_collect_market_data.delay(['AAPL', 'GOOGL', 'MSFT'])")
    print("print(result.get())")
    
    # Initialize database
    print("\n🗄️  Initializing database...")
    init_result = initialize_database.delay()
    print(f"Database initialization task queued: {init_result.id}")
    
    print("\n✅ Celery Market Data Tasks ready!") 