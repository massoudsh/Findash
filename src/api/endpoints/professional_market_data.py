"""
DEPRECATED: This file has been integrated into unified_market_data.py

This module is kept for backward compatibility only.
All functionality is now available in:
- src.api.endpoints.unified_market_data

Please update imports to use:
    from src.api.endpoints.unified_market_data import router as market_data_router
    
This file will be removed in a future version.

Professional Market Data API for Octopus Trading Platform™
Real-time and historical market data with professional-grade quality
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
import yfinance as yf
import pandas as pd
import numpy as np

from src.core.security import get_current_active_user, rate_limit_dependency
from src.core.config import get_settings

settings = get_settings()
router = APIRouter(prefix="/api/market-data", tags=["Professional Market Data"])
logger = logging.getLogger(__name__)

# Professional data models
class MarketQuote(BaseModel):
    """Real-time market quote"""
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None
    timestamp: str

class HistoricalPrice(BaseModel):
    """Historical price data point"""
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    adjusted_close: Optional[float] = None

class TechnicalIndicators(BaseModel):
    """Technical analysis indicators"""
    symbol: str
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_lower: Optional[float] = None
    timestamp: str

class MarketData(BaseModel):
    """Comprehensive market data"""
    quote: MarketQuote
    historical: List[HistoricalPrice]
    technical_indicators: TechnicalIndicators

class OptionsChain(BaseModel):
    """Options chain data"""
    symbol: str
    expiration_date: str
    option_type: str  # "call" or "put"
    strike: float
    bid: float
    ask: float
    last_price: float
    volume: int
    open_interest: int
    implied_volatility: float
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None

def calculate_technical_indicators(df: pd.DataFrame, symbol: str) -> TechnicalIndicators:
    """Calculate technical indicators from price data"""
    try:
        # Simple Moving Averages
        sma_20 = df['Close'].rolling(window=20).mean().iloc[-1] if len(df) >= 20 else None
        sma_50 = df['Close'].rolling(window=50).mean().iloc[-1] if len(df) >= 50 else None
        
        # Exponential Moving Averages
        ema_12 = df['Close'].ewm(span=12).mean().iloc[-1]
        ema_26 = df['Close'].ewm(span=26).mean().iloc[-1]
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1] if len(df) >= 14 else None
        
        # MACD
        macd_line = ema_12 - ema_26
        macd_signal = pd.Series(macd_line).ewm(span=9).mean().iloc[-1]
        
        # Bollinger Bands
        sma_20_series = df['Close'].rolling(window=20).mean()
        std_20 = df['Close'].rolling(window=20).std()
        bollinger_upper = (sma_20_series + (std_20 * 2)).iloc[-1] if len(df) >= 20 else None
        bollinger_lower = (sma_20_series - (std_20 * 2)).iloc[-1] if len(df) >= 20 else None
        
        return TechnicalIndicators(
            symbol=symbol,
            sma_20=float(sma_20) if sma_20 and not pd.isna(sma_20) else None,
            sma_50=float(sma_50) if sma_50 and not pd.isna(sma_50) else None,
            ema_12=float(ema_12) if not pd.isna(ema_12) else None,
            ema_26=float(ema_26) if not pd.isna(ema_26) else None,
            rsi=float(rsi) if rsi and not pd.isna(rsi) else None,
            macd=float(macd_line) if not pd.isna(macd_line) else None,
            macd_signal=float(macd_signal) if not pd.isna(macd_signal) else None,
            bollinger_upper=float(bollinger_upper) if bollinger_upper and not pd.isna(bollinger_upper) else None,
            bollinger_lower=float(bollinger_lower) if bollinger_lower and not pd.isna(bollinger_lower) else None,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        
    except Exception as e:
        logger.error(f"Error calculating technical indicators for {symbol}: {e}")
        return TechnicalIndicators(
            symbol=symbol,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )

@router.get("/quote/{symbol}", response_model=MarketQuote)
async def get_real_time_quote(
    symbol: str,
    current_user: dict = Depends(get_current_active_user),
    _: bool = Depends(rate_limit_dependency)
):
    """
    Get real-time market quote for a symbol
    
    Uses Yahoo Finance API for professional-grade market data
    """
    try:
        # Validate symbol format
        symbol = symbol.upper().strip()
        
        # Fetch real-time data
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Get current price data
        hist = ticker.history(period="2d", interval="1m")
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")
        
        current_price = float(hist['Close'].iloc[-1])
        previous_close = float(info.get('previousClose', hist['Close'].iloc[-2]))
        
        change = current_price - previous_close
        change_percent = (change / previous_close) * 100
        
        quote = MarketQuote(
            symbol=symbol,
            price=current_price,
            change=change,
            change_percent=change_percent,
            volume=int(hist['Volume'].iloc[-1]),
            bid=info.get('bid'),
            ask=info.get('ask'),
            bid_size=info.get('bidSize'),
            ask_size=info.get('askSize'),
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        
        logger.info(f"Real-time quote fetched for {symbol}: ${current_price}")
        return quote
        
    except Exception as e:
        logger.error(f"Error fetching quote for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch quote for {symbol}")

@router.get("/historical/{symbol}", response_model=List[HistoricalPrice])
async def get_historical_data(
    symbol: str,
    period: str = Query("1y", description="Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)"),
    interval: str = Query("1d", description="Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)"),
    current_user: dict = Depends(get_current_active_user)
):
    """
    Get historical price data for a symbol
    """
    try:
        symbol = symbol.upper().strip()
        
        # Fetch historical data
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period, interval=interval)
        
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No historical data found for {symbol}")
        
        # Convert to our format
        historical_data = []
        for date, row in hist.iterrows():
            historical_data.append(HistoricalPrice(
                date=date.strftime("%Y-%m-%d"),
                open=float(row['Open']),
                high=float(row['High']),
                low=float(row['Low']),
                close=float(row['Close']),
                volume=int(row['Volume']),
                adjusted_close=float(row['Close'])  # YFinance adjusts by default
            ))
        
        logger.info(f"Historical data fetched for {symbol}: {len(historical_data)} points")
        return historical_data
        
    except Exception as e:
        logger.error(f"Error fetching historical data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch historical data for {symbol}")

@router.get("/technical/{symbol}", response_model=TechnicalIndicators)
async def get_technical_analysis(
    symbol: str,
    current_user: dict = Depends(get_current_active_user)
):
    """
    Get technical analysis indicators for a symbol
    """
    try:
        symbol = symbol.upper().strip()
        
        # Fetch price data for calculations
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="6mo", interval="1d")
        
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        
        # Calculate technical indicators
        indicators = calculate_technical_indicators(hist, symbol)
        
        logger.info(f"Technical indicators calculated for {symbol}")
        return indicators
        
    except Exception as e:
        logger.error(f"Error calculating technical analysis for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate technical analysis for {symbol}")

@router.get("/comprehensive/{symbol}", response_model=MarketData)
async def get_comprehensive_market_data(
    symbol: str,
    current_user: dict = Depends(get_current_active_user)
):
    """
    Get comprehensive market data including quote, historical, and technical analysis
    """
    try:
        symbol = symbol.upper().strip()
        
        # Fetch all data concurrently
        quote_task = get_real_time_quote(symbol, current_user, True)
        historical_task = get_historical_data(symbol, "3mo", "1d", current_user)
        technical_task = get_technical_analysis(symbol, current_user)
        
        quote, historical, technical = await asyncio.gather(
            quote_task, historical_task, technical_task
        )
        
        comprehensive_data = MarketData(
            quote=quote,
            historical=historical,
            technical_indicators=technical
        )
        
        logger.info(f"Comprehensive market data fetched for {symbol}")
        return comprehensive_data
        
    except Exception as e:
        logger.error(f"Error fetching comprehensive data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch comprehensive data for {symbol}")

@router.get("/options/{symbol}", response_model=List[OptionsChain])
async def get_options_chain(
    symbol: str,
    expiration: Optional[str] = Query(None, description="Expiration date (YYYY-MM-DD)"),
    current_user: dict = Depends(get_current_active_user)
):
    """
    Get options chain data for a symbol
    """
    try:
        symbol = symbol.upper().strip()
        
        # Fetch options data
        ticker = yf.Ticker(symbol)
        
        # Get available expiration dates
        if not hasattr(ticker, 'options') or not ticker.options:
            raise HTTPException(status_code=404, detail=f"No options data available for {symbol}")
        
        # Use specified expiration or nearest one
        if expiration:
            if expiration not in ticker.options:
                raise HTTPException(status_code=400, detail=f"Expiration date {expiration} not available")
            exp_date = expiration
        else:
            exp_date = ticker.options[0]  # Nearest expiration
        
        # Get options chain
        opt = ticker.option_chain(exp_date)
        
        options_data = []
        
        # Process calls
        for _, row in opt.calls.iterrows():
            options_data.append(OptionsChain(
                symbol=symbol,
                expiration_date=exp_date,
                option_type="call",
                strike=float(row['strike']),
                bid=float(row['bid']) if pd.notna(row['bid']) else 0.0,
                ask=float(row['ask']) if pd.notna(row['ask']) else 0.0,
                last_price=float(row['lastPrice']) if pd.notna(row['lastPrice']) else 0.0,
                volume=int(row['volume']) if pd.notna(row['volume']) else 0,
                open_interest=int(row['openInterest']) if pd.notna(row['openInterest']) else 0,
                implied_volatility=float(row['impliedVolatility']) if pd.notna(row['impliedVolatility']) else 0.0
            ))
        
        # Process puts
        for _, row in opt.puts.iterrows():
            options_data.append(OptionsChain(
                symbol=symbol,
                expiration_date=exp_date,
                option_type="put",
                strike=float(row['strike']),
                bid=float(row['bid']) if pd.notna(row['bid']) else 0.0,
                ask=float(row['ask']) if pd.notna(row['ask']) else 0.0,
                last_price=float(row['lastPrice']) if pd.notna(row['lastPrice']) else 0.0,
                volume=int(row['volume']) if pd.notna(row['volume']) else 0,
                open_interest=int(row['openInterest']) if pd.notna(row['openInterest']) else 0,
                implied_volatility=float(row['impliedVolatility']) if pd.notna(row['impliedVolatility']) else 0.0
            ))
        
        logger.info(f"Options chain fetched for {symbol} exp {exp_date}: {len(options_data)} contracts")
        return options_data
        
    except Exception as e:
        logger.error(f"Error fetching options chain for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch options chain for {symbol}")

@router.get("/watchlist")
async def get_market_watchlist(
    current_user: dict = Depends(get_current_active_user)
):
    """
    Get a professional market watchlist with real-time data
    """
    try:
        # Professional trading symbols
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX", "SPY", "QQQ"]
        
        watchlist_data = []
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="2d", interval="1d")
                
                if not hist.empty:
                    current_price = float(hist['Close'].iloc[-1])
                    previous_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_price
                    change = current_price - previous_close
                    change_percent = (change / previous_close) * 100 if previous_close != 0 else 0
                    
                    watchlist_data.append({
                        "symbol": symbol,
                        "price": current_price,
                        "change": change,
                        "change_percent": change_percent,
                        "volume": int(hist['Volume'].iloc[-1])
                    })
                    
            except Exception as e:
                logger.warning(f"Failed to fetch data for {symbol}: {e}")
                continue
        
        logger.info(f"Watchlist data fetched for {len(watchlist_data)} symbols")
        return {
            "watchlist": watchlist_data,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "total_symbols": len(watchlist_data)
        }
        
    except Exception as e:
        logger.error(f"Error fetching watchlist: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch watchlist data")

@router.get("/sectors")
async def get_sector_performance():
    """
    Get sector performance data
    """
    try:
        # Major sector ETFs
        sector_etfs = {
            "Technology": "XLK",
            "Healthcare": "XLV", 
            "Financial": "XLF",
            "Consumer Discretionary": "XLY",
            "Communication Services": "XLC",
            "Industrials": "XLI",
            "Consumer Staples": "XLP",
            "Energy": "XLE",
            "Utilities": "XLU",
            "Real Estate": "XLRE",
            "Materials": "XLB"
        }
        
        sector_performance = []
        
        for sector, etf in sector_etfs.items():
            try:
                ticker = yf.Ticker(etf)
                hist = ticker.history(period="5d", interval="1d")
                
                if not hist.empty:
                    current_price = float(hist['Close'].iloc[-1])
                    week_ago_price = float(hist['Close'].iloc[0])
                    change_percent = ((current_price - week_ago_price) / week_ago_price) * 100
                    
                    sector_performance.append({
                        "sector": sector,
                        "etf_symbol": etf,
                        "price": current_price,
                        "week_change_percent": change_percent
                    })
                    
            except Exception as e:
                logger.warning(f"Failed to fetch data for sector {sector} ({etf}): {e}")
                continue
        
        # Sort by performance
        sector_performance.sort(key=lambda x: x["week_change_percent"], reverse=True)
        
        logger.info(f"Sector performance data fetched for {len(sector_performance)} sectors")
        return {
            "sectors": sector_performance,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
    except Exception as e:
        logger.error(f"Error fetching sector performance: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch sector performance") 