"""
Real Market Data API Endpoints
Uses free data sources to provide real-time market data
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any, Optional
import asyncio
import logging
from datetime import datetime

from src.data_processing.free_data_sources import get_real_market_data, MarketData

logger = logging.getLogger(__name__)
real_data_router = APIRouter()

@real_data_router.get("/market-data/real-time")
async def get_real_time_data(
    symbols: str = Query(..., description="Comma-separated list of symbols (e.g., AAPL,MSFT,BTC-USD)")
) -> Dict[str, Any]:
    """
    Get real-time market data for multiple symbols using free APIs
    """
    try:
        # Parse symbols
        symbol_list = [s.strip().upper() for s in symbols.split(',')]
        
        if len(symbol_list) > 20:  # Limit to prevent abuse
            raise HTTPException(status_code=400, detail="Maximum 20 symbols allowed")
        
        # Fetch real data
        market_data = await get_real_market_data(symbol_list)
        
        # Convert to response format
        response_data = {}
        for symbol, data in market_data.items():
            response_data[symbol] = {
                "symbol": data.symbol,
                "price": data.price,
                "open": data.open,
                "high": data.high,
                "low": data.low,
                "volume": data.volume,
                "change": data.change,
                "change_percent": data.change_percent,
                "timestamp": data.timestamp.isoformat(),
                "source": data.source
            }
        
        return {
            "status": "success",
            "data": response_data,
            "timestamp": datetime.now().isoformat(),
            "symbols_requested": len(symbol_list),
            "symbols_returned": len(response_data)
        }
        
    except Exception as e:
        logger.error(f"Error fetching real-time data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch market data: {str(e)}")

@real_data_router.get("/market-data/portfolio")
async def get_portfolio_data() -> Dict[str, Any]:
    """
    Get real-time data for the default portfolio symbols
    """
    # Default portfolio symbols
    portfolio_symbols = [
        'AAPL', 'TSLA', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',  # Tech stocks
        'BTC-USD', 'ETH-USD', 'TRX-USD', 'LINK-USD', 'CAKE-USD',  # Crypto
        'USDT-USD', 'USDC-USD',  # Stablecoins
        'GLD', 'SLV', 'SPY', 'QQQ'  # Commodities & ETFs
    ]
    
    try:
        market_data = await get_real_market_data(portfolio_symbols)
        
        # Organize by asset class
        organized_data = {
            "tech_stocks": {},
            "crypto": {},
            "stablecoins": {},
            "commodities_etfs": {}
        }
        
        for symbol, data in market_data.items():
            data_dict = {
                "symbol": data.symbol,
                "price": data.price,
                "open": data.open,
                "high": data.high,
                "low": data.low,
                "volume": data.volume,
                "change": data.change,
                "change_percent": data.change_percent,
                "timestamp": data.timestamp.isoformat(),
                "source": data.source
            }
            
            if symbol in ['AAPL', 'TSLA', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']:
                organized_data["tech_stocks"][symbol] = data_dict
            elif symbol in ['BTC-USD', 'ETH-USD', 'TRX-USD', 'LINK-USD', 'CAKE-USD']:
                organized_data["crypto"][symbol] = data_dict
            elif symbol in ['USDT-USD', 'USDC-USD']:
                organized_data["stablecoins"][symbol] = data_dict
            elif symbol in ['GLD', 'SLV', 'SPY', 'QQQ']:
                organized_data["commodities_etfs"][symbol] = data_dict
        
        return {
            "status": "success",
            "data": organized_data,
            "timestamp": datetime.now().isoformat(),
            "total_symbols": len(market_data)
        }
        
    except Exception as e:
        logger.error(f"Error fetching portfolio data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch portfolio data: {str(e)}")

@real_data_router.get("/market-data/top-movers")
async def get_top_movers() -> Dict[str, Any]:
    """
    Get top moving stocks with real data
    """
    # Popular symbols to check for top movers
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'AMD', 'INTC']
    
    try:
        market_data = await get_real_market_data(symbols)
        
        # Sort by change percentage
        movers_data = []
        for symbol, data in market_data.items():
            movers_data.append({
                "symbol": data.symbol,
                "price": data.price,
                "change": data.change,
                "change_percent": data.change_percent,
                "volume": data.volume,
                "source": data.source
            })
        
        # Sort by absolute change percentage
        movers_data.sort(key=lambda x: abs(x['change_percent']), reverse=True)
        
        return {
            "status": "success",
            "gainers": [m for m in movers_data if m['change_percent'] > 0][:5],
            "losers": [m for m in movers_data if m['change_percent'] < 0][:5],
            "most_active": sorted(movers_data, key=lambda x: x['volume'], reverse=True)[:5],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error fetching top movers: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch top movers: {str(e)}")

@real_data_router.get("/market-data/crypto")
async def get_crypto_data() -> Dict[str, Any]:
    """
    Get real-time cryptocurrency data
    """
    crypto_symbols = ['BTC-USD', 'ETH-USD', 'TRX-USD', 'LINK-USD', 'CAKE-USD', 'USDT-USD', 'USDC-USD']
    
    try:
        market_data = await get_real_market_data(crypto_symbols)
        
        # Calculate market summary
        total_market_cap = 0
        total_volume = 0
        gainers = 0
        losers = 0
        
        crypto_data = {}
        for symbol, data in market_data.items():
            crypto_data[symbol] = {
                "symbol": data.symbol,
                "price": data.price,
                "change": data.change,
                "change_percent": data.change_percent,
                "volume": data.volume,
                "source": data.source
            }
            
            total_volume += data.volume
            if data.change_percent > 0:
                gainers += 1
            elif data.change_percent < 0:
                losers += 1
        
        return {
            "status": "success",
            "data": crypto_data,
            "market_summary": {
                "total_volume_24h": total_volume,
                "gainers": gainers,
                "losers": losers,
                "neutral": len(crypto_data) - gainers - losers
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error fetching crypto data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch crypto data: {str(e)}")

@real_data_router.get("/market-data/sources")
def get_data_sources() -> Dict[str, Any]:
    """
    Get information about available free data sources
    """
    return {
        "status": "success",
        "free_sources": {
            "stocks": [
                {
                    "name": "Yahoo Finance",
                    "description": "Free stock data with rate limiting",
                    "rate_limit": "Variable, includes delays",
                    "coverage": "Global stocks, ETFs, indices"
                },
                {
                    "name": "Finnhub",
                    "description": "Free tier with 60 calls/minute",
                    "rate_limit": "60 calls/minute",
                    "coverage": "US stocks, real-time quotes"
                },
                {
                    "name": "Alpha Vantage",
                    "description": "Free tier with 25 calls/day",
                    "rate_limit": "25 calls/day",
                    "coverage": "Global stocks, forex, crypto"
                },
                {
                    "name": "Financial Modeling Prep",
                    "description": "Free tier with 250 calls/day",
                    "rate_limit": "250 calls/day",
                    "coverage": "US stocks, financials, ratios"
                },
                {
                    "name": "Twelve Data",
                    "description": "Free tier with 800 calls/day",
                    "rate_limit": "800 calls/day",
                    "coverage": "Global stocks, forex, crypto"
                }
            ],
            "crypto": [
                {
                    "name": "CoinGecko",
                    "description": "Free crypto data, no API key needed",
                    "rate_limit": "100 calls/minute",
                    "coverage": "10,000+ cryptocurrencies"
                },
                {
                    "name": "Binance Public API",
                    "description": "Free real-time crypto data",
                    "rate_limit": "1200 requests/minute",
                    "coverage": "All Binance trading pairs"
                },
                {
                    "name": "CryptoCompare",
                    "description": "Free tier available",
                    "rate_limit": "Variable",
                    "coverage": "2000+ cryptocurrencies"
                }
            ]
        },
        "fallback_strategy": "The system tries multiple sources in order until one succeeds, ensuring high data availability",
        "caching": "Data is cached for 1 minute to reduce API calls and improve performance"
    } 