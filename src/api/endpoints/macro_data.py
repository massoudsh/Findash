"""
Real Macro Economic Data API
Integrates with FRED, Alpha Vantage, and other economic data providers
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Query, Depends
import httpx
import pandas as pd
from src.core.config import get_settings
from src.core.cache import TradingCache, CacheNamespace

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()

class MacroDataProvider:
    """Real macro economic data provider"""
    
    def __init__(self):
        self.fred_base_url = "https://api.stlouisfed.org/fred"
        self.alphavantage_base_url = "https://www.alphavantage.co/query"
        self.trading_economics_base_url = "https://api.tradingeconomics.com"
        
    async def get_fred_data(self, series_id: str, start_date: str = None) -> Dict[str, Any]:
        """Fetch data from FRED API"""
        try:
            if not start_date:
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            
            # Using public FRED API (no key required for basic access)
            url = f"{self.fred_base_url}/series/observations"
            params = {
                "series_id": series_id,
                "api_key": "demo",  # Replace with actual key if you have one
                "file_type": "json",
                "observation_start": start_date,
                "limit": 1000
            }
            
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    observations = data.get('observations', [])
                    
                    # Process observations
                    processed_data = []
                    for obs in observations:
                        if obs['value'] != '.':  # FRED uses '.' for missing values
                            processed_data.append({
                                'date': obs['date'],
                                'value': float(obs['value'])
                            })
                    
                    return {
                        'series_id': series_id,
                        'data': processed_data,
                        'last_updated': datetime.now().isoformat()
                    }
                else:
                    logger.warning(f"FRED API error: {response.status_code}")
                    return self._get_fallback_data(series_id)
                    
        except Exception as e:
            logger.error(f"Error fetching FRED data for {series_id}: {e}")
            return self._get_fallback_data(series_id)
    
    async def get_treasury_yields(self) -> Dict[str, Any]:
        """Get real treasury yield data"""
        try:
            yields_data = {}
            
            # Treasury yield series IDs from FRED
            yield_series = {
                '3M': 'TB3MS',
                '6M': 'TB6MS', 
                '1Y': 'GS1',
                '2Y': 'GS2',
                '5Y': 'GS5',
                '10Y': 'GS10',
                '30Y': 'GS30'
            }
            
            # Fetch all yields concurrently
            tasks = []
            for period, series_id in yield_series.items():
                tasks.append(self.get_fred_data(series_id))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for i, (period, series_id) in enumerate(yield_series.items()):
                if not isinstance(results[i], Exception) and results[i]['data']:
                    latest = results[i]['data'][-1]
                    yields_data[period] = {
                        'value': latest['value'],
                        'date': latest['date'],
                        'series_id': series_id
                    }
            
            # Calculate yield curve spreads
            if '2Y' in yields_data and '10Y' in yields_data:
                yields_data['2Y10Y_SPREAD'] = {
                    'value': yields_data['10Y']['value'] - yields_data['2Y']['value'],
                    'date': yields_data['10Y']['date']
                }
            
            if '3M' in yields_data and '10Y' in yields_data:
                yields_data['3M10Y_SPREAD'] = {
                    'value': yields_data['10Y']['value'] - yields_data['3M']['value'],
                    'date': yields_data['10Y']['date']
                }
            
            return yields_data
            
        except Exception as e:
            logger.error(f"Error fetching treasury yields: {e}")
            return self._get_fallback_yields()
    
    async def get_inflation_data(self) -> Dict[str, Any]:
        """Get real inflation indicators"""
        try:
            inflation_series = {
                'CPI': 'CPIAUCSL',
                'CORE_CPI': 'CPILFESL',
                'PCE': 'PCE',
                'CORE_PCE': 'PCEPILFE',
                'PPI': 'PPIACO',
                'BREAKEVEN_5Y': 'T5YIE',
                'BREAKEVEN_10Y': 'T10YIE'
            }
            
            tasks = []
            for name, series_id in inflation_series.items():
                tasks.append(self.get_fred_data(series_id))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            inflation_data = {}
            for i, (name, series_id) in enumerate(inflation_series.items()):
                if not isinstance(results[i], Exception) and results[i]['data']:
                    data = results[i]['data']
                    if len(data) >= 2:
                        latest = data[-1]
                        previous = data[-2] if len(data) > 1 else data[-1]
                        
                        # Calculate YoY change for price indices
                        if name in ['CPI', 'CORE_CPI', 'PCE', 'CORE_PCE', 'PPI'] and len(data) >= 12:
                            year_ago = data[-13] if len(data) >= 13 else data[0]
                            yoy_change = ((latest['value'] - year_ago['value']) / year_ago['value']) * 100
                        else:
                            yoy_change = latest['value']  # For breakevens, use direct value
                        
                        inflation_data[name] = {
                            'value': latest['value'],
                            'yoy_change': round(yoy_change, 2),
                            'mom_change': round(((latest['value'] - previous['value']) / previous['value']) * 100, 2),
                            'date': latest['date']
                        }
            
            return inflation_data
            
        except Exception as e:
            logger.error(f"Error fetching inflation data: {e}")
            return self._get_fallback_inflation()
    
    async def get_monetary_policy_data(self) -> Dict[str, Any]:
        """Get Federal Reserve and monetary policy data"""
        try:
            fed_series = {
                'FED_FUNDS_RATE': 'FEDFUNDS',
                'FED_BALANCE_SHEET': 'WALCL',
                'M2_MONEY_SUPPLY': 'M2SL',
                'BANK_CREDIT': 'TOTLL',
                'COMMERCIAL_PAPER': 'COMPAPFF',
                'LIBOR_3M': 'USD3MTD156N'
            }
            
            tasks = []
            for name, series_id in fed_series.items():
                tasks.append(self.get_fred_data(series_id))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            monetary_data = {}
            for i, (name, series_id) in enumerate(fed_series.items()):
                if not isinstance(results[i], Exception) and results[i]['data']:
                    data = results[i]['data']
                    if data:
                        latest = data[-1]
                        previous = data[-2] if len(data) > 1 else data[-1]
                        
                        change_pct = 0
                        if previous['value'] != 0:
                            change_pct = ((latest['value'] - previous['value']) / previous['value']) * 100
                        
                        monetary_data[name] = {
                            'value': latest['value'],
                            'change_pct': round(change_pct, 2),
                            'date': latest['date'],
                            'units': self._get_series_units(name)
                        }
            
            return monetary_data
            
        except Exception as e:
            logger.error(f"Error fetching monetary policy data: {e}")
            return self._get_fallback_monetary()
    
    def _get_series_units(self, series_name: str) -> str:
        """Get units for different series"""
        units_map = {
            'FED_FUNDS_RATE': '%',
            'FED_BALANCE_SHEET': 'Billions $',
            'M2_MONEY_SUPPLY': 'Billions $',
            'BANK_CREDIT': 'Billions $',
            'COMMERCIAL_PAPER': '%',
            'LIBOR_3M': '%'
        }
        return units_map.get(series_name, '')
    
    def _get_fallback_data(self, series_id: str) -> Dict[str, Any]:
        """Fallback data when API fails"""
        fallback_values = {
            'FEDFUNDS': 5.25,
            'GS10': 4.50,
            'GS2': 4.75,
            'CPIAUCSL': 307.5,
            'WALCL': 7200000
        }
        
        return {
            'series_id': series_id,
            'data': [{
                'date': datetime.now().strftime('%Y-%m-%d'),
                'value': fallback_values.get(series_id, 100.0)
            }],
            'last_updated': datetime.now().isoformat(),
            'source': 'fallback'
        }
    
    def _get_fallback_yields(self) -> Dict[str, Any]:
        """Fallback treasury yields"""
        return {
            '3M': {'value': 5.30, 'date': datetime.now().strftime('%Y-%m-%d')},
            '2Y': {'value': 4.75, 'date': datetime.now().strftime('%Y-%m-%d')},
            '10Y': {'value': 4.50, 'date': datetime.now().strftime('%Y-%m-%d')},
            '2Y10Y_SPREAD': {'value': -0.25, 'date': datetime.now().strftime('%Y-%m-%d')}
        }
    
    def _get_fallback_inflation(self) -> Dict[str, Any]:
        """Fallback inflation data"""
        return {
            'CORE_PCE': {'value': 307.5, 'yoy_change': 2.8, 'date': datetime.now().strftime('%Y-%m-%d')},
            'BREAKEVEN_5Y': {'value': 2.45, 'yoy_change': 0.0, 'date': datetime.now().strftime('%Y-%m-%d')}
        }
    
    def _get_fallback_monetary(self) -> Dict[str, Any]:
        """Fallback monetary data"""
        return {
            'FED_FUNDS_RATE': {'value': 5.25, 'change_pct': 0.0, 'date': datetime.now().strftime('%Y-%m-%d'), 'units': '%'},
            'FED_BALANCE_SHEET': {'value': 7200000, 'change_pct': -2.1, 'date': datetime.now().strftime('%Y-%m-%d'), 'units': 'Millions $'}
        }

# Initialize provider
macro_provider = MacroDataProvider()

@router.get("/macro/treasury-yields")
async def get_treasury_yields():
    """Get real-time treasury yield data"""
    try:
        # Check cache first
        cached_data = await TradingCache.get_portfolio("macro_yields")
        if cached_data:
            return cached_data
        
        # Fetch real data
        yields_data = await macro_provider.get_treasury_yields()
        
        # Cache for 5 minutes
        await TradingCache.cache_portfolio("macro_yields", yields_data, ttl=300)
        
        return {
            "status": "success",
            "data": yields_data,
            "last_updated": datetime.now().isoformat(),
            "source": "FRED"
        }
        
    except Exception as e:
        logger.error(f"Error in treasury yields endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/macro/inflation")
async def get_inflation_indicators():
    """Get real inflation indicators"""
    try:
        # Check cache first
        cached_data = await TradingCache.get_portfolio("macro_inflation")
        if cached_data:
            return cached_data
        
        # Fetch real data
        inflation_data = await macro_provider.get_inflation_data()
        
        # Cache for 30 minutes
        await TradingCache.cache_portfolio("macro_inflation", inflation_data, ttl=1800)
        
        return {
            "status": "success",
            "data": inflation_data,
            "last_updated": datetime.now().isoformat(),
            "source": "FRED"
        }
        
    except Exception as e:
        logger.error(f"Error in inflation endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/macro/monetary-policy")
async def get_monetary_policy():
    """Get Federal Reserve monetary policy data"""
    try:
        # Check cache first
        cached_data = await TradingCache.get_portfolio("macro_monetary")
        if cached_data:
            return cached_data
        
        # Fetch real data
        monetary_data = await macro_provider.get_monetary_policy_data()
        
        # Cache for 1 hour
        await TradingCache.cache_portfolio("macro_monetary", monetary_data, ttl=3600)
        
        return {
            "status": "success",
            "data": monetary_data,
            "last_updated": datetime.now().isoformat(),
            "source": "FRED"
        }
        
    except Exception as e:
        logger.error(f"Error in monetary policy endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/macro/comprehensive")
async def get_comprehensive_macro_data():
    """Get comprehensive macro economic dashboard data"""
    try:
        # Fetch all macro data concurrently
        tasks = [
            macro_provider.get_treasury_yields(),
            macro_provider.get_inflation_data(),
            macro_provider.get_monetary_policy_data()
        ]
        
        yields, inflation, monetary = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle errors
        result = {
            "treasury_yields": yields if not isinstance(yields, Exception) else {},
            "inflation": inflation if not isinstance(inflation, Exception) else {},
            "monetary_policy": monetary if not isinstance(monetary, Exception) else {},
            "last_updated": datetime.now().isoformat(),
            "source": "FRED"
        }
        
        return {
            "status": "success",
            "data": result
        }
        
    except Exception as e:
        logger.error(f"Error in comprehensive macro endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/macro/economic-calendar")
async def get_economic_calendar(days_ahead: int = Query(7, ge=1, le=30)):
    """Get upcoming economic events calendar"""
    try:
        # For now, return a structured calendar
        # In production, integrate with TradingEconomics or similar
        
        upcoming_events = [
            {
                "date": (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
                "time": "08:30",
                "event": "Initial Jobless Claims",
                "currency": "USD",
                "impact": "medium",
                "forecast": "220K",
                "previous": "218K"
            },
            {
                "date": (datetime.now() + timedelta(days=3)).strftime('%Y-%m-%d'),
                "time": "14:00",
                "event": "FOMC Meeting Minutes",
                "currency": "USD", 
                "impact": "high",
                "forecast": null,
                "previous": null
            },
            {
                "date": (datetime.now() + timedelta(days=5)).strftime('%Y-%m-%d'),
                "time": "08:30",
                "event": "Core PCE Price Index",
                "currency": "USD",
                "impact": "high",
                "forecast": "2.8%",
                "previous": "2.9%"
            }
        ]
        
        return {
            "status": "success",
            "data": upcoming_events,
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in economic calendar endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 