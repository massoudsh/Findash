from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import asyncio
import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Import our options risk integration module
from options_risk_integration import (
    OptionsPortfolio, OptionPosition, CorrelationAnalyzer, 
    RiskReportGenerator, BlackScholesCalculator, Greeks
)

app = FastAPI(
    title="Quantum Trading Matrix™", 
    description="Advanced Multi-Agent Trading Platform with Options Risk Management",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API
class HealthResponse(BaseModel):
    status: str
    service: str
    timestamp: str
    environment: str

class MarketDataResponse(BaseModel):
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    timestamp: str

class StrategyRequest(BaseModel):
    symbol: str
    strategy_type: str = "momentum"
    period: int = 20

class OptionPositionRequest(BaseModel):
    symbol: str
    option_type: str  # 'call' or 'put'
    strike: float
    expiry_days: int  # days from now
    quantity: int
    premium: float
    underlying_price: float
    risk_free_rate: float = 0.05
    volatility: float = 0.2

class OptionPriceRequest(BaseModel):
    underlying_price: float
    strike: float
    time_to_expiry: float  # in years
    risk_free_rate: float = 0.05
    volatility: float = 0.2
    option_type: str = "call"

class CorrelationRequest(BaseModel):
    symbols: List[str]
    period: str = "1y"

# Global portfolio instance (in production, this would be user-specific)
global_portfolio = OptionsPortfolio()
correlation_analyzer = CorrelationAnalyzer()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for Docker and monitoring."""
    return HealthResponse(
        status="healthy",
        service="Quantum Trading Matrix",
        timestamp=datetime.now().isoformat(),
        environment=os.getenv("ENVIRONMENT", "development")
    )

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to Quantum Trading Matrix™ API", 
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "features": [
            "Market Data Analysis",
            "Options Trading & Risk Management",
            "Portfolio Correlation Analysis",
            "Greeks Calculation",
            "Risk Reporting",
            "Scenario Analysis"
        ]
    }

@app.get("/market-data/{symbol}", response_model=MarketDataResponse)
async def get_market_data(symbol: str):
    """Get real-time market data for a symbol."""
    try:
        ticker = yf.Ticker(symbol.upper())
        info = ticker.info
        hist = ticker.history(period="2d")
        
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No data found for symbol {symbol}")
        
        current_price = float(hist['Close'].iloc[-1])
        previous_price = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_price
        change = current_price - previous_price
        change_percent = (change / previous_price * 100) if previous_price != 0 else 0
        
        return MarketDataResponse(
            symbol=symbol.upper(),
            price=round(current_price, 2),
            change=round(change, 2),
            change_percent=round(change_percent, 2),
            volume=int(hist['Volume'].iloc[-1]),
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching data for {symbol}: {str(e)}")

@app.post("/options/price")
async def calculate_option_price(request: OptionPriceRequest):
    """Calculate option price and Greeks using Black-Scholes model."""
    try:
        # Calculate option price
        option_price = BlackScholesCalculator.option_price(
            request.underlying_price,
            request.strike,
            request.time_to_expiry,
            request.risk_free_rate,
            request.volatility,
            request.option_type
        )
        
        # Calculate Greeks
        greeks = BlackScholesCalculator.calculate_greeks(
            request.underlying_price,
            request.strike,
            request.time_to_expiry,
            request.risk_free_rate,
            request.volatility,
            request.option_type
        )
        
        return {
            "option_price": round(option_price, 4),
            "greeks": {
                "delta": round(greeks.delta, 4),
                "gamma": round(greeks.gamma, 4),
                "theta": round(greeks.theta, 4),
                "vega": round(greeks.vega, 4),
                "rho": round(greeks.rho, 4)
            },
            "inputs": request.dict(),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating option price: {str(e)}")

@app.post("/portfolio/options/add")
async def add_option_position(request: OptionPositionRequest):
    """Add an option position to the portfolio."""
    try:
        expiry_date = datetime.now() + timedelta(days=request.expiry_days)
        
        position = OptionPosition(
            symbol=request.symbol.upper(),
            option_type=request.option_type.lower(),
            strike=request.strike,
            expiry=expiry_date,
            quantity=request.quantity,
            premium=request.premium,
            underlying_price=request.underlying_price,
            risk_free_rate=request.risk_free_rate,
            volatility=request.volatility
        )
        
        global_portfolio.add_position(position)
        
        return {
            "message": "Option position added successfully",
            "position_id": len(global_portfolio.positions) - 1,
            "total_positions": len(global_portfolio.positions),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding position: {str(e)}")

@app.get("/portfolio/options/positions")
async def get_option_positions():
    """Get all option positions in the portfolio."""
    try:
        positions = []
        for i, position in enumerate(global_portfolio.positions):
            # Calculate current option value and Greeks
            time_to_expiry = (position.expiry - datetime.now()).days / 365.0
            
            current_price = BlackScholesCalculator.option_price(
                position.underlying_price, position.strike, time_to_expiry,
                position.risk_free_rate, position.volatility, position.option_type
            )
            
            greeks = BlackScholesCalculator.calculate_greeks(
                position.underlying_price, position.strike, time_to_expiry,
                position.risk_free_rate, position.volatility, position.option_type
            )
            
            positions.append({
                "position_id": i,
                "symbol": position.symbol,
                "option_type": position.option_type,
                "strike": position.strike,
                "expiry": position.expiry.isoformat(),
                "quantity": position.quantity,
                "premium_paid": position.premium,
                "current_price": round(current_price, 4),
                "underlying_price": position.underlying_price,
                "volatility": position.volatility,
                "greeks": {
                    "delta": round(greeks.delta, 4),
                    "gamma": round(greeks.gamma, 4),
                    "theta": round(greeks.theta, 4),
                    "vega": round(greeks.vega, 4),
                    "rho": round(greeks.rho, 4)
                },
                "pnl": round((current_price - position.premium) * position.quantity, 2)
            })
        
        return {
            "positions": positions,
            "total_positions": len(positions),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching positions: {str(e)}")

@app.delete("/portfolio/options/positions/{position_id}")
async def remove_option_position(position_id: int):
    """Remove an option position from the portfolio."""
    try:
        if 0 <= position_id < len(global_portfolio.positions):
            removed_position = global_portfolio.positions[position_id]
            global_portfolio.remove_position(position_id)
            return {
                "message": "Position removed successfully",
                "removed_position": {
                    "symbol": removed_position.symbol,
                    "option_type": removed_position.option_type,
                    "strike": removed_position.strike,
                    "quantity": removed_position.quantity
                },
                "remaining_positions": len(global_portfolio.positions),
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=404, detail="Position not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error removing position: {str(e)}")

@app.get("/portfolio/risk/dashboard")
async def get_risk_dashboard():
    """Get comprehensive risk dashboard for the options portfolio."""
    try:
        if not global_portfolio.positions:
            return {
                "message": "No positions in portfolio",
                "portfolio_summary": {
                    "total_positions": 0,
                    "portfolio_value": 0,
                    "portfolio_var_95": 0,
                    "portfolio_greeks": {
                        "portfolio_delta": 0,
                        "portfolio_gamma": 0,
                        "portfolio_theta": 0,
                        "portfolio_vega": 0,
                        "portfolio_rho": 0
                    }
                }
            }
        
        report_generator = RiskReportGenerator(global_portfolio, correlation_analyzer)
        risk_dashboard = report_generator.generate_risk_dashboard()
        
        # Format the response for JSON serialization
        formatted_dashboard = {
            "portfolio_summary": {
                "total_positions": risk_dashboard['portfolio_summary']['total_positions'],
                "portfolio_value": round(risk_dashboard['portfolio_summary']['portfolio_value'], 2),
                "portfolio_var_95": round(risk_dashboard['portfolio_summary']['portfolio_var_95'], 2),
                "portfolio_greeks": {
                    k: round(v, 4) for k, v in risk_dashboard['portfolio_summary']['portfolio_greeks'].items()
                }
            },
            "concentration_risk": risk_dashboard['concentration_risk'],
            "scenario_analysis": {
                scenario: {
                    "portfolio_value": round(data['portfolio_value'], 2),
                    "pnl": round(data['pnl'], 2),
                    "pnl_percentage": round(data['pnl_percentage'], 2)
                }
                for scenario, data in risk_dashboard['scenario_analysis'].items()
            },
            "risk_alerts": risk_dashboard['risk_alerts'],
            "timestamp": datetime.now().isoformat()
        }
        
        return formatted_dashboard
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating risk dashboard: {str(e)}")

@app.post("/analysis/correlation")
async def analyze_correlations(request: CorrelationRequest):
    """Analyze correlations between assets and generate correlation matrix."""
    try:
        correlation_matrix = correlation_analyzer.calculate_asset_correlations(
            request.symbols
        )
        
        # Convert to dictionary for JSON serialization
        correlation_dict = correlation_matrix.to_dict()
        
        return {
            "correlation_matrix": correlation_dict,
            "symbols": request.symbols,
            "period": request.period,
            "analysis_summary": {
                "highest_correlation": float(correlation_matrix.values[correlation_matrix.values < 1].max()),
                "lowest_correlation": float(correlation_matrix.values.min()),
                "average_correlation": float(correlation_matrix.values[correlation_matrix.values < 1].mean())
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing correlations: {str(e)}")

@app.get("/portfolio/greeks")
async def get_portfolio_greeks():
    """Get aggregated Greeks for the entire options portfolio."""
    try:
        if not global_portfolio.positions:
            return {
                "message": "No positions in portfolio",
                "portfolio_greeks": {
                    "portfolio_delta": 0,
                    "portfolio_gamma": 0,
                    "portfolio_theta": 0,
                    "portfolio_vega": 0,
                    "portfolio_rho": 0
                }
            }
        
        portfolio_greeks = global_portfolio.calculate_portfolio_greeks()
        
        return {
            "portfolio_greeks": {
                k: round(v, 4) for k, v in portfolio_greeks.items()
            },
            "risk_interpretation": {
                "delta": "Price sensitivity - Portfolio will gain/lose $X for every $1 move in underlying",
                "gamma": "Delta sensitivity - How much delta changes as underlying moves",
                "theta": "Time decay - Daily portfolio value loss due to time passage",
                "vega": "Volatility sensitivity - Portfolio gain/loss for 1% volatility change",
                "rho": "Interest rate sensitivity - Portfolio change for 1% rate change"
            },
            "total_positions": len(global_portfolio.positions),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating portfolio Greeks: {str(e)}")

@app.get("/news")
async def get_news():
    """Get financial news (placeholder implementation)."""
    return {
        "news": [
            {
                "title": "Options Trading Sees Increased Activity",
                "summary": "Market volatility drives options trading volumes higher...",
                "timestamp": datetime.now().isoformat(),
                "impact": "HIGH_VOLATILITY"
            },
            {
                "title": "Federal Reserve Policy Update",
                "summary": "Interest rate decisions affecting options pricing...",
                "timestamp": (datetime.now() - timedelta(hours=2)).isoformat(),
                "impact": "INTEREST_RATES"
            }
        ],
        "message": "News module will be implemented with full data pipeline"
    }

@app.post("/analyze-strategy")
async def analyze_strategy(request: StrategyRequest):
    """Analyze trading strategy (simplified implementation)."""
    try:
        ticker = yf.Ticker(request.symbol.upper())
        hist = ticker.history(period="3mo")
        
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No data found for symbol {request.symbol}")
        
        prices = hist['Close'].tolist()
        
        if request.strategy_type == "momentum":
            # Simple momentum calculation
            if len(prices) >= request.period:
                sma = sum(prices[-request.period:]) / request.period
                current_price = prices[-1]
                momentum_signal = "BUY" if current_price > sma else "SELL"
                strength = abs(current_price - sma) / sma * 100
            else:
                momentum_signal = "HOLD"
                strength = 0
                
            return {
                "symbol": request.symbol.upper(),
                "strategy": "momentum",
                "signal": momentum_signal,
                "strength": round(strength, 2),
                "current_price": round(prices[-1], 2),
                "sma": round(sma, 2) if len(prices) >= request.period else None,
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "symbol": request.symbol.upper(),
                "strategy": request.strategy_type,
                "signal": "HOLD",
                "message": "Strategy implementation in progress",
                "timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Strategy analysis error: {str(e)}")

@app.get("/status")
async def get_status():
    """Get system status."""
    return {
        "api_status": "operational",
        "database_status": "connected" if os.getenv("DATABASE_URL") else "not_configured",
        "redis_status": "connected" if os.getenv("REDIS_URL") else "not_configured",
        "portfolio_status": {
            "total_positions": len(global_portfolio.positions),
            "portfolio_value": round(global_portfolio.calculate_portfolio_value(), 2) if global_portfolio.positions else 0
        },
        "modules": {
            "data_collection": "operational",
            "options_trading": "operational",
            "risk_management": "operational",
            "correlation_analysis": "operational",
            "strategies": "basic_implemented",
            "backtesting": "development",
            "paper_trading": "development"
        },
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 