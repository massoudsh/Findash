from fastapi import FastAPI, HTTPException, Depends, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Dict, List, Optional
import asyncio
import os
import sys
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import uuid
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from monitoring.metrics import TradingMetrics
import time

# Add current directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import PostgreSQL repositories instead of SQLAlchemy
from database.repositories import (
    UserRepository, PortfolioRepository, OptionPositionRepository,
    MarketDataRepository, APIKeyRepository, AuditLogRepository
)
from database.postgres_connection import get_db

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

# Security
security = HTTPBearer()

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

class LoginRequest(BaseModel):
    email: str
    password: str

# Initialize repositories
user_repo = UserRepository()
portfolio_repo = PortfolioRepository()
position_repo = OptionPositionRepository()
market_data_repo = MarketDataRepository()
api_key_repo = APIKeyRepository()
audit_repo = AuditLogRepository()
correlation_analyzer = CorrelationAnalyzer()

# Authentication dependency
def get_current_user(request: Request, credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current user from JWT token or API key"""
    token = credentials.credentials
    
    # Try API key first
    api_key_info = api_key_repo.verify_api_key(token)
    if api_key_info:
        user = user_repo.get_user_by_id(api_key_info['user_id'])
        if user and user.is_active:
            # Log API usage
            audit_repo.log_action(
                user_id=user.id,
                action="api_access",
                ip_address=request.client.host,
                user_agent=request.headers.get("user-agent")
            )
            return user
    
    # For now, use demo user if no valid auth (in production, implement proper JWT)
    demo_user = user_repo.get_user_by_email("demo@quantumtrading.com")
    if demo_user:
        return demo_user
    
    raise HTTPException(status_code=401, detail="Invalid authentication")

# Initialize Prometheus instrumentation
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

@app.get("/health")
async def health_check():
    # Update some sample metrics
    TradingMetrics.set_active_connections(len(active_websocket_connections))
    return {"status": "healthy", "timestamp": time.time()}

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
        
        # Save market data to database
        market_data_repo.save_market_data(
            symbol=symbol.upper(),
            open_price=float(hist['Open'].iloc[-1]),
            high_price=float(hist['High'].iloc[-1]),
            low_price=float(hist['Low'].iloc[-1]),
            close_price=current_price,
            volume=int(hist['Volume'].iloc[-1]),
            timestamp=datetime.now()
        )
        
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
async def add_option_position(request: OptionPositionRequest, current_user=Depends(get_current_user)):
    """Add an option position to the portfolio."""
    try:
        # Get user's active portfolio or create one
        portfolios = portfolio_repo.get_user_portfolios(current_user.id)
        if not portfolios:
            portfolio = portfolio_repo.create_portfolio(
                user_id=current_user.id,
                name="Default Portfolio",
                description="Automatically created portfolio"
            )
        else:
            portfolio = portfolios[0]
        
        expiry_date = datetime.now() + timedelta(days=request.expiry_days)
        
        # Create position in database
        position = position_repo.create_position(
            user_id=current_user.id,
            portfolio_id=portfolio.id,
            symbol=request.symbol.upper(),
            option_type=request.option_type.lower(),
            strike_price=request.strike,
            expiry_date=expiry_date,
            quantity=request.quantity,
            premium_paid=request.premium,
            underlying_price=request.underlying_price,
            implied_volatility=request.volatility,
            risk_free_rate=request.risk_free_rate
        )
        
        # Calculate and update Greeks
        time_to_expiry = (expiry_date - datetime.now()).days / 365.0
        greeks = BlackScholesCalculator.calculate_greeks(
            request.underlying_price,
            request.strike,
            time_to_expiry,
            request.risk_free_rate,
            request.volatility,
            request.option_type
        )
        
        # Update position with Greeks
        position_repo.update_position_greeks(
            position.id,
            greeks.delta,
            greeks.gamma,
            greeks.theta,
            greeks.vega,
            greeks.rho
        )
        
        # Log the action
        audit_repo.log_action(
            user_id=current_user.id,
            action="create_option_position",
            resource_type="option_position",
            resource_id=position.id,
            new_values={
                "symbol": request.symbol,
                "option_type": request.option_type,
                "quantity": request.quantity,
                "strike": request.strike
            }
        )
        
        return {
            "message": "Option position added successfully",
            "position_id": position.id,
            "portfolio_id": portfolio.id,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding position: {str(e)}")

@app.get("/portfolio/options/positions")
async def get_option_positions(current_user=Depends(get_current_user)):
    """Get all option positions for the current user."""
    try:
        positions = position_repo.get_user_positions(current_user.id)
        
        position_data = []
        for pos in positions:
            # Calculate current values
            time_to_expiry = max(0, (pos.expiry_date - datetime.now()).days / 365.0)
            
            if time_to_expiry > 0:
                current_price = BlackScholesCalculator.option_price(
                    pos.underlying_price, pos.strike_price, time_to_expiry,
                    pos.risk_free_rate, pos.implied_volatility, pos.option_type
                )
                
                greeks = BlackScholesCalculator.calculate_greeks(
                    pos.underlying_price, pos.strike_price, time_to_expiry,
                    pos.risk_free_rate, pos.implied_volatility, pos.option_type
                )
            else:
                # Expired option
                if pos.option_type == 'call':
                    current_price = max(0, pos.underlying_price - pos.strike_price)
                else:
                    current_price = max(0, pos.strike_price - pos.underlying_price)
                greeks = Greeks(0, 0, 0, 0, 0)
            
            pnl = (current_price - pos.premium_paid) * pos.quantity
            
            position_data.append({
                "position_id": pos.id,
                "symbol": pos.symbol,
                "option_type": pos.option_type,
                "strike": pos.strike_price,
                "expiry": pos.expiry_date.isoformat(),
                "quantity": pos.quantity,
                "premium_paid": pos.premium_paid,
                "current_price": round(current_price, 4),
                "underlying_price": pos.underlying_price,
                "volatility": pos.implied_volatility,
                "greeks": {
                    "delta": round(greeks.delta, 4),
                    "gamma": round(greeks.gamma, 4),
                    "theta": round(greeks.theta, 4),
                    "vega": round(greeks.vega, 4),
                    "rho": round(greeks.rho, 4)
                },
                "pnl": round(pnl, 2),
                "order_status": pos.order_status
            })
        
        return {
            "positions": position_data,
            "total_positions": len(position_data),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching positions: {str(e)}")

@app.delete("/portfolio/options/positions/{position_id}")
async def remove_option_position(position_id: str, current_user=Depends(get_current_user)):
    """Remove an option position from the portfolio."""
    try:
        # Verify position belongs to user
        position = position_repo.get_position_by_id(position_id)
        if not position or position.user_id != current_user.id:
            raise HTTPException(status_code=404, detail="Position not found")
        
        # Log the action before deletion
        audit_repo.log_action(
            user_id=current_user.id,
            action="delete_option_position",
            resource_type="option_position",
            resource_id=position.id,
            old_values={
                "symbol": position.symbol,
                "option_type": position.option_type,
                "quantity": position.quantity
            }
        )
        
        # Delete position
        position_repo.delete_position(position_id)
        
        return {
            "message": "Position removed successfully",
            "removed_position": {
                "symbol": position.symbol,
                "option_type": position.option_type,
                "strike": position.strike_price,
                "quantity": position.quantity
            },
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error removing position: {str(e)}")

@app.get("/portfolio/risk/dashboard")
async def get_risk_dashboard(current_user=Depends(get_current_user)):
    """Get comprehensive risk dashboard for the user's options portfolio."""
    try:
        # Get user's positions
        positions = position_repo.get_user_positions(current_user.id)
        
        if not positions:
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
        
        # Convert database positions to OptionsPortfolio format
        portfolio = OptionsPortfolio()
        for pos in positions:
            opt_pos = OptionPosition(
                symbol=pos.symbol,
                option_type=pos.option_type,
                strike=pos.strike_price,
                expiry=pos.expiry_date,
                quantity=pos.quantity,
                premium=pos.premium_paid,
                underlying_price=pos.underlying_price,
                volatility=pos.implied_volatility,
                risk_free_rate=pos.risk_free_rate
            )
            portfolio.add_position(opt_pos)
        
        # Generate risk report
        report_generator = RiskReportGenerator(portfolio, correlation_analyzer)
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
async def get_portfolio_greeks(current_user=Depends(get_current_user)):
    """Get aggregated Greeks for the user's options portfolio."""
    try:
        positions = position_repo.get_user_positions(current_user.id)
        
        if not positions:
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
        
        # Calculate portfolio Greeks
        total_delta = sum(pos.delta * pos.quantity for pos in positions if pos.delta)
        total_gamma = sum(pos.gamma * pos.quantity for pos in positions if pos.gamma)
        total_theta = sum(pos.theta * pos.quantity for pos in positions if pos.theta)
        total_vega = sum(pos.vega * pos.quantity for pos in positions if pos.vega)
        total_rho = sum(pos.rho * pos.quantity for pos in positions if pos.rho)
        
        return {
            "portfolio_greeks": {
                "portfolio_delta": round(total_delta, 4),
                "portfolio_gamma": round(total_gamma, 4),
                "portfolio_theta": round(total_theta, 4),
                "portfolio_vega": round(total_vega, 4),
                "portfolio_rho": round(total_rho, 4)
            },
            "risk_interpretation": {
                "delta": "Price sensitivity - Portfolio will gain/lose $X for every $1 move in underlying",
                "gamma": "Delta sensitivity - How much delta changes as underlying moves",
                "theta": "Time decay - Daily portfolio value loss due to time passage",
                "vega": "Volatility sensitivity - Portfolio gain/loss for 1% volatility change",
                "rho": "Interest rate sensitivity - Portfolio change for 1% rate change"
            },
            "total_positions": len(positions),
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
    try:
        # Test database connection
        db = get_db()
        db.execute_query("SELECT 1", fetch='one')
        db_status = "connected"
    except:
        db_status = "disconnected"
    
    return {
        "api_status": "operational",
        "database_status": db_status,
        "redis_status": "connected" if os.getenv("REDIS_URL") else "not_configured",
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

@app.get("/metrics")
async def get_metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.on_event("startup")
async def startup_event():
    # Start background task to update metrics
    asyncio.create_task(update_metrics_periodically())

async def update_metrics_periodically():
    while True:
        try:
            # Update portfolio values
            portfolios = await get_all_portfolios()
            for portfolio in portfolios:
                TradingMetrics.update_portfolio_value(
                    portfolio_id=str(portfolio.id),
                    portfolio_name=portfolio.name,
                    value=portfolio.current_value
                )
            
            # Update strategy performances
            strategies = await get_all_strategies()
            for strategy in strategies:
                performance = await calculate_strategy_performance(strategy.id)
                TradingMetrics.update_strategy_performance(
                    strategy_id=str(strategy.id),
                    strategy_name=strategy.name,
                    performance=performance
                )
            
            # Update risk metrics
            risk_data = await calculate_portfolio_risks()
            for portfolio_id, risks in risk_data.items():
                TradingMetrics.update_risk_metric("var", portfolio_id, risks.get("var", 0))
                TradingMetrics.update_risk_metric("sharpe_ratio", portfolio_id, risks.get("sharpe_ratio", 0))
            
            # Update social sentiment
            sentiment_data = await get_social_sentiment()
            for symbol, platforms in sentiment_data.items():
                for platform, score in platforms.items():
                    TradingMetrics.update_social_sentiment(symbol, platform, score)
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
        
        await asyncio.sleep(30)  # Update every 30 seconds

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 