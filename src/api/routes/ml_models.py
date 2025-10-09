"""
Machine Learning Models API Routes for FastAPI service
Handles ML model inference, predictions, and AI-powered analysis
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd

router = APIRouter()

# Pydantic models
class PredictionRequest(BaseModel):
    symbol: str
    model_type: str = Field(..., description="prophet, lstm, transformer, ensemble")
    horizon: int = Field(30, description="Prediction horizon in days")
    features: Optional[Dict[str, Any]] = None

class PredictionResponse(BaseModel):
    symbol: str
    model_type: str
    predictions: List[Dict[str, Union[str, float]]]
    confidence_intervals: Optional[List[Dict[str, float]]] = None
    model_performance: Dict[str, float]
    generated_at: str

class ModelInfo(BaseModel):
    model_id: str
    name: str
    type: str
    status: str  # "active", "training", "inactive"
    accuracy: Optional[float] = None
    last_trained: Optional[str] = None
    supported_symbols: List[str]

class TrainingRequest(BaseModel):
    model_type: str
    symbol: str
    start_date: str
    end_date: str
    hyperparameters: Optional[Dict[str, Any]] = None

class SentimentAnalysisRequest(BaseModel):
    text: Optional[str] = None
    symbol: Optional[str] = None
    sources: List[str] = Field(default=["news", "social", "analyst_reports"])

class SentimentResponse(BaseModel):
    symbol: str
    overall_sentiment: float  # -1 to 1
    sentiment_breakdown: Dict[str, float]
    confidence: float
    sources_analyzed: int
    timestamp: str

@router.get("/models", response_model=List[ModelInfo])
async def list_available_models():
    """List all available ML models"""
    return [
        ModelInfo(
            model_id="prophet_v1",
            name="Prophet Time Series Forecaster",
            type="prophet",
            status="active",
            accuracy=0.85,
            last_trained="2024-11-15T10:30:00Z",
            supported_symbols=["AAPL", "TSLA", "MSFT", "GOOGL"]
        ),
        ModelInfo(
            model_id="lstm_v2",
            name="LSTM Neural Network",
            type="lstm",
            status="active",
            accuracy=0.78,
            last_trained="2024-11-20T14:15:00Z",
            supported_symbols=["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN"]
        ),
        ModelInfo(
            model_id="transformer_v1",
            name="Transformer Price Predictor",
            type="transformer",
            status="training",
            accuracy=None,
            last_trained=None,
            supported_symbols=["AAPL", "TSLA"]
        ),
        ModelInfo(
            model_id="ensemble_v1",
            name="Ensemble Model",
            type="ensemble",
            status="active",
            accuracy=0.89,
            last_trained="2024-11-25T09:00:00Z",
            supported_symbols=["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "NVDA"]
        )
    ]

@router.post("/predict", response_model=PredictionResponse)
async def generate_prediction(request: PredictionRequest):
    """Generate price predictions using specified ML model"""
    
    # Validate model type
    valid_models = ["prophet", "lstm", "transformer", "ensemble"]
    if request.model_type not in valid_models:
        raise HTTPException(status_code=400, detail=f"Invalid model type. Must be one of: {valid_models}")
    
    # Validate horizon
    if request.horizon < 1 or request.horizon > 365:
        raise HTTPException(status_code=400, detail="Horizon must be between 1 and 365 days")
    
    try:
        # In production, this would call actual ML models
        # For now, generate realistic sample predictions
        base_price = 150.0  # Sample base price
        predictions = []
        
        for i in range(request.horizon):
            date = datetime.utcnow() + timedelta(days=i+1)
            # Simple random walk with trend
            price_change = np.random.normal(0.002, 0.02)  # 0.2% drift, 2% volatility
            if i == 0:
                price = base_price * (1 + price_change)
            else:
                price = predictions[-1]["price"] * (1 + price_change)
            
            predictions.append({
                "date": date.strftime("%Y-%m-%d"),
                "price": round(price, 2),
                "confidence": round(np.random.uniform(0.7, 0.95), 2)
            })
        
        # Generate confidence intervals
        confidence_intervals = []
        for pred in predictions:
            lower = pred["price"] * 0.95
            upper = pred["price"] * 1.05
            confidence_intervals.append({
                "date": pred["date"],
                "lower": round(lower, 2),
                "upper": round(upper, 2)
            })
        
        # Model performance metrics
        performance = {
            "mape": 8.5,  # Mean Absolute Percentage Error
            "rmse": 5.2,  # Root Mean Square Error
            "r_squared": 0.85,
            "sharpe_ratio": 1.2
        }
        
        return PredictionResponse(
            symbol=request.symbol,
            model_type=request.model_type,
            predictions=predictions,
            confidence_intervals=confidence_intervals,
            model_performance=performance,
            generated_at=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction generation failed: {str(e)}")

@router.post("/sentiment", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentAnalysisRequest):
    """Analyze market sentiment for a symbol"""
    
    if not request.symbol and not request.text:
        raise HTTPException(status_code=400, detail="Either symbol or text must be provided")
    
    try:
        # In production, this would use real sentiment analysis models
        # For now, generate realistic sample sentiment
        
        # Simulate sentiment analysis
        overall_sentiment = np.random.uniform(-0.3, 0.7)  # Slightly bullish bias
        
        sentiment_breakdown = {
            "news": np.random.uniform(-0.5, 0.8),
            "social_media": np.random.uniform(-0.6, 0.9),
            "analyst_reports": np.random.uniform(-0.2, 0.6),
            "earnings_calls": np.random.uniform(-0.3, 0.5)
        }
        
        confidence = np.random.uniform(0.6, 0.9)
        sources_analyzed = np.random.randint(50, 200)
        
        return SentimentResponse(
            symbol=request.symbol or "UNKNOWN",
            overall_sentiment=round(overall_sentiment, 3),
            sentiment_breakdown={k: round(v, 3) for k, v in sentiment_breakdown.items()},
            confidence=round(confidence, 3),
            sources_analyzed=sources_analyzed,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {str(e)}")

@router.post("/train", response_model=Dict[str, str])
async def start_model_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Start training a new ML model"""
    
    # Validate dates
    try:
        start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(request.end_date, "%Y-%m-%d")
        
        if start_date >= end_date:
            raise HTTPException(status_code=400, detail="End date must be after start date")
        
        if (end_date - start_date).days < 30:
            raise HTTPException(status_code=400, detail="Training period must be at least 30 days")
            
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    
    # Generate training job ID
    job_id = f"training_{request.model_type}_{request.symbol}_{int(datetime.utcnow().timestamp())}"
    
    # In production, start actual training process
    # background_tasks.add_task(train_model_async, request, job_id)
    
    return {
        "job_id": job_id,
        "status": "training_started",
        "estimated_completion": (datetime.utcnow() + timedelta(hours=2)).isoformat(),
        "message": f"Started training {request.model_type} model for {request.symbol}"
    }

@router.get("/training-status/{job_id}", response_model=Dict[str, Any])
async def get_training_status(job_id: str):
    """Get status of a training job"""
    
    # In production, check actual training status
    # For now, simulate different states
    import random
    
    statuses = ["training", "completed", "failed"]
    status = random.choice(statuses)
    
    if status == "training":
        progress = random.randint(10, 90)
        return {
            "job_id": job_id,
            "status": "training",
            "progress": progress,
            "estimated_completion": (datetime.utcnow() + timedelta(minutes=30)).isoformat(),
            "current_epoch": random.randint(5, 50),
            "total_epochs": 100
        }
    elif status == "completed":
        return {
            "job_id": job_id,
            "status": "completed",
            "progress": 100,
            "completed_at": datetime.utcnow().isoformat(),
            "model_accuracy": round(random.uniform(0.75, 0.95), 3),
            "model_id": f"model_{job_id.split('_')[-1]}"
        }
    else:
        return {
            "job_id": job_id,
            "status": "failed",
            "progress": random.randint(10, 60),
            "error": "Training failed due to insufficient data quality",
            "failed_at": datetime.utcnow().isoformat()
        }

@router.get("/performance/{model_id}", response_model=Dict[str, Any])
async def get_model_performance(model_id: str):
    """Get detailed performance metrics for a model"""
    
    return {
        "model_id": model_id,
        "metrics": {
            "accuracy": 0.85,
            "precision": 0.83,
            "recall": 0.87,
            "f1_score": 0.85,
            "mape": 8.5,
            "rmse": 5.2,
            "r_squared": 0.85,
            "sharpe_ratio": 1.2,
            "max_drawdown": -0.15
        },
        "backtesting_results": {
            "total_return": 0.23,
            "annual_return": 0.18,
            "volatility": 0.16,
            "win_rate": 0.58,
            "profit_factor": 1.35
        },
        "last_evaluated": datetime.utcnow().isoformat(),
        "evaluation_period": "2024-01-01 to 2024-11-30"
    }

@router.get("/health", response_model=Dict[str, Any])
async def ml_health():
    """Health check for ML service"""
    return {
        "status": "healthy",
        "service": "ml_models",
        "timestamp": datetime.utcnow().isoformat(),
        "active_models": 4,
        "training_jobs": 1,
        "gpu_available": False,  # Update based on actual hardware
        "memory_usage": "45%",
        "model_versions": {
            "prophet": "1.1.4",
            "sklearn": "1.3.0",
            "torch": "2.0.0"
        }
    } 