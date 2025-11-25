"""
Market Scenario API Endpoints
Scenario matching, analysis, and recommendations
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, Query, Body
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.database.postgres_connection import get_db
from src.core.config import get_settings
from src.core.security import get_current_active_user, TokenData
from src.core.rate_limiter import standard_rate_limit
from src.core.cache import CacheManager, CacheNamespace

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter(prefix="/api/scenarios", tags=["Market Scenarios"])

# Load scenarios from JSON file
SCENARIOS_FILE = Path(__file__).parent.parent.parent.parent / "dataset" / "rag_scenarios_100.json"

def load_scenarios():
    """Load scenarios from JSON file"""
    try:
        with open(SCENARIOS_FILE, 'r') as f:
            data = json.load(f)
            return data.get('scenarios', [])
    except FileNotFoundError:
        logger.warning(f"Scenarios file not found: {SCENARIOS_FILE}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing scenarios JSON: {e}")
        return []

# Cache scenarios in memory
_scenarios_cache = None
_cache_timestamp = None

def get_scenarios():
    """Get scenarios with caching"""
    global _scenarios_cache, _cache_timestamp
    
    # Reload if cache is older than 1 hour
    if _scenarios_cache is None or \
       _cache_timestamp is None or \
       (datetime.utcnow() - _cache_timestamp).total_seconds() > 3600:
        _scenarios_cache = load_scenarios()
        _cache_timestamp = datetime.utcnow()
    
    return _scenarios_cache

# Pydantic models
class ScenarioMatchRequest(BaseModel):
    """Request for scenario matching"""
    price_action: Optional[str] = None
    volatility: Optional[str] = None
    funding_rate: Optional[str] = None
    social_score: Optional[float] = None
    etf_flows: Optional[str] = None
    regime: Optional[str] = None

class ScenarioMatch(BaseModel):
    """Matched scenario with confidence score"""
    scenario_id: str
    title: str
    regime: str
    confidence: float
    matched_fields: List[str]
    scenario: Dict[str, Any]

class ScenarioResponse(BaseModel):
    """Scenario response"""
    scenario_id: str
    title: str
    regime: str
    macro: Dict[str, str]
    crypto: Dict[str, str]
    microstructure: Dict[str, str]
    sentiment: Dict[str, str]
    flows: Dict[str, str]
    geopolitics: Dict[str, str]
    recommended_actions: List[str]
    regime_label: str

@router.get("/", response_model=List[ScenarioResponse])
async def get_scenarios(
    regime: Optional[str] = Query(None, description="Filter by regime"),
    impact: Optional[str] = Query(None, description="Filter by geopolitical impact"),
    limit: int = Query(100, ge=1, le=100, description="Number of scenarios"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    db: Session = Depends(get_db)
):
    """
    Get all scenarios with optional filtering
    """
    try:
        scenarios = get_scenarios()
        
        # Apply filters
        if regime:
            scenarios = [s for s in scenarios if s.get('regime') == regime]
        
        if impact:
            scenarios = [s for s in scenarios if s.get('geopolitics', {}).get('impact', '').lower() == impact.lower()]
        
        # Apply pagination
        paginated = scenarios[offset:offset + limit]
        
        return [ScenarioResponse(**s) for s in paginated]
    except Exception as e:
        logger.error(f"Error fetching scenarios: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{scenario_id}", response_model=ScenarioResponse)
async def get_scenario(
    scenario_id: str,
    db: Session = Depends(get_db)
):
    """
    Get a specific scenario by ID
    """
    try:
        scenarios = get_scenarios()
        scenario = next((s for s in scenarios if s.get('scenario_id') == scenario_id), None)
        
        if not scenario:
            raise HTTPException(status_code=404, detail=f"Scenario {scenario_id} not found")
        
        return ScenarioResponse(**scenario)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching scenario: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/match", response_model=List[ScenarioMatch])
async def match_scenarios(
    request: ScenarioMatchRequest,
    limit: int = Query(5, ge=1, le=20, description="Number of matches to return"),
    db: Session = Depends(get_db)
):
    """
    Match current market conditions to scenarios
    """
    try:
        scenarios = get_scenarios()
        matches = []
        
        for scenario in scenarios:
            matched_fields = []
            confidence = 0.0
            
            # Match on regime
            if request.regime and scenario.get('regime') == request.regime:
                matched_fields.append('regime')
                confidence += 0.3
            
            # Match on crypto metrics
            crypto = scenario.get('crypto', {})
            if request.price_action and request.price_action.lower() in str(crypto.get('price_action', '')).lower():
                matched_fields.append('price_action')
                confidence += 0.2
            
            if request.volatility and request.volatility.lower() == crypto.get('volatility', '').lower():
                matched_fields.append('volatility')
                confidence += 0.15
            
            if request.funding_rate and request.funding_rate.lower() in str(crypto.get('funding_rate', '')).lower():
                matched_fields.append('funding_rate')
                confidence += 0.15
            
            # Match on sentiment
            if request.social_score is not None:
                scenario_score = float(scenario.get('sentiment', {}).get('social_score', 0))
                score_diff = abs(request.social_score - scenario_score)
                if score_diff < 0.2:
                    matched_fields.append('social_score')
                    confidence += 0.1 - (score_diff * 0.5)
            
            # Match on flows
            if request.etf_flows and request.etf_flows.lower() in str(scenario.get('flows', {}).get('etf_flows', '')).lower():
                matched_fields.append('etf_flows')
                confidence += 0.1
            
            if confidence > 0:
                matches.append({
                    'scenario_id': scenario.get('scenario_id'),
                    'title': scenario.get('title'),
                    'regime': scenario.get('regime'),
                    'confidence': min(confidence, 1.0),
                    'matched_fields': matched_fields,
                    'scenario': scenario
                })
        
        # Sort by confidence and return top matches
        matches.sort(key=lambda x: x['confidence'], reverse=True)
        top_matches = matches[:limit]
        
        return [ScenarioMatch(**m) for m in top_matches]
    except Exception as e:
        logger.error(f"Error matching scenarios: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/regimes/list")
async def get_regimes(
    db: Session = Depends(get_db)
):
    """
    Get list of all unique regimes
    """
    try:
        scenarios = get_scenarios()
        regimes = sorted(list(set(s.get('regime') for s in scenarios if s.get('regime'))))
        return {"regimes": regimes, "count": len(regimes)}
    except Exception as e:
        logger.error(f"Error fetching regimes: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats/summary")
async def get_scenario_stats(
    db: Session = Depends(get_db)
):
    """
    Get statistics about scenarios
    """
    try:
        scenarios = get_scenarios()
        
        # Count by regime
        regime_counts = {}
        for scenario in scenarios:
            regime = scenario.get('regime', 'Unknown')
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        # Count by impact
        impact_counts = {}
        for scenario in scenarios:
            impact = scenario.get('geopolitics', {}).get('impact', 'Unknown')
            impact_counts[impact] = impact_counts.get(impact, 0) + 1
        
        return {
            "total_scenarios": len(scenarios),
            "regime_distribution": regime_counts,
            "impact_distribution": impact_counts,
            "last_updated": _cache_timestamp.isoformat() if _cache_timestamp else None
        }
    except Exception as e:
        logger.error(f"Error fetching scenario stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

