"""
Agent Monitoring API Endpoints - Production Version
With Database Persistence, JWT Auth, Rate Limiting, and Caching
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import desc

from src.database.postgres_connection import get_db
from src.database.models import AgentStatus, AgentLog, AgentDecision
from src.core.config import get_settings
from src.core.security import get_current_active_user, TokenData
from src.core.rate_limiter import standard_rate_limit
from src.core.cache import CacheManager, CacheNamespace

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter(prefix="/api/agents", tags=["Agent Monitoring"])

# Initialize cache
cache_manager = CacheManager()

# Pydantic response models
class AgentStatusResponse(BaseModel):
    id: str
    name: str
    status: str
    uptime: int
    tasks_completed: int
    success_rate: float
    avg_execution_time: float
    last_activity: str
    current_task: Optional[str] = None

class AgentLogResponse(BaseModel):
    id: str
    agent_id: str
    agent_name: str
    timestamp: str
    level: str
    message: str
    execution_time: Optional[float] = None

class AgentDecisionResponse(BaseModel):
    id: str
    agent_id: str
    agent_name: str
    symbol: str
    action: str
    confidence: float
    reasoning: List[str]
    timestamp: str
    executed: bool
    result: Optional[Dict[str, Any]] = None

@router.get("/status")
async def get_agent_status(
    agent_id: Optional[str] = Query(None),
    current_user: TokenData = Depends(get_current_active_user),
    rate_limit: bool = Depends(standard_rate_limit),
    db: Session = Depends(get_db)
):
    """Get status of all agents or a specific agent - Requires authentication"""
    cache_key = f"agent_status:{agent_id or 'all'}:{current_user.user_id}"
    
    # Check cache first
    cached = await cache_manager.get(cache_key, namespace=CacheNamespace.API_RESPONSES)
    if cached:
        return cached
    
    try:
        query = db.query(AgentStatus)
        if agent_id:
            query = query.filter(AgentStatus.id == agent_id)
        
        agents = query.order_by(AgentStatus.last_activity.desc()).all()
        
        if not agents:
            # Return empty list or create default agents
            result = {"agents": [], "total_agents": 0, "active_count": 0}
        else:
            result = {
                "agents": [
                    {
                        "id": agent.id,
                        "name": agent.name,
                        "status": agent.status,
                        "uptime": agent.uptime,
                        "tasks_completed": agent.tasks_completed,
                        "success_rate": float(agent.success_rate),
                        "avg_execution_time": float(agent.avg_execution_time),
                        "last_activity": agent.last_activity.isoformat() if agent.last_activity else datetime.utcnow().isoformat(),
                        "current_task": agent.current_task
                    }
                    for agent in agents
                ],
                "total_agents": len(agents),
                "active_count": len([a for a in agents if a.status == 'active'])
            }
        
        # Cache for 30 seconds
        await cache_manager.set(cache_key, result, ttl=30, namespace=CacheNamespace.API_RESPONSES)
        return result
        
    except Exception as e:
        logger.error(f"Error fetching agent status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/logs")
async def get_agent_logs(
    agent_id: Optional[str] = Query(None),
    level: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    current_user: TokenData = Depends(get_current_active_user),
    rate_limit: bool = Depends(standard_rate_limit),
    db: Session = Depends(get_db)
):
    """Get agent activity logs with filtering - Requires authentication"""
    cache_key = f"agent_logs:{agent_id}:{level}:{limit}:{offset}:{current_user.user_id}"
    
    cached = await cache_manager.get(cache_key, namespace=CacheNamespace.API_RESPONSES)
    if cached:
        return cached
    
    try:
        query = db.query(AgentLog).join(AgentStatus, AgentLog.agent_id == AgentStatus.id)
        
        if agent_id:
            query = query.filter(AgentLog.agent_id == agent_id)
        if level:
            query = query.filter(AgentLog.level == level)
        
        total = query.count()
        logs = query.order_by(desc(AgentLog.timestamp)).offset(offset).limit(limit).all()
        
        result = {
            "logs": [
                {
                    "id": log.id,
                    "agent_id": log.agent_id,
                    "agent_name": log.agent.name if log.agent else "Unknown",
                    "timestamp": log.timestamp.isoformat() if log.timestamp else datetime.utcnow().isoformat(),
                    "level": log.level,
                    "message": log.message,
                    "execution_time": float(log.execution_time) if log.execution_time else None
                }
                for log in logs
            ],
            "total": total,
            "page": offset // limit + 1,
            "page_size": limit
        }
        
        # Cache for 10 seconds (logs change frequently)
        await cache_manager.set(cache_key, result, ttl=10, namespace=CacheNamespace.API_RESPONSES)
        return result
        
    except Exception as e:
        logger.error(f"Error fetching agent logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/decisions")
async def get_agent_decisions(
    agent_id: Optional[str] = Query(None),
    symbol: Optional[str] = Query(None),
    limit: int = Query(30, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: TokenData = Depends(get_current_active_user),
    rate_limit: bool = Depends(standard_rate_limit),
    db: Session = Depends(get_db)
):
    """Get trading decisions made by agents - Requires authentication"""
    cache_key = f"agent_decisions:{agent_id}:{symbol}:{limit}:{offset}:{current_user.user_id}"
    
    cached = await cache_manager.get(cache_key, namespace=CacheNamespace.API_RESPONSES)
    if cached:
        return cached
    
    try:
        query = db.query(AgentDecision).join(AgentStatus, AgentDecision.agent_id == AgentStatus.id)
        
        if agent_id:
            query = query.filter(AgentDecision.agent_id == agent_id)
        if symbol:
            query = query.filter(AgentDecision.symbol == symbol)
        
        total = query.count()
        decisions = query.order_by(desc(AgentDecision.timestamp)).offset(offset).limit(limit).all()
        
        result = {
            "decisions": [
                {
                    "id": decision.id,
                    "agent_id": decision.agent_id,
                    "agent_name": decision.agent.name if decision.agent else "Unknown",
                    "symbol": decision.symbol,
                    "action": decision.action,
                    "confidence": float(decision.confidence),
                    "reasoning": decision.reasoning if isinstance(decision.reasoning, list) else [],
                    "timestamp": decision.timestamp.isoformat() if decision.timestamp else datetime.utcnow().isoformat(),
                    "executed": decision.executed,
                    "result": decision.result if isinstance(decision.result, dict) else None
                }
                for decision in decisions
            ],
            "total": total,
            "page": offset // limit + 1,
            "page_size": limit
        }
        
        # Cache for 15 seconds
        await cache_manager.set(cache_key, result, ttl=15, namespace=CacheNamespace.API_RESPONSES)
        return result
        
    except Exception as e:
        logger.error(f"Error fetching agent decisions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance")
async def get_agent_performance(
    agent_id: Optional[str] = Query(None),
    current_user: TokenData = Depends(get_current_active_user),
    rate_limit: bool = Depends(standard_rate_limit),
    db: Session = Depends(get_db)
):
    """Get performance metrics for agents - Requires authentication"""
    cache_key = f"agent_performance:{agent_id}:{current_user.user_id}"
    
    cached = await cache_manager.get(cache_key, namespace=CacheNamespace.API_RESPONSES)
    if cached:
        return cached
    
    try:
        query = db.query(AgentStatus)
        if agent_id:
            query = query.filter(AgentStatus.id == agent_id)
        
        agents = query.all()
        
        if not agents:
            result = {"overall": {}, "by_agent": {}}
        else:
            total_tasks = sum(a.tasks_completed for a in agents)
            total_successes = sum(a.tasks_completed * (a.success_rate / 100) for a in agents)
            avg_success_rate = (total_successes / total_tasks * 100) if total_tasks > 0 else 0
            avg_execution_time = sum(a.avg_execution_time for a in agents) / len(agents) if agents else 0
            
            result = {
                "overall": {
                    "total_tasks": total_tasks,
                    "success_rate": round(avg_success_rate, 2),
                    "avg_execution_time": round(avg_execution_time, 2),
                    "total_agents": len(agents),
                    "active_agents": len([a for a in agents if a.status == 'active'])
                },
                "by_agent": {
                    agent.id: {
                        "name": agent.name,
                        "tasks_completed": agent.tasks_completed,
                        "success_rate": float(agent.success_rate),
                        "avg_execution_time": float(agent.avg_execution_time)
                    }
                    for agent in agents
                }
            }
        
        # Cache for 60 seconds
        await cache_manager.set(cache_key, result, ttl=60, namespace=CacheNamespace.API_RESPONSES)
        return result
        
    except Exception as e:
        logger.error(f"Error fetching agent performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

