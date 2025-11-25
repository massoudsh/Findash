"""
Agent Monitoring API Endpoints
Real-time agent status, logs, and decision tracking
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.database.postgres_connection import get_db
from src.database.models import AgentStatus, AgentLog, AgentDecision
from src.core.config import get_settings
from src.core.security import get_current_active_user, TokenData
from src.core.rate_limiter import standard_rate_limit
from src.core.cache import CacheManager, CacheNamespace
try:
    from src.core.intelligence_orchestrator import IntelligenceOrchestrator
    orchestrator = IntelligenceOrchestrator()
except ImportError:
    orchestrator = None
    logger.warning("IntelligenceOrchestrator not available")

try:
    from src.strategies.strategy_agent import StrategyAgent
    strategy_agent = StrategyAgent(cache) if orchestrator else None
except ImportError:
    strategy_agent = None
    logger.warning("StrategyAgent not available")

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter(prefix="/api/agents", tags=["Agent Monitoring"])

# Initialize cache manager
cache_manager = CacheManager()

# Pydantic models
class AgentStatus(BaseModel):
    id: str
    name: str
    status: str  # 'active', 'idle', 'error', 'training'
    uptime: int
    tasks_completed: int
    success_rate: float
    avg_execution_time: float
    last_activity: str
    current_task: Optional[str] = None

class AgentLog(BaseModel):
    id: str
    agent_id: str
    agent_name: str
    timestamp: str
    level: str  # 'info', 'warning', 'error', 'success'
    message: str
    context: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None

class AgentDecision(BaseModel):
    id: str
    agent_id: str
    agent_name: str
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    confidence: float
    reasoning: List[str]
    timestamp: str
    executed: bool
    result: Optional[Dict[str, Any]] = None

class AgentStatusResponse(BaseModel):
    agents: List[AgentStatus]
    total_agents: int
    active_count: int
    timestamp: str

class AgentLogsResponse(BaseModel):
    logs: List[AgentLog]
    total: int
    page: int
    page_size: int

class AgentDecisionsResponse(BaseModel):
    decisions: List[AgentDecision]
    total: int
    page: int
    page_size: int

# Agent name mapping
AGENT_NAMES = {
    'M1': 'Market Data Agent',
    'M2': 'Sentiment Analysis Agent',
    'M3': 'Risk Management Agent',
    'M4': 'Strategy Agent',
    'M5': 'Deep Learning Agent',
    'M6': 'Prediction Agent',
    'M7': 'Execution Agent',
    'M8': 'Portfolio Agent',
    'M9': 'Compliance Agent',
    'M10': 'Backtesting Agent',
    'M11': 'Optimization Agent'
}

@router.get("/status", response_model=AgentStatusResponse)
async def get_agent_status(
    agent_id: Optional[str] = Query(None, description="Filter by specific agent ID"),
    current_user: TokenData = Depends(get_current_active_user),
    rate_limit: bool = Depends(standard_rate_limit),
    db: Session = Depends(get_db)
):
    """
    Get status of all agents or a specific agent - Requires authentication
    """
    try:
        # In production, fetch from database/cache
        # For now, simulate agent status from orchestrator
        agents = []
        
        # Get agent status from orchestrator (if available)
        agent_tasks = await orchestrator.get_active_tasks() if orchestrator else []
        
        # Mock agent statuses - in production, get from actual agent registry
        mock_agents = [
            {
                'id': 'M1',
                'name': AGENT_NAMES['M1'],
                'status': 'active',
                'uptime': 86400,
                'tasks_completed': 1250,
                'success_rate': 98.5,
                'avg_execution_time': 45,
                'last_activity': datetime.utcnow().isoformat(),
                'current_task': 'Processing market data stream'
            },
            {
                'id': 'M2',
                'name': AGENT_NAMES['M2'],
                'status': 'active',
                'uptime': 86400,
                'tasks_completed': 890,
                'success_rate': 95.2,
                'avg_execution_time': 120,
                'last_activity': datetime.utcnow().isoformat(),
                'current_task': 'Analyzing social sentiment'
            },
            {
                'id': 'M4',
                'name': AGENT_NAMES['M4'],
                'status': 'active',
                'uptime': 86400,
                'tasks_completed': 450,
                'success_rate': 92.8,
                'avg_execution_time': 200,
                'last_activity': datetime.utcnow().isoformat(),
                'current_task': 'Generating trading decision'
            },
            {
                'id': 'M5',
                'name': AGENT_NAMES['M5'],
                'status': 'training',
                'uptime': 7200,
                'tasks_completed': 120,
                'success_rate': 88.5,
                'avg_execution_time': 5000,
                'last_activity': datetime.utcnow().isoformat(),
                'current_task': 'Training TCN model'
            },
            {
                'id': 'M7',
                'name': AGENT_NAMES['M7'],
                'status': 'active',
                'uptime': 86400,
                'tasks_completed': 320,
                'success_rate': 99.1,
                'avg_execution_time': 80,
                'last_activity': datetime.utcnow().isoformat(),
                'current_task': 'Executing order'
            }
        ]
        
        if agent_id:
            agents = [a for a in mock_agents if a['id'] == agent_id]
        else:
            agents = mock_agents
        
        return AgentStatusResponse(
            agents=[AgentStatus(**agent) for agent in agents],
            total_agents=len(agents),
            active_count=len([a for a in agents if a['status'] == 'active']),
            timestamp=datetime.utcnow().isoformat()
        )
    except Exception as e:
        logger.error(f"Error fetching agent status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/logs", response_model=AgentLogsResponse)
async def get_agent_logs(
    agent_id: Optional[str] = Query(None, description="Filter by agent ID"),
    level: Optional[str] = Query(None, description="Filter by log level"),
    limit: int = Query(50, ge=1, le=500, description="Number of logs to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    current_user: TokenData = Depends(get_current_active_user),
    rate_limit: bool = Depends(standard_rate_limit),
    db: Session = Depends(get_db)
):
    """
    Get agent activity logs with filtering and pagination
    """
    try:
        # In production, fetch from database
        # For now, generate mock logs
        logs = []
        
        # Generate mock logs
        for i in range(limit):
            agent_ids = list(AGENT_NAMES.keys())
            selected_agent_id = agent_ids[i % len(agent_ids)] if not agent_id else agent_id
            levels = ['info', 'warning', 'error', 'success']
            selected_level = levels[i % len(levels)] if not level else level
            
            log = {
                'id': f'log_{offset + i}',
                'agent_id': selected_agent_id,
                'agent_name': AGENT_NAMES[selected_agent_id],
                'timestamp': (datetime.utcnow() - timedelta(minutes=i)).isoformat(),
                'level': selected_level,
                'message': f'Agent {selected_agent_id} {selected_level} event',
                'execution_time': 10 + (i % 500)
            }
            
            if (not agent_id or log['agent_id'] == agent_id) and \
               (not level or log['level'] == level):
                logs.append(log)
        
        return AgentLogsResponse(
            logs=[AgentLog(**log) for log in logs],
            total=len(logs),
            page=offset // limit + 1,
            page_size=limit
        )
    except Exception as e:
        logger.error(f"Error fetching agent logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/decisions", response_model=AgentDecisionsResponse)
async def get_agent_decisions(
    agent_id: Optional[str] = Query(None, description="Filter by agent ID"),
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    limit: int = Query(30, ge=1, le=100, description="Number of decisions to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    current_user: TokenData = Depends(get_current_active_user),
    rate_limit: bool = Depends(standard_rate_limit),
    db: Session = Depends(get_db)
):
    """
    Get trading decisions made by agents
    """
    try:
        # In production, fetch from database
        # For now, generate mock decisions
        decisions = []
        
        decision_agents = ['M4', 'M6', 'M8']
        symbols = ['AAPL', 'TSLA', 'MSFT', 'GOOGL', 'BTC-USD', 'ETH-USD']
        actions = ['buy', 'sell', 'hold']
        
        for i in range(limit):
            selected_agent = decision_agents[i % len(decision_agents)] if not agent_id else agent_id
            selected_symbol = symbols[i % len(symbols)] if not symbol else symbol
            
            decision = {
                'id': f'decision_{offset + i}',
                'agent_id': selected_agent,
                'agent_name': AGENT_NAMES.get(selected_agent, f'Agent {selected_agent}'),
                'symbol': selected_symbol,
                'action': actions[i % len(actions)],
                'confidence': 60 + (i % 40),
                'reasoning': [
                    f'Technical indicators show {actions[i % len(actions)]} signal',
                    'Volume analysis supports decision',
                    'Risk metrics within acceptable range'
                ],
                'timestamp': (datetime.utcnow() - timedelta(minutes=i*5)).isoformat(),
                'executed': i % 3 != 0,
                'result': {
                    'price': 100 + (i % 50),
                    'pnl': (i % 1000) - 500,
                    'status': 'success' if i % 5 != 0 else 'failed'
                } if i % 3 != 0 else None
            }
            
            if (not agent_id or decision['agent_id'] == agent_id) and \
               (not symbol or decision['symbol'] == symbol):
                decisions.append(decision)
        
        return AgentDecisionsResponse(
            decisions=[AgentDecision(**d) for d in decisions],
            total=len(decisions),
            page=offset // limit + 1,
            page_size=limit
        )
    except Exception as e:
        logger.error(f"Error fetching agent decisions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance")
async def get_agent_performance(
    agent_id: Optional[str] = Query(None, description="Filter by agent ID"),
    current_user: TokenData = Depends(get_current_active_user),
    rate_limit: bool = Depends(standard_rate_limit),
    db: Session = Depends(get_db)
):
    """
    Get performance metrics for agents
    """
    try:
        # In production, calculate from actual metrics
        performance = {
            'overall': {
                'total_tasks': 10000,
                'success_rate': 95.5,
                'avg_execution_time': 150,
                'total_uptime': 86400
            },
            'by_agent': {}
        }
        
        # Add per-agent metrics
        for agent_id_key, agent_name in AGENT_NAMES.items():
            if not agent_id or agent_id_key == agent_id:
                performance['by_agent'][agent_id_key] = {
                    'name': agent_name,
                    'tasks_completed': 1000 + (hash(agent_id_key) % 500),
                    'success_rate': 90 + (hash(agent_id_key) % 10),
                    'avg_execution_time': 50 + (hash(agent_id_key) % 200)
                }
        
        return performance
    except Exception as e:
        logger.error(f"Error fetching agent performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

