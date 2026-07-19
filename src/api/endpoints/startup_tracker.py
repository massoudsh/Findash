"""
Startup Tracker API (Internal/Admin)
Track GTM (Go-To-Market) hypotheses, customer conversations, and traction metrics
for the Findash team's own product validation. In-memory store; can be replaced with DB.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/startup-tracker", tags=["Startup Tracker"])


# ---------------------------------------------------------------------------
# GTM Hypothesis
# ---------------------------------------------------------------------------

class GTMHypothesisCreate(BaseModel):
    title: str
    statement: str  # e.g. "We believe [segment] will [action] because [reason]"
    target_segment: str
    channel: str
    status: str = Field(default="untested", pattern="^(untested|testing|validated|invalidated)$")
    confidence: int = Field(default=50, ge=0, le=100)
    owner: Optional[str] = None
    notes: Optional[str] = None


class GTMHypothesisResponse(GTMHypothesisCreate):
    id: int
    created_at: str
    updated_at: str


_hypotheses: List[dict] = []
_hyp_next_id = 1


def _next_hyp_id() -> int:
    global _hyp_next_id
    n = _hyp_next_id
    _hyp_next_id += 1
    return n


@router.get("/hypotheses", response_model=List[GTMHypothesisResponse])
async def list_hypotheses():
    """List all GTM hypotheses."""
    return _hypotheses


@router.post("/hypotheses", response_model=GTMHypothesisResponse)
async def create_hypothesis(body: GTMHypothesisCreate):
    """Create a new GTM hypothesis."""
    now = datetime.utcnow().isoformat() + "Z"
    item = {"id": _next_hyp_id(), **body.dict(), "created_at": now, "updated_at": now}
    _hypotheses.append(item)
    logger.info("Created GTM hypothesis id=%s title=%s", item["id"], body.title)
    return item


@router.get("/hypotheses/{hypothesis_id}", response_model=GTMHypothesisResponse)
async def get_hypothesis(hypothesis_id: int):
    for h in _hypotheses:
        if h["id"] == hypothesis_id:
            return h
    raise HTTPException(status_code=404, detail="Hypothesis not found")


@router.put("/hypotheses/{hypothesis_id}", response_model=GTMHypothesisResponse)
async def update_hypothesis(hypothesis_id: int, body: GTMHypothesisCreate):
    for i, h in enumerate(_hypotheses):
        if h["id"] == hypothesis_id:
            now = datetime.utcnow().isoformat() + "Z"
            _hypotheses[i] = {
                "id": hypothesis_id,
                **body.dict(),
                "created_at": h["created_at"],
                "updated_at": now,
            }
            return _hypotheses[i]
    raise HTTPException(status_code=404, detail="Hypothesis not found")


@router.delete("/hypotheses/{hypothesis_id}")
async def delete_hypothesis(hypothesis_id: int):
    for i, h in enumerate(_hypotheses):
        if h["id"] == hypothesis_id:
            _hypotheses.pop(i)
            return {"ok": True}
    raise HTTPException(status_code=404, detail="Hypothesis not found")


# ---------------------------------------------------------------------------
# Customer Conversation
# ---------------------------------------------------------------------------

class CustomerConversationCreate(BaseModel):
    contact_name: str
    company: Optional[str] = None
    role: Optional[str] = None
    channel: str = Field(default="call", pattern="^(call|meeting|email|chat|survey|other)$")
    sentiment: str = Field(default="neutral", pattern="^(positive|neutral|negative)$")
    summary: str
    key_insights: List[str] = Field(default_factory=list)
    hypothesis_id: Optional[int] = None
    conversation_date: Optional[str] = None  # ISO date string, defaults to now on create


class CustomerConversationResponse(CustomerConversationCreate):
    id: int
    created_at: str


_conversations: List[dict] = []
_conv_next_id = 1


def _next_conv_id() -> int:
    global _conv_next_id
    n = _conv_next_id
    _conv_next_id += 1
    return n


@router.get("/conversations", response_model=List[CustomerConversationResponse])
async def list_conversations():
    """List all customer conversations."""
    return _conversations


@router.post("/conversations", response_model=CustomerConversationResponse)
async def create_conversation(body: CustomerConversationCreate):
    """Log a new customer conversation."""
    if body.hypothesis_id is not None and not any(h["id"] == body.hypothesis_id for h in _hypotheses):
        raise HTTPException(status_code=404, detail="Linked hypothesis not found")
    now = datetime.utcnow().isoformat() + "Z"
    data = body.dict()
    if not data.get("conversation_date"):
        data["conversation_date"] = now
    item = {"id": _next_conv_id(), **data, "created_at": now}
    _conversations.append(item)
    logger.info("Logged customer conversation id=%s contact=%s", item["id"], body.contact_name)
    return item


@router.get("/conversations/{conversation_id}", response_model=CustomerConversationResponse)
async def get_conversation(conversation_id: int):
    for c in _conversations:
        if c["id"] == conversation_id:
            return c
    raise HTTPException(status_code=404, detail="Conversation not found")


@router.put("/conversations/{conversation_id}", response_model=CustomerConversationResponse)
async def update_conversation(conversation_id: int, body: CustomerConversationCreate):
    for i, c in enumerate(_conversations):
        if c["id"] == conversation_id:
            data = body.dict()
            if not data.get("conversation_date"):
                data["conversation_date"] = c["conversation_date"]
            _conversations[i] = {"id": conversation_id, **data, "created_at": c["created_at"]}
            return _conversations[i]
    raise HTTPException(status_code=404, detail="Conversation not found")


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: int):
    for i, c in enumerate(_conversations):
        if c["id"] == conversation_id:
            _conversations.pop(i)
            return {"ok": True}
    raise HTTPException(status_code=404, detail="Conversation not found")


# ---------------------------------------------------------------------------
# Traction Metric
# ---------------------------------------------------------------------------

class TractionMetricCreate(BaseModel):
    metric_name: str
    category: str = Field(default="other", pattern="^(users|revenue|engagement|retention|other)$")
    value: float
    unit: Optional[str] = None
    period_start: Optional[str] = None
    period_end: Optional[str] = None
    source: Optional[str] = None
    notes: Optional[str] = None


class TractionMetricResponse(TractionMetricCreate):
    id: int
    recorded_at: str


_traction: List[dict] = []
_traction_next_id = 1


def _next_traction_id() -> int:
    global _traction_next_id
    n = _traction_next_id
    _traction_next_id += 1
    return n


@router.get("/traction", response_model=List[TractionMetricResponse])
async def list_traction():
    """List all traction metric entries."""
    return _traction


@router.post("/traction", response_model=TractionMetricResponse)
async def create_traction(body: TractionMetricCreate):
    """Record a new traction metric data point."""
    now = datetime.utcnow().isoformat() + "Z"
    item = {"id": _next_traction_id(), **body.dict(), "recorded_at": now}
    _traction.append(item)
    logger.info("Recorded traction metric id=%s name=%s value=%s", item["id"], body.metric_name, body.value)
    return item


@router.get("/traction/{metric_id}", response_model=TractionMetricResponse)
async def get_traction(metric_id: int):
    for t in _traction:
        if t["id"] == metric_id:
            return t
    raise HTTPException(status_code=404, detail="Traction metric not found")


@router.put("/traction/{metric_id}", response_model=TractionMetricResponse)
async def update_traction(metric_id: int, body: TractionMetricCreate):
    for i, t in enumerate(_traction):
        if t["id"] == metric_id:
            _traction[i] = {"id": metric_id, **body.dict(), "recorded_at": t["recorded_at"]}
            return _traction[i]
    raise HTTPException(status_code=404, detail="Traction metric not found")


@router.delete("/traction/{metric_id}")
async def delete_traction(metric_id: int):
    for i, t in enumerate(_traction):
        if t["id"] == metric_id:
            _traction.pop(i)
            return {"ok": True}
    raise HTTPException(status_code=404, detail="Traction metric not found")


# ---------------------------------------------------------------------------
# Summary (for admin dashboard cards)
# ---------------------------------------------------------------------------

@router.get("/summary")
async def get_summary():
    """Aggregate counts for the Startup Tracker admin dashboard."""
    validated = sum(1 for h in _hypotheses if h["status"] == "validated")
    invalidated = sum(1 for h in _hypotheses if h["status"] == "invalidated")
    testing = sum(1 for h in _hypotheses if h["status"] == "testing")
    positive_conversations = sum(1 for c in _conversations if c["sentiment"] == "positive")
    latest_traction_by_metric = {}
    for t in _traction:
        name = t["metric_name"]
        if name not in latest_traction_by_metric or t["recorded_at"] > latest_traction_by_metric[name]["recorded_at"]:
            latest_traction_by_metric[name] = t
    return {
        "hypotheses_total": len(_hypotheses),
        "hypotheses_validated": validated,
        "hypotheses_invalidated": invalidated,
        "hypotheses_testing": testing,
        "conversations_total": len(_conversations),
        "conversations_positive": positive_conversations,
        "traction_entries_total": len(_traction),
        "latest_traction": list(latest_traction_by_metric.values()),
    }
