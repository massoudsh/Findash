"""User-scoped investing tools: watchlists, screening, paper trades, events and dividends."""
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional
import json
import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.api.endpoints.iran_market import get_overview
from src.core.security import TokenData, get_optional_user

router = APIRouter(prefix="/api/investor-tools", tags=["Investor Tools"])
DATA_FILE = Path("data/investor_tools.json")


def _load() -> dict:
    if not DATA_FILE.exists():
        return {"watchlists": {}, "paper": {}, "dividends": {}}
    try:
        return json.loads(DATA_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"watchlists": {}, "paper": {}, "dividends": {}}


def _save(data: dict) -> None:
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    DATA_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _user_id(user: Optional[TokenData]) -> str:
    return user.user_id if user else "default"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class WatchlistInput(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    symbols: list[str] = Field(default_factory=list, max_length=100)


@router.get("/watchlists")
def list_watchlists(user: Optional[TokenData] = Depends(get_optional_user)):
    return _load()["watchlists"].get(_user_id(user), [])


@router.post("/watchlists", status_code=201)
def create_watchlist(payload: WatchlistInput, user: Optional[TokenData] = Depends(get_optional_user)):
    data, uid = _load(), _user_id(user)
    items = data["watchlists"].setdefault(uid, [])
    if len(items) >= 5:
        raise HTTPException(422, "حداکثر پنج واچ‌لیست برای هر کاربر مجاز است")
    item = {"id": str(uuid.uuid4()), "name": payload.name.strip(), "symbols": sorted(set(payload.symbols)), "updated_at": _now()}
    items.append(item)
    _save(data)
    return item


@router.put("/watchlists/{watchlist_id}")
def update_watchlist(watchlist_id: str, payload: WatchlistInput, user: Optional[TokenData] = Depends(get_optional_user)):
    data, uid = _load(), _user_id(user)
    for item in data["watchlists"].get(uid, []):
        if item["id"] == watchlist_id:
            item.update(name=payload.name.strip(), symbols=sorted(set(payload.symbols)), updated_at=_now())
            _save(data)
            return item
    raise HTTPException(404, "واچ‌لیست پیدا نشد")


@router.delete("/watchlists/{watchlist_id}", status_code=204)
def delete_watchlist(watchlist_id: str, user: Optional[TokenData] = Depends(get_optional_user)):
    data, uid = _load(), _user_id(user)
    items = data["watchlists"].get(uid, [])
    filtered = [item for item in items if item["id"] != watchlist_id]
    if len(filtered) == len(items):
        raise HTTPException(404, "واچ‌لیست پیدا نشد")
    data["watchlists"][uid] = filtered
    _save(data)


@router.get("/screener")
async def screener(category: Optional[str] = None, min_change: Optional[float] = None, max_change: Optional[float] = None, query: Optional[str] = None, sort: Literal["change_desc", "change_asc", "price_desc"] = "change_desc"):
    """Screen live-supported Iranian assets. Technical and equity-only fields are omitted until a verified source is connected."""
    items = (await get_overview())["items"]
    result = [item for item in items if item["available"]]
    if category:
        result = [item for item in result if item["category"] == category]
    if min_change is not None:
        result = [item for item in result if item["change_pct"] >= min_change]
    if max_change is not None:
        result = [item for item in result if item["change_pct"] <= max_change]
    if query:
        needle = query.casefold()
        result = [item for item in result if needle in item["symbol"].casefold() or needle in item["label"].casefold()]
    key, reverse = ({"change_desc": ("change_pct", True), "change_asc": ("change_pct", False), "price_desc": ("price", True)})[sort]
    return {"items": sorted(result, key=lambda item: item[key] or 0, reverse=reverse), "cached_at": (await get_overview())["cached_at"]}


class PaperOrderInput(BaseModel):
    symbol: str = Field(min_length=1, max_length=30)
    side: Literal["buy", "sell"]
    quantity: float = Field(gt=0)
    price: float = Field(gt=0)
    fee: float = Field(default=0, ge=0)
    thesis: str = Field(default="", max_length=1000)
    stop_loss: Optional[float] = Field(default=None, gt=0)
    take_profit: Optional[float] = Field(default=None, gt=0)
    tags: list[str] = Field(default_factory=list, max_length=10)


@router.get("/paper")
def get_paper_account(user: Optional[TokenData] = Depends(get_optional_user)):
    data, uid = _load(), _user_id(user)
    return data["paper"].setdefault(uid, {"cash": 100_000_000, "orders": [], "journal": []})


@router.post("/paper/orders", status_code=201)
def place_paper_order(payload: PaperOrderInput, user: Optional[TokenData] = Depends(get_optional_user)):
    data, uid = _load(), _user_id(user)
    account = data["paper"].setdefault(uid, {"cash": 100_000_000, "orders": [], "journal": []})
    value = payload.quantity * payload.price + payload.fee
    if payload.side == "buy" and value > account["cash"]:
        raise HTTPException(422, "موجودی حساب آزمایشی کافی نیست")
    account["cash"] += -value if payload.side == "buy" else value - payload.fee
    order = {"id": str(uuid.uuid4()), **payload.model_dump(), "value": value, "executed_at": _now(), "mode": "paper"}
    account["orders"].insert(0, order)
    account["journal"].insert(0, {"id": order["id"], "symbol": payload.symbol, "thesis": payload.thesis, "stop_loss": payload.stop_loss, "take_profit": payload.take_profit, "tags": payload.tags, "created_at": order["executed_at"]})
    _save(data)
    return order


@router.get("/events")
def market_events(kind: Optional[str] = None, symbol: Optional[str] = None):
    """Calendar contract; items are explicitly marked manual until a verified provider is configured."""
    events = [
        {"id": "manual-1", "title": "نمونه رویداد بازار", "kind": "manual", "symbol": None, "starts_at": None, "source": "ثبت دستی", "verified": False},
    ]
    if kind:
        events = [event for event in events if event["kind"] == kind]
    if symbol:
        events = [event for event in events if event["symbol"] == symbol]
    return {"items": events, "source_status": "برای رویدادهای رسمی، اتصال منبع تأییدشده لازم است"}


class DividendInput(BaseModel):
    symbol: str = Field(min_length=1, max_length=30)
    amount: float = Field(gt=0)
    currency: Literal["IRT", "IRR"] = "IRT"
    paid_at: str
    status: Literal["received", "expected"] = "received"
    note: str = Field(default="", max_length=500)


@router.get("/dividends")
def list_dividends(user: Optional[TokenData] = Depends(get_optional_user)):
    items = _load()["dividends"].get(_user_id(user), [])
    received = sum(item["amount"] for item in items if item["status"] == "received")
    expected = sum(item["amount"] for item in items if item["status"] == "expected")
    return {"items": items, "summary": {"received": received, "expected": expected}}


@router.post("/dividends", status_code=201)
def create_dividend(payload: DividendInput, user: Optional[TokenData] = Depends(get_optional_user)):
    data, uid = _load(), _user_id(user)
    item = {"id": str(uuid.uuid4()), **payload.model_dump(), "created_at": _now()}
    data["dividends"].setdefault(uid, []).insert(0, item)
    _save(data)
    return item


@router.delete("/dividends/{dividend_id}", status_code=204)
def delete_dividend(dividend_id: str, user: Optional[TokenData] = Depends(get_optional_user)):
    data, uid = _load(), _user_id(user)
    items = data["dividends"].get(uid, [])
    filtered = [item for item in items if item["id"] != dividend_id]
    if len(filtered) == len(items):
        raise HTTPException(404, "ثبت سود نقدی پیدا نشد")
    data["dividends"][uid] = filtered
    _save(data)
