"""
FastAPI router — /api/assets
Endpoints:
  GET  /api/assets              → all assets (optional ?category=gold)
  GET  /api/assets/{symbol}     → single asset detail
  GET  /api/assets/{symbol}/history?days=30 → OHLCV history
  GET  /api/assets/usd-rate     → current USD/Toman rate
  POST /api/assets/portfolio    → add asset to user portfolio
  GET  /api/assets/portfolio    → user's portfolio
"""
from fastapi import APIRouter, Depends, Query, HTTPException, status
from typing import Optional

# from src.database import get_db, get_redis   ← your existing dep providers
# from src.auth import get_current_user         ← your existing auth dep
from src.schemas.asset_schema import (
    AssetDetail,
    AssetListResponse,
    AssetHistory,
    PortfolioAssetCreate,
    PortfolioAsset,
)
from src.services.asset_service import AssetService, SYMBOL_MAP
from src.models.asset import ASSET_SEEDS

router = APIRouter(prefix="/api/assets", tags=["assets"])


# ─── Dependency ─────────────────────────────────────────────────────────────

async def get_asset_service(
    # db=Depends(get_db),
    # redis=Depends(get_redis),
) -> AssetService:
    """
    Replace commented deps with your project's actual dep providers.
    For now returns a service without DB persistence (cache-only mode).
    """
    import redis.asyncio as aioredis
    r = aioredis.from_url("redis://localhost:6379", decode_responses=True)
    return AssetService(redis_client=r, db_session=None)


# ─── Routes ─────────────────────────────────────────────────────────────────

@router.get("", response_model=AssetListResponse, summary="همه دارایی‌ها")
async def list_assets(
    category: Optional[str] = Query(None, description="gold | silver | currency | real_estate | crypto"),
    service: AssetService = Depends(get_asset_service),
):
    """
    لیست قیمت لحظه‌ای تمام دارایی‌های ایرانی.
    نتایج از Redis cache (TTL: 60s) سرو می‌شوند.
    """
    assets = await service.get_all_prices(category=category)
    usd_rate = await service.get_usd_to_toman()

    from datetime import datetime
    return AssetListResponse(
        assets=assets,
        usd_to_toman=usd_rate,
        last_updated=datetime.utcnow().isoformat(),
    )


@router.get("/usd-rate", summary="نرخ دلار به تومان")
async def usd_rate(service: AssetService = Depends(get_asset_service)):
    rate = await service.get_usd_to_toman()
    return {"usd_to_toman": rate}


@router.get("/{symbol}", response_model=AssetDetail, summary="جزئیات یک دارایی")
async def get_asset(
    symbol: str,
    service: AssetService = Depends(get_asset_service),
):
    symbol = symbol.upper()
    meta = SYMBOL_MAP.get(symbol)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")

    price_data = await service.get_price(symbol)
    if not price_data:
        raise HTTPException(status_code=503, detail="Price data temporarily unavailable")

    return AssetDetail(**meta, **price_data)


@router.get("/{symbol}/history", response_model=AssetHistory, summary="تاریخچه قیمت")
async def get_history(
    symbol: str,
    days: int = Query(30, ge=1, le=365, description="تعداد روز"),
    service: AssetService = Depends(get_asset_service),
):
    symbol = symbol.upper()
    if symbol not in SYMBOL_MAP:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")

    data = await service.get_history(symbol, days=days)
    return AssetHistory(
        symbol=symbol,
        interval="1d",
        data=data,
    )


@router.post("/portfolio", response_model=PortfolioAsset, status_code=status.HTTP_201_CREATED,
             summary="افزودن دارایی به پورتفولیو")
async def add_to_portfolio(
    body: PortfolioAssetCreate,
    # current_user=Depends(get_current_user),
    service: AssetService = Depends(get_asset_service),
):
    symbol = body.symbol.upper()
    if symbol not in SYMBOL_MAP:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")

    price_data = await service.get_price(symbol)
    current_price = price_data["price_toman"] if price_data else body.buy_price

    current_value    = body.quantity * current_price
    profit_loss      = current_value - (body.quantity * body.buy_price)
    profit_loss_pct  = (profit_loss / (body.quantity * body.buy_price)) * 100 if body.buy_price else 0

    # TODO: persist to DB via service when DB session is wired
    return PortfolioAsset(
        **body.model_dump(),
        id=1,
        current_price=current_price,
        current_value=round(current_value, 0),
        profit_loss=round(profit_loss, 0),
        profit_loss_percent=round(profit_loss_pct, 2),
    )
