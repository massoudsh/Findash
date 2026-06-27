"""
Asset Service — fetches Iranian market prices from tgju.org
with Redis caching and PostgreSQL persistence.

TGJU public endpoint:
  GET https://api.tgju.org/v1/market/indicator/summary-table-data/{key}

Response shape (relevant fields):
  data[0][0]  = current price (Rial)
  data[0][1]  = change amount
  data[0][2]  = change percent
  data[0][3]  = min today
  data[0][4]  = max today
  data[0][12] = last update timestamp
"""
import json
import logging
from datetime import datetime, timedelta
from typing import Optional

import httpx
import redis.asyncio as aioredis

from src.models.asset import ASSET_SEEDS, AssetPriceSnapshot

logger = logging.getLogger(__name__)

TGJU_BASE   = "https://api.tgju.org/v1/market/indicator/summary-table-data"
CACHE_TTL   = 60          # seconds — refresh price every 60s
RIAL_FACTOR = 10          # tgju returns Rial → divide by 10 for Toman
HEADERS     = {"User-Agent": "OctopusTradingPlatform/1.0"}


# ─── Symbol → source_key map ────────────────────────────────────────────────
SYMBOL_MAP: dict[str, dict] = {s["symbol"]: s for s in ASSET_SEEDS}


class AssetService:
    def __init__(self, redis_client: aioredis.Redis, db_session=None):
        self.redis = redis_client
        self.db    = db_session

    # ─── Public API ─────────────────────────────────────────────────────────

    async def get_all_prices(self, category: Optional[str] = None) -> list[dict]:
        """Return all asset prices, filtered by category if given."""
        symbols = [
            s for s in ASSET_SEEDS
            if category is None or s["category"] == category
        ]
        results = []
        for asset in symbols:
            price = await self.get_price(asset["symbol"])
            if price:
                results.append({**asset, **price})
        return results

    async def get_price(self, symbol: str) -> Optional[dict]:
        """Return latest price for a symbol (cache-first)."""
        cache_key = f"asset:price:{symbol}"

        # 1. Try Redis cache
        cached = await self.redis.get(cache_key)
        if cached:
            return json.loads(cached)

        # 2. Fetch from TGJU
        asset_meta = SYMBOL_MAP.get(symbol)
        if not asset_meta:
            return None

        price_data = await self._fetch_tgju(asset_meta["source_key"])
        if not price_data:
            return None

        # 3. Store in cache
        await self.redis.setex(cache_key, CACHE_TTL, json.dumps(price_data))

        # 4. Persist snapshot (fire-and-forget)
        if self.db:
            await self._upsert_snapshot(symbol, price_data)

        return price_data

    async def get_history(self, symbol: str, days: int = 30) -> list[dict]:
        """Return OHLCV history from DB for the given symbol."""
        if not self.db:
            return []
        from sqlalchemy import select, and_
        from src.models.asset import AssetPriceHistory

        cutoff = datetime.utcnow() - timedelta(days=days)
        result = await self.db.execute(
            select(AssetPriceHistory)
            .where(and_(
                AssetPriceHistory.symbol == symbol,
                AssetPriceHistory.timestamp >= cutoff
            ))
            .order_by(AssetPriceHistory.timestamp)
        )
        rows = result.scalars().all()
        return [
            {
                "timestamp": r.timestamp.isoformat(),
                "open":  r.open,
                "high":  r.high,
                "low":   r.low,
                "close": r.close,
            }
            for r in rows
        ]

    async def get_usd_to_toman(self) -> float:
        """Return current USD → Toman rate."""
        price = await self.get_price("USD")
        if price:
            return price["price_toman"]
        return 0.0

    # ─── Internal helpers ────────────────────────────────────────────────────

    async def _fetch_tgju(self, source_key: str) -> Optional[dict]:
        url = f"{TGJU_BASE}/{source_key}"
        try:
            async with httpx.AsyncClient(timeout=8, headers=HEADERS) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                payload = resp.json()

            row = payload.get("data", [[]])[0]
            if not row:
                return None

            price_rial   = float(str(row[0]).replace(",", "")) if row[0] else 0
            price_toman  = round(price_rial / RIAL_FACTOR, 0)
            change       = float(str(row[1]).replace(",", "")) / RIAL_FACTOR if row[1] else 0
            change_pct   = float(str(row[2]).replace(",", "").replace("%", "")) if row[2] else 0
            low          = float(str(row[3]).replace(",", "")) / RIAL_FACTOR if row[3] else 0
            high         = float(str(row[4]).replace(",", "")) / RIAL_FACTOR if row[4] else 0
            updated_at   = row[12] if len(row) > 12 else datetime.utcnow().isoformat()

            return {
                "price":              price_toman,
                "price_toman":        price_toman,
                "change_24h":         round(change, 0),
                "change_percent_24h": round(change_pct, 2),
                "high_24h":           high,
                "low_24h":            low,
                "updated_at":         str(updated_at),
                "source":             "tgju.org",
            }
        except Exception as e:
            logger.warning(f"TGJU fetch failed for {source_key}: {e}")
            return None

    async def _upsert_snapshot(self, symbol: str, data: dict):
        """Insert or update AssetPriceSnapshot row."""
        try:
            from sqlalchemy import select
            result = await self.db.execute(
                select(AssetPriceSnapshot).where(AssetPriceSnapshot.symbol == symbol)
            )
            snapshot = result.scalar_one_or_none()

            if snapshot:
                snapshot.price              = data["price_toman"]
                snapshot.price_toman        = data["price_toman"]
                snapshot.change_24h         = data["change_24h"]
                snapshot.change_percent_24h = data["change_percent_24h"]
                snapshot.high_24h           = data["high_24h"]
                snapshot.low_24h            = data["low_24h"]
                snapshot.updated_at         = datetime.utcnow()
            else:
                self.db.add(AssetPriceSnapshot(
                    symbol              = symbol,
                    price               = data["price_toman"],
                    price_toman         = data["price_toman"],
                    change_24h          = data["change_24h"],
                    change_percent_24h  = data["change_percent_24h"],
                    high_24h            = data["high_24h"],
                    low_24h             = data["low_24h"],
                ))
            await self.db.commit()
        except Exception as e:
            logger.error(f"DB upsert failed for {symbol}: {e}")
            await self.db.rollback()
