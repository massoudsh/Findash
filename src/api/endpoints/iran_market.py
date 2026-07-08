"""
Iran Market Data API
داده‌های بازار ایران: ارز، طلا، سکه، کریپتو، شاخص بورس
منابع: tgju.org (ارز/طلا) + Nobitex (کریپتو)
"""
import asyncio
import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter
import httpx

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/iran-market", tags=["Iran Market"])

# ── Cache: 30-second TTL ──────────────────────────────────────────────────────
_cache: Dict[str, Any] = {}
_cache_ts: Dict[str, float] = {}
CACHE_TTL = 60  # seconds — کاربر: بروزرسانی هر ۶۰ ثانیه


def _is_fresh(key: str) -> bool:
    import time
    return key in _cache_ts and (time.time() - _cache_ts[key]) < CACHE_TTL


async def _fetch(url: str, timeout: int = 8) -> Optional[Dict]:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(url, headers={"User-Agent": "findash/1.0"})
            r.raise_for_status()
            return r.json()
    except Exception as e:
        logger.warning(f"fetch error {url}: {e}")
        return None


# ── tgju helpers ─────────────────────────────────────────────────────────────
TGJU_INDICATORS = {
    "USD-IRR":      "price_dollar_rl",
    "EUR-IRR":      "price_euro",
    "GBP-IRR":      "price_gbp",
    "GOLD18-IRT":   "geram18",
    "GOLD24-IRT":   "geram24",
    "COIN-IRT":     "sekeb",
    "HALFCOIN-IRT": "sekenim",
    "QUARTERCOIN-IRT": "sekerob",
}

TGJU_BASE = "https://api.tgju.org/v1/market/indicator/summary-table-data"

NOBITEX_SYMBOLS = {
    "BTC-IRT":  "BTCIRT",
    "ETH-IRT":  "ETHIRT",
    "USDT-IRT": "USDTIRT",
    "BNB-IRT":  "BNBIRT",
    "TRX-IRT":  "TRXIRT",
}


async def _tgju_price(indicator: str) -> Optional[Dict]:
    data = await _fetch(f"{TGJU_BASE}/{indicator}")
    if not data or "data" not in data:
        return None
    rows = data["data"]
    if not rows:
        return None
    row = rows[0]
    try:
        price = float(str(row.get("p", row.get("price", 0))).replace(",", ""))
        change_pct = float(str(row.get("dp", row.get("d", 0))).replace(",", "").replace("%", ""))
        return {"price": price, "change_pct": change_pct}
    except Exception:
        return None


async def _nobitex_stats() -> Optional[Dict]:
    data = await _fetch("https://api.nobitex.ir/market/stats?srcCurrency=btc,eth,usdt,bnb,trx&dstCurrency=rls")
    if not data or "stats" not in data:
        return None
    return data["stats"]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/overview")
async def get_overview():
    """
    شاخص‌های کلیدی بازار ایران — ارز، طلا، سکه، کریپتو
    """
    import time

    cache_key = "overview"
    if _is_fresh(cache_key):
        return _cache[cache_key]

    # fetch tgju + nobitex concurrently
    tgju_tasks = {sym: _tgju_price(ind) for sym, ind in TGJU_INDICATORS.items()}
    nobitex_task = _nobitex_stats()

    results_tgju = await asyncio.gather(*tgju_tasks.values(), return_exceptions=True)
    nobitex_stats = await nobitex_task

    tgju_data: Dict[str, Any] = {}
    for sym, result in zip(tgju_tasks.keys(), results_tgju):
        if isinstance(result, dict):
            tgju_data[sym] = result

    # map Nobitex crypto prices (prices are in Rial → divide by 10 for Toman)
    crypto_data: Dict[str, Any] = {}
    if nobitex_stats:
        for sym, nb_sym in NOBITEX_SYMBOLS.items():
            key = nb_sym.lower()
            if key in nobitex_stats:
                s = nobitex_stats[key]
                try:
                    price_rial = float(s.get("latest", 0))
                    price_toman = price_rial / 10
                    open_price = float(s.get("dayOpen", price_rial)) / 10
                    change_pct = ((price_toman - open_price) / open_price * 100) if open_price else 0
                    crypto_data[sym] = {"price": price_toman, "change_pct": round(change_pct, 2)}
                except Exception:
                    pass

    items = []
    labels = {
        "USD-IRR":         ("دلار آمریکا", "💵", "currency"),
        "EUR-IRR":         ("یورو", "💶", "currency"),
        "GBP-IRR":         ("پوند", "💷", "currency"),
        "GOLD18-IRT":      ("طلا ۱۸ عیار", "🥇", "gold"),
        "GOLD24-IRT":      ("طلا ۲۴ عیار", "🥇", "gold"),
        "COIN-IRT":        ("سکه بهار آزادی", "🪙", "coin"),
        "HALFCOIN-IRT":    ("نیم‌سکه", "🪙", "coin"),
        "QUARTERCOIN-IRT": ("ربع‌سکه", "🪙", "coin"),
        "BTC-IRT":         ("بیت‌کوین", "₿", "crypto"),
        "ETH-IRT":         ("اتریوم", "Ξ", "crypto"),
        "USDT-IRT":        ("تتر", "₮", "crypto"),
        "BNB-IRT":         ("بایننس کوین", "🔶", "crypto"),
        "TRX-IRT":         ("ترون", "🔷", "crypto"),
    }

    all_data = {**tgju_data, **crypto_data}

    for sym, (label, icon, category) in labels.items():
        entry = all_data.get(sym)
        items.append({
            "symbol": sym,
            "label": label,
            "icon": icon,
            "category": category,
            "price": entry["price"] if entry else None,
            "change_pct": entry["change_pct"] if entry else None,
            "up": (entry["change_pct"] >= 0) if entry else None,
            "available": entry is not None,
        })

    payload = {"items": items, "cached_at": time.time()}
    _cache[cache_key] = payload
    _cache_ts[cache_key] = time.time()
    return payload


@router.get("/prices")
async def get_prices(symbols: Optional[str] = None):
    """
    قیمت لحظه‌ای چند نماد — symbols=BTC-IRT,ETH-IRT,...
    """
    overview = await get_overview()
    if not symbols:
        return overview

    wanted = set(symbols.upper().split(","))
    filtered = [item for item in overview["items"] if item["symbol"] in wanted]
    return {"items": filtered}


@router.get("/assets")
async def get_assets(category: Optional[str] = None):
    """
    لیست همه دارایی‌های ایرانی تعریف‌شده در AssetsConfig
    category: currency | gold | coin | crypto | bourse (اختیاری — برای فیلتر)
    """
    from src.core.assets_config import AssetsConfig, Sector

    SECTOR_TO_CATEGORY: dict = {
        "currency": Sector.CURRENCY,
        "gold": Sector.GOLD,
        "crypto": Sector.IRAN_CRYPTO,
        "bourse": Sector.IRAN_BOURSE,
        "commodity": Sector.IRAN_COMMODITY,
    }

    iranian_syms = set(
        AssetsConfig.IRANIAN_CURRENCY_ASSETS
        + AssetsConfig.IRANIAN_GOLD_ASSETS
        + AssetsConfig.IRANIAN_CRYPTO_ASSETS
        + AssetsConfig.IRANIAN_BOURSE_ASSETS
    )

    result = []
    for sym in iranian_syms:
        asset = AssetsConfig.ASSETS.get(sym)
        if not asset:
            continue
        cat = next(
            (k for k, v in SECTOR_TO_CATEGORY.items() if asset.sector == v),
            "other"
        )
        if category and cat != category:
            continue
        result.append({
            "symbol": asset.symbol,
            "name": asset.name,
            "asset_type": asset.asset_type.value,
            "sector": asset.sector.value,
            "currency": asset.currency,
            "category": cat,
            "trading_hours": asset.trading_hours,
            "volatility": asset.volatility_category,
            "liquidity": asset.liquidity_category,
            "tick_size": asset.tick_size,
            "min_trade_size": asset.min_trade_size,
        })

    # sort: currency → gold → crypto → bourse
    order = ["currency", "gold", "coin", "crypto", "bourse", "other"]
    result.sort(key=lambda x: order.index(x["category"]) if x["category"] in order else 99)
    return {"assets": result, "total": len(result)}


@router.get("/ticker")
async def get_ticker():
    """
    TickerBar: ۵ نماد اصلی برای نوار بالای داشبورد
    """
    overview = await get_overview()
    ticker_syms = ["USD-IRR", "GOLD18-IRT", "COIN-IRT", "BTC-IRT", "TEDPIX"]
    items = {item["symbol"]: item for item in overview["items"]}
    result = []
    for sym in ticker_syms:
        if sym in items:
            result.append(items[sym])
        elif sym == "TEDPIX":
            # TEDPIX not in tgju/nobitex — placeholder
            result.append({
                "symbol": "TEDPIX",
                "label": "شاخص کل",
                "icon": "📈",
                "category": "bourse",
                "price": None,
                "change_pct": None,
                "up": None,
                "available": False,
            })
    return {"items": result}
