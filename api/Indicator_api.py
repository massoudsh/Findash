"""
indicator_api.py – lightweight FastAPI micro‑service that serves key technical
indicators (MACD, RSI, OBV, Stochastic, SMA, EMA, ATR, VWAP) for any symbol
supported by the CCXT library (default: Binance).

Install requirements:
    pip install fastapi uvicorn pandas pandas_ta ccxt

Run locally:
    uvicorn indicator_api:app --reload

Example request:
    GET http://localhost:8000/indicators?symbol=BTC/USDT&interval=1h&exchange=binance

This returns a JSON payload with the latest indicator values.
"""
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import pandas as pd
import pandas_ta as ta
import ccxt
from datetime import datetime

app = FastAPI(title="Technical Indicator API", version="0.1.0")

# Map of exchange identifiers to CCXT exchange classes\SUPPORTED_EXCHANGES = {name: getattr(ccxt, name) for name in ccxt.exchanges}

class IndicatorResponse(BaseModel):
    symbol: str
    exchange: str
    interval: str
    timestamp: int
    datetime: str
    indicators: dict


def fetch_ohlcv(exchange_id: str, symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
    """Fetch OHLCV data from an exchange via CCXT and return as a DataFrame."""
    if exchange_id not in SUPPORTED_EXCHANGES:
        raise ValueError(f"Exchange '{exchange_id}' is not supported by ccxt.")

    exchange_class = SUPPORTED_EXCHANGES[exchange_id]
    exchange = exchange_class({"enableRateLimit": True})
    exchange.load_markets()
    # Normalise symbol (ccxt uses e.g. BTC/USDT)
    if symbol not in exchange.markets:
        raise ValueError(f"Symbol '{symbol}' not available on {exchange_id}.")

    # CCXT returns a list: [ timestamp, open, high, low, close, volume ]
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a suite of technical indicators onto the OHLCV DataFrame."""
    df = df.copy()
    df.set_index("ts", inplace=True)

    # Simple & Exponential Moving Averages
    df["sma20"] = ta.sma(df["close"], length=20)
    df["ema12"] = ta.ema(df["close"], length=12)
    df["ema26"] = ta.ema(df["close"], length=26)

    # MACD (returns MACD line, signal line, histogram)
    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    df = pd.concat([df, macd], axis=1)

    # RSI
    df["rsi14"] = ta.rsi(df["close"], length=14)

    # OBV
    df["obv"] = ta.obv(df["close"], df["volume"])

    # Stochastic Oscillator (%K and %D)
    stoch = ta.stoch(df["high"], df["low"], df["close"], k=14, d=3)
    df = pd.concat([df, stoch], axis=1)

    # Average True Range (ATR)
    df["atr14"] = ta.atr(df["high"], df["low"], df["close"], length=14)

    # Volume‑Weighted Average Price (VWAP)
    df["vwap"] = ta.vwap(df["high"], df["low"], df["close"], df["volume"])

    df.reset_index(inplace=True)
    return df


@app.get("/indicators", response_model=IndicatorResponse)
async def get_indicators(
    symbol: str = Query(..., description="Trading pair, e.g. BTC/USDT"),
    interval: str = Query("1h", description="Candle duration, e.g. 1m, 5m, 1h, 1d"),
    exchange: str = Query("binance", description="Exchange id used by ccxt, e.g. binance, kucoin")
):
    """Return the most recent indicator values as JSON."""
    try:
        df = fetch_ohlcv(exchange, symbol, interval)
        df = compute_indicators(df)
    except Exception as err:
        raise HTTPException(status_code=400, detail=str(err))

    latest = df.iloc[-1]
    indicators = {
        "sma20": latest["sma20"],
        "ema12": latest["ema12"],
        "ema26": latest["ema26"],
        "macd": latest["MACD_12_26_9"],
        "macd_signal": latest["MACDs_12_26_9"],
        "macd_hist": latest["MACDh_12_26_9"],
        "rsi14": latest["rsi14"],
        "obv": latest["obv"],
        "stoch_k": latest.get("STOCHk_14_3_3"),
        "stoch_d": latest.get("STOCHd_14_3_3"),
        "atr14": latest["atr14"],
        "vwap": latest["vwap"],
    }

    # Convert NumPy types to native Python for JSON serialisation
    indicators = {k: (None if pd.isna(v) else float(v)) for k, v in indicators.items()}

    return IndicatorResponse(
        symbol=symbol,
        exchange=exchange,
        interval=interval,
        timestamp=int(latest["ts"].timestamp() * 1000),
        datetime=latest["ts"].isoformat(),
        indicators=indicators,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("indicator_api:app", host="0.0.0.0", port=8000, reload=True)
