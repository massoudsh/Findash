from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel

from src.data_processing.collection_tasks import fetch_latest_news_task, fetch_intraday_prices_task

router = APIRouter()

class PriceRequest(BaseModel):
    symbol: str
    interval: str = '5min'

@router.post("/fetch_news", status_code=202)
def trigger_fetch_news():
    """Triggers an asynchronous job to fetch the latest financial news."""
    task = fetch_latest_news_task.delay()
    return {"message": "News fetching job started.", "task_id": task.id}

@router.post("/fetch_prices", status_code=202)
def trigger_fetch_prices(request: PriceRequest):
    """Triggers an asynchronous job to fetch intraday prices for a symbol."""
    task = fetch_intraday_prices_task.delay(symbol=request.symbol, interval=request.interval)
    return {"message": f"Price fetching job for {request.symbol} started.", "task_id": task.id} 