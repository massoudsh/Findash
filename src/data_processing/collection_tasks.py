import logging
from typing import Dict, Any

from src.core.celery_app import celery_app
from .scraping.news_scraper import NewsScraper
from .api_clients.alpha_vantage_client import AlphaVantageClient

logger = logging.getLogger(__name__)

@celery_app.task(name="data_collection.fetch_latest_news")
def fetch_latest_news_task(limit: int = 10) -> Dict[str, Any]:
    """
    Celery task to fetch the latest financial news.
    """
    logger.info("Kicking off task to fetch latest news.")
    try:
        scraper = NewsScraper()
        news_articles = scraper.fetch_latest_news(limit=limit)
        return {"status": "success", "count": len(news_articles), "articles": news_articles}
    except Exception as e:
        logger.error(f"Error in fetch_latest_news_task: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


@celery_app.task(name="data_collection.fetch_intraday_prices")
def fetch_intraday_prices_task(symbol: str, interval: str = '5min') -> Dict[str, Any]:
    """
    Celery task to fetch intraday pricing data for a given symbol.
    """
    logger.info(f"Kicking off task to fetch intraday prices for {symbol}.")
    try:
        client = AlphaVantageClient()
        data = client.fetch_intraday_data(symbol=symbol, interval=interval)
        
        # We can add data cleaning, validation, and storage logic here later.
        
        # For now, just return a success message and the raw data.
        return {"status": "success", "symbol": symbol, "data": data}
    except Exception as e:
        logger.error(f"Error in fetch_intraday_prices_task for {symbol}: {e}", exc_info=True)
        return {"status": "error", "message": str(e)} 