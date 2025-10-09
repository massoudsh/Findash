from typing import List
import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel
import logging

from src.core.logging_config import setup_logging
from src.core.exceptions import APIError

setup_logging(log_level='INFO')
logger = logging.getLogger(__name__)

class NewsArticle(BaseModel):
    title: str
    summary: str
    url: str

def scrape_fintech_news(limit: int = 5) -> List[NewsArticle]:
    """Scrape latest fintech news articles from Finextra."""
    url = "https://www.finextra.com/news/latestannouncements.aspx"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        news_items = soup.find_all('div', class_='item')
        news_list = []
        for item in news_items[:limit]:
            title_tag = item.find('a')
            summary_tag = item.find('div', class_='summary')
            if not title_tag or not summary_tag:
                continue
            title = title_tag.get_text(strip=True)
            summary = summary_tag.get_text(strip=True)
            link = title_tag['href'] if title_tag.has_attr('href') else url
            news_list.append(NewsArticle(title=title, summary=summary, url=link))
        return news_list
    except requests.RequestException as e:
        logger.error(f"Failed to fetch news: {e}")
        raise APIError(f"Failed to fetch news: {str(e)}") 