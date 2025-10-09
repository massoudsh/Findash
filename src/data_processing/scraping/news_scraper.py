import requests
from bs4 import BeautifulSoup
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class NewsScraper:
    """
    A scraper for fetching news articles from financial news websites.
    """
    
    def __init__(self, url: str = "https://www.finextra.com/news/latestannouncements.aspx"):
        self.url = url

    def fetch_latest_news(self, limit: int = 5) -> List[Dict[str, str]]:
        """
        Fetches the latest news articles from the specified URL.

        Args:
            limit (int): The maximum number of news articles to return.

        Returns:
            A list of dictionaries, where each dictionary represents a news article.
        """
        logger.info(f"Fetching latest news from {self.url}")
        try:
            response = requests.get(self.url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")
            news_items = soup.find_all('div', class_='item')

            news_list = []
            for item in news_items[:limit]:
                title_tag = item.find('a')
                summary_tag = item.find('div', class_='summary')

                if title_tag and summary_tag:
                    title = title_tag.get_text(strip=True)
                    summary = summary_tag.get_text(strip=True)
                    news_list.append({'title': title, 'summary': summary})
            
            logger.info(f"Successfully fetched {len(news_list)} news articles.")
            return news_list

        except requests.RequestException as e:
            logger.error(f"Failed to retrieve news from {self.url}: {e}")
            return []
        except Exception as e:
            logger.error(f"An error occurred during news scraping: {e}")
            return [] 