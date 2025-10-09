from typing import Dict, Any
import requests
from pydantic import BaseModel
import logging

from src.core.logging_config import setup_logging
from src.core.exceptions import APIError

setup_logging(log_level='INFO')
logger = logging.getLogger(__name__)

class RedditSentiment(BaseModel):
    subreddit: str
    positive: int
    negative: int
    neutral: int
    top_keywords: list[str]
    sample_size: int

def analyze_reddit_sentiment(subreddit: str, limit: int = 100) -> RedditSentiment:
    """Analyze sentiment of recent posts in a subreddit using simple keyword analysis."""
    url = f"https://www.reddit.com/r/{subreddit}/new.json?limit={limit}"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; sentiment-bot/1.0)"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        posts = [post['data']['title'] for post in data['data']['children']]
        # Simple sentiment analysis (placeholder: count positive/negative/neutral keywords)
        positive_keywords = ["good", "great", "bull", "moon", "win", "profit"]
        negative_keywords = ["bad", "bear", "down", "loss", "crash", "fail"]
        positive = negative = neutral = 0
        keyword_counter = {}
        for title in posts:
            title_lower = title.lower()
            found = False
            for word in positive_keywords:
                if word in title_lower:
                    positive += 1
                    keyword_counter[word] = keyword_counter.get(word, 0) + 1
                    found = True
            for word in negative_keywords:
                if word in title_lower:
                    negative += 1
                    keyword_counter[word] = keyword_counter.get(word, 0) + 1
                    found = True
            if not found:
                neutral += 1
        top_keywords = sorted(keyword_counter, key=keyword_counter.get, reverse=True)[:5]
        return RedditSentiment(
            subreddit=subreddit,
            positive=positive,
            negative=negative,
            neutral=neutral,
            top_keywords=top_keywords,
            sample_size=len(posts)
        )
    except requests.RequestException as e:
        logger.error(f"Failed to fetch Reddit data: {e}")
        raise APIError(f"Failed to fetch Reddit data: {str(e)}")
    except Exception as e:
        logger.error(f"Reddit sentiment analysis failed: {e}")
        raise APIError(f"Reddit sentiment analysis failed: {str(e)}") 