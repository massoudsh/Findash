import pytest
import requests
from unittest.mock import Mock, patch
from ..Scraping import fetch_real_time_data, scrape_fintech_news

def test_fetch_real_time_data_success(requests_mock):
    """Test successful API call to Alpha Vantage"""
    # Mock response data
    mock_data = {
        "Time Series (1min)": {
            "2024-02-14 12:00:00": {
                "4. close": "150.0",
                "5. volume": "1000"
            }
        }
    }
    
    # Setup mock
    requests_mock.get(
        "https://www.alphavantage.co/query",
        json=mock_data
    )
    
    # Test the function
    result = fetch_real_time_data("AAPL")
    
    assert result["symbol"] == "AAPL"
    assert result["latest_price"] == "150.0"
    assert result["volume"] == "1000"

def test_fetch_real_time_data_error(requests_mock):
    """Test error handling in Alpha Vantage API call"""
    # Setup mock to raise exception
    requests_mock.get(
        "https://www.alphavantage.co/query",
        status_code=500
    )
    
    # Test the function
    result = fetch_real_time_data("AAPL")
    
    assert "error" in result
    assert "API request failed" in result["error"]

def test_scrape_fintech_news_success(requests_mock, sample_html_content):
    """Test successful news scraping"""
    # Setup mock
    requests_mock.get(
        "https://www.finextra.com/news/latestannouncements.aspx",
        text=sample_html_content
    )
    
    # Test the function
    result = scrape_fintech_news()
    
    assert isinstance(result, list)
    assert len(result) > 0
    assert "title" in result[0]
    assert "summary" in result[0]

def test_scrape_fintech_news_error(requests_mock):
    """Test error handling in news scraping"""
    # Setup mock to raise exception
    requests_mock.get(
        "https://www.finextra.com/news/latestannouncements.aspx",
        status_code=500
    )
    
    # Test the function
    result = scrape_fintech_news()
    
    assert isinstance(result, dict)
    assert "error" in result 