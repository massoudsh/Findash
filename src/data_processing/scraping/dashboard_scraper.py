import logging
from typing import List
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

logger = logging.getLogger(__name__)

class DashboardScraper:
    """
    A scraper for fetching data from a dynamic dashboard using Selenium.
    """
    def __init__(self, dashboard_url: str, chrome_driver_path: str):
        self.dashboard_url = dashboard_url
        self.chrome_driver_path = chrome_driver_path

    def scrape_stock_prices(self, timeout: int = 10) -> List[str]:
        """
        Scrapes stock prices from the dashboard.
        
        Args:
            timeout (int): The maximum time to wait for elements to load.

        Returns:
            A list of strings, each representing a stock price.
        """
        logger.info(f"Scraping dashboard at {self.dashboard_url}")
        
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        
        service = Service(self.chrome_driver_path)
        
        try:
            with webdriver.Chrome(service=service, options=options) as driver:
                driver.get(self.dashboard_url)
                
                wait = WebDriverWait(driver, timeout)
                stock_elements = wait.until(
                    EC.presence_of_all_elements_located((By.CLASS_NAME, 'stock-price'))
                )
                
                prices = [stock.text for stock in stock_elements]
                logger.info(f"Successfully scraped {len(prices)} stock prices.")
                return prices
        except Exception as e:
            logger.error(f"Failed to scrape dashboard: {e}", exc_info=True)
            return [] 