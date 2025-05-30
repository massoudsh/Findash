# UNsloth.py

# Import necessary libraries
from unsloth.data import YahooFinance
from unsloth.strategy import Backtest, Strategy
from django.http import JsonResponse
from celery import shared_task
import duckdb

# Fetching data for a specific stock
data = YahooFinance().get("AAPL", start="2023-01-01", end="2023-12-31", interval="1d")

class MyStrategy(Strategy):
    def init(self):
        # Initialize the Simple Moving Average (SMA) indicator
        self.sma = self.indicator(self.data.close.rolling(20).mean)

    def next(self):
        # Trading logic based on SMA
        if self.data.close[-1] > self.sma[-1]:
            self.buy()
        elif self.data.close[-1] < self.sma[-1]:
            self.sell()

# Running the backtest
bt = Backtest(data, MyStrategy)
bt.run()
bt.plot()

# Database connection and data insertion
conn = duckdb.connect('trading_data.duckdb')
conn.execute("CREATE TABLE IF NOT EXISTS stock_data (date DATE, close FLOAT)")
conn.execute("INSERT INTO stock_data SELECT * FROM dataframe", dataframe=data)

def fetch_stock_data(request):
    # Fetch stock data and return as JSON response
    data = YahooFinance().get("AAPL", start="2023-01-01", end="2023-12-31", interval="1d")
    return JsonResponse(data.to_dict())

@shared_task
def update_stock_data():
    # Task to update stock data
    data = YahooFinance().get("AAPL", start="2023-01-01", end="2023-12-31", interval="1d")
    # Save to DB or perform analysis
