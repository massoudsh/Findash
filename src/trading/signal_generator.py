import pandas as pd
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class SignalGenerator:
    """
    Generates trading signals based on technical analysis indicators.
    """
    def __init__(self, data: pd.DataFrame):
        if data.empty:
            raise ValueError("Data provided to SignalGenerator cannot be empty.")
        self.data = self._calculate_indicators(data)

    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculates all necessary technical indicators."""
        df = data.copy()
        
        # SMA
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['SMA50'] = df['Close'].rolling(window=50).mean()

        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).fillna(0)
        loss = -delta.where(delta < 0, 0).fillna(0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        return df.dropna()

    def generate_signals_for_period(self) -> pd.DataFrame:
        """
        Generates buy/sell/hold signals for each timestamp in the data.
        Returns a DataFrame with a 'signal' column (1 for Buy, -1 for Sell, 0 for Hold).
        """
        signals = pd.DataFrame(index=self.data.index)
        signals['signal'] = 0

        # Golden Cross / Death Cross
        signals['SMA_signal'] = 0
        signals.loc[self.data['SMA20'] > self.data['SMA50'], 'SMA_signal'] = 1
        signals.loc[self.data['SMA20'] < self.data['SMA50'], 'SMA_signal'] = -1

        # RSI
        signals['RSI_signal'] = 0
        signals.loc[self.data['RSI'] < 30, 'RSI_signal'] = 1
        signals.loc[self.data['RSI'] > 70, 'RSI_signal'] = -1
        
        # MACD
        signals['MACD_signal'] = 0
        signals.loc[self.data['MACD'] > self.data['Signal_Line'], 'MACD_signal'] = 1
        signals.loc[self.data['MACD'] < self.data['Signal_Line'], 'MACD_signal'] = -1
        
        # Combine signals (simple majority vote)
        signals['signal'] = signals[['SMA_signal', 'RSI_signal', 'MACD_signal']].mean(axis=1).apply(np.sign)

        return signals[['signal']] 