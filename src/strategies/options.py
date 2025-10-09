import pandas as pd
from typing import Dict
from .base import Strategy

class VolatilitySpreadStrategy(Strategy):
    """
    An options strategy that capitalizes on the spread between
    implied volatility (IV) and historical volatility (HV).

    - If IV > HV by a certain threshold, it suggests options may be overpriced,
      signaling a good time to sell volatility (e.g., sell a straddle or strangle).
    - If HV > IV, it suggests options may be underpriced, signaling a
      good time to buy volatility (e.g., buy a straddle or strangle).
    """
    def __init__(self, volatility_spread_threshold: float = 0.1):
        """
        Args:
            volatility_spread_threshold (float): The minimum difference between IV and HV
                                                 to trigger a signal.
        """
        self.volatility_spread_threshold = volatility_spread_threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generates BUY/SELL signals based on the IV/HV spread.

        Args:
            data (pd.DataFrame): DataFrame with 'implied_volatility' and
                                 'historical_volatility' columns.
        Returns:
            pd.DataFrame: Original DataFrame with a new 'signal' column.
                          'BUY' means buy volatility (e.g., long straddle).
                          'SELL' means sell volatility (e.g., short straddle).
        """
        if not all(col in data.columns for col in ['implied_volatility', 'historical_volatility']):
            raise ValueError("Input DataFrame must contain 'implied_volatility' and 'historical_volatility' columns.")

        df = data.copy()
        df['volatility_spread'] = df['implied_volatility'] - df['historical_volatility']

        df['signal'] = 'HOLD'

        # Signal to SELL volatility (options may be overpriced)
        df.loc[df['volatility_spread'] > self.volatility_spread_threshold, 'signal'] = 'SELL'

        # Signal to BUY volatility (options may be underpriced)
        df.loc[df['volatility_spread'] < -self.volatility_spread_threshold, 'signal'] = 'BUY'

        return df

    def get_parameters(self) -> Dict:
        return {
            "volatility_spread_threshold": self.volatility_spread_threshold
        } 
 