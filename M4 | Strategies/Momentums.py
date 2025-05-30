import talib
import numpy as np

def momentum_strategy(prices):
    rsi = talib.RSI(np.array(prices), timeperiod=14)
    if rsi[-1] < 30:
        return "BUY"
    elif rsi[-1] > 70:
        return "SELL"
    return "HOLD"

def mean_reversion_strategy(prices):
    upper_band, middle_band, lower_band = talib.BBANDS(np.array(prices), timeperiod=20)
    if prices[-1] < lower_band[-1]:
        return "BUY"
    elif prices[-1] > upper_band[-1]:
        return "SELL"
    return "HOLD"