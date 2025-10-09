from dataclasses import dataclass

@dataclass
class PositionRisk:
    """
    A data class to hold risk metrics for a single position.
    """
    symbol: str
    position_size: float
    current_price: float
    entry_price: float
    stop_loss: float
    take_profit: float
    max_drawdown: float
    var_95: float  # Value at Risk (95% confidence)
    sharpe_ratio: float 