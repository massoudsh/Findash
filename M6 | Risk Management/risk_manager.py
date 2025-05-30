import numpy as np
from typing import Dict, List, Optional
import pandas as pd
from dataclasses import dataclass
import logging
from datetime import datetime

@dataclass
class PositionRisk:
    symbol: str
    position_size: float
    current_price: float
    entry_price: float
    stop_loss: float
    take_profit: float
    max_drawdown: float
    var_95: float  # Value at Risk (95% confidence)
    sharpe_ratio: float

class RiskManager:
    def __init__(self, 
                 max_portfolio_risk: float = 0.02,  # 2% max risk per portfolio
                 max_position_risk: float = 0.01,   # 1% max risk per position
                 max_correlation: float = 0.7):      # Maximum correlation between positions
        self.max_portfolio_risk = max_portfolio_risk
        self.max_position_risk = max_position_risk
        self.max_correlation = max_correlation
        self.positions: Dict[str, PositionRisk] = {}
        self.logger = logging.getLogger(__name__)

    def calculate_position_size(self, 
                              capital: float,
                              entry_price: float,
                              stop_loss: float,
                              symbol: str) -> float:
        """Calculate the appropriate position size based on risk parameters."""
        risk_amount = capital * self.max_position_risk
        price_risk = abs(entry_price - stop_loss)
        position_size = risk_amount / price_risk
        
        self.logger.info(f"Calculated position size for {symbol}: {position_size}")
        return position_size

    def calculate_value_at_risk(self, 
                              returns: np.ndarray,
                              confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk using historical simulation method."""
        if len(returns) < 100:
            self.logger.warning("Insufficient data for reliable VaR calculation")
            return 0.0
            
        return np.percentile(returns, (1 - confidence_level) * 100)

    def check_correlation_risk(self, 
                             new_symbol: str,
                             historical_data: Dict[str, pd.DataFrame]) -> bool:
        """Check if adding a new position would exceed correlation limits."""
        if not self.positions:
            return True

        new_returns = historical_data[new_symbol]['returns']
        for symbol in self.positions:
            existing_returns = historical_data[symbol]['returns']
            correlation = np.corrcoef(new_returns, existing_returns)[0, 1]
            
            if abs(correlation) > self.max_correlation:
                self.logger.warning(f"High correlation ({correlation:.2f}) between {new_symbol} and {symbol}")
                return False
                
        return True

    def evaluate_trade(self,
                      symbol: str,
                      entry_price: float,
                      stop_loss: float,
                      take_profit: float,
                      historical_data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate a potential trade against risk parameters."""
        # Calculate basic metrics
        risk_reward_ratio = (take_profit - entry_price) / (entry_price - stop_loss)
        returns = historical_data['returns'].values
        volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
        var_95 = self.calculate_value_at_risk(returns)
        
        # Calculate Sharpe Ratio (assuming risk-free rate of 2%)
        excess_returns = returns - 0.02/252  # Daily risk-free rate
        sharpe_ratio = np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)
        
        return {
            'risk_reward_ratio': risk_reward_ratio,
            'volatility': volatility,
            'var_95': var_95,
            'sharpe_ratio': sharpe_ratio
        }

    def can_take_trade(self,
                      symbol: str,
                      entry_price: float,
                      stop_loss: float,
                      take_profit: float,
                      capital: float,
                      historical_data: Dict[str, pd.DataFrame]) -> tuple[bool, str]:
        """Determine if a trade can be taken based on all risk parameters."""
        # Check correlation risk
        if not self.check_correlation_risk(symbol, historical_data):
            return False, "Correlation risk too high"

        # Calculate position size
        position_size = self.calculate_position_size(capital, entry_price, stop_loss, symbol)
        
        # Evaluate trade metrics
        metrics = self.evaluate_trade(symbol, entry_price, stop_loss, take_profit, historical_data[symbol])
        
        # Check risk-reward ratio
        if metrics['risk_reward_ratio'] < 2:
            return False, "Risk-reward ratio below minimum threshold"
            
        # Check portfolio VaR
        portfolio_var = self._calculate_portfolio_var()
        if portfolio_var > self.max_portfolio_risk:
            return False, "Portfolio VaR exceeds maximum threshold"
            
        return True, "Trade meets risk parameters"

    def _calculate_portfolio_var(self) -> float:
        """Calculate the overall portfolio Value at Risk."""
        if not self.positions:
            return 0.0
            
        position_vars = [pos.var_95 * pos.position_size for pos in self.positions.values()]
        return np.sum(position_vars)

    def update_position_risk(self,
                           symbol: str,
                           current_price: float,
                           historical_data: pd.DataFrame) -> None:
        """Update risk metrics for an existing position."""
        if symbol not in self.positions:
            self.logger.warning(f"Position {symbol} not found in risk manager")
            return

        position = self.positions[symbol]
        returns = historical_data['returns'].values
        
        position.current_price = current_price
        position.var_95 = self.calculate_value_at_risk(returns)
        position.max_drawdown = (position.entry_price - current_price) / position.entry_price
        
        self.logger.info(f"Updated risk metrics for {symbol}: VaR={position.var_95:.4f}, DrawDown={position.max_drawdown:.4f}") 