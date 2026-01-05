"""
DEPRECATED: This file has been integrated into portfolio_manager.py

This module is kept for backward compatibility only.
All functionality is now available in:
- src.portfolio.portfolio_manager.PortfolioOptimizer (unified class)

Please update imports to use:
    from src.portfolio.portfolio_manager import PortfolioOptimizer
    
This file will be removed in a future version.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Re-export from unified module for backward compatibility
from .portfolio_manager import PortfolioOptimizer

__all__ = ['PortfolioOptimizer'] 