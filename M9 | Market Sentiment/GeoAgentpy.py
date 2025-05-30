import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import json
from datetime import datetime, timedelta
import sys
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import defaultdict

# System integration - import from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from M6___Risk_Management.portfolio_optimizer import PortfolioOptimizer
    from M4___Strategies.strategy_framework import StrategySignal
    from M3___Real_Time_Processing.event_processor import EventProcessor
    from M11___Visulization.visualization_engine import create_heatmap, plot_time_series
except ImportError:
    logging.warning("Some Quantum Trading Matrix modules could not be imported. Limited functionality available.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)), "geopolitical_agent.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("GeopoliticalAgent")

# Constants
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
API_ENDPOINTS = {
    "GPI": "https://visionofhumanity.org/wp-content/uploads/2023/06/GPI-2023-overall-scores-and-rankings.csv",  # Example
    "CPI": "https://www.transparency.org/en/cpi/2022/index/",  # Example
    "WGI": "https://info.worldbank.org/governance/wgi/",  # Example
    "ACLED": "https://acleddata.com/data-export-tool/",  # Example
    "GDELT": "https://www.gdeltproject.org/data.html#rawdatafiles"  # Example
}

# =============================
# 1. Enhanced Data Acquisition
# =============================
def fetch_index_data(api_url: str, index_name: str = None, cache: bool = True) -> pd.DataFrame:
    """
    Fetch geopolitical index data from the specified API with caching support.
    
    Args:
        api_url: URL of the API endpoint
        index_name: Name of the index for caching purposes
        cache: Whether to use cached data if available
        
    Returns:
        DataFrame with index data
    """
    # Create cache directory if needed
    cache_dir = os.path.join(DEFAULT_DATA_DIR, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_file = None
    if index_name:
        cache_file = os.path.join(cache_dir, f"{index_name.lower()}_data.csv")
    
    # Check cache first if enabled
    if cache and cache_file and os.path.exists(cache_file):
        cache_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if datetime.now() - cache_time < timedelta(days=1):  # Cache for 1 day
            logger.info(f"Loading {index_name} data from cache")
            return pd.read_csv(cache_file)
    
    # Fetch fresh data
    try:
        logger.info(f"Fetching data from {api_url}")
    response = requests.get(api_url)
        response.raise_for_status()  # Raise error for bad status codes
        
        if response.headers.get('content-type') == 'application/json':
            data = pd.DataFrame(response.json())
    else:
            # Assuming CSV data
            data = pd.read_csv(response.text)
        
        # Cache the data if caching is enabled
        if cache and cache_file:
            logger.info(f"Caching {index_name} data")
            data.to_csv(cache_file, index=False)
            
        return data
        
    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
        
        # If cache exists but is outdated, use it as fallback
        if cache_file and os.path.exists(cache_file):
            logger.warning(f"Using outdated cache for {index_name} as fallback")
            return pd.read_csv(cache_file)
            
        # Create dummy data as last resort
        logger.warning(f"Creating dummy data for {index_name}")
        return pd.DataFrame({
            'Country': ['USA', 'China', 'Germany', 'Japan', 'UK'],
            'Value': [0.8, 0.7, 0.9, 0.85, 0.82],
            'Rank': [10, 15, 5, 8, 9]
        })

# =============================
# 2. Enhanced Data Processing
# =============================
def normalize_index(data: pd.DataFrame, column: str, invert: bool = False) -> pd.DataFrame:
    """
    Normalize a column of index values to a scale of 0 to 1.
    
    Args:
        data: DataFrame containing the index data
        column: Column name to normalize
        invert: Whether to invert the normalization (1 = good, 0 = bad)
        
    Returns:
        DataFrame with an additional normalized column
    """
    if column not in data.columns:
        logger.error(f"Column {column} not found in data")
        return data
        
    min_val = data[column].min()
    max_val = data[column].max()
    
    # Check for division by zero
    if max_val == min_val:
        data[f"{column}_normalized"] = 0.5
    else:
    data[f"{column}_normalized"] = (data[column] - min_val) / (max_val - min_val)
    
    # Invert if needed (e.g., for indices where higher values mean worse outcomes)
    if invert:
        data[f"{column}_normalized"] = 1 - data[f"{column}_normalized"]
        
    return data

def clean_country_names(data: pd.DataFrame, name_column: str = 'Country') -> pd.DataFrame:
    """
    Standardize country names across different datasets.
    
    Args:
        data: DataFrame containing country data
        name_column: Name of the column containing country names
        
    Returns:
        DataFrame with standardized country names
    """
    # Common country name variations
    country_map = {
        'United States': 'USA',
        'United States of America': 'USA',
        'US': 'USA',
        'United Kingdom': 'UK',
        'Great Britain': 'UK',
        'Russian Federation': 'Russia',
        "People's Republic of China": 'China',
        'Korea, Republic of': 'South Korea',
        'Korea, Democratic People\'s Republic of': 'North Korea',
        'Iran, Islamic Republic of': 'Iran'
    }
    
    if name_column in data.columns:
        data[name_column] = data[name_column].replace(country_map)
    
    return data

# =============================
# 3. Enhanced Geopolitical Agent
# =============================
class GeopoliticalAgent:
    """
    Advanced geopolitical risk analysis agent for trading decisions.
    
    This agent:
    1. Collects data from multiple geopolitical indices
    2. Assesses country and regional risks
    3. Generates trading signals based on geopolitical events
    4. Integrates with the broader trading platform
    5. Provides visualization of geopolitical risks
    """
    
    def __init__(self, 
                 data_sources: Dict[str, str] = None, 
                 data_dir: str = DEFAULT_DATA_DIR,
                 auto_fetch: bool = True):
        """
        Initialize the GeopoliticalAgent with data sources and configuration.
        
        Args:
            data_sources: Dictionary of index names and their API URLs
            data_dir: Directory to store data and results
            auto_fetch: Whether to automatically fetch data on initialization
        """
        # Initialize directories
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize data sources
        self.data_sources = data_sources or API_ENDPOINTS
        logger.info(f"Initialized GeopoliticalAgent with {len(self.data_sources)} data sources")
        
        # Initialize data storage
        self.data = {}
        self.risk_scores = {}
        self.country_mapping = self._load_country_to_market_mapping()
        self.last_update = {}
        
        # Initialize integration components
        self.signals = []
        self.event_history = []
        
        # Fetch data if auto_fetch is enabled
        if auto_fetch:
            self.fetch_all_data()
            
    def _load_country_to_market_mapping(self) -> Dict[str, List[str]]:
        """
        Load mapping of countries to related markets/tickers.
        
        Returns:
            Dictionary mapping country names to related market tickers
        """
        mapping_file = os.path.join(self.data_dir, "country_market_mapping.json")
        
        # Default basic mapping
        default_mapping = {
            'USA': ['SPY', 'QQQ', 'DIA', 'IWM', 'USD'],
            'China': ['FXI', 'MCHI', 'KWEB', 'CNY'],
            'Japan': ['EWJ', 'DXJ', 'JPY'],
            'Germany': ['EWG', 'DAX', 'EUR'],
            'UK': ['EWU', 'GBP'],
            'Russia': ['RSX', 'RUB'],
            'Brazil': ['EWZ', 'BRL'],
            'India': ['INDA', 'INR'],
            'Australia': ['EWA', 'AUD'],
            'Canada': ['EWC', 'CAD'],
            'South Korea': ['EWY', 'KRW'],
            'Mexico': ['EWW', 'MXN'],
            'Global': ['VT', 'ACWI', 'VEU']
        }
        
        try:
            if os.path.exists(mapping_file):
                with open(mapping_file, 'r') as f:
                    mapping = json.load(f)
                logger.info(f"Loaded country-market mapping from {mapping_file}")
                return mapping
        except Exception as e:
            logger.warning(f"Could not load country-market mapping: {e}")
            
        return default_mapping
    
    def fetch_all_data(self) -> None:
        """
        Fetch data from all configured data sources.
        """
        for name, url in self.data_sources.items():
            try:
                logger.info(f"Fetching data for {name}...")
                self.data[name] = fetch_index_data(url, name)
                self.last_update[name] = datetime.now()
            except Exception as e:
                logger.error(f"Error fetching {name} data: {e}")
    
    def preprocess_data(self) -> None:
        """
        Preprocess all data sources for consistency and analysis.
        """
        for name, df in self.data.items():
            try:
                logger.info(f"Preprocessing data for {name}...")
                
                # Clean country names
                df = clean_country_names(df)
                
                # Determine value column (may vary by dataset)
                value_col = next((col for col in df.columns if col in 
                                ['Value', 'Score', 'Index', 'value', 'score', 'index']), df.columns[1])
                
                # Determine whether to invert based on index type
                # For some indices, higher values are better (e.g., HDI)
                # For others, higher values are worse (e.g., CPI, where higher = more corrupt)
                invert = name in ['CPI', 'ACLED']
                
                # Normalize the data
                df = normalize_index(df, value_col, invert)
                
                # Update the processed data
                self.data[name] = df
                
            except Exception as e:
                logger.error(f"Error preprocessing {name} data: {e}")
    
    def calculate_risk_scores(self) -> Dict[str, float]:
        """
        Calculate comprehensive geopolitical risk scores for all countries.
        
        Returns:
            Dictionary mapping countries to risk scores (0-1, higher = riskier)
        """
        # Collect all countries across all indices
        all_countries = set()
        for df in self.data.values():
            country_col = next((col for col in df.columns if col in 
                             ['Country', 'country', 'Country Name', 'Nation']), df.columns[0])
            all_countries.update(df[country_col].unique())
        
        # Calculate risk scores for each country
        risk_scores = {}
        for country in all_countries:
            try:
                # Collect normalized scores from each index
                scores = []
                weights = []
                
                for name, df in self.data.items():
                    country_col = next((col for col in df.columns if col in 
                                     ['Country', 'country', 'Country Name', 'Nation']), df.columns[0])
                    
                    # Find normalized column
                    norm_cols = [col for col in df.columns if 'normalized' in col]
                    if not norm_cols:
                        continue
                        
                    # Get data for this country
                    country_data = df[df[country_col] == country]
                    if len(country_data) == 0:
                        continue
                        
                    # Get normalized value
                    norm_val = country_data[norm_cols[0]].values[0]
                    
                    # Weight by index importance (could be adjusted)
                    if name == 'GPI':  # Global Peace Index
                        weight = 0.5
                    elif name == 'CPI':  # Corruption Perception Index
                        weight = 0.3
                    else:
                        weight = 0.2
                        
                    scores.append(norm_val)
                    weights.append(weight)
                
                # Calculate weighted average if we have scores
                if scores:
                    # Normalize weights to sum to 1
                    weights = [w/sum(weights) for w in weights]
                    risk_score = sum(s * w for s, w in zip(scores, weights))
                    risk_scores[country] = risk_score
                    
            except Exception as e:
                logger.error(f"Error calculating risk for {country}: {e}")
        
        self.risk_scores = risk_scores
        return risk_scores
    
    def assess_risk(self, country: str) -> float:
        """
        Retrieve the geopolitical risk score for a specific country.
        
        Args:
            country: Name of the country to assess
            
        Returns:
            Risk score (0 to 1, where higher means riskier)
        """
        # If we haven't calculated risk scores yet, do it now
        if not self.risk_scores:
            self.calculate_risk_scores()
            
        # Return the risk score if available
        if country in self.risk_scores:
            return self.risk_scores[country]
            
        # Try to find similar country names
        for c in self.risk_scores:
            if country.lower() in c.lower() or c.lower() in country.lower():
                logger.warning(f"Using {c} as a match for {country}")
                return self.risk_scores[c]
                
        # Return global average if no match found
        logger.warning(f"No risk data found for {country}, using global average")
        if self.risk_scores:
            return sum(self.risk_scores.values()) / len(self.risk_scores)
        else:
            return 0.5  # Default to medium risk
    
    def get_affected_markets(self, country: str) -> List[str]:
        """
        Get list of markets/tickers affected by a country's geopolitical situation.
        
        Args:
            country: The country to analyze
            
        Returns:
            List of tickers related to the country
        """
        # Look for exact match
        if country in self.country_mapping:
            return self.country_mapping[country]
            
        # Look for partial matches
        for c, markets in self.country_mapping.items():
            if country.lower() in c.lower() or c.lower() in country.lower():
                return markets
                
        # If no match, return global markets
        return self.country_mapping.get('Global', ['SPY', 'VT', 'ACWI'])
    
    def generate_trading_signals(self, risk_threshold: float = 0.7) -> List[Dict]:
        """
        Generate trading signals based on geopolitical risk assessment.
        
        Args:
            risk_threshold: Threshold above which to generate risk signals
            
        Returns:
            List of trading signals with affected markets
        """
        signals = []
        
        # If we haven't calculated risk scores yet, do it now
        if not self.risk_scores:
            self.calculate_risk_scores()
            
        # Assess each country's risk level
        for country, risk in self.risk_scores.items():
            # High risk countries generate signals
            if risk > risk_threshold:
                affected_markets = self.get_affected_markets(country)
                
                for market in affected_markets:
                    # Higher risk = stronger signal
                    signal_strength = (risk - risk_threshold) / (1 - risk_threshold)
                    
                    # Create signal
                    signal = {
                        'type': 'RISK_ALERT',
                        'source': 'geopolitical',
                        'country': country,
                        'market': market,
                        'direction': 'BEARISH',
                        'strength': signal_strength,
                        'timestamp': datetime.now().isoformat(),
                        'risk_score': risk,
                        'reason': f"High geopolitical risk in {country} affecting {market}"
                    }
                    
                    signals.append(signal)
                    logger.info(f"Generated {signal['direction']} signal for {market} due to {country} risk")
        
        self.signals = signals
        return signals
    
    def integrate_with_risk_manager(self, risk_manager=None) -> Dict[str, float]:
        """
        Integrate geopolitical risk assessment with the risk management system.
        
        Args:
            risk_manager: Risk management module instance (optional)
            
        Returns:
            Dictionary of risk adjustments by market
        """
        # Calculate risk adjustments based on geopolitical factors
        risk_adjustments = {}
        
        for country, risk in self.risk_scores.items():
            affected_markets = self.get_affected_markets(country)
            
            for market in affected_markets:
                # Calculate risk adjustment factor (0.8-1.2 range)
                # Higher country risk = higher adjustment factor
                adjustment = 0.8 + 0.4 * risk  # Scales from 0.8 (low risk) to 1.2 (high risk)
                
                if market in risk_adjustments:
                    # Take the higher adjustment if multiple countries affect this market
                    risk_adjustments[market] = max(risk_adjustments[market], adjustment)
                else:
                    risk_adjustments[market] = adjustment
        
        # If risk_manager is provided, apply the adjustments
        if risk_manager:
            logger.info("Applying geopolitical risk adjustments to risk manager")
            try:
                risk_manager.apply_external_risk_factors(risk_adjustments)
            except Exception as e:
                logger.error(f"Error applying risk adjustments: {e}")
        
        return risk_adjustments
    
    def integrate_with_strategy(self, strategy=None) -> List[Dict]:
        """
        Integrate geopolitical insights with trading strategy.
        
        Args:
            strategy: Strategy module instance (optional)
            
        Returns:
            List of strategy signals incorporating geopolitical factors
        """
        # Generate trading signals if we haven't already
        if not self.signals:
            self.generate_trading_signals()
        
        # Convert to strategy signals if strategy module is available
        strategy_signals = []
        
        if strategy and hasattr(strategy, 'add_signal'):
            logger.info("Integrating geopolitical signals with strategy module")
            try:
                for signal in self.signals:
                    # Convert our signal format to the strategy's format
                    strategy_signal = StrategySignal(
                        ticker=signal['market'],
                        signal_type=signal['direction'],
                        confidence=signal['strength'],
                        source="GeopoliticalAgent",
                        metadata={
                            'country': signal['country'],
                            'risk_score': signal['risk_score'],
                            'reason': signal['reason']
                        }
                    )
                    strategy.add_signal(strategy_signal)
                    strategy_signals.append(strategy_signal)
            except Exception as e:
                logger.error(f"Error integrating with strategy module: {e}")
        
        return strategy_signals

    # =============================
    # 4. Visualization Methods
    # =============================
    def plot_risk_scores(self, countries: List[str] = None, 
                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize the geopolitical risk scores for multiple countries.
        
        Args:
            countries: List of country names to assess (None = top 10 riskiest)
            save_path: Path to save the plot (if None, display only)
            
        Returns:
            Matplotlib figure object
        """
        # If we haven't calculated risk scores yet, do it now
        if not self.risk_scores:
            self.calculate_risk_scores()
            
        # If no countries specified, use the top 10 riskiest
        if not countries:
            countries = sorted(self.risk_scores.items(), 
                             key=lambda x: x[1], reverse=True)[:10]
            countries = [c for c, _ in countries]
            
        # Filter for countries we have data for
        countries = [c for c in countries if c in self.risk_scores]
        if not countries:
            logger.error("No valid countries to plot")
            return None
            
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Get scores
        scores = [self.risk_scores[country] for country in countries]
        
        # Create color gradient (green to red)
        colors = [(0.2, 0.8, 0.2, 1.0) if s < 0.5 else 
                 (min(1.0, 2 * s - 0.5), max(0.0, 1.5 - 2 * s), 0.0, 1.0) 
                 for s in scores]
        
        # Plot bars
        bars = ax.bar(countries, scores, color=colors)
        
        # Add value labels on top of bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Add styling
        ax.set_xlabel('Country', fontsize=12, fontweight='bold')
        ax.set_ylabel('Risk Score', fontsize=12, fontweight='bold')
        ax.set_title('Geopolitical Risk Assessment', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add risk level indicators
        ax.axhspan(0, 0.3, alpha=0.2, color='green', label='Low Risk')
        ax.axhspan(0.3, 0.7, alpha=0.2, color='yellow', label='Medium Risk')
        ax.axhspan(0.7, 1.0, alpha=0.2, color='red', label='High Risk')
        
        ax.legend()
        
        plt.tight_layout()
        
        # Save or display
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Risk score plot saved to {save_path}")
        
        return fig
    
    def create_risk_heatmap(self, regions: bool = True, 
                          save_path: Optional[str] = None) -> None:
        """
        Create a world heatmap of geopolitical risk.
        
        Args:
            regions: Whether to aggregate by regions instead of individual countries
            save_path: Path to save the heatmap image
        """
        try:
            # We'll use the visualization module if available
            if 'create_heatmap' in globals():
                logger.info("Creating geopolitical risk heatmap using visualization module")
                
                # Prepare data for heatmap
                risk_data = pd.DataFrame(list(self.risk_scores.items()), 
                                       columns=['Country', 'Risk'])
                
                # Create title and description
                title = "Global Geopolitical Risk Heatmap"
                description = f"Generated on {datetime.now().strftime('%Y-%m-%d')} | " \
                              f"Data sources: {', '.join(self.data.keys())}"
                
                # Call the visualization module's heatmap function
                create_heatmap(
                    data=risk_data,
                    location_column='Country',
                    value_column='Risk',
                    title=title,
                    description=description,
                    colorscale='Reds',
                    aggregate_regions=regions,
                    filename=save_path
                )
                
            else:
                # Fallback to a basic matplotlib map if visualization module not available
                logger.warning("Visualization module not available, using basic plot")
                
                fig, ax = plt.subplots(figsize=(15, 8))
                ax.text(0.5, 0.5, "Geopolitical Risk Heatmap\n\n(Visualization module required for actual map)",
                       ha='center', va='center', fontsize=14)
                
                if save_path:
                    plt.savefig(save_path)
                    
        except Exception as e:
            logger.error(f"Error creating risk heatmap: {e}")
    
    def save_analysis_results(self, filename: Optional[str] = None) -> str:
        """
        Save risk analysis results to disk.
        
        Args:
            filename: Filename to save data (if None, generates based on timestamp)
            
        Returns:
            Path to saved file
        """
        if not self.risk_scores:
            logger.error("No risk scores to save")
            return None
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"geopolitical_risk_{timestamp}.json"
        
        # Create full path
        file_path = os.path.join(self.data_dir, filename)
        
        try:
            # Prepare data for saving
            save_data = {
                'risk_scores': self.risk_scores,
                'signals': self.signals,
                'timestamp': datetime.now().isoformat(),
                'data_sources': list(self.data.keys())
            }
            
            # Save to JSON
            with open(file_path, 'w') as f:
                json.dump(save_data, f, indent=4)
            
            logger.info(f"Analysis results saved to {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error saving analysis results: {e}")
            return None

    def load_analysis_results(self, file_path: str) -> bool:
        """
        Load previously saved analysis results.
        
        Args:
            file_path: Path to saved analysis results
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(file_path, 'r') as f:
                loaded_data = json.load(f)
            
            # Load data
            self.risk_scores = loaded_data.get('risk_scores', {})
            self.signals = loaded_data.get('signals', [])
            
            logger.info(f"Loaded analysis results from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading analysis results: {e}")
            return False

# =============================
# 5. System Integration Functions
# =============================
def integrate_with_quantum_trading_matrix(agent: GeopoliticalAgent) -> Dict:
    """
    Integrate the GeopoliticalAgent with the Quantum Trading Matrix system.
    
    Args:
        agent: Initialized GeopoliticalAgent instance
        
    Returns:
        Dictionary with integration results
    """
    results = {
        'risk_adjustments': {},
        'trading_signals': [],
        'strategy_signals': [],
        'visualizations': []
    }
    
    logger.info("Starting integration with Quantum Trading Matrix")
    
    # 1. Generate risk scores
    agent.calculate_risk_scores()
    
    # 2. Generate trading signals
    results['trading_signals'] = agent.generate_trading_signals()
    
    # 3. Integrate with risk management
    try:
        # Import dynamically to avoid circular imports
        from M6___Risk_Management.risk_manager import RiskManager
        risk_manager = RiskManager()
        results['risk_adjustments'] = agent.integrate_with_risk_manager(risk_manager)
    except ImportError:
        logger.warning("Risk management module not available, skipping integration")
        results['risk_adjustments'] = agent.integrate_with_risk_manager()
    
    # 4. Integrate with strategy
    try:
        # Import dynamically to avoid circular imports
        from M4___Strategies.strategy_manager import StrategyManager
        strategy_manager = StrategyManager()
        results['strategy_signals'] = agent.integrate_with_strategy(strategy_manager)
    except ImportError:
        logger.warning("Strategy module not available, skipping integration")
        results['strategy_signals'] = agent.integrate_with_strategy()
    
    # 5. Generate visualizations
    visuals_dir = os.path.join(agent.data_dir, "visualizations")
    os.makedirs(visuals_dir, exist_ok=True)
    
    # Risk score chart
    risk_chart_path = os.path.join(visuals_dir, "geopolitical_risk_chart.png")
    agent.plot_risk_scores(save_path=risk_chart_path)
    results['visualizations'].append(risk_chart_path)
    
    # Risk heatmap
    risk_map_path = os.path.join(visuals_dir, "geopolitical_risk_map.png")
    agent.create_risk_heatmap(save_path=risk_map_path)
    results['visualizations'].append(risk_map_path)
    
    # 6. Save results
    results_path = agent.save_analysis_results()
    results['results_file'] = results_path
    
    logger.info("Geopolitical agent integration complete")
    return results

# =============================
# 6. Main Program
# =============================
if __name__ == "__main__":
    # Ensure data directory exists
    os.makedirs(DEFAULT_DATA_DIR, exist_ok=True)
    
    # Alternative data sources (more realistic for a production system)
    # For demo purposes, we'll use dummy URLs. In production, use real data sources
    data_sources = {
        "GPI": "https://visionofhumanity.org/wp-content/uploads/2023/06/GPI-2023-overall-scores-and-rankings.csv",
        "CPI": "https://www.transparency.org/en/cpi/2022/index/",
        "WGI": "https://info.worldbank.org/governance/wgi/",
        "ACLED": "https://acleddata.com/data-export-tool/",
        "GDELT": "https://www.gdeltproject.org/data.html#rawdatafiles"
    }

    # Initialize the agent
    print("Initializing GeopoliticalAgent...")
    agent = GeopoliticalAgent(data_sources, auto_fetch=False)
    
    # For demo purposes, load dummy data
    print("Loading sample data...")
    dummy_data = {
        'GPI': pd.DataFrame({
            'Country': ['USA', 'China', 'Russia', 'Germany', 'Japan', 'UK', 'France', 'India', 'Brazil', 'Canada'],
            'Value': [0.7, 0.6, 0.85, 0.4, 0.3, 0.5, 0.45, 0.65, 0.55, 0.35],
            'Rank': [10, 15, 5, 30, 40, 20, 25, 12, 18, 35]
        }),
        'CPI': pd.DataFrame({
            'Country': ['USA', 'China', 'Russia', 'Germany', 'Japan', 'UK', 'France', 'India', 'Brazil', 'Canada'],
            'Value': [0.3, 0.6, 0.7, 0.2, 0.15, 0.25, 0.3, 0.5, 0.55, 0.2],
            'Rank': [25, 78, 136, 10, 8, 11, 22, 85, 96, 13]
        })
    }
    
    # For demo purposes, use sample data
    agent.data = dummy_data

    # Preprocess the data
    print("Preprocessing data...")
    agent.preprocess_data()

    # Calculate risk scores
    print("Calculating risk scores...")
    risk_scores = agent.calculate_risk_scores()
    
    # Print risk scores for specific countries
    countries = ["USA", "Russia", "Germany", "China", "Japan"]
    print("\nGeopolitical Risk Scores:")
    for country in countries:
        risk_score = agent.assess_risk(country)
        print(f"Risk Score for {country}: {risk_score:.2f}")

    # Generate trading signals
    print("\nGenerating trading signals...")
    signals = agent.generate_trading_signals(risk_threshold=0.6)
    if signals:
        print(f"Generated {len(signals)} trading signals")
        for signal in signals[:3]:  # Show first 3 signals
            print(f"  {signal['direction']} signal for {signal['market']} due to {signal['country']} risk")
    else:
        print("No trading signals generated")

    # Visualize the risk scores
    print("\nCreating risk score visualization...")
    agent.plot_risk_scores(countries)
    
    # Save analysis results
    print("\nSaving analysis results...")
    agent.save_analysis_results()
    
    # Demonstrate integration with Quantum Trading Matrix
    print("\nIntegrating with Quantum Trading Matrix...")
    try:
        results = integrate_with_quantum_trading_matrix(agent)
        print("Integration complete")
    except Exception as e:
        print(f"Error during integration: {e}")