import vaex
import pandas as pd
import numpy as np

# Initialize Vaex for financial data processing
def init_vaex_financial():
    """Initialize Vaex configuration optimized for financial applications"""
    # Set memory optimization parameters
    vaex.settings.main.thread_count = 8  # Adjust based on your CPU
    vaex.settings.main.chunk_size = 1_000_000  # Optimize for financial time series
    
def load_financial_data(file_path, format='csv'):
    """Load financial data into Vaex dataframe with optimized settings"""
    if format.lower() == 'csv':
        df = vaex.from_csv(file_path, 
                          chunk_size=1_000_000,
                          convert=True)  # Convert to efficient arrow format
    elif format.lower() == 'parquet':
        df = vaex.open(file_path)
    else:
        raise ValueError("Supported formats are 'csv' and 'parquet'")
    
    return df

def optimize_financial_calculations(df):
    """Apply financial-specific optimizations to Vaex dataframe"""
    # Enable memory mapping for large datasets
    df = df.memory_mapped()
    
    # Pre-compute commonly used financial calculations
    if 'close' in df.columns:
        # Calculate returns using Vaex's efficient computations
        df['returns'] = df.close.diff() / df.close.shift(1)
        
        # Add rolling statistics
        df['rolling_mean'] = df.close.rolling(window=20).mean()
        df['rolling_std'] = df.close.rolling(window=20).std()
        
    return df

def create_financial_features(df):
    """Create common financial features using Vaex's efficient computations"""
    # Technical indicators
    if all(col in df.columns for col in ['high', 'low', 'close']):
        # True Range
        df['tr'] = df.eval('max(high - low, abs(high - close.shift(1)), abs(low - close.shift(1)))')
        
        # Average True Range (14-period)
        df['atr'] = df.tr.rolling(window=14).mean()
        
        # Relative Strength Index (14-period)
        delta = df.close.diff()
        gain = delta.clip(min=0)
        loss = -delta.clip(max=0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
    
    return df

# Example usage:
"""
# Initialize Vaex
init_vaex_financial()

# Load data
df = load_financial_data('financial_data.csv')

# Apply optimizations
df = optimize_financial_calculations(df)

# Create technical indicators
df = create_financial_features(df)

# Efficient filtering and aggregation
filtered_df = df[df.volume > 1000000]
daily_stats = filtered_df.groupby('date', agg={'close': 'mean', 'volume': 'sum'})
"""
