#!/usr/bin/env python3
"""
API Keys Configuration for Multi-Source Market Data Collection
Octopus Trading Platform

Instructions:
1. Get your API keys from the respective providers
2. Replace the placeholder values below with your actual API keys
3. Keep this file secure and never commit it to version control

Free API Key Resources:
- Alpha Vantage: https://www.alphavantage.co/support/#api-key (Free: 5 calls/min)
- IEX Cloud: https://iexcloud.io/ (Free: 500k calls/month)
- Polygon.io: https://polygon.io/ (Free: 5 calls/min)
- Finnhub: https://finnhub.io/ (Free: 60 calls/min)
"""

# API Configuration
API_KEYS = {
    # Alpha Vantage - Premium fundamental data and technical indicators
    # Free tier: 5 API calls per minute, 500 calls per day
    # Premium plans start at $49.99/month
    'alpha_vantage': 'your_alpha_vantage_api_key_here',
    
    # IEX Cloud - Real-time and historical market data
    # Free tier: 500,000 API calls per month
    # Paid plans start at $9/month
    'iex_cloud': 'your_iex_cloud_token_here',
    
    # Polygon.io - High-frequency and professional market data
    # Free tier: 5 API calls per minute
    # Professional plans start at $99/month
    'polygon': 'your_polygon_api_key_here',
    
    # Finnhub - Financial data and news
    # Free tier: 60 API calls per minute
    # Premium plans start at $39.99/month
    'finnhub': 'your_finnhub_api_key_here'
}

# Rate limiting configuration for each source
RATE_LIMITS = {
    'yahoo': {
        'calls_per_minute': 100,
        'delay_seconds': 0.6,
        'cost': 'Free'
    },
    'alpha_vantage': {
        'calls_per_minute': 5,  # Free tier
        'delay_seconds': 12,
        'cost': 'Free tier: 5/min, Premium: $49.99/month'
    },
    'iex_cloud': {
        'calls_per_minute': 100,
        'delay_seconds': 0.6,
        'cost': 'Free tier: 500k/month, Plans from $9/month'
    },
    'polygon': {
        'calls_per_minute': 5,  # Free tier
        'delay_seconds': 12,
        'cost': 'Free tier: 5/min, Plans from $99/month'
    },
    'finnhub': {
        'calls_per_minute': 60,
        'delay_seconds': 1,
        'cost': 'Free tier: 60/min, Plans from $39.99/month'
    }
}

# Data quality scores (0.0 to 1.0)
DATA_QUALITY_SCORES = {
    'yahoo': 0.85,          # Good quality, free source
    'alpha_vantage': 0.95,  # Excellent quality, premium features
    'iex_cloud': 0.90,      # High quality, real-time data
    'polygon': 0.92,        # Professional quality, high-frequency
    'finnhub': 0.88         # Good quality, comprehensive data
}

# Source capabilities matrix
SOURCE_CAPABILITIES = {
    'yahoo': {
        'real_time': True,
        'historical': True,
        'fundamental': True,
        'options': False,
        'crypto': True,
        'forex': True,
        'news': False
    },
    'alpha_vantage': {
        'real_time': True,
        'historical': True,
        'fundamental': True,
        'options': False,
        'crypto': True,
        'forex': True,
        'news': False
    },
    'iex_cloud': {
        'real_time': True,
        'historical': True,
        'fundamental': True,
        'options': False,
        'crypto': True,
        'forex': False,
        'news': True
    },
    'polygon': {
        'real_time': True,
        'historical': True,
        'fundamental': False,
        'options': True,
        'crypto': True,
        'forex': True,
        'news': False
    },
    'finnhub': {
        'real_time': True,
        'historical': True,
        'fundamental': True,
        'options': False,
        'crypto': True,
        'forex': True,
        'news': True
    }
}

def get_api_key(source: str) -> str:
    """Get API key for a specific source"""
    return API_KEYS.get(source, '')

def is_source_configured(source: str) -> bool:
    """Check if a source has been configured with an API key"""
    key = API_KEYS.get(source, '')
    return key and not key.endswith('_here')

def get_configured_sources() -> list:
    """Get list of sources that have been configured with API keys"""
    return [source for source in API_KEYS.keys() if is_source_configured(source)]

def validate_configuration():
    """Validate API key configuration and show status"""
    print("🔑 API Key Configuration Status")
    print("=" * 50)
    
    configured_sources = []
    
    for source, key in API_KEYS.items():
        if is_source_configured(source):
            status = "✅ Configured"
            configured_sources.append(source)
        else:
            status = "❌ Not configured"
        
        rate_limit = RATE_LIMITS.get(source, {})
        quality = DATA_QUALITY_SCORES.get(source, 0)
        
        print(f"{source:15} | {status:15} | Quality: {quality:.2f}")
        print(f"{'':15} | {rate_limit.get('cost', 'N/A')}")
        print()
    
    print(f"📊 Summary: {len(configured_sources)}/{len(API_KEYS)} sources configured")
    
    if configured_sources:
        print(f"✅ Active sources: {', '.join(configured_sources)}")
    else:
        print("⚠️  No sources configured. At least Yahoo Finance will work without API keys.")
    
    return configured_sources

if __name__ == "__main__":
    validate_configuration() 