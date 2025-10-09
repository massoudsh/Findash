"""
Asset Configuration for Quantum Trading Matrix™
Centralized asset definitions and metadata
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional


class AssetType(Enum):
    """Asset type classifications"""
    STOCK = "stock"
    ETF = "etf"
    CRYPTOCURRENCY = "cryptocurrency"
    STABLECOIN = "stablecoin"
    COMMODITY = "commodity"
    FOREX = "forex"
    OPTION = "option"
    FUTURE = "future"


class Sector(Enum):
    """Sector classifications"""
    TECHNOLOGY = "technology"
    HEALTHCARE = "healthcare"
    FINANCIALS = "financials"
    CONSUMER_DISCRETIONARY = "consumer_discretionary"
    CONSUMER_STAPLES = "consumer_staples"
    ENERGY = "energy"
    INDUSTRIALS = "industrials"
    MATERIALS = "materials"
    REAL_ESTATE = "real_estate"
    UTILITIES = "utilities"
    TELECOMMUNICATIONS = "telecommunications"
    CRYPTOCURRENCY = "cryptocurrency"
    STABLECOIN = "stablecoin"
    COMMODITIES = "commodities"
    DIVERSIFIED = "diversified"


@dataclass
class AssetMetadata:
    """Asset metadata and configuration"""
    symbol: str
    name: str
    asset_type: AssetType
    sector: Sector
    currency: str = "USD"
    exchange: Optional[str] = None
    description: Optional[str] = None
    market_cap_category: Optional[str] = None  # "large", "mid", "small", "micro"
    volatility_category: Optional[str] = None  # "low", "medium", "high", "extreme"
    liquidity_category: Optional[str] = None   # "high", "medium", "low"
    trading_hours: Optional[str] = None        # "24/7", "market_hours", "extended"
    min_trade_size: float = 0.01
    tick_size: float = 0.01


class AssetsConfig:
    """Centralized asset configuration"""
    
    # Core asset definitions
    ASSETS: Dict[str, AssetMetadata] = {
        # Traditional Stocks
        "AAPL": AssetMetadata(
            symbol="AAPL",
            name="Apple Inc.",
            asset_type=AssetType.STOCK,
            sector=Sector.TECHNOLOGY,
            exchange="NASDAQ",
            description="Technology company focusing on consumer electronics",
            market_cap_category="large",
            volatility_category="medium",
            liquidity_category="high",
            trading_hours="market_hours"
        ),
        "TSLA": AssetMetadata(
            symbol="TSLA",
            name="Tesla Inc.",
            asset_type=AssetType.STOCK,
            sector=Sector.CONSUMER_DISCRETIONARY,
            exchange="NASDAQ",
            description="Electric vehicle and clean energy company",
            market_cap_category="large",
            volatility_category="high",
            liquidity_category="high",
            trading_hours="market_hours"
        ),
        "MSFT": AssetMetadata(
            symbol="MSFT",
            name="Microsoft Corporation",
            asset_type=AssetType.STOCK,
            sector=Sector.TECHNOLOGY,
            exchange="NASDAQ",
            description="Technology corporation focusing on software and cloud services",
            market_cap_category="large",
            volatility_category="medium",
            liquidity_category="high",
            trading_hours="market_hours"
        ),
        "GOOGL": AssetMetadata(
            symbol="GOOGL",
            name="Alphabet Inc.",
            asset_type=AssetType.STOCK,
            sector=Sector.TECHNOLOGY,
            exchange="NASDAQ",
            description="Technology conglomerate specializing in internet services",
            market_cap_category="large",
            volatility_category="medium",
            liquidity_category="high",
            trading_hours="market_hours"
        ),
        "AMZN": AssetMetadata(
            symbol="AMZN",
            name="Amazon.com Inc.",
            asset_type=AssetType.STOCK,
            sector=Sector.CONSUMER_DISCRETIONARY,
            exchange="NASDAQ",
            description="E-commerce and cloud computing company",
            market_cap_category="large",
            volatility_category="medium",
            liquidity_category="high",
            trading_hours="market_hours"
        ),
        "NVDA": AssetMetadata(
            symbol="NVDA",
            name="NVIDIA Corporation",
            asset_type=AssetType.STOCK,
            sector=Sector.TECHNOLOGY,
            exchange="NASDAQ",
            description="Graphics processing and AI chip manufacturer",
            market_cap_category="large",
            volatility_category="high",
            liquidity_category="high",
            trading_hours="market_hours"
        ),
        
        # ETFs
        "SPY": AssetMetadata(
            symbol="SPY",
            name="SPDR S&P 500 ETF Trust",
            asset_type=AssetType.ETF,
            sector=Sector.DIVERSIFIED,
            exchange="NYSE",
            description="ETF tracking the S&P 500 index",
            market_cap_category="large",
            volatility_category="low",
            liquidity_category="high",
            trading_hours="market_hours"
        ),
        "QQQ": AssetMetadata(
            symbol="QQQ",
            name="Invesco QQQ Trust",
            asset_type=AssetType.ETF,
            sector=Sector.TECHNOLOGY,
            exchange="NASDAQ",
            description="ETF tracking the NASDAQ-100 index",
            market_cap_category="large",
            volatility_category="medium",
            liquidity_category="high",
            trading_hours="market_hours"
        ),
        
        # Cryptocurrencies
        "BTC-USD": AssetMetadata(
            symbol="BTC-USD",
            name="Bitcoin",
            asset_type=AssetType.CRYPTOCURRENCY,
            sector=Sector.CRYPTOCURRENCY,
            description="The first and largest cryptocurrency by market cap",
            market_cap_category="large",
            volatility_category="extreme",
            liquidity_category="high",
            trading_hours="24/7",
            min_trade_size=0.00001,
            tick_size=0.01
        ),
        "ETH-USD": AssetMetadata(
            symbol="ETH-USD",
            name="Ethereum",
            asset_type=AssetType.CRYPTOCURRENCY,
            sector=Sector.CRYPTOCURRENCY,
            description="Smart contract platform and cryptocurrency",
            market_cap_category="large",
            volatility_category="extreme",
            liquidity_category="high",
            trading_hours="24/7",
            min_trade_size=0.0001,
            tick_size=0.01
        ),
        "TRX-USD": AssetMetadata(
            symbol="TRX-USD",
            name="TRON",
            asset_type=AssetType.CRYPTOCURRENCY,
            sector=Sector.CRYPTOCURRENCY,
            description="Decentralized blockchain platform",
            market_cap_category="mid",
            volatility_category="extreme",
            liquidity_category="medium",
            trading_hours="24/7",
            min_trade_size=0.1,
            tick_size=0.0001
        ),
        "LINK-USD": AssetMetadata(
            symbol="LINK-USD",
            name="Chainlink",
            asset_type=AssetType.CRYPTOCURRENCY,
            sector=Sector.CRYPTOCURRENCY,
            description="Decentralized oracle network",
            market_cap_category="mid",
            volatility_category="extreme",
            liquidity_category="high",
            trading_hours="24/7",
            min_trade_size=0.01,
            tick_size=0.001
        ),
        "CAKE-USD": AssetMetadata(
            symbol="CAKE-USD",
            name="PancakeSwap",
            asset_type=AssetType.CRYPTOCURRENCY,
            sector=Sector.CRYPTOCURRENCY,
            description="Decentralized exchange token on BSC",
            market_cap_category="small",
            volatility_category="extreme",
            liquidity_category="medium",
            trading_hours="24/7",
            min_trade_size=0.01,
            tick_size=0.001
        ),
        
        # Stablecoins
        "USDT-USD": AssetMetadata(
            symbol="USDT-USD",
            name="Tether",
            asset_type=AssetType.STABLECOIN,
            sector=Sector.STABLECOIN,
            description="USD-pegged stablecoin",
            market_cap_category="large",
            volatility_category="low",
            liquidity_category="high",
            trading_hours="24/7",
            min_trade_size=0.01,
            tick_size=0.0001
        ),
        "USDC-USD": AssetMetadata(
            symbol="USDC-USD",
            name="USD Coin",
            asset_type=AssetType.STABLECOIN,
            sector=Sector.STABLECOIN,
            description="Regulated USD-backed stablecoin",
            market_cap_category="large",
            volatility_category="low",
            liquidity_category="high",
            trading_hours="24/7",
            min_trade_size=0.01,
            tick_size=0.0001
        ),
        
        # Commodities
        "GLD": AssetMetadata(
            symbol="GLD",
            name="SPDR Gold Trust",
            asset_type=AssetType.COMMODITY,
            sector=Sector.COMMODITIES,
            exchange="NYSE",
            description="Gold-backed ETF",
            market_cap_category="large",
            volatility_category="medium",
            liquidity_category="high",
            trading_hours="market_hours"
        ),
        "SLV": AssetMetadata(
            symbol="SLV",
            name="iShares Silver Trust",
            asset_type=AssetType.COMMODITY,
            sector=Sector.COMMODITIES,
            exchange="NYSE",
            description="Silver-backed ETF",
            market_cap_category="mid",
            volatility_category="high",
            liquidity_category="high",
            trading_hours="market_hours"
        ),
    }
    
    # Asset groups for easy access
    WATCHLIST_DEFAULT = [
        "AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "NVDA",
        "BTC-USD", "ETH-USD", "USDT-USD", "USDC-USD", 
        "TRX-USD", "LINK-USD", "CAKE-USD", "GLD", "SLV"
    ]
    
    CRYPTO_ASSETS = [
        "BTC-USD", "ETH-USD", "TRX-USD", "LINK-USD", "CAKE-USD"
    ]
    
    STABLECOIN_ASSETS = [
        "USDT-USD", "USDC-USD"
    ]
    
    COMMODITY_ASSETS = [
        "GLD", "SLV"
    ]
    
    STOCK_ASSETS = [
        "AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "NVDA"
    ]
    
    ETF_ASSETS = [
        "SPY", "QQQ", "GLD", "SLV"
    ]
    
    # Sector mappings
    SECTOR_MAPPING = {
        asset.symbol: asset.sector.value 
        for asset in ASSETS.values()
    }
    
    # Currency mappings
    CURRENCY_MAPPING = {
        asset.symbol: asset.currency 
        for asset in ASSETS.values()
    }
    
    @classmethod
    def get_asset(cls, symbol: str) -> Optional[AssetMetadata]:
        """Get asset metadata by symbol"""
        return cls.ASSETS.get(symbol)
    
    @classmethod
    def get_assets_by_type(cls, asset_type: AssetType) -> List[AssetMetadata]:
        """Get all assets of a specific type"""
        return [asset for asset in cls.ASSETS.values() if asset.asset_type == asset_type]
    
    @classmethod
    def get_assets_by_sector(cls, sector: Sector) -> List[AssetMetadata]:
        """Get all assets in a specific sector"""
        return [asset for asset in cls.ASSETS.values() if asset.sector == sector]
    
    @classmethod
    def get_trading_symbols(cls) -> List[str]:
        """Get all tradeable symbols"""
        return list(cls.ASSETS.keys())
    
    @classmethod
    def is_crypto(cls, symbol: str) -> bool:
        """Check if symbol is a cryptocurrency"""
        asset = cls.get_asset(symbol)
        return asset and asset.asset_type in [AssetType.CRYPTOCURRENCY, AssetType.STABLECOIN]
    
    @classmethod
    def is_24_7_trading(cls, symbol: str) -> bool:
        """Check if asset trades 24/7"""
        asset = cls.get_asset(symbol)
        return asset and asset.trading_hours == "24/7"
    
    @classmethod
    def get_volatility_category(cls, symbol: str) -> Optional[str]:
        """Get volatility category for symbol"""
        asset = cls.get_asset(symbol)
        return asset.volatility_category if asset else None


# Export for easy imports
__all__ = [
    'AssetType', 'Sector', 'AssetMetadata', 'AssetsConfig'
] 