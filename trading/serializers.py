from rest_framework import serializers
from .models import Portfolio, Position, Trade, Strategy, StrategyPerformance

class PortfolioSerializer(serializers.ModelSerializer):
    class Meta:
        model = Portfolio
        fields = ['id', 'name', 'description', 'initial_balance', 'current_balance', 'created_at', 'updated_at']
        read_only_fields = ['current_balance', 'created_at', 'updated_at']

class PositionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Position
        fields = ['id', 'portfolio', 'symbol', 'quantity', 'average_price', 'current_price', 'unrealized_pnl', 'created_at', 'updated_at']
        read_only_fields = ['current_price', 'unrealized_pnl', 'created_at', 'updated_at']

class TradeSerializer(serializers.ModelSerializer):
    class Meta:
        model = Trade
        fields = ['id', 'portfolio', 'symbol', 'trade_type', 'quantity', 'price', 'total_amount', 'timestamp']
        read_only_fields = ['total_amount', 'timestamp']

class StrategySerializer(serializers.ModelSerializer):
    class Meta:
        model = Strategy
        fields = ['id', 'name', 'description', 'parameters', 'is_active', 'created_at', 'updated_at']
        read_only_fields = ['created_at', 'updated_at']

class StrategyPerformanceSerializer(serializers.ModelSerializer):
    class Meta:
        model = StrategyPerformance
        fields = ['id', 'strategy', 'portfolio', 'start_date', 'end_date', 'total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'created_at']
        read_only_fields = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'created_at'] 