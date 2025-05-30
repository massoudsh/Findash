from django.db import models
from django.contrib.auth import get_user_model

User = get_user_model()

class Portfolio(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='portfolios')
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    initial_capital = models.DecimalField(max_digits=15, decimal_places=2)
    current_value = models.DecimalField(max_digits=15, decimal_places=2)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.user.username}'s {self.name} Portfolio"

class Position(models.Model):
    portfolio = models.ForeignKey(Portfolio, on_delete=models.CASCADE, related_name='positions')
    symbol = models.CharField(max_length=10)
    quantity = models.DecimalField(max_digits=15, decimal_places=4)
    average_price = models.DecimalField(max_digits=15, decimal_places=2)
    current_price = models.DecimalField(max_digits=15, decimal_places=2)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.symbol} - {self.quantity} shares"

class Trade(models.Model):
    TRADE_TYPES = (
        ('BUY', 'Buy'),
        ('SELL', 'Sell'),
    )

    portfolio = models.ForeignKey(Portfolio, on_delete=models.CASCADE, related_name='trades')
    symbol = models.CharField(max_length=10)
    trade_type = models.CharField(max_length=4, choices=TRADE_TYPES)
    quantity = models.DecimalField(max_digits=15, decimal_places=4)
    price = models.DecimalField(max_digits=15, decimal_places=2)
    timestamp = models.DateTimeField(auto_now_add=True)
    commission = models.DecimalField(max_digits=10, decimal_places=2, default=0)

    def __str__(self):
        return f"{self.trade_type} {self.quantity} {self.symbol} @ {self.price}"

class Strategy(models.Model):
    STRATEGY_TYPES = (
        ('MOMENTUM', 'Momentum'),
        ('MEAN_REVERSION', 'Mean Reversion'),
        ('TREND_FOLLOWING', 'Trend Following'),
        ('RISK_AWARE', 'Risk Aware'),
    )

    name = models.CharField(max_length=100)
    strategy_type = models.CharField(max_length=20, choices=STRATEGY_TYPES)
    description = models.TextField()
    parameters = models.JSONField(default=dict)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.name} ({self.strategy_type})"

class StrategyPerformance(models.Model):
    strategy = models.ForeignKey(Strategy, on_delete=models.CASCADE, related_name='performance')
    portfolio = models.ForeignKey(Portfolio, on_delete=models.CASCADE, related_name='strategy_performance')
    start_date = models.DateTimeField()
    end_date = models.DateTimeField()
    initial_capital = models.DecimalField(max_digits=15, decimal_places=2)
    final_capital = models.DecimalField(max_digits=15, decimal_places=2)
    sharpe_ratio = models.DecimalField(max_digits=10, decimal_places=4)
    max_drawdown = models.DecimalField(max_digits=10, decimal_places=4)
    returns = models.JSONField()  # Store daily returns
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.strategy.name} Performance ({self.start_date.date()} - {self.end_date.date()})" 