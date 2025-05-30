from django.db import models
from trading.models import Portfolio, Strategy

class RiskMetrics(models.Model):
    portfolio = models.ForeignKey(Portfolio, on_delete=models.CASCADE, related_name='risk_metrics')
    strategy = models.ForeignKey(Strategy, on_delete=models.CASCADE, related_name='risk_metrics')
    date = models.DateTimeField()
    volatility = models.DecimalField(max_digits=10, decimal_places=4)
    sharpe_ratio = models.DecimalField(max_digits=10, decimal_places=4)
    sortino_ratio = models.DecimalField(max_digits=10, decimal_places=4)
    max_drawdown = models.DecimalField(max_digits=10, decimal_places=4)
    var_95 = models.DecimalField(max_digits=10, decimal_places=4)  # Value at Risk (95%)
    expected_shortfall = models.DecimalField(max_digits=10, decimal_places=4)
    beta = models.DecimalField(max_digits=10, decimal_places=4)
    alpha = models.DecimalField(max_digits=10, decimal_places=4)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('portfolio', 'strategy', 'date')

    def __str__(self):
        return f"Risk Metrics for {self.portfolio.name} - {self.date.date()}"

class RiskLimit(models.Model):
    portfolio = models.ForeignKey(Portfolio, on_delete=models.CASCADE, related_name='risk_limits')
    strategy = models.ForeignKey(Strategy, on_delete=models.CASCADE, related_name='risk_limits')
    max_position_size = models.DecimalField(max_digits=10, decimal_places=4)  # As percentage of portfolio
    max_sector_exposure = models.DecimalField(max_digits=10, decimal_places=4)  # As percentage of portfolio
    max_drawdown_limit = models.DecimalField(max_digits=10, decimal_places=4)  # As percentage
    var_limit = models.DecimalField(max_digits=10, decimal_places=4)  # Value at Risk limit
    target_sharpe = models.DecimalField(max_digits=10, decimal_places=4)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ('portfolio', 'strategy')

    def __str__(self):
        return f"Risk Limits for {self.portfolio.name} - {self.strategy.name}"

class RiskAlert(models.Model):
    ALERT_TYPES = (
        ('DRAWDOWN', 'Drawdown Limit'),
        ('VAR', 'Value at Risk'),
        ('VOLATILITY', 'Volatility'),
        ('EXPOSURE', 'Exposure'),
        ('OTHER', 'Other'),
    )

    SEVERITY_LEVELS = (
        ('LOW', 'Low'),
        ('MEDIUM', 'Medium'),
        ('HIGH', 'High'),
        ('CRITICAL', 'Critical'),
    )

    portfolio = models.ForeignKey(Portfolio, on_delete=models.CASCADE, related_name='risk_alerts')
    strategy = models.ForeignKey(Strategy, on_delete=models.CASCADE, related_name='risk_alerts')
    alert_type = models.CharField(max_length=20, choices=ALERT_TYPES)
    severity = models.CharField(max_length=10, choices=SEVERITY_LEVELS)
    message = models.TextField()
    threshold = models.DecimalField(max_digits=10, decimal_places=4)
    current_value = models.DecimalField(max_digits=10, decimal_places=4)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    resolved_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"{self.alert_type} Alert for {self.portfolio.name} - {self.severity}"

class CorrelationMatrix(models.Model):
    portfolio = models.ForeignKey(Portfolio, on_delete=models.CASCADE, related_name='correlation_matrices')
    date = models.DateTimeField()
    matrix = models.JSONField()  # Store correlation matrix as JSON
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('portfolio', 'date')

    def __str__(self):
        return f"Correlation Matrix for {self.portfolio.name} - {self.date.date()}" 