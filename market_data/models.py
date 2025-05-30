from django.db import models

class Stock(models.Model):
    symbol = models.CharField(max_length=10, unique=True)
    name = models.CharField(max_length=100)
    sector = models.CharField(max_length=50)
    industry = models.CharField(max_length=50)
    exchange = models.CharField(max_length=20)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.symbol} - {self.name}"

class StockPrice(models.Model):
    stock = models.ForeignKey(Stock, on_delete=models.CASCADE, related_name='prices')
    date = models.DateTimeField()
    open_price = models.DecimalField(max_digits=15, decimal_places=2)
    high_price = models.DecimalField(max_digits=15, decimal_places=2)
    low_price = models.DecimalField(max_digits=15, decimal_places=2)
    close_price = models.DecimalField(max_digits=15, decimal_places=2)
    volume = models.BigIntegerField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('stock', 'date')
        indexes = [
            models.Index(fields=['stock', 'date']),
        ]

    def __str__(self):
        return f"{self.stock.symbol} - {self.date.date()}"

class MarketIndex(models.Model):
    symbol = models.CharField(max_length=10, unique=True)
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.symbol} - {self.name}"

class MarketIndexPrice(models.Model):
    index = models.ForeignKey(MarketIndex, on_delete=models.CASCADE, related_name='prices')
    date = models.DateTimeField()
    open_price = models.DecimalField(max_digits=15, decimal_places=2)
    high_price = models.DecimalField(max_digits=15, decimal_places=2)
    low_price = models.DecimalField(max_digits=15, decimal_places=2)
    close_price = models.DecimalField(max_digits=15, decimal_places=2)
    volume = models.BigIntegerField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('index', 'date')
        indexes = [
            models.Index(fields=['index', 'date']),
        ]

    def __str__(self):
        return f"{self.index.symbol} - {self.date.date()}"

class TechnicalIndicator(models.Model):
    INDICATOR_TYPES = (
        ('MA', 'Moving Average'),
        ('RSI', 'Relative Strength Index'),
        ('MACD', 'Moving Average Convergence Divergence'),
        ('BB', 'Bollinger Bands'),
        ('VOL', 'Volume'),
    )

    stock = models.ForeignKey(Stock, on_delete=models.CASCADE, related_name='technical_indicators')
    indicator_type = models.CharField(max_length=10, choices=INDICATOR_TYPES)
    date = models.DateTimeField()
    value = models.DecimalField(max_digits=15, decimal_places=4)
    parameters = models.JSONField(default=dict)  # Store indicator parameters
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('stock', 'indicator_type', 'date')
        indexes = [
            models.Index(fields=['stock', 'indicator_type', 'date']),
        ]

    def __str__(self):
        return f"{self.stock.symbol} - {self.indicator_type} - {self.date.date()}" 