from django.db import models
from trading.models import Portfolio, Strategy
from risk_management.models import RiskMetrics

class Dashboard(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    layout = models.JSONField()  # Store dashboard layout configuration
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name

class Chart(models.Model):
    CHART_TYPES = (
        ('LINE', 'Line Chart'),
        ('CANDLESTICK', 'Candlestick Chart'),
        ('BAR', 'Bar Chart'),
        ('SCATTER', 'Scatter Plot'),
        ('HEATMAP', 'Heatmap'),
        ('PIE', 'Pie Chart'),
    )

    dashboard = models.ForeignKey(Dashboard, on_delete=models.CASCADE, related_name='charts')
    name = models.CharField(max_length=100)
    chart_type = models.CharField(max_length=20, choices=CHART_TYPES)
    data_source = models.CharField(max_length=100)  # API endpoint or data source
    configuration = models.JSONField()  # Store chart configuration
    position = models.JSONField()  # Store chart position in dashboard
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.name} ({self.chart_type})"

class ChartData(models.Model):
    chart = models.ForeignKey(Chart, on_delete=models.CASCADE, related_name='data')
    date = models.DateTimeField()
    data = models.JSONField()  # Store chart data
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('chart', 'date')
        indexes = [
            models.Index(fields=['chart', 'date']),
        ]

    def __str__(self):
        return f"Data for {self.chart.name} - {self.date.date()}"

class UserPreference(models.Model):
    user = models.OneToOneField('auth.User', on_delete=models.CASCADE, related_name='visualization_preferences')
    default_dashboard = models.ForeignKey(Dashboard, on_delete=models.SET_NULL, null=True, blank=True)
    theme = models.CharField(max_length=20, default='light')
    chart_defaults = models.JSONField(default=dict)  # Store default chart configurations
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Preferences for {self.user.username}"

class SavedView(models.Model):
    user = models.ForeignKey('auth.User', on_delete=models.CASCADE, related_name='saved_views')
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    filters = models.JSONField()  # Store view filters
    sort_order = models.JSONField()  # Store sort order
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.name} - {self.user.username}" 