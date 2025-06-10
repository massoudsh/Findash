# Trading Platform Monitoring Setup

This directory contains the complete monitoring infrastructure for the Quantum Trading Matrix™ platform using Prometheus and Grafana.

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Trading API   │    │   Prometheus    │    │     Grafana     │
│   (Port 8000)   │───▶│   (Port 9090)   │───▶│   (Port 3001)   │
│   /metrics      │    │   Scrapes       │    │   Visualizes    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Custom Metrics  │    │ Alerting Rules  │    │   Dashboards    │
│ - Trades        │    │ - Performance   │    │ - Real-time     │
│ - Portfolio     │    │ - Risk Alerts   │    │ - Trading Data  │
│ - Risk Metrics  │    │ - System Health │    │ - System Stats  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Components

### 1. Prometheus Configuration (`prometheus.yml`)
- **Scrape Intervals**: API metrics every 5s, system metrics every 15s
- **Targets**: 
  - Trading API (`api:8000/metrics`)
  - Node Exporter (`node-exporter:9100`)
  - PostgreSQL Exporter (optional)
  - Redis Exporter (optional)

### 2. Trading Metrics (`metrics.py`)
Custom metrics specifically designed for trading platforms:

#### Trading Metrics
- `trading_platform_trades_total` - Total trades executed (by symbol, side, status)
- `trading_platform_portfolio_value_usd` - Portfolio values in USD
- `trading_platform_trade_execution_seconds` - Trade execution time histogram
- `trading_platform_strategy_performance_percent` - Strategy performance percentages

#### Risk & Analysis Metrics
- `trading_platform_risk_metrics` - Various risk metrics (VaR, Sharpe ratio, etc.)
- `trading_platform_social_sentiment_score` - Social media sentiment scores
- `trading_platform_model_accuracy_percent` - ML model accuracy tracking

#### System Metrics
- `trading_platform_active_connections` - Active WebSocket connections
- `trading_platform_api_request_duration_seconds` - API response time histogram
- `trading_platform_market_data_latency_seconds` - Market data feed latency

### 3. Grafana Dashboards

#### Main Trading Dashboard (`trading-platform-dashboard.json`)
**Panels include:**
- **Trade Rate**: Real-time trading activity per 5-minute intervals
- **Portfolio Values**: Current portfolio values in USD (table format)
- **Strategy Performance**: Performance percentages over time
- **Risk Metrics**: VaR, Sharpe ratio, and other risk indicators
- **API Response Times**: 95th and 50th percentile response times
- **Social Sentiment**: Sentiment scores from various platforms
- **Active Connections**: WebSocket connection count
- **ML Model Accuracy**: Current model performance metrics
- **Market Data Latency**: Feed latency monitoring

### 4. Alerting Rules (`prometheus/rules/trading-alerts.yml`)

#### Critical Alerts
- **Portfolio Value Drop**: >10% drop in 1 hour
- **High Risk Metric**: VaR exceeding $50,000
- **High Trade Failure Rate**: >10% failure rate

#### Warning Alerts
- **High API Response Time**: >2 seconds (95th percentile)
- **High Market Data Latency**: >1 second
- **Low Model Accuracy**: <70% accuracy
- **System Resource Issues**: CPU >80%, Memory >80%

# Disk space usage
- alert: HighDiskUsage
  expr: (node_filesystem_size_bytes{mountpoint="/"} - node_filesystem_free_bytes{mountpoint="/"}) / node_filesystem_size_bytes{mountpoint="/"} * 100 > 90
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "High disk usage detected"
    description: "Disk usage is {{ $value }}% on instance {{ $labels.instance }}"

# API error rate
- alert: HighAPIErrorRate
  expr: sum(rate(trading_platform_api_request_duration_seconds_count{status_code=~\"5..\"}[5m])) / sum(rate(trading_platform_api_request_duration_seconds_count[5m])) > 0.05
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "High API error rate"
    description: "API 5xx error rate is {{ $value | humanizePercentage }} for the last 5 minutes"

## Quick Start

### 1. Start the Monitoring Stack
```bash
docker-compose up -d prometheus grafana node-exporter
```

### 2. Access Interfaces
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3001 (admin/admin123)

### 3. Verify Metrics Collection
```bash
# Check if API metrics are being scraped
curl http://localhost:8000/metrics

# Check Prometheus targets
curl http://localhost:9090/api/v1/targets
```

### 4. Import Dashboard
The trading dashboard is automatically provisioned, but you can also manually import:
1. Go to Grafana → Dashboards → Import
2. Upload `grafana/dashboards/trading-platform-dashboard.json`

## Monitoring Best Practices

### 1. Metric Naming Convention
```python
# Use consistent prefixes
trading_platform_<component>_<metric_name>_<unit>

# Examples:
trading_platform_portfolio_value_usd
trading_platform_api_request_duration_seconds
trading_platform_trades_total
```

### 2. Label Strategy
```python
# Use meaningful labels for filtering/grouping
TRADES_TOTAL.labels(
    symbol="AAPL",
    side="buy", 
    status="executed"
).inc()
```

### 3. Histogram vs Counter vs Gauge
- **Counter**: Cumulative metrics (total trades, total errors)
- **Gauge**: Point-in-time values (portfolio value, active connections)
- **Histogram**: Distribution of values (response times, execution times)

## Customization

### Adding New Metrics
1. Define metric in `metrics.py`:
```python
NEW_METRIC = Gauge(
    'trading_platform_new_metric',
    'Description of the metric',
    ['label1', 'label2']
)
```

2. Update metric in your code:
```python
from monitoring.metrics import TradingMetrics
TradingMetrics.update_new_metric(label1="value1", label2="value2", value=123.45)
```

3. Add to Grafana dashboard:
```json
{
  "expr": "trading_platform_new_metric",
  "legendFormat": "{{label1}} - {{label2}}"
}
```

### Adding New Alerts
Edit `prometheus/rules/trading-alerts.yml`:
```yaml
- alert: NewAlert
  expr: trading_platform_new_metric > threshold
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "New alert triggered"
    description: "Metric value is {{ $value }}"
```

## Troubleshooting

### Common Issues

1. **Metrics not appearing in Prometheus**
   - Check if `/metrics` endpoint is accessible
   - Verify Prometheus configuration syntax
   - Check Docker network connectivity

2. **Grafana dashboard empty**
   - Verify Prometheus datasource configuration
   - Check metric names match exactly
   - Ensure time range includes data

3. **Alerts not firing**
   - Check alerting rules syntax in Prometheus UI
   - Verify expression returns expected values
   - Check alert conditions match your data

### Debug Commands
```bash
# Check Prometheus config
docker exec prometheus promtool check config /etc/prometheus/prometheus.yml

# Check alerting rules
docker exec prometheus promtool check rules /etc/prometheus/rules/*.yml

# View logs
docker logs prometheus
docker logs grafana
```

## Security Considerations

1. **Change default passwords**:
   - Grafana admin password (GF_SECURITY_ADMIN_PASSWORD)
   
2. **Network security**:
   - All services use internal Docker network
   - Only necessary ports exposed

3. **Data retention**:
   - Prometheus retains data for 200h by default
   - Adjust based on your storage capacity

## Performance Optimization

1. **Scrape intervals**:
   - API metrics: 5s (high frequency for trading data)
   - System metrics: 15s (standard)
   
2. **Metric cardinality**:
   - Avoid high-cardinality labels (like timestamps)
   - Use consistent label values
   
3. **Storage**:
   - Monitor Prometheus disk usage
   - Consider remote storage for long-term retention

## Integration with Alerting

To add external alerting (Slack, email, etc.):

1. Add Alertmanager to `docker-compose.yml`
2. Configure notification channels in `alertmanager.yml`
3. Update Prometheus to use Alertmanager

Example Alertmanager configuration:
```yaml
# alertmanager.yml
route:
  group_by: ['alertname']
  receiver: 'web.hook'

receivers:
- name: 'web.hook'
  slack_configs:
  - api_url: 'YOUR_SLACK_WEBHOOK_URL'
    channel: '#trading-alerts'
``` 