# Grafana Access Guide

## Login Credentials

- **URL**: http://localhost:3001
- **Username**: `admin`
- **Password**: `admin`

## Viewing Data Flow Dashboard

### Step 1: Login to Grafana
1. Open http://localhost:3001 in your browser
2. Enter username: `admin`
3. Enter password: `admin`
4. Click "Log in"

### Step 2: Navigate to Kafka + Redis Integration Dashboard
1. Click on **"Dashboards"** in the left sidebar
2. Click on **"Infrastructure"** folder
3. Click on **"Kafka + Redis + Celery Integration"**

### Step 3: Start Data Streaming (if not already running)

To see real data flowing, start the market data services:

```bash
# Start market data producer and consumer
docker-compose -f docker-compose-complete.yml up -d \
  market-data-producer \
  market-data-consumer

# Verify they're running
docker ps | grep market-data
```

### Step 4: View Metrics

The dashboard will show:
- **Kafka Message Throughput**: Messages produced/consumed per second
- **Kafka Latency**: Produce/consume latency (p95)
- **Redis Cache Performance**: Hit rate and cache operations
- **Redis Pub/Sub**: Message distribution rate
- **Market Data Updates**: Updates per symbol
- **Processing Lag**: End-to-end latency
- **Celery Tasks**: Task execution from Kafka stream
- **Pipeline Health**: Overall system status

## Available Dashboards

1. **Executive Overview** - High-level trading metrics
2. **Trading Operations** - Trading activity and performance
3. **Risk Management** - Risk metrics and alerts
4. **System Infrastructure** - System health and resources
5. **Celery Monitoring** - Celery worker and task metrics
6. **Kafka + Redis + Celery Integration** ⭐ - Complete data flow pipeline

## Data Sources

- **Prometheus**: http://prometheus:9090 (metrics collection)
- **PostgreSQL**: db:5432 (trading database)
- **Redis**: redis:6379 (cache and pub/sub)

## Troubleshooting

### No Data Showing?

1. **Check Prometheus targets**:
   ```bash
   docker exec octopus-prometheus wget -qO- http://localhost:9090/api/v1/targets | python3 -m json.tool
   ```

2. **Check if services are running**:
   ```bash
   docker ps | grep -E "market-data|celery|prometheus"
   ```

3. **Check Prometheus metrics**:
   ```bash
   curl http://localhost:9090/api/v1/query?query=kafka_messages_produced_total
   ```

4. **Restart services if needed**:
   ```bash
   docker-compose -f docker-compose-complete.yml restart \
     market-data-producer \
     market-data-consumer \
     prometheus \
     grafana
   ```

### Can't Login?

If admin/admin doesn't work, reset the password:
```bash
docker exec octopus-grafana grafana cli admin reset-admin-password admin
```

### Dashboard Not Loading?

1. Check if dashboards are provisioned:
   ```bash
   docker exec octopus-grafana ls -la /etc/grafana/provisioning/dashboards/
   ```

2. Restart Grafana:
   ```bash
   docker restart octopus-grafana
   ```

## Quick Test Query

In Grafana, go to **Explore** and try:
```promql
rate(kafka_messages_produced_total[5m])
```

This shows Kafka message production rate.

