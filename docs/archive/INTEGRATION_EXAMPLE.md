# Kafka + Redis + Prometheus + Grafana + Flower Integration Example

## Overview

This example demonstrates a complete real-time market data streaming pipeline that integrates:
- **Kafka** - Message streaming and event distribution
- **Redis** - Caching and pub/sub task allocation
- **Prometheus** - Metrics collection
- **Grafana** - Visualization and monitoring
- **Flower** - Celery task monitoring

## Architecture Flow

```
Market Data Source
    ↓
Kafka Producer → market-data-stream topic
    ↓
Kafka Consumer → Processes messages
    ↓
Redis Cache → Stores latest prices
    ↓
Redis Pub/Sub → Allocates tasks to Celery workers
    ↓
Celery Tasks → Process and store data
    ↓
Prometheus ← Metrics from all components
    ↓
Grafana ← Visualizes metrics
    ↓
Flower ← Monitors Celery tasks
```

## Components

### 1. Kafka Producer (`market-data-producer`)
- Produces real-time market data events to Kafka
- Symbols: AAPL, MSFT, GOOGL, AMZN, TSLA, etc.
- Updates every 1 second per symbol
- Exposes Prometheus metrics on port 8001

### 2. Kafka Consumer (`market-data-consumer`)
- Consumes market data from Kafka
- Updates Redis cache
- Publishes to Redis pub/sub for Celery task allocation
- Triggers Celery tasks via pub/sub pattern
- Exposes Prometheus metrics on port 8002

### 3. Redis
- **Cache**: Stores latest market data (5min TTL)
- **Pub/Sub**: Distributes tasks to Celery workers
- **Metrics**: Tracks cache hits/misses and pub/sub messages

### 4. Celery Tasks
- `data_processing.update_market_data` - Processes market data updates
- Triggered via Redis pub/sub from Kafka consumer
- Updates database and cache
- Tracked in Flower

### 5. Prometheus
- Scrapes metrics from:
  - Kafka producer (port 8001)
  - Kafka consumer (port 8002)
  - Celery metrics (port 9540)
  - Redis exporter (port 9121)

### 6. Grafana Dashboard
- **Kafka Message Throughput** - Messages produced/consumed per second
- **Kafka Latency** - Produce/consume latency (p95)
- **Redis Cache Performance** - Hit rate and cache operations
- **Redis Pub/Sub** - Message distribution rate
- **Market Data Updates** - Updates per symbol
- **Processing Lag** - End-to-end latency
- **Celery Tasks** - Task execution from Kafka stream
- **Pipeline Health** - Overall system status

### 7. Flower
- Monitors Celery tasks triggered by Kafka events
- Shows task execution, success/failure rates
- Real-time task monitoring

## Running the Integration

### 1. Start All Services

```bash
docker-compose -f docker-compose-complete.yml up -d
```

### 2. Start Market Data Services

```bash
# Producer (generates market data)
docker-compose -f docker-compose-complete.yml up -d market-data-producer

# Consumer (processes market data)
docker-compose -f docker-compose-complete.yml up -d market-data-consumer
```

### 3. Verify Services

```bash
# Check Kafka topic
docker exec octopus-kafka kafka-topics.sh --bootstrap-server localhost:9092 --list | grep market-data

# Check Redis cache
docker exec octopus-redis redis-cli KEYS "market_data:*"

# Check Prometheus targets
curl http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | select(.labels.job | contains("market"))'
```

### 4. View Dashboards

- **Grafana**: http://localhost:3001
  - Dashboard: "Kafka + Redis + Celery Integration"
  - Login: admin/admin

- **Flower**: http://localhost:5555
  - View Celery tasks triggered by Kafka events

- **Prometheus**: http://localhost:9090
  - Query: `rate(kafka_messages_produced_total[5m])`

## Metrics Exposed

### Kafka Metrics
- `kafka_messages_produced_total` - Messages produced
- `kafka_messages_consumed_total` - Messages consumed
- `kafka_produce_latency_seconds` - Produce latency
- `kafka_consume_latency_seconds` - Consume latency

### Redis Metrics
- `redis_cache_hits_total` - Cache hits
- `redis_cache_misses_total` - Cache misses
- `redis_pubsub_messages_total` - Pub/sub messages

### Market Data Metrics
- `market_data_updates_total` - Updates processed
- `market_data_lag_seconds` - Processing lag

### Celery Metrics
- `celery_task_started_total` - Tasks started
- `celery_task_succeeded_total` - Tasks succeeded
- `celery_task_failed_total` - Tasks failed
- `celery_task_duration_seconds` - Task duration

## Example Workflow

1. **Producer** generates market data for AAPL: $175.50
2. **Kafka** receives and stores the event in `market-data-stream` topic
3. **Consumer** reads from Kafka
4. **Redis** caches the latest price: `market_data:AAPL:latest`
5. **Redis Pub/Sub** publishes task allocation: `tasks:market_data:AAPL`
6. **Celery Worker** picks up task via pub/sub
7. **Celery Task** processes and stores data
8. **Prometheus** collects metrics from all components
9. **Grafana** visualizes the entire pipeline
10. **Flower** shows task execution in real-time

## Testing the Integration

### Manual Test

```python
from src.infrastructure.market_data_stream import IntegratedMarketDataService

service = IntegratedMarketDataService(
    kafka_servers='localhost:9092',
    redis_url='redis://localhost:6379/0'
)

# Run for 60 seconds
service.simulate_market_data_stream(
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    duration=60
)
```

### Verify Data Flow

```bash
# Check Kafka messages
docker exec octopus-kafka kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic market-data-stream \
  --from-beginning \
  --max-messages 10

# Check Redis cache
docker exec octopus-redis redis-cli GET "market_data:AAPL:latest"

# Check Celery tasks in Flower
# Open http://localhost:5555 and look for "update_market_data" tasks
```

## Monitoring Queries

### Prometheus Queries

```promql
# Total messages in pipeline
sum(rate(kafka_messages_produced_total[5m]))

# End-to-end latency
market_data_lag_seconds

# Cache hit rate
rate(redis_cache_hits_total[5m]) / 
(rate(redis_cache_hits_total[5m]) + rate(redis_cache_misses_total[5m]))

# Task success rate
rate(celery_task_succeeded_total{task_name="update_market_data"}[5m]) /
rate(celery_task_started_total{task_name="update_market_data"}[5m])
```

## Troubleshooting

### Kafka Not Receiving Messages
```bash
# Check Kafka is running
docker ps | grep kafka

# Check topic exists
docker exec octopus-kafka kafka-topics.sh --bootstrap-server localhost:9092 --list
```

### Redis Not Caching
```bash
# Check Redis connection
docker exec octopus-redis redis-cli PING

# Check cache keys
docker exec octopus-redis redis-cli KEYS "market_data:*"
```

### Celery Tasks Not Running
```bash
# Check Celery worker logs
docker logs octopus-celery-worker --tail 50

# Check Flower
# Open http://localhost:5555
```

### Metrics Not Appearing
```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Check metrics endpoint
curl http://localhost:8001/metrics | grep kafka
```

## Performance Tuning

1. **Kafka**: Adjust batch size and compression
2. **Redis**: Tune cache TTL and memory limits
3. **Celery**: Scale workers based on queue length
4. **Prometheus**: Adjust scrape intervals for high volume

## Next Steps

1. Add more symbols to the stream
2. Implement backpressure handling
3. Add alerting rules in Prometheus
4. Scale consumers horizontally
5. Add dead letter queue for failed messages

