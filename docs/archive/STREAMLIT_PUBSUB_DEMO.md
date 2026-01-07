# Streamlit BTC Price Pub/Sub Demo

## Overview

This Streamlit dashboard demonstrates the Redis Pub/Sub mechanism for real-time BTC price updates. It runs on **port 8500** and shows how subscribers receive messages from the `btc_price_updates` channel.

## Features

- ✅ **Real-time Subscription**: Subscribes to Redis `btc_price_updates` channel
- ✅ **Live Price Display**: Shows BTC price updates as they arrive
- ✅ **Price Chart**: Visualizes price changes over time
- ✅ **Message Log**: Displays recent messages received
- ✅ **Metrics Dashboard**: Shows connection status and statistics

## How to Run

### Option 1: Direct Python (Recommended)

```bash
# Install Streamlit if not already installed
pip install streamlit plotly redis

# Run the dashboard
streamlit run streamlit_btc_pubsub_demo.py --server.port 8500
```

### Option 2: Docker

Add to `docker-compose-complete.yml`:

```yaml
streamlit-demo:
  build:
    context: .
    dockerfile: Dockerfile.streamlit
  container_name: octopus-streamlit-demo
  restart: unless-stopped
  ports:
    - "8500:8500"
  environment:
    - REDIS_URL=redis://redis:6379/0
  depends_on:
    - redis
  networks:
    - octopus-network
```

## Access

Once running, access the dashboard at:
- **URL**: http://localhost:8500

## How It Works

### 1. **Publisher** (Already Running)
- Celery task (`market_data.fetch_btc_price_realtime`) publishes to `btc_price_updates` every 5 seconds
- Located in: `src/data_processing/market_data_tasks.py`

### 2. **Subscriber** (This Dashboard)
- Subscribes to `btc_price_updates` channel
- Receives messages instantly via Redis Pub/Sub
- Updates UI in real-time

### 3. **Data Flow**

```
┌─────────────────┐
│  Celery Task    │
│  (Every 5s)     │
└────────┬────────┘
         │
         │ redis.publish('btc_price_updates', data)
         │
         ▼
┌─────────────────┐
│  Redis Channel  │
│ btc_price_      │
│ updates         │
└────────┬────────┘
         │
         │ Message broadcast
         │
         ▼
┌─────────────────┐
│ Streamlit Demo  │
│ (Port 8500)     │
│ Subscriber      │
└─────────────────┘
```

## UI Components

### Main Dashboard
- **Price Display**: Large, colorful BTC price with 24h change
- **Price Chart**: Real-time line chart showing price over time
- **Metrics**: Current price, change, source, message count

### Sidebar
- **Controls**: Start/Stop subscription buttons
- **Status**: Connection and subscription status 
- **Channel Info**: Redis channel details

### Message Log
- Shows last 10 messages received
- Expandable details for each message
- Timestamp and price information

## Benefits Demonstrated

1. **Real-time Updates**: No polling needed - instant delivery
2. **Low Latency**: Messages arrive immediately when published
3. **Scalability**: Multiple subscribers can listen to same channel
4. **Decoupling**: Publisher doesn't know about subscribers

## Troubleshooting

### No Messages Received?

1. **Check Celery Task**:
   ```bash
   docker logs octopus-celery-worker | grep "Published BTC"
   ```

2. **Check Redis Channel**:
   ```bash
   docker exec octopus-redis redis-cli PUBSUB CHANNELS
   docker exec octopus-redis redis-cli PUBSUB NUMSUB btc_price_updates
   ```

3. **Verify Redis Connection**:
   - Dashboard should show "✅ Connected to Redis" in sidebar
   - If not, ensure Redis is running on `localhost:6379`

### Messages Not Updating?

- Click "Stop Subscribing" then "Start Subscribing" again
- Check that Celery task is running and publishing
- Verify Redis connection is stable

## Comparison: Pub/Sub vs Polling

| Aspect | Pub/Sub (This Demo) | Polling |
|--------|---------------------|---------|
| **Latency** | Instant (0-5s) | Up to 5s delay |
| **Server Load** | Low (push only) | High (constant requests) |
| **Efficiency** | Efficient | Wasteful |
| **Real-time** | ✅ Yes | ❌ No |

## Next Steps

1. **Add More Subscribers**: Create additional services that subscribe to the same channel
2. **WebSocket Integration**: Bridge Redis Pub/Sub → WebSocket → Frontend
3. **Alert System**: Subscribe and send alerts when price changes significantly
4. **Analytics**: Subscribe and log all price updates to database

## Code Location

- **Dashboard**: `streamlit_btc_pubsub_demo.py`
- **Publisher**: `src/data_processing/market_data_tasks.py` (line 347-355)
- **Redis Config**: Uses `redis://localhost:6379/0`

