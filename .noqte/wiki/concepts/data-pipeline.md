# Data Pipeline

> pipeline ورود داده بازار از منابع خارجی تا نمایش در frontend.

## مراحل
```
Market Data APIs → Kafka Producer → Kafka Topic (market-data-stream)
→ Kafka Consumer → Redis Cache (SETEX, TTL:300s)
→ Redis Pub/Sub (tasks:market_data:{symbol})
→ IntelligenceOrchestrator → CeleryPubSubAllocator
→ Celery Worker → ML Processing → TimescaleDB
→ WebSocket Broadcast → Frontend Display
```

## نکات مهم
- TTL کش Redis: 300 ثانیه
- Celery Worker 1: `update_market_data`
- Celery Worker 2: `train_model`, `predict_price`
- Celery Worker 3: `calculate_var`, `execute_strategy`
- نتایج از طریق WebSocket به frontend ارسال می‌شود

## اجزای درگیر
- [[entities/data-layer]] — Kafka، Redis، TimescaleDB
- [[entities/orchestrator]] — پردازش و routing
- [[entities/frontend]] — دریافت و نمایش

## منابع کد
- `MyProjects/Octopus/README.md:140` — flowchart pipeline
- `MyProjects/Octopus/README.md:282` — sequence diagram کامل
