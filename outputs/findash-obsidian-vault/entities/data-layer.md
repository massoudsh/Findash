# Data Layer

> لایه داده پلتفرم اختاپوس — PostgreSQL، TimescaleDB، Redis، Kafka.

## مسئولیت‌ها
- ذخیره‌سازی دائمی داده‌های بازار، معاملات، کاربران
- کش کردن داده‌های ریل‌تایم با TTL
- استریمینگ رویدادهای بازار از طریق Kafka
- Pub/Sub برای هماهنگی Worker ها

## اجزاء

### PostgreSQL + TimescaleDB
- جداول: `market_data`, `portfolio`, `trades`, `users`, `ml_models`
- TimescaleDB برای time-series داده‌های بازار
- Migration با Alembic

### Redis Cache
- کلید: `market_data:{symbol}:latest`
- TTL: **300 ثانیه** (pipeline اصلی Kafka → Redis) | **60 ثانیه** (assets feature: `asset_service.py`)
- Pub/Sub کانال‌ها: `tasks:market_data:*`, `worker:*`

### Kafka
- Topic: `market-data-stream`
- Producer: انتشار رویداد بازار
- Consumer: دریافت و cache + trigger Celery task

## وابستگی‌ها
- [[entities/orchestrator]] — خواندن/نوشتن از Redis
- [[entities/backend]] — query های PostgreSQL
- [[concepts/data-pipeline]] — flow کامل داده

## منابع کد
- `MyProjects/Octopus/README.md:246` — دیاگرام Orchestrator و Redis
