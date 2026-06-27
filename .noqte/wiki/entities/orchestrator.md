# Orchestrator

> IntelligenceOrchestrator — هماهنگ‌کننده مرکزی ۱۱ AI Agent در پلتفرم اختاپوس.

## مسئولیت‌ها
- دریافت task از Redis Pub/Sub و توزیع بین Agentها
- مسیریابی وظایف با اولویت‌بندی
- خواندن نتایج از Redis Cache

## Agentهای فعال (M1–M5 مستند)
| Agent | عملکرد |
|-------|--------|
| M1: Data Collector | جمع‌آوری داده بازار |
| M2: Data Warehouse | ذخیره‌سازی و warehouse |
| M3: Real-time Processor | پردازش ریل‌تایم |
| M4: Strategy Agent | تحلیل و پیشنهاد استراتژی |
| M5: ML Models | اجرای مدل‌های یادگیری ماشین |

## مسیر داده
```
Kafka → Kafka Consumer → Redis Cache/Pub/Sub
→ Orchestrator → CeleryPubSubAllocator
→ Celery Worker (3 Worker) → PostgreSQL + Redis
```

## Celery Workers
| Worker | صف‌ها | وظایف |
|--------|--------|--------|
| Worker 1 | data_processing, portfolio | update_market_data |
| Worker 2 | ml_training, prediction | train_model, predict_price |
| Worker 3 | risk, strategies | calculate_var, execute_strategy |

## مانیتورینگ
- Flower: پورت `5555` — مانیتور task های Celery
- Prometheus: پورت `9090` — متریک‌ها
- Grafana: پورت `3001` — داشبورد

## وابستگی‌ها
- [[entities/data-layer]] — Redis Pub/Sub و PostgreSQL
- [[entities/backend]] — submit task

## منابع کد
- `MyProjects/Octopus/docs/orchestrator-architecture-detailed.md` — معماری کامل
- `MyProjects/Octopus/docs/orchestrator-architecture.md` — مرجع سریع
