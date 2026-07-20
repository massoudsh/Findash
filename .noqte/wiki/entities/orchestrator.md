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

## ⚠️ Gap شناخته‌شده: کلاس `IntelligenceOrchestrator` (`src/core/intelligence_orchestrator.py`)
این کلاس (استفاده‌شده در `tests/test_intelligence_orchestrator.py`) در `__init__` عمداً ۴ رفرنس ایجنت (`strategy_agent`, `ml_agent`, `prediction_agent`, `sentiment_agent`) را `None` می‌گذارد و `initialize_agents()` صرفاً `pass` است — هیچ‌کدام از ایجنت‌های واقعی هرگز واقعاً wire نمی‌شوند. متدهای `_get_strategy_intelligence`/`_get_ml_intelligence`/`_get_prediction_intelligence`/`_get_sentiment_intelligence`/`_build_consensus`/`_identify_uncertainty_factors` نیز فعلاً stub هستند (مقادیر ثابت برمی‌گردانند).
کلاس‌های واقعی موجودند و API‌شان دقیقاً با انتظار تست مطابقت دارد: `StrategyAgent` (`src/strategies/strategy_agent.py`)، `DeepLearningAgent` (`src/training/transformer_models.py`)، `AdvancedPredictionAgent` (`src/prediction/advanced_prediction_agent.py`)، `MarketSentimentAgent` (`src/analytics/sentiment_agent.py`) — همه `__init__(self, cache: TradingCache)` می‌گیرند. اما `DeepLearningAgent`/`MarketSentimentAgent` به `torch` و `AdvancedPredictionAgent` به `prophet` نیاز دارند که در کانتینر sandbox فعلی نصب نیستند (نصب‌شان سنگین/کند است؛ طبق قانون پروژه باید روی سرور SSH انجام شود). رفع کامل این gap یک تسک جداگانه است (نصب dependency روی سرور واقعی + پیاده‌سازی منطق واقعی متدهای stub بالا).

## منابع کد
- `MyProjects/Octopus/docs/orchestrator-architecture-detailed.md` — معماری کامل
- `MyProjects/Octopus/docs/orchestrator-architecture.md` — مرجع سریع
- `src/core/intelligence_orchestrator.py` — پیاده‌سازی کد (agent-wiring gap بالا)
