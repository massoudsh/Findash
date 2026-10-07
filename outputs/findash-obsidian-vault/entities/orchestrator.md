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

## ✅ رفع‌شده (`b3f05f2`): agent-wiring کلاس `IntelligenceOrchestrator` (`src/core/intelligence_orchestrator.py`)
`__init__` اکنون واقعاً `self.strategy_agent = StrategyAgent(self.cache)` را می‌سازد (فقط به sklearn نیاز دارد، همیشه real). برای `ml_agent`/`prediction_agent`/`sentiment_agent` (به‌ترتیب `DeepLearningAgent`، `AdvancedPredictionAgent`، `MarketSentimentAgent`) import با `try/except ImportError` گارد شده — اگر `torch`/`prophet`+`cv2`/`transformers` نصب نباشند (مثل کانتینر sandbox فعلی)، fallback به کلاس‌های stub سبک (`_StubMLAgent`, `_StubPredictionAgent`, `_StubSentimentAgent`) که همان اسامی متد async را دارند (پس همچنان non-None و قابل `patch.object` هستند)، دقیقاً مطابق الگوی `CVXPY_AVAILABLE` در `src/portfolio/portfolio_manager.py`.
متدهای `_get_strategy_intelligence`/`_get_ml_intelligence`/`_get_prediction_intelligence`/`_get_sentiment_intelligence` اکنون واقعاً روی agent مربوطه صدا می‌زنند (`generate_trading_decision`, `ensemble_predict`+`detect_anomalies`, `generate_comprehensive_prediction`+`get_pattern_signals`, `get_sentiment_summary`). `_build_consensus` یک رأی‌گیری وزن‌دار واقعی است (`agent_weights` × confidence × جهت رأی؛ آستانه `±0.2` برای BUY/SELL/HOLD). `_calculate_unified_risk` و `_identify_uncertainty_factors` نیز منطق واقعی دارند (نه مقادیر ثابت). `generate_intelligence_report` اکنون گزارش را با `await self.cache.set(...)` کش می‌کند. نتیجه: تمام ۱۴ تست `test_intelligence_orchestrator.py` سبز.
`DeepLearningAgent`/`AdvancedPredictionAgent`/`MarketSentimentAgent` واقعی (نه stub) همچنان نیاز به نصب `torch`/`prophet`/`cv2`/`transformers` روی سرور واقعی SSH دارند (بیلد سنگین، خارج از scope کانتینر sandbox) — تا آن زمان orchestrator با نسخهٔ stub این ۳ ایجنت کار می‌کند (منطقاً صحیح، ولی بدون هوش واقعی ML/prediction/sentiment).

## ✅ رفع‌شده (2026-07-22): کرش `RuntimeError: no running event loop` هنگام نصب واقعی torch/transformers
چون `intelligence_orchestrator` یک singleton سطح-ماژول است (`src/core/intelligence_orchestrator.py:729`، در import-time ساخته می‌شود، قبل از اجرای هر event loop)، وقتی `torch` واقعاً نصب باشد (مسیر production/SSH واقعی، نه سناریوی همیشگی sandbox)، `DeepLearningAgent.__init__` (`src/training/transformer_models.py`) بدون گارد `asyncio.create_task(self._initialize_models())` صدا می‌زد → کل import اپ (`main_refactored.py`) کرش می‌کرد. الگوی مشابه در `MarketSentimentAgent.__init__` (`src/analytics/sentiment_agent.py`، `asyncio.create_task(self.finbert.initialize())`) برای `transformers`. هر دو با `try: asyncio.get_running_loop(); asyncio.create_task(...) except RuntimeError: log warning` گارد شدند — مصرف‌کننده‌های `self.models` از قبل چک `if model_name not in self.models` دارند، پس بی‌خطر. کشف این باگ فقط با نصب واقعی `torch` (که در sandbox معمولاً نصب نیست) ممکن بود.

## منابع کد
- `MyProjects/Octopus/docs/orchestrator-architecture-detailed.md` — معماری کامل
- `MyProjects/Octopus/docs/orchestrator-architecture.md` — مرجع سریع
- `src/core/intelligence_orchestrator.py` — پیاده‌سازی کد (agent wiring رفع‌شده، stub fallback برای ml/prediction/sentiment)
