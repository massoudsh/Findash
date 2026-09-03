"""
Celery Tasks — ارزیابی دوره‌ای هشدار قیمت (#19) و سیاست ریسک (#22)

منطق واقعی ارزیابی در خود ماژول endpoint قرار دارد (price_alerts.py /
risk_policy.py) تا هم از طریق Celery beat (این فایل) و هم از طریق
endpoint دستی ادمین (POST /evaluate, /evaluate-all) قابل فراخوانی باشد
بدون تکرار کد. زمان‌بندی در beat_schedule داخل core/celery_app.py تنظیم شده.
"""

import asyncio
import logging

from src.core.celery_app import celery_app
from src.database.postgres_connection import SessionLocal

logger = logging.getLogger(__name__)


@celery_app.task(name="notifications.evaluate_price_alerts")
def evaluate_price_alerts_task():
    from src.api.endpoints.price_alerts import evaluate_price_alerts

    db = SessionLocal()
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            count = loop.run_until_complete(evaluate_price_alerts(db))
        finally:
            loop.close()
        logger.info(f"Price alerts evaluated: {count} triggered")
        return {"triggered": count}
    except Exception as e:
        logger.error(f"Price alert evaluation failed: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}
    finally:
        db.close()


@celery_app.task(name="notifications.evaluate_risk_policies")
def evaluate_risk_policies_task():
    from src.api.endpoints.risk_policy import evaluate_all_risk_policies

    db = SessionLocal()
    try:
        count = evaluate_all_risk_policies(db)
        logger.info(f"Risk policies evaluated: {count} new breaches")
        return {"new_breaches": count}
    except Exception as e:
        logger.error(f"Risk policy evaluation failed: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}
    finally:
        db.close()
